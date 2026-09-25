"""
LOFA benchmark runner — graded ladder of solver-robustness stages.

Each stage is run in its OWN subprocess with a timeout (some baseline defects
hang forever, e.g. the continuous-mode restart loop), and writes a JSON record
to results/<stage>.json. `--all` runs every stage and appends a dated section
to RESULTS.md (untracked) keyed to the current git describe.

Stages (see README.md for the expectation table):
  A  ballpark-guess steady solve          (guess robustness)
  B  expert-guess steady solve            (non-regression guard)
  C  choreographed transient, loose atol  (non-regression guard — legacy recipe)
  D  natural-event transient              (event machinery)
  E  choreographed transient, tight atol  (smoothness/Jacobian)
  F  realistic power (84 kW)              (steady + transient)

Usage:
  conda run -n stream-env python benchmarks/lofa/run.py --stage C
  conda run -n stream-env python benchmarks/lofa/run.py --all
"""
import argparse
import json
import os
import subprocess
import sys
import time as _time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS_DIR = os.path.join(HERE, "results")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TIMEOUTS = dict(A=600, B=600, C=1200, D=1200, E=1200, F=1800, G=1800)
STAGES = "ABCDEFG"


# ── helpers ───────────────────────────────────────────────────────────────────

def _steady(agr, guess):
    from stream.jacobians import ALG_jacobian
    vec = agr.solve_steady(guess, jac=ALG_jacobian(agr))
    return vec


def _steady_metrics(agr, K, refs, vec):
    st = agr.save(vec)
    ch, fu = refs["channel"], refs["fuel"]
    return dict(
        mdot_channel=float(st[K.name][K.component_edge(ch)]),
        T_outlet=float(np.asarray(st[ch.name]["T_cool"])[-1]),
        T_wall_outlet=float(np.asarray(st[fu.name]["T_wall_left"])[-1]),
        h_left_outlet=float(np.asarray(st[ch.name]["h_left"])[-1]),
    )


def _check_steady(m, power):
    import case
    dT = power / (MDOT_CP := case.MDOT0 * 4179)
    ok_T = abs(m["T_outlet"] - (case.TIN + dT)) < 10.0
    ok_wall = m["T_wall_outlet"] > m["T_outlet"]
    return ok_T and ok_wall, f"T_out ok={ok_T}, wall>cool={ok_wall}"


def _transient_metrics(agr, K, refs, sol):
    ch, fl = refs["channel"], refs["flapper"]
    t = sol.time
    mdot_ch = agr.at_times(sol, K, K.component_edge(ch))
    mdot_fl = agr.at_times(sol, K, K.component_edge(fl))
    t_wall = agr.at_times(sol, refs["fuel"], "T_wall_left")[:, -1]
    open_idx = np.flatnonzero(np.abs(mdot_fl) > 0.005)
    return dict(
        t_end_reached=float(t[-1]),
        mdot_channel_final=float(mdot_ch[-1]),
        flapper_opened=bool(len(open_idx)),
        t_flapper_open=(float(t[open_idx[0]]) if len(open_idx) else None),
        peak_T_wall=float(np.max(t_wall)),
        reversal=bool(mdot_ch[-1] < 0),
    )


def _choreographed_transient(agr, K, refs, steady_vec, atol_val, rtol):
    """The legacy recipe: coast down with flapper closed, manually pre-open at
    the estimated time, continue through opening and reversal. Avoids the
    event machinery entirely (that is what stage D exercises)."""
    import case
    from stream.jacobians import DAE_jacobian
    from stream.aggregator.solution import Solution
    from stream.solvers import TransientRuntimeError

    refs["pump"].p = 0.0
    atol = np.full(len(agr), atol_val)
    t1_end = case.T_OPEN_ESTIMATE - 2.0

    sol1 = agr.solve(steady_vec, time=np.linspace(0.0, t1_end, 500),
                     jacfn=DAE_jacobian(agr), atol=atol, rtol=rtol)

    fl = refs["flapper"]
    fl.t_open = np.inf
    fl._flag = False
    fl.stop_on_open = False
    fl.open(case.T_OPEN_ESTIMATE)

    partial = None
    try:
        sol2 = agr.solve(sol1.data[-1], time=np.linspace(t1_end, 400.0, 1500),
                         jacfn=DAE_jacobian(agr), atol=atol, rtol=rtol,
                         max_steps=100000)
    except TransientRuntimeError as e:
        partial = f"phase-2 TransientRuntimeError at t={e.t[-1]:.2f}s: {e.message}" \
            if e.t is not None else f"phase-2 TransientRuntimeError (no partial data): {e}"
        sol2 = Solution(e.t, e.y) if e.t is not None else sol1
    m = _transient_metrics(agr, K, refs, sol2)
    return m, partial


# ── stages ────────────────────────────────────────────────────────────────────

def stage_A():
    """Ballpark-guess steady solve at REALISTIC power (no expert thermal pre-solve).

    At demo power (1 kW) the system is near-isothermal and a uniform guess is
    trivially close to the solution (verified: passes at baseline). The real
    guess-robustness question is at realistic power, where the expert guess
    works (stage F's steady sub-step) but a ballpark guess must too."""
    import case
    agr, K, refs = case.build(power=case.POWER_REALISTIC)
    guess = case.ballpark_guess(agr, K, refs)
    vec0 = agr.load(guess)  # load failures surface here
    vec = _steady(agr, guess)
    m = _steady_metrics(agr, K, refs, vec)
    ok, detail = _check_steady(m, case.POWER_REALISTIC)
    return dict(status="PASS" if ok else "FAIL", metrics=m, detail=detail)


def stage_B():
    """Expert-guess steady solve (legacy recipe) — must always pass."""
    import case
    agr, K, refs = case.build(power=case.POWER_DEMO)
    vec = _steady(agr, case.expert_guess(refs, case.POWER_DEMO))
    m = _steady_metrics(agr, K, refs, vec)
    ok, detail = _check_steady(m, case.POWER_DEMO)
    return dict(status="PASS" if ok else "FAIL", metrics=m, detail=detail)


def stage_C():
    """Choreographed transient at loose atol=1e-1 (legacy recipe) — must always pass."""
    import case
    agr, K, refs = case.build(power=case.POWER_DEMO)
    vec = _steady(agr, case.expert_guess(refs, case.POWER_DEMO))
    m, partial = _choreographed_transient(agr, K, refs, vec, atol_val=1e-1, rtol=1e-3)
    ok = m["flapper_opened"] and m["reversal"] and partial is None
    return dict(status="PASS" if ok else "FAIL", metrics=m,
                detail=partial or "opened + reversed")


def stage_D():
    """Natural event path (stop_on_open=True + continuous=True): output-grid
    invariance of the flapper opening time.

    The defect this stage exposes: the boolean root function cannot be localized
    by IDA, so the opening time is latched at whatever evaluation point first
    sees the condition — verified at baseline to be exact OUTPUT-GRID points,
    i.e. the physics depends on how densely the user asked for plot points.
    PASS requires all runs completing with reversal AND the latched t_open
    agreeing across three incommensurate output grids to within 0.02 s."""
    import case
    from stream.jacobians import DAE_jacobian

    def natural_run(n_out):
        agr, K, refs = case.build(power=case.POWER_DEMO, stop_on_open=True)
        vec = _steady(agr, case.expert_guess(refs, case.POWER_DEMO))
        refs["pump"].p = 0.0
        atol = np.full(len(agr), 1e-1)
        sol = agr.solve(vec, time=np.linspace(0.0, 400.0, n_out),
                        jacfn=DAE_jacobian(agr), atol=atol, rtol=1e-3,
                        max_steps=100000, continuous=True)
        m = _transient_metrics(agr, K, refs, sol)
        m["t_open_latched"] = float(refs["flapper"].t_open)
        m["n_out"] = n_out
        return m

    runs = [natural_run(n) for n in (2000, 1461, 3571)]
    t_opens = [m["t_open_latched"] for m in runs]
    spread = max(t_opens) - min(t_opens)
    complete = all(m["flapper_opened"] and m["reversal"] and m["t_end_reached"] >= 399.0
                   for m in runs)
    ok = complete and spread <= 0.02
    return dict(status="PASS" if ok else "FAIL",
                metrics=dict(runs=runs, t_open_spread_s=round(spread, 4)),
                detail=f"complete={complete}; t_open latched at "
                       f"{'/'.join(f'{t:.3f}' for t in t_opens)}s across grids "
                       f"(spread {spread:.3f}s, must be <=0.02)")


def stage_E():
    """Choreographed transient at proper tolerances (atol=rtol=1e-6)."""
    import case
    agr, K, refs = case.build(power=case.POWER_DEMO)
    vec = _steady(agr, case.expert_guess(refs, case.POWER_DEMO))
    m, partial = _choreographed_transient(agr, K, refs, vec, atol_val=1e-6, rtol=1e-6)
    ok = m["flapper_opened"] and m["reversal"] and partial is None
    return dict(status="PASS" if ok else "FAIL", metrics=m,
                detail=partial or "opened + reversed at tight tolerance")


def stage_F():
    """Realistic power (83.6 kW): expert steady, then choreographed transient."""
    import case
    agr, K, refs = case.build(power=case.POWER_REALISTIC)
    vec = _steady(agr, case.expert_guess(refs, case.POWER_REALISTIC))
    m_ss = _steady_metrics(agr, K, refs, vec)
    ok_ss, d_ss = _check_steady(m_ss, case.POWER_REALISTIC)
    m_tr, partial = _choreographed_transient(agr, K, refs, vec, atol_val=1e-1, rtol=1e-3)
    ok = ok_ss and m_tr["flapper_opened"] and m_tr["reversal"] and partial is None
    return dict(status="PASS" if ok else "FAIL",
                metrics=dict(steady=m_ss, transient=m_tr),
                detail=f"steady[{d_ss}]; transient[{partial or 'opened + reversed'}]")


def stage_G():
    """General multichannel LOFA (capstone): 4 channels of different power and
    geometry between shared plena, flapper NC leg, scram decay power, staggered
    reversal. Sub-stages recorded independently; PASS requires all of them plus
    the physically-provable ordering t_rev(hot) < t_rev(warm)."""
    import case
    from stream.jacobians import DAE_jacobian
    from stream.aggregator.solution import Solution
    from stream.solvers import TransientRuntimeError

    sub = {}
    # 1 — wiring: construction + gravity closure (must pass even at baseline)
    agr, K, refs = case.build_general()
    sub["wiring"] = "PASS"

    # 2 — expert-guess steady state at full power (167.6 kW total)
    vec = _steady(agr, case.expert_guess_general(refs))
    st = agr.save(vec)
    ch = refs["channels"]
    mdots = {k: float(st[K.name][K.component_edge(c)]) for k, c in ch.items()}
    t_out_hot = float(np.asarray(st[ch["hot"].name]["T_cool"])[-1])
    t_wall_hot = float(np.asarray(st[refs["fuels"]["hot"].name]["T_wall_left"])[-1])
    sub["steady"] = dict(mdots=mdots, T_out_hot=t_out_hot, T_wall_hot=t_wall_hot,
                         ok=bool(t_wall_hot > t_out_hot and mdots["hot"] > 0))

    # 3 — scram + coastdown transient (choreographed pre-open, legacy loose atol).
    # NC development is slow here (parallel low-resistance paths + flywheel):
    # opening ~100 s, staggered reversals at ~1000/1500/2000 s — physically
    # realistic for pool reactors, hence the long 2600 s window.
    case.scram(agr, refs)
    atol = np.full(len(agr), 1e-1)
    t_open = case.gen_t_open_from(mdots_pump := float(st[K.name][K.component_edge(refs["pump"])]))
    sub["t_open_used"] = round(t_open, 1)
    t1_end = t_open - 2.0
    partial = None
    try:
        sol1 = agr.solve(vec, time=np.linspace(0.0, t1_end, 300),
                         jacfn=DAE_jacobian(agr), atol=atol, rtol=1e-3)
    except TransientRuntimeError as e:
        partial = (f"phase-1 coastdown (flapper closed) died at t={e.t[-1]:.2f}s: {e.message}"
                   if e.t is not None else f"phase-1 coastdown died (no partial): {e}")
        sub["transient"] = partial
        sub["t_reversal"] = None
        return dict(status="FAIL", metrics=sub,
                    detail=f"steady ok={sub['steady']['ok']}; {partial}")
    fl = refs["flapper"]
    fl.t_open, fl._flag, fl.stop_on_open = np.inf, False, False
    fl.open(t_open)
    try:
        sol2 = agr.solve(sol1.data[-1], time=np.linspace(t1_end, 2600.0, 2000),
                         jacfn=DAE_jacobian(agr), atol=atol, rtol=1e-3,
                         max_steps=200000)
    except TransientRuntimeError as e:
        partial = (f"transient died at t={e.t[-1]:.2f}s: {e.message}"
                   if e.t is not None else f"transient died (no partial): {e}")
        sol2 = Solution(e.t, e.y) if e.t is not None else sol1
    sub["transient"] = partial or f"completed to t={float(sol2.time[-1]):.1f}s"

    # 4 — staggered-reversal physics checks
    t2 = sol2.time
    t_rev, m_final = {}, {}
    for k, c in ch.items():
        m = agr.at_times(sol2, K, K.component_edge(c))
        idx = np.flatnonzero(m < -1e-4)
        t_rev[k] = float(t2[idx[0]]) if len(idx) else None
        m_final[k] = float(m[-1])
    sub["t_reversal"] = t_rev
    sub["mdot_final"] = m_final
    heated_reversed = all(t_rev[k] is not None for k in ("hot", "warm", "wide"))
    ordering = (heated_reversed and t_rev["hot"] < t_rev["warm"])
    bypass_downcomer = t_rev["bypass"] is None and m_final["bypass"] > 0
    sub["bypass_stayed_downcomer"] = bypass_downcomer
    peak_T_wall = float(np.max(agr.at_times(sol2, refs["fuels"]["hot"], "T_wall_left")))
    sub["peak_T_wall_hot"] = peak_T_wall

    ok = (sub["steady"]["ok"] and partial is None and heated_reversed
          and ordering and bypass_downcomer)
    return dict(status="PASS" if ok else "FAIL", metrics=sub,
                detail=f"steady ok={sub['steady']['ok']}; {sub['transient']}; "
                       f"t_rev={t_rev}; hot<warm={ordering}")


# ── orchestration ─────────────────────────────────────────────────────────────

def run_stage_inprocess(stage: str) -> dict:
    t0 = _time.time()
    try:
        rec = {"A": stage_A, "B": stage_B, "C": stage_C, "D": stage_D,
               "E": stage_E, "F": stage_F, "G": stage_G}[stage]()
    except Exception:
        rec = dict(status="CRASH", metrics={},
                   detail=traceback.format_exc(limit=8).strip().splitlines()[-1],
                   trace=traceback.format_exc(limit=20))
    rec.update(stage=stage, wall_s=round(_time.time() - t0, 1))
    return rec


def run_stage_subprocess(stage: str) -> dict:
    t0 = _time.time()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--stage", stage],
                           capture_output=True, text=True, timeout=TIMEOUTS[stage])
        path = os.path.join(RESULTS_DIR, f"{stage}.json")
        if os.path.exists(path):
            return json.load(open(path))
        return dict(stage=stage, status="CRASH", metrics={}, wall_s=round(_time.time() - t0, 1),
                    detail=(p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else "no output")
    except subprocess.TimeoutExpired:
        rec = dict(stage=stage, status="TIMEOUT", metrics={},
                   wall_s=TIMEOUTS[stage], detail=f"killed after {TIMEOUTS[stage]}s")
        json.dump(rec, open(os.path.join(RESULTS_DIR, f"{stage}.json"), "w"), indent=1)
        return rec


def _git_describe():
    def g(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return f"{g('describe', '--tags', '--always')} ({g('rev-parse', '--abbrev-ref', 'HEAD')} @ {g('rev-parse', '--short', 'HEAD')})"


def write_report(records):
    import datetime
    lines = [f"\n## {datetime.date.today().isoformat()} — `{_git_describe()}`\n",
             "| Stage | Status | Wall (s) | Detail |", "|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['stage']} | **{r['status']}** | {r['wall_s']} | {r['detail']} |")
    lines.append("")
    for r in records:
        if r.get("metrics"):
            lines.append(f"- **{r['stage']}** metrics: `{json.dumps(r['metrics'])}`")
    with open(os.path.join(HERE, "RESULTS.md"), "a") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the RESULTS.md section from existing results/*.json")
    args = ap.parse_args()

    if args.report:
        records = [json.load(open(os.path.join(RESULTS_DIR, f"{s}.json"))) for s in STAGES]
        write_report(records)
        print("Report appended to RESULTS.md")
        sys.exit(0)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.stage:
        rec = run_stage_inprocess(args.stage)
        json.dump(rec, open(os.path.join(RESULTS_DIR, f"{args.stage}.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "trace"}))
        sys.exit(0)

    if args.all:
        records = []
        for s in STAGES:
            print(f"── stage {s} (timeout {TIMEOUTS[s]}s)...", flush=True)
            r = run_stage_subprocess(s)
            print(f"   {r['status']} in {r['wall_s']}s — {r['detail']}", flush=True)
            records.append(r)
        write_report(records)
        print("Report appended to RESULTS.md")
