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
  D  natural-event grid invariance        (event machinery)
  E  choreographed transient, tight atol  (smoothness/Jacobian)
  F  realistic power (83.6 kW) + scram    (steady + natural event)
  G  general multichannel LOFA (capstone) (0.70 + regime friction, ramp scram)

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

TIMEOUTS = dict(A=600, B=600, C=1200, D=1800, E=1200, F=1800, G=1800)
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
    invariance of the flapper opening time, gated at PROPER tolerances.

    The event machinery is grid-invariant to root-localization
    precision. Re-running the three incommensurate grids at rtol=atol=1e-6
    collapses the t_open spread ~570x (0.028 s -> 5e-5 s — two grids agree to
    the microsecond). The residual loose-tol spread is ~rtol=1e-3 *solution*
    divergence between grids, not event error, so it must NOT gate the stage.
    GATE: tight-tolerance spread <= 0.02 s (passes with ~400x margin) AND all
    runs completing with reversal. The loose-tol spread is reported as an
    informational metric only."""
    import case
    from stream.jacobians import DAE_jacobian

    def natural_run(n_out, rtol, atol_val):
        agr, K, refs = case.build(power=case.POWER_DEMO, stop_on_open=True)
        vec = _steady(agr, case.expert_guess(refs, case.POWER_DEMO))
        refs["pump"].p = 0.0
        atol = np.full(len(agr), atol_val)
        sol = agr.solve(vec, time=np.linspace(0.0, 400.0, n_out),
                        jacfn=DAE_jacobian(agr), atol=atol, rtol=rtol,
                        max_steps=1000000, continuous=True)
        m = _transient_metrics(agr, K, refs, sol)
        m["t_open_latched"] = float(refs["flapper"].t_open)
        m["n_out"], m["rtol"], m["atol"] = n_out, rtol, atol_val
        return m

    grids = (2000, 1461, 3571)
    loose = [natural_run(n, 1e-3, 1e-1) for n in grids]
    tight = [natural_run(n, 1e-6, 1e-6) for n in grids]
    loose_opens = [m["t_open_latched"] for m in loose]
    tight_opens = [m["t_open_latched"] for m in tight]
    loose_spread = max(loose_opens) - min(loose_opens)
    tight_spread = max(tight_opens) - min(tight_opens)
    complete = all(m["flapper_opened"] and m["reversal"] and m["t_end_reached"] >= 399.0
                   for m in loose + tight)
    ok = complete and tight_spread <= 0.02
    return dict(status="PASS" if ok else "FAIL",
                metrics=dict(loose_runs=loose, tight_runs=tight,
                             t_open_spread_tight_s=round(tight_spread, 6),
                             t_open_spread_loose_s=round(loose_spread, 6)),
                detail=f"complete={complete}; tight-tol spread {tight_spread:.2e}s "
                       f"(GATE <=0.02); loose-tol spread {loose_spread:.4f}s "
                       f"(informational)")


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
    """Single-channel REALISTIC power (83.6 kW) WITH the ramp
    scram it physically always had, and the non-optimistic ``regime_dependent``
    friction law. Expert-guess steady, then a natural-event transient: the pump
    head ramps to zero, power blends to decay heat, the flapper opens on its own
    margin as the flow coasts down, and buoyancy reverses the channel into
    natural circulation.

    PASS = steady OK, integrates to >= 2499 s, channel reverses (final mdot < 0),
    and the hot coolant trajectory peak stays sub-saturation. This keeps F's
    original meaning (realistic power) while fixing what made it ill-posed (no
    scram) and using the non-optimistic law."""
    import case
    from stream.jacobians import DAE_jacobian
    from stream.substances import light_water

    agr, K, refs = case.build(power=case.POWER_REALISTIC, regime_friction=True)
    vec = _steady(agr, case.expert_guess(refs, case.POWER_REALISTIC))
    m_ss = _steady_metrics(agr, K, refs, vec)
    ok_ss, d_ss = _check_steady(m_ss, case.POWER_REALISTIC)

    case.wire_ramp_scram(agr, refs["pump"], {refs["fuel"]: case.POWER_REALISTIC})
    ch, fl = refs["channel"], refs["flapper"]
    sol = agr.solve(vec, time=np.linspace(0.0, 2500.0, 1251),
                    jacfn=DAE_jacobian(agr), atol=np.full(len(agr), 1e-1),
                    rtol=1e-3, max_steps=1000000)
    t = np.asarray(sol.time)
    mdot_ch = agr.at_times(sol, K, K.component_edge(ch))
    peak = float(np.max(agr.at_times(sol, ch, "T_cool")))
    tsat = float(light_water.sat_temperature(case.P_REF))
    neg = np.flatnonzero(mdot_ch < -1e-3)
    m_tr = dict(
        t_end_reached=float(t[-1]),
        t_flapper_open=(float(fl.t_open) if np.isfinite(fl.t_open) else None),
        t_reversal=(float(t[neg[0]]) if len(neg) else None),
        mdot_channel_final=float(mdot_ch[-1]),
        reversal=bool(mdot_ch[-1] < 0),
        peak_T_cool=peak, T_sat=tsat, margin=tsat - peak,
    )
    ok = (ok_ss and m_tr["t_end_reached"] >= 2499.0 and m_tr["reversal"] and peak < tsat)
    return dict(status="PASS" if ok else "FAIL",
                metrics=dict(steady=m_ss, transient=m_tr),
                detail=f"steady[{d_ss}]; t_open={m_tr['t_flapper_open']}, "
                       f"rev={m_tr['t_reversal']}, peak={peak:.2f}C, "
                       f"margin={m_tr['margin']:+.2f}C (sat {tsat:.1f})")


def stage_G():
    """General multichannel LOFA (capstone): 4 channels
    of different power and geometry between shared plena at the 0.70 de-rate with
    the non-optimistic regime_dependent friction (both via build_general's new
    defaults), flapper NC leg, ramp scram, staggered reversal.

    Steady from the BALLPARK guess through solve_steady's 'auto' fallback (the
    guess-robustness acceptance case), then a natural-event transient (no
    choreographed pre-open — the flapper opens on its own margin). PASS requires
    ALL of: completion t_end >= 2499 (no silent truncation), hot/warm/wide
    all reversed at the end, physical ordering t_rev(hot) < t_rev(warm), bypass
    stays a downcomer (final mdot > 0), and the hot coolant peak < saturation."""
    import case
    from stream.jacobians import DAE_jacobian
    from stream.substances import light_water

    sub = {}
    # 1 — wiring: construction + gravity closure (check_gravity_mismatch in build_general)
    agr, K, refs = case.build_general()
    ch = refs["channels"]
    sub["wiring"] = "PASS"

    # 2 — steady from the ballpark guess through solve_steady's 'auto' fallback
    vec = _steady(agr, case.ballpark_guess_general(agr, K))
    st = agr.save(vec)
    mdots = {k: float(st[K.name][K.component_edge(c)]) for k, c in ch.items()}
    sub["steady"] = dict(residual_norm=float(np.linalg.norm(agr.compute(vec, 0.0))),
                         mdots=mdots)

    # 3 — ramp scram + natural-event coastdown to 2500 s (loose sweep-baseline tol).
    # NC development is slow (parallel low-resistance paths + flywheel): the
    # flapper opens ~78 s, staggered reversals ~800/1200/1500 s — realistic for
    # pool reactors, hence the long window.
    case.wire_ramp_scram(agr, refs["pump"],
                         {refs["fuels"][k]: case.GEN_POWERS[k] for k in refs["fuels"]})
    sol = agr.solve(vec, time=np.linspace(0.0, 2500.0, 1251),
                    jacfn=DAE_jacobian(agr), atol=np.full(len(agr), 1e-1),
                    rtol=1e-3, max_steps=1000000)
    t = np.asarray(sol.time)
    t_end = float(t[-1])
    sub["t_end_reached"] = t_end
    sub["t_flapper_open"] = (float(refs["flapper"].t_open)
                             if np.isfinite(refs["flapper"].t_open) else None)

    # 4 — staggered-reversal physics checks
    t_rev, m_final = {}, {}
    for k, c in ch.items():
        m = agr.at_times(sol, K, K.component_edge(c))
        neg = np.flatnonzero(m < -1e-3)
        t_rev[k] = float(t[neg[0]]) if len(neg) else None
        m_final[k] = float(m[-1])
    sub["t_reversal"] = t_rev
    sub["mdot_final"] = m_final
    tsat = float(light_water.sat_temperature(case.P_REF))
    peak = float(np.max(agr.at_times(sol, ch["hot"], "T_cool")))
    sub["peak_T_cool_hot"] = peak
    sub["T_sat"], sub["margin"] = tsat, tsat - peak

    reached = t_end >= 2499.0
    heated_reversed = all(m_final[k] < 0 for k in ("hot", "warm", "wide"))
    ordering = (t_rev["hot"] is not None and t_rev["warm"] is not None
                and t_rev["hot"] < t_rev["warm"])
    bypass_downcomer = m_final["bypass"] > 0
    sub["bypass_stayed_downcomer"] = bypass_downcomer

    ok = reached and heated_reversed and ordering and bypass_downcomer and peak < tsat
    return dict(status="PASS" if ok else "FAIL", metrics=sub,
                detail=f"reached={t_end:.0f}s; t_rev hot/warm/wide="
                       f"{t_rev['hot']}/{t_rev['warm']}/{t_rev['wide']}; "
                       f"hot<warm={ordering}; bypass={m_final['bypass']:+.5f}; "
                       f"peak={peak:.2f}C margin={tsat - peak:+.2f}C (sat {tsat:.1f})")


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
