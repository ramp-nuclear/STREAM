"""Rungs of the pool-reactor loss-of-flow case. ``--rung R`` runs one rung in this process
and writes results/<rung>.json; ``--all`` runs each rung in its own spawned process with a
timeout and appends a dated table to RESULTS.md. Rung 6 always runs its configurations in
separate subprocesses."""
import argparse
import datetime
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import time as _time
import traceback
from pathlib import Path

import numpy as np

from stream.composition import guess_steady_state
from stream.examples.lofa import HEATED, build, guess, steady, steady_checks
from stream.examples.lofa.params import NC_SIGNS, Params

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lofa_bench_metrics", HERE / "metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

RESULTS_DIR = HERE / "results"

RUNGS = ("1", "2", "3", "4", "5", "6")
CONFIG_TIMEOUT = 3600
TIMEOUTS = {"1": 600, "2": 600, "3": 2400, "4": 1800, "5": 2400, "6": 4 * CONFIG_TIMEOUT + 300}
TITLES = {"1": "steady, ballpark guess", "2": "steady, library guess", "3": "both pumps trip, 1 h", "4": "one pump trips, 30 min",
          "5": "stuck flapper", "6": "event-time invariance"}


def _library_guess(agr, K, refs) -> np.ndarray:
    return agr.load(guess_steady_state(agr, K, overrides={refs["trip"].name: {"flag": 0.0}}))


def _steady_rung(kind: str) -> tuple[dict, np.ndarray]:
    agr, K, refs = build(Params())
    y0 = _library_guess(agr, K, refs) if kind == "library" else guess(agr, K, refs, kind)
    y = steady(agr, y0)
    chk = steady_checks(agr, K, refs, y)
    ok = chk["T_out_error_max"] < 0.5 and chk["split_error_max"] < 0.15 and chk["pool_flow"] * NC_SIGNS["pool"] < 0 \
        and chk["inlet_matches_pool"] and chk["wall_above_coolant"]
    m = {"T_out_error_max": chk["T_out_error_max"], "split_error_max": chk["split_error_max"], "pool_flow": chk["pool_flow"],
         "head": chk["heads"]["A"], "mdot": chk["mdot"], "residual_entry": float(np.linalg.norm(agr.compute(y0, 0.0))),
         "residual_steady": float(np.linalg.norm(agr.compute(y, 0.0)))}
    return {"status": "PASS" if ok else "FAIL", "metrics": m,
            "detail": f"T_out err {chk['T_out_error_max']:.2f} C, split err {chk['split_error_max']:.3f}, head {chk['heads']['A']:.0f} Pa"}, y


def rung_1():
    return _steady_rung("ballpark")[0]


def rung_2():
    return _steady_rung("library")[0]


def _result_path(rung: str) -> str:
    return os.path.join(RESULTS_DIR, f"{rung}.json")


def _record_rung(p, t_end, n, passes, label):
    rec = metrics.run(p, t_end=t_end, n=n)
    ok, reasons = True, []
    for name, cond in passes(rec).items():
        if not cond:
            ok, reasons = False, reasons + [name]
    detail = f"{label}: status {rec.status}; " + ("all conditions hold" if ok else "failed " + ", ".join(reasons))
    if rec.error:
        detail += f"; {rec.error}"
    return {"status": "PASS" if ok else "FAIL", "metrics": {k: v for k, v in rec.__dict__.items() if k != "traceback"}, "detail": detail}


def rung_3():
    def passes(r):
        return {
            "completed": r.status == "completed", "reached": r.t_last >= 3600.0 * (1 - 1e-9), "no_warnings": r.n_warnings == 0,
            "tripped": r.t_trip is not None, "opened": r.t_open is not None,
            "valves": r.t_valve_A is not None and r.t_valve_B is not None,
            "reversed_all": all(getattr(r, f"t_rev_{k}") is not None for k in HEATED),
            "buoyancy_order": r.reversal_in_buoyancy_order,
            "pool_downcomer": r.final_pool * NC_SIGNS["pool"] > 0 and r.final_flapper * NC_SIGNS["flapper"] > 0,
            "margin": r.margin > 0,
            "legs_shut": abs(r.final_leg_A) < 0.01 * r.mdot_total / 2 and abs(r.final_leg_B) < 0.01 * r.mdot_total / 2,
        }
    r = _record_rung(Params(scenario="both"), 3600.0, 1801, passes, "both pumps")
    m = r["metrics"]
    half = "never" if m["t_half"] is None else f"{m['t_half']:.1f} s"
    r["detail"] += f"; riser flow halves at {half}, flywheel estimate {m['t_half_hand']:.1f} s after the pump trip"
    return r


def rung_4():
    def passes(r):
        return {
            "completed": r.status == "completed", "reached": r.t_last >= 1800.0 * (1 - 1e-9), "no_warnings": r.n_warnings == 0,
            "valve_B_shut": r.t_valve_B is not None, "valve_A_open": r.t_valve_A is None,
            "flapper_shut": r.t_open is None, "leg_B_leak": abs(r.final_leg_B) < 0.01 * r.mdot_total / 2,
            "settled": abs(r.final_leg_A) > 0.5 * r.mdot_total,
        }
    return _record_rung(Params(scenario="one"), 1800.0, 901, passes, "one pump")


def rung_5():
    def passes(r):
        return {"saturation": r.status == "saturation", "before_end": r.t_last < 3600.0,
                "named": ("SaturationReachedError" in r.error)}
    return _record_rung(Params(scenario="stuck", flapper_stuck=True), 3600.0, 1801, passes, "stuck flapper")


def _stderr_tail(stderr, lines: int = 30) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return "\n".join([ln for ln in (stderr or "").splitlines() if ln.strip()][-lines:])


def _isolated_row(atol_rel: float, n: int, t_end: float) -> dict:
    out = _result_path(f"6-{atol_rel:g}-{n}")
    if os.path.exists(out):
        os.remove(out)
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
    cmd = [sys.executable, str(HERE / "metrics.py"), "--params", "{}", "--out", out, "--atol-rel", str(atol_rel),
           "--rtol", str(atol_rel), "--t-end", str(t_end), "--n", str(n)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CONFIG_TIMEOUT, env=env, check=False)
    except subprocess.TimeoutExpired as e:
        tail = _stderr_tail(e.stderr)
        return {"status": "timeout", "t_trip": None, "t_open": None, "wall_s": CONFIG_TIMEOUT, "error": tail, "traceback": tail}
    if not os.path.exists(out):
        tail = _stderr_tail(proc.stderr)
        return {"status": "crash", "t_trip": None, "t_open": None, "wall_s": None,
                "error": f"no record written, exit code {proc.returncode}: {tail}", "traceback": tail}
    with open(out) as fh:
        return json.load(fh)


def rung_6():
    rows = {(a, n): _isolated_row(a, n, 300.0) for a in (1e-4, 1e-5) for n in (151, 601)}
    def spread(key, atol_rel):
        vals = [rows[(atol_rel, n)].get(key) for n in (151, 601)]
        return None if any(v is None for v in vals) else abs(vals[0] - vals[1])
    def bracket(atol_rel):
        vals = [rows[(atol_rel, n)].get("t_open") for n in (151, 601)]
        return None if any(v is None for v in vals) else 10 * atol_rel * max(vals)
    def shift(key):
        a, b = rows[(1e-4, 601)].get(key), rows[(1e-5, 601)].get(key)
        return None if a is None or b is None else abs(a - b)
    m = {"trip_grid_spread": {a: spread("t_trip", a) for a in (1e-4, 1e-5)},
         "open_grid_spread": {a: spread("t_open", a) for a in (1e-4, 1e-5)},
         "grid_bracket": {a: bracket(a) for a in (1e-4, 1e-5)},
         "trip_tolerance_shift": shift("t_trip"), "open_tolerance_shift": shift("t_open"),
         "statuses": {f"{a:g}, {n}": r["status"] for (a, n), r in rows.items()},
         "errors": {f"{a:g}, {n}": r.get("error", "") for (a, n), r in rows.items() if r["status"] != "completed"},
         "walls": {f"{a:g}, {n}": r.get("wall_s") for (a, n), r in rows.items()}}
    def within(key, a):
        return m[key][a] is not None and m["grid_bracket"][a] is not None and m[key][a] < m["grid_bracket"][a]
    ok = all(r["status"] == "completed" for r in rows.values()) \
        and all(within("trip_grid_spread", a) and within("open_grid_spread", a) for a in (1e-4, 1e-5)) \
        and m["trip_tolerance_shift"] is not None and m["trip_tolerance_shift"] < 0.1 \
        and m["open_tolerance_shift"] is not None and m["open_tolerance_shift"] < 0.1
    return {"status": "PASS" if ok else "FAIL", "metrics": m, "detail": json.dumps(m)}


def run_rung_inprocess(rung: str) -> dict:
    match rung:
        case "1":
            fn = rung_1
        case "2":
            fn = rung_2
        case "3":
            fn = rung_3
        case "4":
            fn = rung_4
        case "5":
            fn = rung_5
        case "6":
            fn = rung_6
        case _:
            raise ValueError(f"rung must be one of {RUNGS}, got {rung!r}")
    start = _time.perf_counter()
    try:
        rec = fn()
    except Exception as e:  # noqa: BLE001
        rec = {"status": "CRASH", "metrics": {}, "detail": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    rec["wall_s"] = _time.perf_counter() - start
    rec["rung"], rec["title"] = rung, TITLES[rung]
    return rec


def _child(rung: str, out: str) -> None:
    rec = run_rung_inprocess(rung)
    with open(out, "w") as f:
        json.dump(rec, f, default=str)


def run_all() -> list[dict]:
    ctx = multiprocessing.get_context("spawn")
    rows = []
    for rung in RUNGS:
        out = _result_path(rung)
        if os.path.exists(out):
            os.remove(out)
        proc = ctx.Process(target=_child, args=(rung, out))
        start = _time.perf_counter()
        proc.start()
        proc.join(TIMEOUTS[rung])
        if proc.is_alive():
            proc.terminate()
            proc.join()
            rows.append({"rung": rung, "title": TITLES[rung], "status": "TIMEOUT", "wall_s": TIMEOUTS[rung], "detail": f"no result within {TIMEOUTS[rung]} s", "metrics": {}})
            continue
        if not os.path.exists(out):
            rows.append({"rung": rung, "title": TITLES[rung], "status": "CRASH", "wall_s": _time.perf_counter() - start,
                         "detail": f"no record written, exit code {proc.exitcode}", "metrics": {}})
            continue
        with open(out) as f:
            rows.append(json.load(f))
    return rows


def write_report(records):
    """Append a dated table of ``records`` to RESULTS.md, headed by the commit of the library that ran."""
    identity = metrics.library_identity()
    library = identity["commit"][:7] + (" with uncommitted changes" if identity["dirty"] else "")
    lines = [f"\n## {datetime.datetime.now().astimezone().date().isoformat()} library `{library}`\n", "| Rung | Title | Status | Wall (s) | Detail |", "|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['rung']} | {r['title']} | {r['status']} | {r['wall_s']:.0f} | {r['detail']} |")
    with open(HERE / "RESULTS.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=list(RUNGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.report:
        records = []
        for r in RUNGS:
            with open(_result_path(r)) as fh:
                records.append(json.load(fh))
        write_report(records)
    elif args.rung:
        rec = run_rung_inprocess(args.rung)
        with open(_result_path(args.rung), "w") as fh:
            json.dump(rec, fh, indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("traceback", "metrics")}))
    elif args.all:
        records = run_all()
        for r in records:
            print(f"rung {r['rung']} ({r['title']}): {r['status']} in {r['wall_s']:.0f} s: {r['detail']}", flush=True)
        write_report(records)
