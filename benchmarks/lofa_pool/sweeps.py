"""Stress sweep of the pool-reactor case: one spawned worker process per case, one CSV per axis.
Each axis has a wall-clock limit (``--timeout``, default ``AXIS_TIMEOUT`` seconds); an axis that
exceeds it raises ``multiprocessing.TimeoutError``."""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
from pathlib import Path

import pandas as pd

from stream.examples.lofa.params import Params

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lofa_bench_metrics", HERE / "metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

RESULTS = HERE / "results"
STATUSES = ("completed", "saturation", "failed")
AXIS_TIMEOUT = 7200.0
OPTION_DEFAULTS = {"guess": "ballpark", "seed": 0, "band": 0.3, "atol_rel": 1e-4, "rtol": 1e-4, "t_end": 3600.0, "n": 1801}


def _axis(name, values, make, options=None):
    return lambda: [(f"{name}={v}", make(v), dict(options or {})) for v in values]


AXES: dict[str, object] = {
    "power": _axis("power_scale", (0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6), lambda v: Params(power_scale=v)),
    "pool": _axis("pool_area_scale", (0.1, 0.3), lambda v: Params(pool_area_scale=v)),
    "tank": _axis("tank_volume_scale", (0.5, 2.0), lambda v: Params(tank_volume_scale=v)),
    "inertia": _axis("inertia_scale", (0.5, 2.0), lambda v: Params(inertia_scale=v)),
    "trip_delay": _axis("trip_delay", (0.2, 1.0, 2.0), lambda v: Params(trip_delay=v)),
    "rod_worth": _axis("rod_worth_dollars", (5.0, 15.0), lambda v: Params(rod_worth_dollars=v)),
    "feedback": _axis("feedback_scale", (0.0, 0.5, 2.0), lambda v: Params(feedback_scale=v)),
    "flapper": _axis("flapper_fraction", (0.1, 0.5), lambda v: Params(flapper_fraction=v)),
    "valve": _axis("valve_reverse_ratio", (1e4, 1e6), lambda v: Params(valve_reverse_ratio=v)),
    "hx_outlet": _axis("hx_outlet", (25.0, 55.0), lambda v: Params(hx_outlet=v)),
    "riser": _axis("basement_drop", (5.3, 9.3), lambda v: Params(basement_drop=v)),
    "gap_hot": _axis("gap_hot", (0.0016, 0.0020), lambda v: Params(gap_hot=v)),
    "htc": _axis("htc", ("regime", "dittus"), lambda v: Params(htc=v)),
    "friction": _axis("friction", ("blasius",), lambda v: Params(friction=v)),
    "cells": _axis("axial_cells", (5, 20, 40), lambda v: Params(axial_cells=v)),
    "guesses": lambda: [*((kind, Params(), {"guess": kind}) for kind in ("ballpark", "expert")),
                        *((f"perturbed-{s}", Params(), {"guess": "perturbed", "seed": s, "band": 0.3}) for s in range(5))],
    "tolerance": lambda: [(f"atol={a:g}", Params(), {"atol_rel": a, "rtol": a}) for a in (1e-3, 1e-5, 1e-6)],
    "grid": lambda: [(f"n={n}", Params(), {"n": n}) for n in (901, 3601)],
    "one_pump": _axis("trip_threshold", (0.6, 0.7, 0.9), lambda v: Params(scenario="one", trip_threshold=v)),
    "stuck": lambda: [("stuck-half-power", Params(scenario="stuck", flapper_stuck=True, power_scale=0.5), {})],
}


EXPECTED_SATURATION = {("power", "power_scale=1.6"), ("gap_hot", "gap_hot=0.0016"), ("stuck", "stuck-half-power")}


def run_case(job: tuple[str, str, Params, dict]) -> dict:
    """One sweep row, run in a worker process: ``(axis, label, params, options)`` to a flat row."""
    axis, label, params, options = job
    options = OPTION_DEFAULTS | options
    run_options = {("guess_kind" if k == "guess" else k): v for k, v in options.items()}
    rec = metrics.run(params, **run_options)
    return metrics.record_to_row(params, rec, axis=axis, label=label, **options)


def run_axis(name: str, workers: int = 10, timeout: float | None = None) -> pd.DataFrame:
    jobs = [(name, label, params, options) for label, params, options in AXES[name]()]
    ctx = multiprocessing.get_context("spawn")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        result = pool.map_async(run_case, jobs)
        rows = result.get(timeout)
    df = pd.DataFrame(rows)
    df["passed"] = passed(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    return df


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def passed(df: pd.DataFrame) -> pd.Series:
    """A case passes when it completes without warnings, or stops at saturation where
    ``EXPECTED_SATURATION`` says the physics should."""
    expected = pd.Series([(a, label) in EXPECTED_SATURATION for a, label in zip(df["axis"], df["label"])], index=df.index)
    clean = (df["status"] == "completed") & (df["n_warnings"] == 0)
    return clean | ((df["status"] == "saturation") & expected)


def summary(frames) -> pd.DataFrame:
    """Status counts per axis, plus ``passed``, ``warned`` (completed with warnings) and
    ``unexpected`` (any other outcome: an unexpected saturation or a failure)."""
    df = pd.concat(list(frames), ignore_index=True)
    ok = passed(df)
    warned = (df["status"] == "completed") & (df["n_warnings"] != 0)
    counts = pd.crosstab(df["axis"], df["status"]).reindex(columns=list(STATUSES), fill_value=0)
    counts.insert(0, "cases", df.groupby("axis").size())
    counts["passed"] = ok.groupby(df["axis"]).sum()
    counts["warned"] = warned.groupby(df["axis"]).sum()
    counts["unexpected"] = (~ok & ~warned).groupby(df["axis"]).sum()
    counts.columns.name = None
    return counts.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the pool-reactor stress sweep.")
    parser.add_argument("--axis", default="all")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=AXIS_TIMEOUT,
                        help=f"wall-clock limit of one axis in seconds (default {AXIS_TIMEOUT:g})")
    args = parser.parse_args(argv)
    names = list(AXES) if args.axis == "all" else [args.axis]
    frames = []
    for name in names:
        frames.append(run_axis(name, workers=args.workers, timeout=args.timeout))
        print(f"{name}: {frames[-1].status.value_counts().to_dict()}", flush=True)
    print(summary(frames).to_string(index=False))


if __name__ == "__main__":
    main()
