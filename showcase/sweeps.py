"""Stress sweeps of the loss-of-flow case: one subprocess per case, one CSV per axis.

Each axis is a list of ``(label, params, options)`` cases, where ``options`` may hold any of
:data:`OPTION_DEFAULTS` and is passed to ``python -m showcase.lofa_case``. A case that hangs
or crashes still yields a row, with status ``timeout`` or ``failed``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from showcase.lofa_case import DP0, Params, Record, record_to_row

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
PY = sys.executable
STATUSES = ("completed", "saturation", "failed", "timeout")
OPTION_DEFAULTS = dict(guess_kind="toolkit", seed=0, band=0.3, atol_rel=1e-4, rtol=1e-4, t_end=2500.0, n=1251)
FLAGS = dict(guess_kind="--guess", seed="--seed", band="--band", atol_rel="--atol-rel", rtol="--rtol",
             t_end="--t-end", n="--n")
DEFAULT_TIMEOUT = 900.0
TIMEOUTS = {"tolerance": 1800.0}
STDERR_TAIL = 30


def _cases(axis, values, make, options=None):
    return [(f"{axis}={v}", make(v), dict(options or {})) for v in values]


AXES: dict[str, list] = {
    "power": _cases("power_scale", np.round(np.arange(0.30, 1.001, 0.05), 2), lambda v: Params(power_scale=float(v))),
    "pump_head": _cases("pump_head", (0.5, 0.75, 1.0, 1.5, 2.0), lambda v: Params(pump_head=v * DP0)),
    "inertia": _cases("inertia", (0.3, 0.5, 1.0, 2.0, 3.0), lambda v: Params(inertia=v * 3e5)),
    "gaps": [
        ("hot=1.5mm", Params(gaps=(0.0015, 0.002, 0.003, 0.004)), {}),
        ("hot=2.5mm", Params(gaps=(0.0025, 0.002, 0.003, 0.004)), {}),
        ("hot=3mm", Params(gaps=(0.003, 0.002, 0.003, 0.004)), {}),
        ("wide=2.5mm", Params(gaps=(0.002, 0.002, 0.0025, 0.004)), {}),
        ("wide=4mm", Params(gaps=(0.002, 0.002, 0.004, 0.004)), {}),
        ("bypass=3mm", Params(gaps=(0.002, 0.002, 0.003, 0.003)), {}),
        ("bypass=6mm", Params(gaps=(0.002, 0.002, 0.003, 0.006)), {}),
    ],
    "hx_outlet": _cases("hx_outlet", (25.0, 30.0, 40.0, 50.0, 60.0), lambda v: Params(hx_outlet=v)),
    "pressure": _cases("p_ref", (1.5e5, 2e5, 3e5, 4e5), lambda v: Params(p_ref=v)),
    "flapper_threshold": _cases("flapper_fraction", (0.1, 0.2, 0.3, 0.4, 0.5), lambda v: Params(flapper_fraction=v)),
    "flapper_area": _cases("flapper_area", (0.3, 1.0, 3.0), lambda v: Params(flapper_area=v * 1e-3)),
    "friction": _cases("friction", ("regime", "blasius"), lambda v: Params(friction=v)),
    "htc": _cases("htc", ("dittus", "regime"), lambda v: Params(htc=v)),
    "tolerance": [(f"atol={a:g},rtol={r:g}", Params(), dict(atol_rel=a, rtol=r))
                  for a, r in ((1e-1, 1e-3), (1e-2, 1e-3), (1e-3, 1e-4), (1e-4, 1e-4), (1e-5, 1e-5), (1e-6, 1e-6))],
    "scram": [*_cases("ramp_width", (0.1, 0.5, 2.0, 5.0), lambda v: Params(ramp_width=v)),
              *_cases("t_scram", (1.0, 3.0, 10.0, 30.0), lambda v: Params(t_scram=v))],
    "guesses": [*((kind, Params(), dict(guess_kind=kind)) for kind in ("toolkit", "seed", "ballpark", "expert")),
                *((f"perturbed-{s}", Params(), dict(guess_kind="perturbed", seed=s, band=0.3)) for s in range(20))],
    "grid": [(f"n={n}", Params(), dict(n=n)) for n in (626, 1251, 2501)],
}


def _extras(axis: str, label: str, options: dict) -> dict:
    return {"axis": axis, "label": label} | OPTION_DEFAULTS | options


COLUMNS = list(record_to_row(Params(), Record.empty(""), **_extras("", "", {})))


def _stderr_row(params: Params, stderr: str, extras: dict) -> dict:
    lines = [line for line in stderr.splitlines() if line.strip()]
    rec = Record.empty("failed", error=lines[-1] if lines else "no output and no stderr")
    rec.traceback = "\n".join(lines[-STDERR_TAIL:])
    return record_to_row(params, rec, **extras)


def run_case(axis: str, label: str, params: Params, options: dict, timeout: float) -> dict:
    """Run one case in a fresh interpreter and return its flat row (``COLUMNS``)."""
    extras = _extras(axis, label, options)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "row.json"
        cmd = [PY, "-m", "showcase.lofa_case", "--params", json.dumps(asdict(params)), "--out", str(out),
               "--axis", axis, "--label", label]
        for key, value in options.items():
            cmd += [FLAGS[key], str(value)]
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=ROOT, env=env)
        except subprocess.TimeoutExpired:
            row = record_to_row(params, Record.empty("timeout", error=f"exceeded {timeout:g} s"), **extras)
        else:
            if out.exists():
                row = json.loads(out.read_text()) | extras
            else:
                row = _stderr_row(params, proc.stderr, extras)
    return {k: row.get(k) for k in COLUMNS}


def run_axis(name: str, workers: int = 12, timeout: float | None = None) -> pd.DataFrame:
    """Run every case of ``AXES[name]`` on ``workers`` threads, write ``RESULTS/<name>.csv``
    and return the table; ``timeout`` (s per case) defaults to ``TIMEOUTS`` or ``DEFAULT_TIMEOUT``."""
    limit = TIMEOUTS.get(name, DEFAULT_TIMEOUT) if timeout is None else timeout
    cases = AXES[name]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda case: run_case(name, *case, limit), cases))
    df = pd.DataFrame(rows, columns=COLUMNS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    return df


def load(name: str) -> pd.DataFrame:
    """The stored table of axis ``name``."""
    return pd.read_csv(RESULTS / f"{name}.csv")


def summary(frames) -> pd.DataFrame:
    """One row per axis: the number of cases and of each status in ``STATUSES``."""
    df = pd.concat(list(frames), ignore_index=True)
    counts = pd.crosstab(df["axis"], df["status"]).reindex(columns=list(STATUSES), fill_value=0)
    counts.insert(0, "cases", df.groupby("axis").size())
    counts.columns.name = None
    return counts.reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run loss-of-flow stress sweeps.")
    parser.add_argument("--axis", default="all", help=f"one of {', '.join(AXES)}, or 'all'")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    names = list(AXES) if args.axis == "all" else [args.axis]
    unknown = [n for n in names if n not in AXES]
    if unknown:
        parser.error(f"unknown axis {unknown[0]!r}")
    frames = []
    for name in names:
        frames.append(run_axis(name, workers=args.workers))
        print(f"{name}: {frames[-1].status.value_counts().to_dict()}", flush=True)
    print(summary(frames).to_string(index=False))


if __name__ == "__main__":
    main()
