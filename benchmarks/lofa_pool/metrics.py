"""Run one scenario of the loss-of-flow case and write a flat record as JSON: the package's
record plus the library that ran, the warnings raised, the wall time and the traceback of a
failure.

    python benchmarks/lofa_pool/metrics.py --params '{"power_scale": 1.2}' --out row.json
"""
import argparse
import json
import os
import subprocess
import time
import traceback
import warnings
from dataclasses import asdict, dataclass

import numpy as np

import stream
from stream.examples.lofa import Record, both_pumps, one_pump, stuck_flapper
from stream.examples.lofa.params import Params
from stream.examples.lofa.system import GUESS_KINDS

SCENARIO_RUNS = {"both": both_pumps, "one": one_pump, "stuck": stuck_flapper}


@dataclass
class BenchRecord(Record):
    """A :class:`Record` with the library identity, the warnings, the wall time and a traceback."""

    library_path: str = ""
    library_commit: str = ""
    library_dirty: bool | None = None
    n_warnings: int = 0
    warnings: str = ""
    wall_s: float = np.nan
    traceback: str = ""
    guess_kind: str = ""


def library_identity() -> dict:
    """``path`` of the ``stream`` package, its checkout's ``commit`` (``"unknown"`` without git)
    and whether the package has uncommitted changes (``dirty``, None without git)."""
    root = os.path.dirname(os.path.realpath(stream.__file__))

    def git(*args):
        try:
            proc = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", ".")
    return dict(path=root, commit=commit or "unknown", dirty=None if status is None else bool(status))


def run(p: Params, *, guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3, atol_rel: float = 1e-4,
        rtol: float = 1e-4, t_end: float = 3600.0, n: int = 1801) -> BenchRecord:
    """Run ``p.scenario`` and never raise on a solver failure: the record says ``failed`` and
    carries the error and traceback instead."""
    scenario = SCENARIO_RUNS[p.scenario]
    identity = library_identity()
    rec = BenchRecord(**asdict(Record.empty("failed")), guess_kind=guess_kind, **{f"library_{k}": v for k, v in identity.items()})
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _, inner = scenario(p, t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol, guess_kind=guess_kind, seed=seed, band=band)
            for k, v in asdict(inner).items():
                setattr(rec, k, v)
        except Exception as e:
            rec.status, rec.error, rec.traceback = "failed", f"{type(e).__name__}: {e}", traceback.format_exc()
    rec.wall_s = time.perf_counter() - start
    rec.n_warnings = len(caught)
    rec.warnings = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught[:3])
    return rec


def record_to_row(p: Params, rec: Record, **extra) -> dict:
    """One flat table row: ``Params`` fields prefixed ``p_``, the record's fields, then ``extra``."""
    return {f"p_{k}": v for k, v in asdict(p).items()} | asdict(rec) | extra


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def params_from_json(text: str) -> Params:
    return Params(**json.loads(text))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one loss-of-flow scenario and write its row as JSON.")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--guess", default="ballpark", choices=GUESS_KINDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--band", type=float, default=0.3)
    parser.add_argument("--atol-rel", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=3600.0)
    parser.add_argument("--n", type=int, default=1801)
    parser.add_argument("--axis", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    p = params_from_json(args.params)
    rec = run(p, guess_kind=args.guess, seed=args.seed, band=args.band, atol_rel=args.atol_rel, rtol=args.rtol,
              t_end=args.t_end, n=args.n)
    row = record_to_row(p, rec, axis=args.axis, label=args.label, guess=args.guess, seed=args.seed, band=args.band,
                        atol_rel=args.atol_rel, rtol=args.rtol, t_end=args.t_end, n=args.n)
    with open(args.out, "w") as f:
        json.dump({k: _json_safe(v) for k, v in row.items()}, f)


if __name__ == "__main__":
    main()
