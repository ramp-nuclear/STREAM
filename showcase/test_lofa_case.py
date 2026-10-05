import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "lofa"))

from showcase import lofa_case as lc


def test_default_matches_benchmark():
    import case
    agr_b, K_b, refs_b = case.build_general()
    agr, K, refs = lc.build(lc.Params())
    assert len(agr) == len(agr_b)
    y_b = agr_b.solve_steady(case.ballpark_guess_general(agr_b, K_b))
    y = lc.steady(agr, lc.guess(agr, K, refs, "ballpark"))
    for k in ("hot", "warm", "wide", "bypass"):
        mb = float(agr_b.save(y_b)[K_b.name][K_b.component_edge(refs_b["channels"][k])])
        m = float(agr.save(y)[K.name][K.component_edge(refs["channels"][k])])
        assert abs(m - mb) < 1e-9


def test_run_default_completes():
    rec = lc.run(lc.Params(), t_end=2500.0)
    assert rec.status == "completed"
    assert rec.n_warnings == 0
    assert rec.t_rev_hot is not None and rec.t_rev_warm is not None
    assert rec.t_rev_hot < rec.t_rev_warm
    assert rec.final_bypass > 0
    assert rec.margin > 0


def test_saturation_is_named():
    rec = lc.run(lc.Params(power_scale=1.0), t_end=2500.0)
    assert rec.status == "saturation"
    assert "SaturationReachedError" in rec.error
    assert rec.t_last > 2000


def test_run_records_failure(monkeypatch):
    def boom(*a, **k):
        raise ValueError("synthetic")
    monkeypatch.setattr(lc, "transient", boom)
    rec = lc.run(lc.Params(), t_end=10.0, n=6)
    assert rec.status == "failed"
    assert "ValueError" in rec.error and "synthetic" in rec.error
    assert "synthetic" in rec.traceback


def test_guess_kinds_all_converge():
    agr, K, refs = lc.build(lc.Params())
    ref = lc.steady(agr, lc.guess(agr, K, refs, "toolkit"))
    for kind in ("seed", "ballpark", "expert", "perturbed"):
        y = lc.steady(agr, lc.guess(agr, K, refs, kind, seed=3))
        assert np.max(np.abs(y - ref)) < 1e-6 * (1 + np.max(np.abs(ref)))


def test_row_is_flat():
    rec = lc.run(lc.Params(), t_end=10.0, n=6)
    row = lc.record_to_row(lc.Params(), rec, axis="smoke", label="x")
    assert row["p_power_scale"] == 0.70 and row["axis"] == "smoke"
    assert all(not isinstance(v, (dict, list, tuple)) for v in row.values())
