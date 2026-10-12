import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "lofa_pool"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"lofa_bench_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_modules_do_not_touch_sys_path():
    before = list(sys.path)
    for name in ("metrics", "run", "sweeps", "physics_checks", "top_flapper"):
        _load(name)
    assert sys.path == before


def test_row_is_flat_and_carries_params():
    metrics = _load("metrics")
    from stream.examples.lofa import Record
    from stream.examples.lofa.params import Params
    rec = Record.empty("failed", error="x")
    row = metrics.record_to_row(Params(scenario="one"), rec, axis="smoke", label="a")
    assert row["p_scenario"] == "one" and row["axis"] == "smoke"
    assert all(not isinstance(v, (dict, list, tuple)) for v in row.values())
    assert metrics.params_from_json(json.dumps({"power_scale": 0.5})).power_scale == 0.5


def test_run_records_a_synthetic_failure(monkeypatch):
    metrics = _load("metrics")
    from stream.examples.lofa import scenarios
    from stream.examples.lofa.params import Params

    def boom(*a, **k):
        raise ValueError("synthetic")

    monkeypatch.setattr(scenarios, "transient", boom)
    rec = metrics.run(Params(), t_end=20.0, n=11)
    assert rec.status == "failed"
    assert "ValueError" in rec.error and "synthetic" in rec.traceback
    assert rec.library_path and np.isfinite(rec.wall_s)
