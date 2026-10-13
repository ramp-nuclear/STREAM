import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "lofa_pool"
SILENT_RUNG = "4"


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


class _FakeProcess:
    def __init__(self, target, args):
        self.rung, self.out = args
        self.exitcode = None

    def start(self):
        if self.rung == SILENT_RUNG:
            self.exitcode = -11
            return
        with open(self.out, "w") as f:
            json.dump({"rung": self.rung, "status": "PASS", "wall_s": 0.0, "detail": "", "metrics": {}}, f)
        self.exitcode = 0

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def test_runner_records_a_rung_whose_child_wrote_nothing_as_a_crash(monkeypatch, tmp_path):
    run = _load("run")
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run.multiprocessing, "get_context", lambda method: SimpleNamespace(Process=_FakeProcess))
    rows = run.run_all()
    assert [r["rung"] for r in rows] == list(run.RUNGS)
    crashed = [r for r in rows if r["status"] == "CRASH"]
    assert len(crashed) == 1 and crashed[0]["rung"] == SILENT_RUNG
    assert crashed[0]["detail"] == "no record written, exit code -11"
    assert all(r["status"] == "PASS" for r in rows if r["rung"] != SILENT_RUNG)
