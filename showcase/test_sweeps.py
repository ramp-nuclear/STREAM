import pandas as pd
from showcase import sweeps, lofa_case as lc


def test_axes_cover_spec():
    expected = {"power", "pump_head", "inertia", "gaps", "hx_outlet", "pressure", "flapper_threshold",
                "flapper_area", "friction", "htc", "tolerance", "scram", "guesses", "grid"}
    assert expected <= set(sweeps.AXES)
    assert all(len(v) >= 2 for v in sweeps.AXES.values())


def test_sweep_timeout_row(tmp_path, monkeypatch):
    monkeypatch.setattr(sweeps, "RESULTS", tmp_path)
    monkeypatch.setitem(sweeps.AXES, "_slow", [("slow", lc.Params(), dict(n=1251))])
    df = sweeps.run_axis("_slow", workers=1, timeout=1)
    assert list(df.status) == ["timeout"]
    assert (tmp_path / "_slow.csv").exists()


def test_sweep_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(sweeps, "RESULTS", tmp_path)
    monkeypatch.setitem(sweeps.AXES, "_smoke", [("short", lc.Params(), dict(n=6, t_end=10.0)),
                                                ("short2", lc.Params(inertia=2e5), dict(n=6, t_end=10.0))])
    df = sweeps.run_axis("_smoke", workers=2, timeout=600)
    assert set(df.label) == {"short", "short2"}
    assert "p_inertia" in df.columns
    assert set(df.status) == {"completed"}
    assert list(df.columns) == list(sweeps.load("_smoke").columns)
