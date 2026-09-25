import numpy as np
import pandas as pd
import pytest
from conftest import frame, make_case_frame

from stream.viz.labels import Label
from stream.viz.sweep import Coords, FrameList, Sweep, run_cases


def test_run_cases_keeps_alignment_and_messages():
    def model(power, mdot):
        if power > 5e6:
            raise RuntimeError(f"no convergence at {power}")
        return make_case_frame(power, mdot)

    cases = [dict(power=4e6, mdot=0.3), dict(power=8e6, mdot=0.3)]
    with pytest.warns(UserWarning, match="case 1 .*power=8e\\+06.*no convergence"):
        out_cases, frames = run_cases(model, cases)
    assert out_cases == cases
    assert isinstance(frames, FrameList)
    assert frames[1] is None and frames[0] is not None
    assert frames.failures == {1: "no convergence at 8000000.0"}


def test_run_cases_only_catches_listed_exceptions():
    def model(power):
        raise KeyError("boom")

    with pytest.raises(KeyError):
        run_cases(model, [dict(power=1.0)], catch=(ValueError,))


def test_table_has_case_and_parameter_columns(sweep):
    assert sweep.parameters == ("power", "mdot")
    assert list(sweep.table.columns[:3]) == ["case", "power", "mdot"]
    assert set(sweep.table.columns[3:]) == {"calculation", "variable", "i", "j", "value"}
    assert sweep.table.case.nunique() == 8
    assert sweep.n_cases == 8
    assert str(sweep.table.calculation.dtype) == "category"
    assert not sweep.banded
    assert sweep.holes == {}


def test_levels_are_sorted_unique(sweep):
    assert list(sweep.levels("power")) == [2e6, 4e6, 6e6, 8e6]
    assert list(sweep.levels("mdot")) == [0.2, 0.3]
    with pytest.raises(ValueError, match="depth.*power, mdot"):
        sweep.levels("depth")


def test_case_of(sweep):
    assert sweep.case_of(3) == dict(power=4e6, mdot=0.3)


def test_hole_recorded_and_messages_travel(grid_cases, agr):
    frames = FrameList(make_case_frame(**c) for c in grid_cases)
    frames[7] = None
    frames.failures = {7: "diverged"}
    with pytest.warns(UserWarning, match="case 7 .*diverged"):
        s = Sweep(grid_cases, frames, agr)
    assert s.holes == {7: "diverged"}
    assert 7 not in set(s.table.case)


def test_misaligned_lists_raise(grid_cases, agr):
    with pytest.raises(ValueError, match="8 cases.*7 frames"):
        Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases[:-1]], agr)


def test_non_float_case_value_raises(agr):
    with pytest.raises(TypeError, match="case 0.*'tag'.*'up'"):
        Sweep([dict(power=1.0, tag="up")], [make_case_frame(1.0, 0.3)], agr)


def test_integers_become_floats(agr):
    s = Sweep([dict(power=2)], [make_case_frame(2.0, 0.3)], agr)
    assert s.cases[0]["power"] == 2.0 and isinstance(s.cases[0]["power"], float)


def test_differing_keys_raise(agr):
    cases = [dict(power=1.0, mdot=0.3), dict(power=1.0)]
    with pytest.raises(ValueError, match="case 1.*mdot"):
        Sweep(cases, [make_case_frame(1.0, 0.3)] * 2, agr)


def test_transient_frame_refused(agr):
    df = make_case_frame(1.0, 0.3).assign(time=0.0)
    with pytest.raises(ValueError, match="case 0.*time.*steady"):
        Sweep([dict(power=1.0)], [df], agr)


def test_missing_columns_refused(agr):
    df = make_case_frame(1.0, 0.3).drop(columns=["j"])
    with pytest.raises(ValueError, match="case 0.*'j'"):
        Sweep([dict(power=1.0)], [df], agr)


def test_banded_predicate_and_zero_columns(grid_cases, agr):
    frames = [make_case_frame(**c, banded=True) for c in grid_cases]
    s = Sweep(grid_cases, frames, agr)
    assert s.banded
    zero = [f.assign(sys=0, stat=0) for f in frames]
    assert not Sweep(grid_cases, zero, agr).banded
    assert "sys" not in Sweep(grid_cases, zero, agr).table.columns


def test_mixed_banded_frames_warn_and_get_zero_bands(grid_cases, agr):
    frames = [make_case_frame(**c, banded=True) for c in grid_cases]
    frames[2] = make_case_frame(**grid_cases[2])
    with pytest.warns(UserWarning, match="case 2 .*no nonzero uncertainty"):
        s = Sweep(grid_cases, frames, agr)
    assert s.banded
    sub = s.table[s.table.case == 2]
    assert (sub.sys == 0).all() and (sub.stat == 0).all()
    assert s.table.sys.dtype == float


def test_negative_band_raises(agr):
    df = make_case_frame(1.0, 0.3, banded=True)
    df.loc[2, "stat"] = -1.0
    with pytest.raises(ValueError, match="case 0.*stat.*row 2"):
        Sweep([dict(power=1.0)], [df], agr)


def test_non_finite_rows_dropped_with_warning(agr):
    df = make_case_frame(1.0, 0.3)
    df.loc[1, "value"] = np.nan
    with pytest.warns(UserWarning, match="case 0.*row 1"):
        s = Sweep([dict(power=1.0)], [df], agr)
    assert s.bad_rows == {0: [1]}
    assert len(s.table) == len(df) - 1


def test_single_case_sweep(agr):
    s = Sweep.single(make_case_frame(4e6, 0.3), agr)
    assert s.parameters == ()
    assert s.n_cases == 1
    assert s.case_of(0) == {}


def test_labels_merge_over_defaults(sweep):
    assert sweep.labels["T_cool"] == Label("Coolant temperature", "°C")
    sweep.labels = {"T_cool": ("Bulk", "K")}
    assert sweep.labels["T_cool"] == Label("Bulk", "K")
    assert sweep.labels["q, left"].unit == "W/m$^2$"


def test_input_frames_untouched(grid_cases, agr):
    frames = [make_case_frame(**c) for c in grid_cases]
    copies = [f.copy() for f in frames]
    Sweep(grid_cases, frames, agr)
    for f, c in zip(frames, copies):
        pd.testing.assert_frame_equal(f, c)


def test_select_exact_partial_and_empty(sweep):
    assert list(sweep.select(dict(power=4e6, mdot=0.3))) == [3]
    assert list(sweep.select(dict(power=4e6))) == [2, 3]
    assert list(sweep.select({})) == list(range(8))
    assert list(sweep.select(None)) == list(range(8))


def test_select_float_tolerance(sweep):
    assert list(sweep.select(dict(mdot=0.3 + 1e-12))) == [1, 3, 5, 7]


def test_select_unknown_parameter_raises(sweep):
    with pytest.raises(ValueError, match="depth.*power, mdot"):
        sweep.select(dict(depth=1.0))


def test_select_off_grid_raises_and_lists_levels(sweep):
    with pytest.raises(ValueError, match="mdot=0.35.*0.2, 0.3"):
        sweep.select(dict(mdot=0.35))


def test_select_nearest_warns_with_the_case_used(sweep):
    with pytest.warns(UserWarning, match="mdot=0.35.*mdot=0.3"):
        idx = sweep.select(dict(power=4e6, mdot=0.35), nearest=True)
    assert list(idx) == [3]


def test_select_includes_holes(holed_sweep):
    assert list(holed_sweep.select(dict(mdot=0.3))) == [1, 3, 5, 7]
    assert holed_sweep.is_hole(7) and not holed_sweep.is_hole(5)


def test_rank(sweep):
    assert sweep.rank("T_cool", "CC") == 1
    assert sweep.rank("power", "CC") == 0
    assert sweep.rank("T", "plate") == 2


def test_names_resolve_defaults_and_validate(sweep):
    assert sweep.names("T_cool") == (["T_cool"], ["CC"])
    assert sweep.names("T") == (["T"], ["plate"])
    assert sweep.names(["T_cool", "CHFR"], "CC") == (["T_cool", "CHFR"], ["CC"])
    with pytest.raises(ValueError, match="'OSVR'.*CC.*T_cool"):
        sweep.names("OSVR", "CC")
    with pytest.raises(ValueError, match="'hot'.*CC, plate"):
        sweep.names("T_cool", "hot")


def test_coords_channel(sweep):
    c = sweep.coords("CC")
    assert isinstance(c, Coords)
    assert c.names == ("z",)
    np.testing.assert_allclose(c.centers[0], [0.05, 0.15, 0.25, 0.35])
    np.testing.assert_allclose(c.weights[0], 0.1)
    assert c.mask is None


def test_coords_plate(sweep):
    c = sweep.coords("plate")
    assert c.names == ("z", "x")
    np.testing.assert_allclose(c.centers[1], [0.5e-3, 1.5e-3, 2.5e-3])
    np.testing.assert_allclose(c.weights[0], [0.5, 0.5])
    assert c.mask.shape == (2, 3)


def test_coords_of_a_calculation_the_aggregator_lacks_names_it(agr):
    s = Sweep.single(frame([("ghost", "v", 0, j, float(j)) for j in range(3)]), agr)
    with pytest.raises(ValueError, match="'ghost'.*aggregator.*CC, plate"):
        s.coords("ghost")


def test_coords_unknown_calculation_is_none(agr):
    agr["blob"] = object()
    s = Sweep.single(frame([("blob", "v", 0, j, float(j)) for j in range(3)]), agr)
    assert s.coords("blob") is None
    assert s.coords("blob") is None


def test_zero_band_warning_fires_once_for_all_cases(agr):
    cases = [dict(power=1.0), dict(power=2.0), dict(power=3.0)]
    frames = [
        make_case_frame(1.0, 0.3),
        make_case_frame(2.0, 0.3, banded=True),
        make_case_frame(3.0, 0.3),
    ]
    with pytest.warns(UserWarning) as record:
        Sweep(cases, frames, agr)
    zero_band = [w for w in record.list if "no nonzero uncertainty" in str(w.message)]
    assert len(zero_band) == 1
    assert "case 0" in str(zero_band[0].message)
    assert "case 2" in str(zero_band[0].message)


def test_non_finite_warning_fires_once_for_all_cases(agr):
    df0 = make_case_frame(1.0, 0.3)
    df0.loc[1, "value"] = np.nan
    df1 = make_case_frame(2.0, 0.3)
    df1.loc[2, "value"] = np.nan
    with pytest.warns(UserWarning) as record:
        Sweep([dict(power=1.0), dict(power=2.0)], [df0, df1], agr)
    non_finite = [w for w in record.list if "non-finite" in str(w.message)]
    assert len(non_finite) == 1
    assert "case 0" in str(non_finite[0].message)
    assert "case 1" in str(non_finite[0].message)


def test_reserved_parameter_name_raises(agr):
    with pytest.raises(ValueError, match="'value'.*case, calculation, variable, i, j, value, sys, stat"):
        Sweep([dict(power=1.0, value=2.0)], [make_case_frame(1.0, 0.3)], agr)


def test_categorical_columns_survive_concat_with_differing_calculations(agr):
    df0 = frame([("CC", "T_cool", 0, 0, 40.0)])
    df1 = frame([("plate", "T", 0, 0, 50.0)])
    s = Sweep([dict(power=1.0), dict(power=2.0)], [df0, df1], agr)
    assert str(s.table.calculation.dtype) == "category"
    assert str(s.table.variable.dtype) == "category"
    assert set(s.table.calculation.cat.categories) == {"CC", "plate"}


def test_coords_plate_without_bounds_gets_unit_weights(agr):
    class ThinPlate:
        def __init__(self):
            self.z_centers = np.array([0.25, 0.75])
            self.x_centers = np.array([0.5e-3, 1.5e-3, 2.5e-3])

    agr["thin"] = ThinPlate()
    s = Sweep.single(frame([("thin", "v", 0, j, float(j)) for j in range(3)]), agr)
    c = s.coords("thin")
    assert c.names == ("z", "x")
    assert c.bounds == (None, None)
    np.testing.assert_allclose(c.weights[0], [1.0, 1.0])
    np.testing.assert_allclose(c.weights[1], [1.0, 1.0, 1.0])
    assert c.mask is None
