import numpy as np
import pytest

from stream.viz import at
from stream.viz.labels import Labels
from stream.viz.results import Field, Result, check_units, y_axis_text


def test_profile_single_case(sweep):
    r = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    assert isinstance(r, Result) and r.kind == "profile"
    assert r.x == "z"
    assert list(r.frame.columns) == ["quantity", "calculation", "power", "mdot", "case", "z", "lower", "value", "upper", "inner_lower", "inner_upper"]
    assert len(r.frame) == 4
    np.testing.assert_allclose(r.frame.z, [0.05, 0.15, 0.25, 0.35])
    np.testing.assert_allclose(r.frame.value, [44.0, 49.0, 54.0, 59.0])
    assert (r.frame.lower == r.frame.value).all()
    assert r.dims == {}
    assert r.fixed == dict(power=4e6, mdot=0.3)
    assert r.calculation == "CC"
    assert r.ylabel == "Coolant temperature [°C]"
    assert r.subtitle == "Power = 4e+06 W, Mass flow = 0.3 kg/s, CC"
    assert not r.banded


def test_profile_family_has_free_dimension(sweep):
    r = sweep.profile("CHFR", of="CC", where=dict(mdot=0.3))
    assert r.dims == {"power": [2e6, 4e6, 6e6, 8e6]}
    assert r.numeric == frozenset({"power"})
    assert r.fixed == dict(mdot=0.3)
    assert len(r.frame) == 16


def test_profile_two_quantities_and_unit_check(sweep):
    r = sweep.profile(["T_cool", "CHFR"], of="CC", where=dict(power=4e6, mdot=0.3))
    assert r.dims == {"quantity": ["T_cool", "CHFR"]}
    sweep.labels = {"CHFR": ("CHF ratio", "")}
    with pytest.raises(ValueError, match="'CHFR'.*dimensionless.*'T_cool'.*°C"):
        sweep.profile(["T_cool", "CHFR"], of="CC", where=dict(power=4e6, mdot=0.3))


def test_profile_rejects_rank_0_and_2(sweep):
    with pytest.raises(ValueError, match="'power'.*scalar.*reduce"):
        sweep.profile("power", of="CC", where=dict(power=4e6, mdot=0.3))
    with pytest.raises(ValueError, match="'T'.*field.*axis"):
        sweep.profile("T", of="plate", where=dict(power=4e6, mdot=0.3))


def test_profile_uses_cell_index_without_coordinates(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    agr["blob"] = object()
    s = Sweep.single(frame([("blob", "v", 0, j, float(j)) for j in range(3)]), agr)
    r = s.profile("v")
    assert r.x == "cell"
    np.testing.assert_array_equal(r.frame.cell, [0, 1, 2])


def test_profile_skips_holes_with_warning(holed_sweep):
    with pytest.warns(UserWarning, match="case 7 .*power=8e\\+06, mdot=0.3"):
        r = holed_sweep.profile("T_cool", of="CC", where=dict(mdot=0.3))
    assert r.dims == {"power": [2e6, 4e6, 6e6]}


def test_profile_leaves_a_gap_for_a_dropped_cell(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    rows = [("CC", "T_cool", 0, j, 40.0 + 5 * j) for j in range(4)]
    rows[2] = ("CC", "T_cool", 0, 2, np.nan)
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep.single(frame(rows), agr)
    r = s.profile("T_cool")
    assert len(r.frame) == 3
    np.testing.assert_allclose(r.frame.z, [0.05, 0.15, 0.35])
    np.testing.assert_allclose(r.frame.value, [40.0, 45.0, 55.0])


def test_profile_of_calculations_with_different_x_names_raises(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    agr["blob"] = object()
    rows = [("CC", "v", 0, j, float(j)) for j in range(4)] + [("blob", "v", 0, j, float(j)) for j in range(3)]
    s = Sweep.single(frame(rows), agr)
    with pytest.raises(ValueError, match="CC along z.*blob along cell|blob along cell.*CC along z"):
        s.profile("v", of=["CC", "blob"])


def test_profile_without_a_solved_case_names_the_selection(holed_sweep):
    with pytest.warns(UserWarning, match="case 7"):
        with pytest.raises(ValueError, match="no solved case.*power=8e\\+06, mdot=0.3.*'T_cool'"):
            holed_sweep.profile("T_cool", of="CC", where=dict(power=8e6, mdot=0.3))


def test_profile_envelopes_from_bands(banded_sweep):
    r = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3), sigma=2.0)
    assert r.banded
    np.testing.assert_allclose(r.frame.upper - r.frame.value, 0.1 + 2 * 0.05)
    r0 = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3), uncertainty=False)
    assert not r0.banded and (r0.frame.upper == r0.frame.value).all()


def test_field_single_case(sweep):
    f = sweep.field("T", where=dict(power=4e6, mdot=0.3))
    assert isinstance(f, Field) and f.kind == "field"
    assert f.arrays["value"].shape == (2, 3)
    np.testing.assert_allclose(f.arrays["value"][1], [60.0, 61.0, 62.0])
    np.testing.assert_allclose(f.z_bounds, [0.0, 0.5, 1.0])
    assert f.mask.shape == (2, 3)
    assert f.quantity == "T" and f.calculation == "plate"
    assert list(f.frame.columns[-7:]) == ["z", "x", "lower", "value", "upper", "inner_lower", "inner_upper"]
    assert len(f.frame) == 6


def test_field_values_are_the_three_arrays(sweep):
    f = sweep.field("T", where=dict(power=4e6, mdot=0.3))
    lower, value, upper = f.values()
    np.testing.assert_array_equal(value, f.arrays["value"])
    np.testing.assert_array_equal(lower, f.arrays["lower"])
    np.testing.assert_array_equal(upper, f.arrays["upper"])


def test_field_requires_one_case(sweep):
    with pytest.raises(ValueError, match="4 cases.*one"):
        sweep.field("T", where=dict(mdot=0.3))


def test_field_takes_one_quantity_of_one_calculation(sweep):
    with pytest.raises(ValueError, match="one quantity of one calculation.*'T_cool'.*'CHFR'"):
        sweep.field(["T_cool", "CHFR"], of="CC", where=dict(power=4e6, mdot=0.3))


def test_field_fills_a_dropped_cell_with_nan(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    rows = [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    rows[4] = ("plate", "T", 1, 1, np.nan)
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep.single(frame(rows), agr)
    f = s.field("T")
    assert f.arrays["value"].shape == (2, 3)
    assert np.isnan(f.arrays["value"][1, 1])
    np.testing.assert_allclose(f.arrays["value"][0], [50.0, 51.0, 52.0])
    assert len(f.frame) == 6


def test_mesh_shorter_than_the_frame_is_named(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    channel = Sweep.single(frame([("CC", "T_cool", 0, j, float(j)) for j in range(5)]), agr)
    with pytest.raises(ValueError, match="'T_cool'.*cell 4.*4 cells"):
        channel.profile("T_cool")
    plate = Sweep.single(frame([("plate", "T", i, j, 1.0) for i in range(3) for j in range(3)]), agr)
    with pytest.raises(ValueError, match="'T'.*row 2.*2 rows"):
        plate.field("T")


def test_reduce_min_versus_power(sweep):
    r = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    assert r.kind == "curve" and r.x == "power"
    assert list(r.frame.columns) == ["quantity", "calculation", "power", "mdot", "case", "lower", "value", "upper", "inner_lower", "inner_upper"]
    np.testing.assert_allclose(r.frame.power, [2e6, 4e6, 6e6, 8e6])
    np.testing.assert_allclose(r.frame.value, 1.5 * 4e6 / np.array([2e6, 4e6, 6e6, 8e6]))
    assert r.dims == {} and r.fixed == dict(mdot=0.3)
    assert r.reducer == "min"
    assert r.ylabel == "min CHFR"
    assert r.ylabel_source == ("CHFR",) and r.sigma == 1.0


def test_reduce_keeps_values_on_their_own_coordinate_after_a_dropped_cell(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    rows = [("CC", "T_cool", 0, j, v) for j, v in enumerate((np.nan, 45.0, 50.0, 55.0))]
    cases = [dict(power=2e6), dict(power=4e6)]
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep(cases, [frame(rows), frame(rows)], agr)
    assert s.reduce("T_cool", "where_min", versus="power").frame.value.iloc[0] == pytest.approx(0.15)
    assert s.reduce("T_cool", "mean", versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(cell=2), versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(z=0.25), versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(z=0.35), versus="power").frame.value.iloc[0] == 55.0


def test_reduce_with_axis_survives_a_dropped_cell(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    rows = [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    rows[1] = ("plate", "T", 0, 1, np.nan)
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep.single(frame(rows), agr)
    r = s.reduce("T", "max", axis="x")
    assert r.x == "z"
    np.testing.assert_allclose(r.frame.value, [52.0, 62.0])


def test_a_quantity_missing_in_one_case_leaves_the_others_alone(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    live = [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    dead = [("plate", "T", i, j, np.nan) for i in range(2) for j in range(3)]
    cases = [dict(power=2e6), dict(power=4e6)]
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep(cases, [frame(live), frame(dead)], agr)
    f = s.field("T", where=dict(power=4e6))
    assert f.arrays["value"].shape == (2, 3) and np.isnan(f.arrays["value"]).all()
    r = s.reduce("T", "max", versus="power")
    np.testing.assert_allclose(r.frame.value, [62.0, np.nan])


def test_reduce_versus_refuses_a_reducer_that_leaves_a_profile(sweep):
    with pytest.raises(ValueError, match="'T'.*plate.*leaves a profile.*axis="):
        sweep.reduce("T", at(z=0.25), versus="power", where=dict(mdot=0.3))


def test_reduce_free_parameter_becomes_dimension(sweep):
    r = sweep.reduce("CHFR", "min", versus="power")
    assert r.dims == {"mdot": [0.2, 0.3]}
    assert len(r.frame) == 8


def test_reduce_several_quantities(sweep):
    r = sweep.reduce(["CHFR", "T_cool"], "max", versus="power", where=dict(mdot=0.3))
    assert r.dims == {"quantity": ["CHFR", "T_cool"]}
    assert r.ylabel == "max CHFR, Coolant temperature [°C]"


def test_reduce_scalar_passes_through(sweep):
    r = sweep.reduce("power", "min", versus="mdot", where=dict(power=4e6))
    np.testing.assert_allclose(r.frame.value, [4e6, 4e6])
    assert r.ylabel == "Power [W]"


def test_reduce_where_min_and_at(sweep):
    r = sweep.reduce("CHFR", "where_min", versus="power", where=dict(mdot=0.3))
    np.testing.assert_allclose(r.frame.value, 0.25)
    assert r.ylabel == "where_min CHFR [m]"
    r2 = sweep.reduce("T_cool", at(z=0.35), versus="power", where=dict(mdot=0.3))
    np.testing.assert_allclose(r2.frame.value, [57.0, 59.0, 61.0, 63.0])
    assert r2.ylabel == "at z=0.35 Coolant temperature [°C]"


def test_where_with_axis_reports_the_collapsed_coordinate(sweep):
    r = sweep.reduce("T", "where_max", axis="x", where=dict(power=4e6, mdot=0.3))
    assert r.coordinate == "x"
    assert r.ylabel == "where_max Temperature [m]"
    sweep.labels = {"x": ("x", "mm", 1e3)}
    scaled = sweep.reduce("T", "where_max", axis="x", where=dict(power=4e6, mdot=0.3))
    assert scaled.ylabel == "where_max Temperature [mm]"


def test_level_space_holds_every_parameter_of_the_sweep(sweep):
    r = sweep.reduce("CHFR", "min", versus="power")
    assert r.level_space == {"power": [2e6, 4e6, 6e6, 8e6], "mdot": [0.2, 0.3]}
    assert sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).level_space == r.level_space


def test_reduce_with_axis_returns_profile(sweep):
    r = sweep.reduce("T", "max", axis="x", where=dict(power=4e6, mdot=0.3))
    assert r.kind == "profile" and r.x == "z"
    np.testing.assert_allclose(r.frame.value, [52.0, 62.0])
    with pytest.raises(ValueError, match="versus.*axis"):
        sweep.reduce("T", "max", axis="x", versus="power", where=dict(mdot=0.3))


def test_reduce_requires_versus_without_axis(sweep):
    with pytest.raises(ValueError, match="versus"):
        sweep.reduce("CHFR", "min")


def test_reduce_versus_must_be_free(sweep):
    with pytest.raises(ValueError, match="power.*where"):
        sweep.reduce("CHFR", "min", versus="power", where=dict(power=4e6))


def test_reduce_skips_holes(holed_sweep):
    with pytest.warns(UserWarning, match="case 7"):
        r = holed_sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    np.testing.assert_allclose(r.frame.power, [2e6, 4e6, 6e6])


def test_reduce_without_a_solved_case_names_the_selection(grid_cases, agr):
    from conftest import make_case_frame

    from stream.viz.sweep import Sweep

    frames = [None if case["mdot"] == 0.3 else make_case_frame(**case) for case in grid_cases]
    with pytest.warns(UserWarning, match="holes"):
        s = Sweep(grid_cases, frames, agr)
    with pytest.warns(UserWarning, match="case 1"):
        with pytest.raises(ValueError, match="no solved case.*mdot=0.3.*'CHFR'"):
            s.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))


def test_reduce_envelopes(banded_sweep):
    r = banded_sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    np.testing.assert_allclose(r.frame.value - r.frame.lower, 0.15)
    np.testing.assert_allclose(r.frame.upper - r.frame.value, 0.15)


def test_y_axis_text_and_check_units():
    labels = Labels({"a": ("A", "K"), "b": ("B", "K"), "c": ("C", "")})
    assert y_axis_text(labels, ["a"]) == "A [K]"
    assert y_axis_text(labels, ["a", "b"], "min") == "min A, B [K]"
    assert y_axis_text(labels, ["c"], "max") == "max C"
    assert y_axis_text(labels, ["a", "zzz"]) == "A, zzz [K]"
    check_units(labels, ["a", "zzz"])
    with pytest.raises(ValueError, match="'a'.*K.*'c'.*dimensionless"):
        check_units(labels, ["a", "c"])


def test_check_units_refuses_two_scales_on_one_axis():
    labels = Labels({"a": ("A", "W", 1.0), "b": ("B", "W", 1e-6)})
    with pytest.raises(ValueError, match="'a'.*scaled by 1.*'b'.*1e-06"):
        check_units(labels, ["a", "b"])
