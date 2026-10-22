import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import FakeChannel, channel_rows, frame, make_case_frame

from stream.viz import Result, Sweep, at, at_least, at_most


def test_profile_single_case(sweep):
    r = sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3})
    assert isinstance(r, Result) and r.kind == "profile"
    assert r.x == "z"
    assert list(r.frame.columns) == ["quantity", "calculation", "power", "mdot", "case", "z", "lower", "value", "upper", "inner_lower", "inner_upper", "filled"]
    assert len(r.frame) == 4
    np.testing.assert_allclose(r.frame.z, [0.05, 0.15, 0.25, 0.35])
    np.testing.assert_allclose(r.frame.value, [44.0, 49.0, 54.0, 59.0])
    assert (r.frame.lower == r.frame.value).all()
    assert r.dims == {}
    assert r.fixed == {"power": 4e6, "mdot": 0.3}
    assert r.calculation == "CC"
    assert r.ylabel == "Coolant temperature [°C]"
    assert r.subtitle == "Power = 4e+06 W, Mass flow = 0.3 kg/s, CC"


def test_profile_family_has_free_dimension(sweep):
    r = sweep.profile("CHFR", of="CC", where={"mdot": 0.3})
    assert r.dims == {"power": [2e6, 4e6, 6e6, 8e6]}
    assert r.fixed == {"mdot": 0.3}
    assert len(r.frame) == 16


def test_profile_two_quantities_and_unit_check(sweep):
    r = sweep.profile(["T_cool", "CHFR"], of="CC", where={"power": 4e6, "mdot": 0.3})
    assert r.dims == {"quantity": ["T_cool", "CHFR"]}
    sweep.labels = {"CHFR": ("CHF ratio", "")}
    with pytest.raises(ValueError, match="'CHFR'.*dimensionless.*'T_cool'.*°C"):
        sweep.profile(["T_cool", "CHFR"], of="CC", where={"power": 4e6, "mdot": 0.3})


def test_profile_rejects_rank_0_and_2(sweep):
    with pytest.raises(ValueError, match="'power'.*scalar.*reduce"):
        sweep.profile("power", of="CC", where={"power": 4e6, "mdot": 0.3})
    with pytest.raises(ValueError, match="'T'.*field.*axis"):
        sweep.profile("T", of="plate", where={"power": 4e6, "mdot": 0.3})


def test_profile_uses_cell_index_without_coordinates(agr):
    agr["blob"] = object()
    s = Sweep.single(frame([("blob", "v", 0, j, float(j)) for j in range(3)]), agr)
    r = s.profile("v")
    assert r.x == "cell"
    np.testing.assert_array_equal(r.frame.cell, [0, 1, 2])


def test_profile_skips_holes_with_warning(holed_sweep):
    with pytest.warns(UserWarning, match="case 7 .*power=8e\\+06, mdot=0.3"):
        r = holed_sweep.profile("T_cool", of="CC", where={"mdot": 0.3})
    assert r.dims == {"power": [2e6, 4e6, 6e6]}


def test_profile_leaves_a_gap_for_a_dropped_cell(agr):
    rows = [("CC", "T_cool", 0, j, 40.0 + 5 * j) for j in range(4)]
    rows[2] = ("CC", "T_cool", 0, 2, np.nan)
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep.single(frame(rows), agr)
    r = s.profile("T_cool")
    assert len(r.frame) == 3
    np.testing.assert_allclose(r.frame.z, [0.05, 0.15, 0.35])
    np.testing.assert_allclose(r.frame.value, [40.0, 45.0, 55.0])


def test_profile_of_calculations_with_different_x_names_raises(agr):
    agr["blob"] = object()
    rows = [("CC", "v", 0, j, float(j)) for j in range(4)] + [("blob", "v", 0, j, float(j)) for j in range(3)]
    s = Sweep.single(frame(rows), agr)
    with pytest.raises(ValueError, match="CC along z.*blob along cell|blob along cell.*CC along z"):
        s.profile("v", of=["CC", "blob"])


def test_profile_without_a_solved_case_names_the_selection(holed_sweep):
    with pytest.warns(UserWarning, match="case 7"), pytest.raises(ValueError, match="no solved case.*power=8e\\+06, mdot=0.3.*'T_cool'"):
        holed_sweep.profile("T_cool", of="CC", where={"power": 8e6, "mdot": 0.3})


def test_profile_envelopes_from_bands(banded_sweep):
    r = banded_sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3}, sigma=2.0)
    np.testing.assert_allclose(r.frame.upper - r.frame.value, 0.1 + 2 * 0.05)
    np.testing.assert_allclose(r.frame.value - r.frame.lower, 0.1 + 2 * 0.05)
    np.testing.assert_allclose(r.frame.inner_upper - r.frame.value, 0.1)
    np.testing.assert_allclose(r.frame.value - r.frame.inner_lower, 0.1)
    r0 = banded_sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3}, uncertainty=False)
    assert (r0.frame.upper == r0.frame.value).all() and (r0.frame.inner_lower == r0.frame.value).all()


def test_field_single_case(sweep):
    f = sweep.field("T", where={"power": 4e6, "mdot": 0.3})
    assert isinstance(f, Result) and f.kind == "field"
    assert f.arrays["value"].shape == (2, 3)
    np.testing.assert_allclose(f.arrays["value"][1], [60.0, 61.0, 62.0])
    np.testing.assert_allclose(f.z_bounds, [0.0, 0.5, 1.0])
    assert f.mask.shape == (2, 3)
    assert f.quantity == "T" and f.calculation == "plate"
    assert list(f.frame.columns[-8:]) == ["z", "x", "lower", "value", "upper", "inner_lower", "inner_upper", "filled"]
    assert len(f.frame) == 6


def test_field_values_are_the_three_arrays(sweep):
    f = sweep.field("T", where={"power": 4e6, "mdot": 0.3})
    lower, value, upper = f.values()
    np.testing.assert_array_equal(value, f.arrays["value"])
    np.testing.assert_array_equal(lower, f.arrays["lower"])
    np.testing.assert_array_equal(upper, f.arrays["upper"])


def test_field_requires_one_case(sweep):
    with pytest.raises(ValueError, match="4 cases.*one"):
        sweep.field("T", where={"mdot": 0.3})


def test_field_takes_one_quantity_of_one_calculation(sweep):
    with pytest.raises(ValueError, match="one quantity of one calculation.*'T_cool'.*'CHFR'"):
        sweep.field(["T_cool", "CHFR"], of="CC", where={"power": 4e6, "mdot": 0.3})


def test_field_fills_a_dropped_cell_with_nan(agr):
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
    channel = Sweep.single(frame([("CC", "T_cool", 0, j, float(j)) for j in range(5)]), agr)
    with pytest.raises(ValueError, match="'T_cool'.*cell 4.*4 cells"):
        channel.profile("T_cool")
    plate = Sweep.single(frame([("plate", "T", i, j, 1.0) for i in range(3) for j in range(3)]), agr)
    with pytest.raises(ValueError, match="'T'.*row 2.*2 rows"):
        plate.field("T")


def test_reduce_min_versus_power(sweep):
    r = sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3})
    assert r.kind == "curve" and r.x == "power"
    assert list(r.frame.columns) == ["quantity", "calculation", "power", "mdot", "case", "lower", "value", "upper", "inner_lower", "inner_upper", "filled"]
    np.testing.assert_allclose(r.frame.power, [2e6, 4e6, 6e6, 8e6])
    np.testing.assert_allclose(r.frame.value, 1.5 * 4e6 / np.array([2e6, 4e6, 6e6, 8e6]))
    assert r.dims == {} and r.fixed == {"mdot": 0.3}
    assert r.ylabel == "min CHFR"
    assert r.question.sigma == 1.0


def test_reduce_keeps_values_on_their_own_coordinate_after_a_dropped_cell(agr):
    rows = [("CC", "T_cool", 0, j, v) for j, v in enumerate((np.nan, 45.0, 50.0, 55.0))]
    cases = [{"power": 2e6}, {"power": 4e6}]
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep(cases, [frame(rows), frame(rows)], agr)
    assert s.reduce("T_cool", "where_min", versus="power").frame.value.iloc[0] == pytest.approx(0.15)
    assert s.reduce("T_cool", "mean", versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(cell=2), versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(z=0.25), versus="power").frame.value.iloc[0] == 50.0
    assert s.reduce("T_cool", at(z=0.35), versus="power").frame.value.iloc[0] == 55.0


def test_reduce_with_axis_survives_a_dropped_cell(agr):
    rows = [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    rows[1] = ("plate", "T", 0, 1, np.nan)
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep.single(frame(rows), agr)
    r = s.reduce("T", "max", axis="x")
    assert r.x == "z"
    np.testing.assert_allclose(r.frame.value, [52.0, 62.0])


def test_a_quantity_missing_in_one_case_leaves_the_others_alone(agr):
    live = [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    dead = [("plate", "T", i, j, np.nan) for i in range(2) for j in range(3)]
    cases = [{"power": 2e6}, {"power": 4e6}]
    with pytest.warns(UserWarning, match="non-finite"):
        s = Sweep(cases, [frame(live), frame(dead)], agr)
    f = s.field("T", where={"power": 4e6})
    assert f.arrays["value"].shape == (2, 3) and np.isnan(f.arrays["value"]).all()
    r = s.reduce("T", "max", versus="power")
    np.testing.assert_allclose(r.frame.value, [62.0, np.nan])


def test_reduce_versus_refuses_a_reducer_that_leaves_a_profile(sweep):
    with pytest.raises(ValueError, match="'T'.*plate.*leaves a profile.*axis="):
        sweep.reduce("T", at(z=0.25), versus="power", where={"mdot": 0.3})


def test_reduce_free_parameter_becomes_dimension(sweep):
    r = sweep.reduce("CHFR", "min", versus="power")
    assert r.dims == {"mdot": [0.2, 0.3]}
    assert len(r.frame) == 8


def test_reduce_several_quantities(sweep):
    r = sweep.reduce(["CHFR", "T_cool"], "max", versus="power", where={"mdot": 0.3})
    assert r.dims == {"quantity": ["CHFR", "T_cool"]}
    assert r.ylabel == "max CHFR, Coolant temperature [°C]"


def test_reduce_scalar_passes_through(sweep):
    r = sweep.reduce("power", "min", versus="mdot", where={"power": 4e6})
    np.testing.assert_allclose(r.frame.value, [4e6, 4e6])
    assert r.ylabel == "Power [W]"


def test_reduce_where_min_and_at(sweep):
    r = sweep.reduce("CHFR", "where_min", versus="power", where={"mdot": 0.3})
    np.testing.assert_allclose(r.frame.value, 0.25)
    assert r.ylabel == "where_min CHFR [m]"
    r2 = sweep.reduce("T_cool", at(z=0.35), versus="power", where={"mdot": 0.3})
    np.testing.assert_allclose(r2.frame.value, [57.0, 59.0, 61.0, 63.0])
    assert r2.ylabel == "at z=0.35 Coolant temperature [°C]"


def test_where_with_axis_reports_the_collapsed_coordinate(sweep):
    r = sweep.reduce("T", "where_max", axis="x", where={"power": 4e6, "mdot": 0.3})
    assert r.ylabel == "where_max Temperature [m]"
    sweep.labels = {"x": ("x", "mm", 1e3)}
    scaled = sweep.reduce("T", "where_max", axis="x", where={"power": 4e6, "mdot": 0.3})
    assert scaled.ylabel == "where_max Temperature [mm]"


def test_reduce_with_axis_returns_profile(sweep):
    r = sweep.reduce("T", "max", axis="x", where={"power": 4e6, "mdot": 0.3})
    assert r.kind == "profile" and r.x == "z"
    np.testing.assert_allclose(r.frame.value, [52.0, 62.0])
    with pytest.raises(ValueError, match="versus.*axis"):
        sweep.reduce("T", "max", axis="x", versus="power", where={"mdot": 0.3})


def test_reduce_requires_versus_without_axis(sweep):
    with pytest.raises(ValueError, match="versus"):
        sweep.reduce("CHFR", "min")


def test_reduce_versus_must_be_free(sweep):
    with pytest.raises(ValueError, match="power.*where"):
        sweep.reduce("CHFR", "min", versus="power", where={"power": 4e6})


def test_reduce_skips_holes(holed_sweep):
    with pytest.warns(UserWarning, match="case 7"):
        r = holed_sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3})
    np.testing.assert_allclose(r.frame.power, [2e6, 4e6, 6e6])


def test_reduce_without_a_solved_case_names_the_selection(grid_cases, agr):
    frames = [None if case["mdot"] == 0.3 else make_case_frame(**case) for case in grid_cases]
    with pytest.warns(UserWarning, match="holes"):
        s = Sweep(grid_cases, frames, agr)
    with pytest.warns(UserWarning, match="case 1"), pytest.raises(ValueError, match="no solved case.*mdot=0.3.*'CHFR'"):
        s.reduce("CHFR", "min", versus="power", where={"mdot": 0.3})


def test_reduce_envelopes(banded_sweep):
    r = banded_sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3})
    np.testing.assert_allclose(r.frame.value - r.frame.lower, 0.15)
    np.testing.assert_allclose(r.frame.upper - r.frame.value, 0.15)


def test_quantities_scaled_differently_cannot_share_an_axis(sweep):
    sweep.labels = {"T_cool": ("Coolant", "°C", 1.0), "CHFR": ("CHF ratio", "°C", 10.0)}
    with pytest.raises(ValueError, match="'CHFR'.*scaled by 10.*'T_cool' by 1"):
        sweep.profile(["T_cool", "CHFR"], of="CC", where={"power": 4e6, "mdot": 0.3})


def test_reduce_with_two_parameters_is_a_map(sweep):
    r = sweep.reduce("CHFR", "min", versus=("power", "mdot"))
    assert r.kind == "map" and r.x == "power" and r.y == "mdot" and r.dims == {}
    assert len(r.frame) == 8 and r.holes == []
    assert r.ylabel == "min CHFR"


def test_map_refusals(sweep, agr):
    with pytest.raises(ValueError, match="versus= as a map takes two different parameters, like versus=\\('power', 'mdot'\\)"):
        sweep.reduce("CHFR", "min", versus=("power", "power"))
    with pytest.raises(ValueError, match="'Tin' is not a sweep parameter"):
        sweep.reduce("CHFR", "min", versus=("power", "Tin"))
    with pytest.raises(ValueError, match="mdot is a map axis and cannot also be fixed in where="):
        sweep.reduce("CHFR", "min", versus=("power", "mdot"), where={"mdot": 0.3})
    cs = [{"power": p, "wall_law": w} for w in ("a", "b") for p in (2e6, 4e6)]
    cat = Sweep(cs, [make_case_frame(c["power"], 0.3) for c in cs], agr)
    with pytest.raises(ValueError, match="wall_law is categorical"):
        cat.reduce("CHFR", "min", versus=("power", "wall_law"))


def test_reduce_lists_unfilled_holes_in_the_selection(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr)
    with pytest.warns(UserWarning, match="skipping unsolved"):
        r = sweep.reduce("CHFR", "min", versus=("power", "mdot"))
    assert r.holes == [grid_cases[3]]
    with pytest.warns(UserWarning, match="skipping unsolved"):
        assert sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3}).holes == [grid_cases[3]]
    assert sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.2}).holes == []


def test_map_axis_with_one_level_is_refused(agr):
    cs = [{"power": p, "mdot": 0.3} for p in (2e6, 4e6)]
    one = Sweep(cs, [make_case_frame(**c) for c in cs], agr)
    with pytest.raises(ValueError, match="mdot has one level; a map needs at least two levels on each axis"):
        one.reduce("CHFR", "min", versus=("power", "mdot"))


def test_filled_cases_are_selected_and_flagged(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    r = sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3})
    assert len(r.frame) == 4 and list(r.frame.sort_values("power").filled) == [False, True, False, False]
    assert r.holes == []
    p = sweep.profile("CHFR", where={"power": 4e6, "mdot": 0.3})
    assert p.frame.filled.all()
    f = sweep.field("T", of="plate", where={"power": 4e6, "mdot": 0.3})
    assert f.frame.filled.all() and f.subtitle.endswith(", interpolated")


def two_channel_sweep(agr, grid_cases, **limits):
    def make(power, mdot):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        rows += [("hot", "CHFR", 0, j, chfr[j] * 0.8) for j in range(4)]
        rows += [("hot", "T_cool", 0, j, T[j] + 5.0) for j in range(4)]
        return frame(rows)

    agr["hot"] = FakeChannel([0.0, 0.1, 0.2, 0.3, 0.4])
    return Sweep(grid_cases, [make(**c) for c in grid_cases], agr, limits=limits)


def test_reduce_limiting_keeps_the_binding_calculation(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.3))
    both = sweep.reduce("CHFR", "min", versus="power", of=["CC", "hot"])
    r = sweep.reduce("CHFR", "min", versus="power", of=["CC", "hot"], limiting=True)
    assert r.binding and "calculation" not in r.dims and len(r.frame) == 8
    assert set(r.frame.calculation) == {"hot"} and set(r.frame.binder) == {"CHFR of hot"}
    hot = both.frame[both.frame.calculation == "hot"].sort_values(["mdot", "power"])
    np.testing.assert_allclose(r.frame.sort_values(["mdot", "power"]).value, hot.value)


def test_reduce_limiting_refusals(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.3))
    with pytest.raises(ValueError, match="limiting needs a limit for 'T_cool'; declare it in limits="):
        sweep.reduce("T_cool", "max", versus="power", of=["CC", "hot"], limiting=True)
    with pytest.raises(ValueError, match="limiting=True but only CHFR of CC is reduced; nothing to collapse"):
        sweep.reduce("CHFR", "min", versus="power", of="CC", limiting=True)
    with pytest.raises(ValueError, match="limiting applies to versus=, not axis="):
        sweep.reduce("T", "max", of="plate", axis="x", limiting=True)
    mixed = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.3), power=at_most(5e6))
    with pytest.raises(
        ValueError,
        match=r"limiting across quantities needs them on the same side of their limits; "
        r"CHFR is at least 1\.3 and power is at most 5e\+06",
    ):
        mixed.reduce(["CHFR", "power"], "min", versus="power", of="CC", limiting=True)


def test_reduce_limiting_on_a_map(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.3))
    r = sweep.reduce("CHFR", "min", versus=("power", "mdot"), of=["CC", "hot"], limiting=True)
    assert r.kind == "map" and len(r.frame) == 8 and set(r.frame.calculation) == {"hot"}
    fig, _ax = r.plot()
    plt.close(fig)


def test_reduce_limiting_refuses_a_where_reducer(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.3))
    with pytest.raises(
        ValueError, match=r"where_min gives a location, not a value of CHFR; use min, max, mean or at\(\) with limiting=True"
    ):
        sweep.reduce("CHFR", "where_min", versus="power", of=["CC", "hot"], limiting=True)


def test_reduce_limiting_at_most_keeps_the_largest(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, T_cool=at_most(100.0))
    both = sweep.reduce("T_cool", "max", versus="power", of=["CC", "hot"])
    r = sweep.reduce("T_cool", "max", versus="power", of=["CC", "hot"], limiting=True)
    assert set(r.frame.binder) == {"T_cool of hot"}
    hot = both.frame[both.frame.calculation == "hot"].sort_values(["mdot", "power"])
    np.testing.assert_allclose(r.frame.sort_values(["mdot", "power"]).value, hot.value)


def osvr_sweep(agr, grid_cases, **limits):
    def make(power, mdot):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        rows += [("CC", "OSVR", 0, j, chfr[j] * (power / 4e6) ** 0.5) for j in range(4)]
        return frame(rows)

    return Sweep(grid_cases, [make(**c) for c in grid_cases], agr, limits=limits)


def test_reduce_limiting_across_quantities_needs_one_limit_value(agr, grid_cases):
    sweep = osvr_sweep(agr, grid_cases, CHFR=at_least(1.3), OSVR=at_least(1.1))
    with pytest.raises(
        ValueError,
        match=r"limiting across quantities needs one limit value; CHFR is at least 1\.3 and OSVR is at least 1\.1; "
        r"required\(\.\.\., limiting=True\) compares them in the walked parameter instead",
    ):
        sweep.reduce(["CHFR", "OSVR"], "min", versus="power", limiting=True)
    same = osvr_sweep(agr, grid_cases, CHFR=at_least(1.3), OSVR=at_least(1.3))
    r = same.reduce(["CHFR", "OSVR"], "min", versus="power", limiting=True)
    assert set(r.frame.binder) == {"CHFR", "OSVR"}
    fig, ax = r.plot(color="mdot", show_limit="shade")
    assert len([p for p in ax.patches if p.get_hatch()]) == 1
    assert len([ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]) == 1
    plt.close(fig)


def test_map_accepts_a_list_pair(sweep):
    r = sweep.reduce("CHFR", "min", versus=["power", "mdot"])
    assert r.kind == "map" and r.x == "power" and r.y == "mdot"
    with pytest.raises(ValueError, match="versus= as a map takes two different parameters"):
        sweep.reduce("CHFR", "min", versus=["power"])
    with pytest.raises(ValueError, match="5 is not a sweep parameter; the parameters are power, mdot"):
        sweep.reduce("CHFR", "min", versus=5)


def test_a_result_keeps_the_labels_it_was_asked_under(sweep):
    where = {"power": 4e6, "mdot": 0.3}
    r = sweep.profile("T_cool", where=where)
    sweep.labels = {"T_cool": ("Bulk temperature", "K")}
    assert r.ylabel == "Coolant temperature [°C]"
    assert sweep.profile("T_cool", where=where).ylabel == "Bulk temperature [K]"
    _, ax = r.plot()
    assert ax.get_ylabel() == "Coolant temperature [°C]"


def chfr_at(result, power):
    return result.frame[result.frame.power == power].value.iloc[0]


def test_named_reducers_collapse_a_profile(sweep):
    where = {"mdot": 0.3}
    assert chfr_at(sweep.reduce("CHFR", "min", versus="power", where=where), 4e6) == pytest.approx(1.5)
    assert chfr_at(sweep.reduce("CHFR", "max", versus="power", where=where), 4e6) == pytest.approx(3.0)
    assert chfr_at(sweep.reduce("CHFR", "mean", versus="power", where=where), 4e6) == pytest.approx(2.075)
    assert chfr_at(sweep.reduce("CHFR", "where_min", versus="power", where=where), 4e6) == pytest.approx(0.25)
    assert chfr_at(sweep.reduce("CHFR", "where_max", versus="power", where=where), 4e6) == pytest.approx(0.05)


def test_mean_is_weighted_by_cell_length(agr):
    agr["uneven"] = FakeChannel([0.0, 0.2, 0.6])
    agr["blob"] = object()
    rows = [("uneven", "v", 0, 0, 1.0), ("uneven", "v", 0, 1, 4.0), ("blob", "w", 0, 0, 1.0), ("blob", "w", 0, 1, 4.0)]
    s = Sweep([{"power": 1.0}, {"power": 2.0}], [frame(rows), frame(rows)], agr)
    assert s.reduce("v", "mean", versus="power").frame.value.iloc[0] == pytest.approx(3.0)
    assert s.reduce("w", "mean", versus="power").frame.value.iloc[0] == pytest.approx(2.5)


def test_at_picks_the_nearest_cell_or_the_named_one(sweep, agr):
    where = {"mdot": 0.3}
    assert chfr_at(sweep.reduce("CHFR", at(z=0.26), versus="power", where=where), 4e6) == pytest.approx(1.5)
    assert chfr_at(sweep.reduce("CHFR", at(cell=3), versus="power", where=where), 4e6) == pytest.approx(1.8)
    with pytest.raises(ValueError, match="cell 9 is outside the 4 cells"):
        sweep.reduce("CHFR", at(cell=9), versus="power", where=where)
    agr["blob"] = object()
    blob = Sweep([{"power": 1.0}, {"power": 2.0}], [frame([("blob", "v", 0, j, 1.0) for j in range(3)])] * 2, agr)
    with pytest.raises(ValueError, match=r"at\(z=...\) needs z coordinates"):
        blob.reduce("v", at(z=0.1), versus="power")


def test_a_callable_reducer_gives_its_value_and_its_name(sweep):
    def span(values, coords):
        return float(values[-1] - values[0])

    r = sweep.reduce("CHFR", span, versus="power", where={"mdot": 0.3})
    assert chfr_at(r, 4e6) == pytest.approx(-1.2)
    assert r.ylabel == "span CHFR"


def test_a_field_collapses_along_one_axis(sweep):
    where = {"power": 4e6, "mdot": 0.3}
    r = sweep.reduce("T", "min", axis="z", where=where)
    assert r.x == "x"
    np.testing.assert_allclose(r.frame.value, [50.0, 51.0, 52.0])
    with pytest.raises(ValueError, match="where_max on a field needs an axis; pass axis='z' or axis='x'"):
        sweep.reduce("T", "where_max", versus="power", where={"mdot": 0.3})
    with pytest.raises(ValueError, match="axis 'y' is not one of z, x"):
        sweep.reduce("T", "max", axis="y", where=where)


def test_an_unknown_reducer_raises(sweep):
    with pytest.raises(ValueError, match="unknown reducer 'median'; use one of min, max, mean"):
        sweep.reduce("CHFR", "median", versus="power")
