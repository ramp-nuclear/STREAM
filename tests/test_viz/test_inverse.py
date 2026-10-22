import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import FakeChannel, channel_rows, frame, make_case_frame

from stream.viz import Sweep, at, at_least, at_most


def scalar_sweep(agr, values, depths=(1.0,)):
    cs = [{"power": float(p), "mdot": m, "depth": d} for p in range(1, len(values) + 1) for m in (0.2, 0.3) for d in depths]
    return Sweep(cs, [frame([("CC", "q", 0, 0, values[int(c["power"]) - 1] * c["depth"])]) for c in cs], agr)


def test_required_interpolates_linearly_between_grid_points(agr):
    r = scalar_sweep(agr, [4.0, 2.0, 1.0, 0.5]).required("q", equals=1.5, solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, 2.5)


def test_required_pchip_follows_a_curved_walk_closer(agr):
    s = scalar_sweep(agr, [8.0 / x for x in (1.0, 2.0, 3.0, 4.0, 5.0)])
    linear = s.required("q", equals=3.0, solve_for="power", versus="mdot").frame.value.iloc[0]
    pchip = s.required("q", equals=3.0, solve_for="power", versus="mdot", method="pchip").frame.value.iloc[0]
    assert 2.0 < pchip < 3.0
    assert abs(pchip - 8.0 / 3.0) < abs(linear - 8.0 / 3.0)


def test_required_warns_on_several_crossings_and_on_none(agr):
    s = scalar_sweep(agr, [1.0, 3.0, 1.0, 3.0])
    with pytest.warns(UserWarning, match=r"q at mdot=0\.[23], depth=1: 3 crossings of 2 at 1.5, 2.5, 3.5; using the first"):
        r = s.required("q", equals=2.0, solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, 1.5)
    with pytest.warns(UserWarning, match=r"q at mdot=0\.[23], depth=1: the curve never reaches 9 over 1 to 4"):
        r = s.required("q", equals=9.0, solve_for="power", versus="mdot")
    assert np.isnan(r.frame.value).all()


def test_required_at_a_parameter_on_and_off_the_grid(agr):
    s = scalar_sweep(agr, [1.0, 2.0, 3.0, 4.0], depths=(1.0, 2.0))
    np.testing.assert_allclose(s.required("q", equals=3.0, solve_for="power", versus="mdot", at={"depth": 2.0}).frame.value, 1.5)
    with pytest.warns(UserWarning, match="depth=1.5 is not on the grid; interpolating between 1 and 2"):
        r = s.required("q", equals=3.0, solve_for="power", versus="mdot", at={"depth": 1.5})
    np.testing.assert_allclose(r.frame.value, 2.0)
    assert r.fixed == {"depth": 1.5}
    with pytest.raises(ValueError, match="depth=3 is outside the grid; the values of depth are 1, 2"):
        s.required("q", equals=3.0, solve_for="power", versus="mdot", at={"depth": 3.0})


def test_required_power_for_min_chfr(sweep):
    r = sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert r.kind == "curve" and r.x == "mdot"
    assert list(r.frame.columns) == ["quantity", "calculation", "mdot", "lower", "value", "upper", "inner_lower", "inner_upper", "safe", "filled"]
    np.testing.assert_allclose(r.frame.mdot, [0.2, 0.3])
    np.testing.assert_allclose(r.frame.value, [3e6, 4e6], rtol=1e-6)
    assert r.ylabel == "Power [W]"
    assert r.question.reducer == "min"
    assert r.dims == {} and r.fixed == {}
    assert r.question.targets == {"CHFR": 1.5} and r.question.quantities == ("CHFR",)
    assert r.subtitle == "min CHFR = 1.5, CC"
    fig, ax = r.plot()
    assert ax.get_title() == "min CHFR = 1.5, CC"
    plt.close(fig)


def test_required_with_free_dimension_and_at(agr):
    cases = [{"power": p, "mdot": m, "depth": d} for p in (2e6, 4e6, 8e6) for m in (0.2, 0.3) for d in (1.0, 2.0)]
    s = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr)
    r = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert r.dims == {"depth": [1.0, 2.0]}
    assert len(r.frame) == 4
    with pytest.warns(UserWarning, match="depth=1.5"):
        r2 = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", at={"depth": 1.5})
    assert r2.dims == {} and r2.fixed == {"depth": 1.5}
    assert len(r2.frame) == 2


def test_required_role_conflicts(sweep):
    with pytest.raises(ValueError, match="solve_for.*versus"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="power")
    with pytest.raises(ValueError, match="power.*where"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", where={"power": 4e6})


def test_required_needs_two_points(holed_sweep):
    cases = [{"power": p, "mdot": 0.3} for p in (2e6, 4e6)]
    frames = [holed_sweep.frames[1], None]
    with pytest.warns(UserWarning):
        s = Sweep(cases, frames, holed_sweep.agr)
    with pytest.warns(UserWarning, match="skipping unsolved"), pytest.raises(ValueError, match="power.*1 solved point"):
        s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")


def test_required_bracket_over_hole_warns(holed_sweep):
    with pytest.warns(UserWarning) as record:
        r = holed_sweep.required("CHFR", equals=0.9, reduce="min", solve_for="power", versus="mdot")
    assert any("case 7" in str(w.message) for w in record)
    assert len([w for w in record if "skipping unsolved" in str(w.message)]) == 1
    assert np.isnan(r.frame[r.frame.mdot == 0.3].value.iloc[0])


def test_required_bracket_spanning_a_hole_names_the_case(agr):
    cases = [{"power": p, "mdot": 0.3} for p in (2e6, 4e6, 6e6)]
    frames = [make_case_frame(c["power"], c["mdot"]) for c in cases]
    frames[1] = None
    with pytest.warns(UserWarning, match="holes"):
        s = Sweep(cases, frames, agr)
    with pytest.warns(UserWarning) as record:
        r = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert any("interpolated across missing case 1 (power=4e+06, mdot=0.3)" in str(w.message) for w in record)
    assert r.frame.value.iloc[0] == pytest.approx(5e6)


def test_required_quantity_target(agr):
    def rows(power):
        rising = [("CC", "hot", 0, j, power / 1e6 + j) for j in range(4)]
        falling = [("CC", "cold", 0, j, 10.0 - power / 1e6 + j) for j in range(4)]
        return frame(rising + falling)

    cases = [{"power": p, "mdot": m} for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3)]
    s = Sweep(cases, [rows(c["power"]) for c in cases], agr)
    r = s.required("hot", equals="cold", reduce="max", solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, [5e6, 5e6])
    assert r.question.targets == {"hot": "cold"} and r.question.quantities == ("hot",)
    assert r.subtitle == "max hot = cold, CC"
    assert r.ylabel == "Power [W]"


def test_required_rejects_an_unknown_method(sweep):
    with pytest.raises(ValueError, match="'cubic'.*'linear'.*'pchip'"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", method="cubic")


def test_required_rejects_a_parameter_fixed_twice(agr):
    cases = [{"power": p, "mdot": m, "depth": d} for p in (2e6, 4e6) for m in (0.2, 0.3) for d in (1.0, 2.0)]
    s = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr)
    with pytest.raises(ValueError, match="depth.*where.*at"):
        s.required(
            "CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot",
            where={"depth": 1.0}, at={"depth": 1.0},
        )


def test_required_envelopes_are_the_three_walks(banded_sweep):
    r = banded_sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    row = r.frame[r.frame.mdot == 0.3].iloc[0]
    assert row.lower < row.value < row.upper
    rising = banded_sweep.required("T_cool", equals=60.0, reduce="max", solve_for="power", versus="mdot")
    up = rising.frame[rising.frame.mdot == 0.3].iloc[0]
    assert up.value == pytest.approx(5e6)
    assert up.lower <= up.value <= up.upper
    assert up.inner_lower <= up.value <= up.inner_upper


def limited_sweep(agr, grid_cases, **limits):
    return Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr, limits=limits)


def test_required_without_equals_uses_the_declared_limit(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5))
    r = sweep.required("CHFR", solve_for="power", versus="mdot")
    explicit = sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, explicit.frame.value)
    assert list(r.frame.safe) == ["below", "below"]
    assert r.safe_side == "below"
    assert r.question.targets == {"CHFR": 1.5}


def test_required_without_equals_and_without_limit_raises(sweep):
    with pytest.raises(ValueError, match="no limit declared for 'CHFR'; pass equals= or declare it in limits="):
        sweep.required("CHFR", solve_for="power", versus="mdot")


def test_required_explicit_equals_has_no_side(sweep):
    r = sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot")
    assert r.frame.safe.isna().all() and r.safe_side is None
    assert r.subtitle == "min CHFR = 1.5, CC"


def test_required_dict_targets_and_reducers(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases)
    r = sweep.required(["CHFR", "T_cool"], equals={"CHFR": 1.5, "T_cool": 60.0},
                       reduce={"CHFR": "min", "T_cool": "max"}, solve_for="power", versus="mdot")
    assert set(r.frame.quantity) == {"CHFR", "T_cool"}
    chfr = sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot").frame
    np.testing.assert_allclose(r.frame[r.frame.quantity == "CHFR"].value.to_numpy(), chfr.value.to_numpy())
    assert r.question.targets == {"CHFR": 1.5, "T_cool": 60.0}
    assert "min CHFR = 1.5, max Coolant temperature = 60 °C" in r.subtitle


def test_required_dict_with_wrong_keys_raises(sweep):
    with pytest.raises(ValueError, match="equals= has keys CHFR, OSVr but the quantities are CHFR, T_cool; give one entry per quantity"):
        sweep.required(["CHFR", "T_cool"], equals={"CHFR": 1.5, "OSVr": 1.1}, solve_for="power", versus="mdot")


def test_required_default_reducer_follows_the_side(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, T_cool=at_most(60.0))
    r = sweep.required("T_cool", solve_for="power", versus="mdot")
    explicit = sweep.required("T_cool", equals=60.0, reduce="max", solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, explicit.frame.value)
    np.testing.assert_allclose(r.frame.value, [5e6, 5e6])
    assert list(r.frame.safe) == ["below", "below"]
    assert r.question.reducer_of("T_cool") == "max"


def test_required_safe_above_when_the_quantity_improves_with_solve_for(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5))
    with pytest.warns(UserWarning):
        r = sweep.required("CHFR", solve_for="mdot", versus="power")
    assert set(r.frame.safe[r.frame.value.notna()]) == {"above"}


def test_required_missing_solve_for_raises(sweep):
    with pytest.raises(ValueError, match="required\\(\\) needs solve_for="):
        sweep.required("CHFR", equals=1.5, versus="mdot")


def test_required_limiting_keeps_the_binding_quantity(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_most(60.0))
    both = sweep.required(["CHFR", "T_cool"], equals={"CHFR": 1.5, "T_cool": 60.0}, solve_for="power", versus="mdot")
    r = sweep.required(["CHFR", "T_cool"], equals={"CHFR": 1.5, "T_cool": 60.0}, solve_for="power", versus="mdot", limiting=True)
    assert r.binding is True and "quantity" not in r.dims and len(r.frame) == 2
    assert list(r.frame.binder) == ["CHFR", "CHFR"]
    chfr = both.frame[both.frame.quantity == "CHFR"].sort_values("mdot")
    np.testing.assert_allclose(r.frame.sort_values("mdot").value, chfr.value)
    assert r.subtitle.startswith("limiting of CHFR, Coolant temperature")


def test_required_limiting_needs_sides(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5))
    with pytest.raises(ValueError, match="limiting needs the side of every limit; declare T_cool in limits="):
        sweep.required(
            ["CHFR", "T_cool"], equals={"CHFR": 1.5, "T_cool": 60.0}, reduce={"CHFR": "min", "T_cool": "max"},
            solve_for="power", versus="mdot", limiting=True,
        )


def test_required_limiting_with_nothing_to_collapse_raises(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5))
    with pytest.raises(ValueError, match="limiting=True but only CHFR of CC is walked; nothing to collapse"):
        sweep.required("CHFR", solve_for="power", versus="mdot", limiting=True)


def test_required_limiting_with_disagreeing_sides_raises(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_least(45.0))
    with pytest.raises(ValueError, match="CHFR is safe below the required power and T_cool above it; the feasible region is an interval, not one curve"):
        sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot", limiting=True)


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


def test_required_limiting_skips_a_quantity_safe_everywhere(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_most(1e9))
    with pytest.warns(UserWarning, match="the curve never reaches 1e\\+09"):
        both = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot")
    with pytest.warns(UserWarning, match="never reaches"):
        r = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot", limiting=True)
    assert list(both.frame[both.frame.quantity == "T_cool"].safe) == ["everywhere", "everywhere"]
    assert list(r.frame.binder) == ["CHFR", "CHFR"]
    chfr = both.frame[both.frame.quantity == "CHFR"].sort_values("mdot")
    np.testing.assert_allclose(r.frame.sort_values("mdot").value, chfr.value)


def test_required_limiting_with_a_quantity_unsafe_everywhere_gives_nan(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_most(10.0))
    with pytest.warns(UserWarning, match="never reaches 10"):
        r = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot", limiting=True)
    assert len(r.frame) == 2 and r.frame.value.isna().all()
    assert list(r.frame.safe) == ["nowhere", "nowhere"]
    assert list(r.frame.binder) == ["T_cool", "T_cool"]


def test_required_limiting_across_calculations_names_the_calculation(agr, grid_cases):
    sweep = two_channel_sweep(agr, grid_cases, CHFR=at_least(1.5))
    both = sweep.required("CHFR", solve_for="power", versus="mdot", of=["CC", "hot"])
    r = sweep.required("CHFR", solve_for="power", versus="mdot", of=["CC", "hot"], limiting=True)
    assert r.binding and "calculation" not in r.dims and len(r.frame) == 2
    assert set(r.frame.binder) == {"CHFR of hot"}
    hot = both.frame[both.frame.calculation == "hot"].sort_values("mdot")
    np.testing.assert_allclose(r.frame.sort_values("mdot").value, hot.value)


def test_required_quantity_target_with_a_limit_has_a_side(agr):
    def rows(power):
        rising = [("CC", "hot", 0, j, power / 1e6 + j) for j in range(4)]
        falling = [("CC", "cold", 0, j, 10.0 - power / 1e6 + j) for j in range(4)]
        return frame(rising + falling)

    cases = [{"power": p, "mdot": m} for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3)]
    s = Sweep(cases, [rows(c["power"]) for c in cases], agr, limits={"hot": at_most(0.0)})
    r = s.required("hot", equals="cold", reduce="max", solve_for="power", versus="mdot")
    assert list(r.frame.safe) == ["below", "below"] and r.safe_side == "below"


def test_safe_side_from_the_first_point_after_a_crossing_on_the_grid_start(agr, grid_cases):
    sweep = limited_sweep(agr, grid_cases, T_cool=at_most(57.0))
    r = sweep.required("T_cool", solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, [2e6, 2e6])
    assert list(r.frame.safe) == ["below", "below"]


def test_required_crossing_on_a_filled_case(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    with pytest.warns(UserWarning, match="rests on interpolated case 3"):
        r = sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot")
    assert list(r.frame.sort_values("mdot").filled) == [False, True]


def test_required_subtitle_prints_the_reducer_position_with_its_label(sweep):
    r = sweep.required("T_cool", equals=60.0, reduce=at(z=0.35), solve_for="power", versus="mdot")
    assert r.subtitle == "at z=0.35 Coolant temperature = 60 °C, CC"
    fig, ax = r.plot(labels={"z": ("z", "mm", 1e3, ".0f")})
    assert ax.get_title() == "at z=350 Coolant temperature = 60 °C, CC"
    plt.close(fig)
