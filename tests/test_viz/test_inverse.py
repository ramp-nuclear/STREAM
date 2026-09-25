import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from stream.viz.inverse import crossings, first_crossing, slice_at


def test_crossings_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 2.0, 1.0, 0.5])
    assert crossings(x, y, 1.5) == [pytest.approx(2.5)]
    assert crossings(x, y, 4.0) == [pytest.approx(1.0)]
    assert crossings(x, y, 5.0) == []
    assert crossings(x, np.array([1.0, 3.0, 1.0, 3.0]), 2.0) == [pytest.approx(v) for v in (1.5, 2.5, 3.5)]


def test_crossings_pchip_is_monotone_between_points():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 8.0 / x
    (c,) = crossings(x, y, 3.0, method="pchip")
    assert 2.0 < c < 3.0
    assert abs(c - 8.0 / 3.0) < abs(crossings(x, y, 3.0)[0] - 8.0 / 3.0)


def test_first_crossing_warnings():
    x = np.array([1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match="CHFR at mdot=0.3.*2 crossings.*1.5"):
        assert first_crossing(x, np.array([1.0, 3.0, 1.0]), 2.0, "linear", "CHFR at mdot=0.3") == pytest.approx(1.5)
    with pytest.warns(UserWarning, match="CHFR at mdot=0.3.*never reaches 9.*1 to 3"):
        assert np.isnan(first_crossing(x, np.array([1.0, 3.0, 1.0]), 9.0, "linear", "CHFR at mdot=0.3"))


def test_slice_at_on_grid_filters(sweep):
    curve = sweep.reduce("CHFR", "min", versus="power").frame
    out = slice_at(curve, "mdot", 0.3, sweep)
    assert set(out.mdot) == {0.3} and len(out) == 4


def test_slice_at_off_grid_blends_and_warns(sweep):
    curve = sweep.reduce("CHFR", "min", versus="power").frame
    with pytest.warns(UserWarning, match="mdot=0.25.*0.2.*0.3"):
        out = slice_at(curve, "mdot", 0.25, sweep)
    lo = curve[curve.mdot == 0.2].sort_values("power").value.to_numpy()
    hi = curve[curve.mdot == 0.3].sort_values("power").value.to_numpy()
    np.testing.assert_allclose(out.sort_values("power").value, 0.5 * (lo + hi))
    assert set(out.mdot) == {0.25}


def test_slice_at_outside_grid_raises(sweep):
    curve = sweep.reduce("CHFR", "min", versus="power").frame
    with pytest.raises(ValueError, match="mdot=0.5.*0.2.*0.3"):
        slice_at(curve, "mdot", 0.5, sweep)


def test_required_power_for_min_chfr(sweep):
    r = sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert r.kind == "curve" and r.x == "mdot"
    assert list(r.frame.columns) == ["quantity", "calculation", "mdot", "lower", "value", "upper", "inner_lower", "inner_upper"]
    np.testing.assert_allclose(r.frame.mdot, [0.2, 0.3])
    np.testing.assert_allclose(r.frame.value, [3e6, 4e6], rtol=1e-6)
    assert r.ylabel == "Power [W]"
    assert r.reducer is None and r.walk_reducer == "min"
    assert r.ylabel_source == ("power",)
    assert r.dims == {} and r.fixed == {}
    assert r.target == 1.5 and r.walk_quantity == "CHFR"
    assert r.subtitle == "min CHFR = 1.5, CC"
    fig, ax = r.plot()
    assert ax.get_title() == "min CHFR = 1.5, CC"
    plt.close(fig)


def test_required_with_free_dimension_and_at(agr):
    from conftest import make_case_frame

    from stream.viz.sweep import Sweep

    cases = [dict(power=p, mdot=m, depth=d) for p in (2e6, 4e6, 8e6) for m in (0.2, 0.3) for d in (1.0, 2.0)]
    s = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr)
    r = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert r.dims == {"depth": [1.0, 2.0]}
    assert len(r.frame) == 4
    with pytest.warns(UserWarning, match="depth=1.5"):
        r2 = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", at=dict(depth=1.5))
    assert r2.dims == {} and r2.fixed == dict(depth=1.5)
    assert len(r2.frame) == 2


def test_required_role_conflicts(sweep):
    with pytest.raises(ValueError, match="solve_for.*versus"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="power")
    with pytest.raises(ValueError, match="power.*where"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", where=dict(power=4e6))


def test_required_needs_two_points(holed_sweep):
    cases = [dict(power=p, mdot=0.3) for p in (2e6, 4e6)]
    frames = [holed_sweep.frames[1], None]
    from stream.viz.sweep import Sweep

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
    from conftest import make_case_frame

    from stream.viz.sweep import Sweep

    cases = [dict(power=p, mdot=0.3) for p in (2e6, 4e6, 6e6)]
    frames = [make_case_frame(c["power"], c["mdot"]) for c in cases]
    frames[1] = None
    with pytest.warns(UserWarning, match="holes"):
        s = Sweep(cases, frames, agr)
    with pytest.warns(UserWarning) as record:
        r = s.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    assert any("interpolated across missing case 1 (power=4e+06, mdot=0.3)" in str(w.message) for w in record)
    assert r.frame.value.iloc[0] == pytest.approx(5e6)


def test_required_quantity_target(agr):
    from conftest import frame

    from stream.viz.sweep import Sweep

    def rows(power):
        rising = [("CC", "hot", 0, j, power / 1e6 + j) for j in range(4)]
        falling = [("CC", "cold", 0, j, 10.0 - power / 1e6 + j) for j in range(4)]
        return frame(rising + falling)

    cases = [dict(power=p, mdot=m) for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3)]
    s = Sweep(cases, [rows(c["power"]) for c in cases], agr)
    r = s.required("hot", equals="cold", reduce="max", solve_for="power", versus="mdot")
    np.testing.assert_allclose(r.frame.value, [5e6, 5e6])
    assert r.target == "cold" and r.walk_quantity == "hot"
    assert r.subtitle == "max hot = cold, CC"
    assert r.ylabel == "Power [W]"


def test_required_rejects_an_unknown_method(sweep):
    with pytest.raises(ValueError, match="'cubic'.*'linear'.*'pchip'"):
        sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot", method="cubic")


def test_required_rejects_a_parameter_fixed_twice(agr):
    from conftest import make_case_frame

    from stream.viz.sweep import Sweep

    cases = [dict(power=p, mdot=m, depth=d) for p in (2e6, 4e6) for m in (0.2, 0.3) for d in (1.0, 2.0)]
    s = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr)
    with pytest.raises(ValueError, match="depth.*where.*at"):
        s.required(
            "CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot",
            where=dict(depth=1.0), at=dict(depth=1.0),
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
