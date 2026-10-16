from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from conftest import channel_rows, frame, make_case_frame

from stream.solvers import AlgRuntimeError
from stream.viz import Label, Sweep, at_least, cases, run_cases


def test_run_cases_returns_the_frames_in_case_order():
    cases = [{"power": 4e6, "mdot": 0.3}, {"power": 8e6, "mdot": 0.2}]
    frames = run_cases(make_case_frame, cases)
    assert isinstance(frames, list) and len(frames) == 2 and frames.failures == {}
    for f, c in zip(frames, cases):
        pd.testing.assert_frame_equal(f, make_case_frame(**c))


def test_run_cases_turns_a_solver_failure_into_a_hole_with_its_message():
    def model(power, mdot):
        if power > 5e6:
            raise AlgRuntimeError(f"no convergence at {power}")
        return make_case_frame(power, mdot)

    cases = [{"power": 4e6, "mdot": 0.3}, {"power": 8e6, "mdot": 0.3}]
    with pytest.warns(UserWarning, match="case 1 .*power=8e\\+06.*no convergence"):
        frames = run_cases(model, cases)
    assert frames[1] is None and frames[0] is not None
    assert frames.failures == {1: "no convergence at 8000000.0"}


def test_run_cases_lets_other_exceptions_propagate():
    def buggy(power):
        raise RuntimeError("a bug in the model")

    def diverging(power):
        raise AlgRuntimeError("diverged")

    with pytest.raises(RuntimeError, match="a bug in the model"):
        run_cases(buggy, [{"power": 1.0}])
    with pytest.raises(AlgRuntimeError):
        run_cases(diverging, [{"power": 1.0}], catch=(ValueError,))


def test_run_cases_with_a_thread_pool_matches_the_serial_run():
    def model(power, mdot):
        if power == 6e6:
            raise AlgRuntimeError("diverged")
        return make_case_frame(power, mdot)

    cases = [{"power": p, "mdot": m} for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3)]
    with pytest.warns(UserWarning, match="2 of 8 cases failed"):
        serial = run_cases(model, cases)
    with ThreadPoolExecutor(max_workers=4) as pool, pytest.warns(UserWarning, match="2 of 8 cases failed"):
        threaded = run_cases(model, cases, mapper=pool.map)
    assert threaded.failures == serial.failures == {4: "diverged", 5: "diverged"}
    for a, b in zip(serial, threaded):
        if a is None:
            assert b is None
        else:
            pd.testing.assert_frame_equal(a, b)


def test_table_has_case_and_parameter_columns(sweep):
    assert sweep.parameters == ("power", "mdot")
    assert list(sweep.table.columns[:3]) == ["case", "power", "mdot"]
    assert set(sweep.table.columns[3:]) == {"calculation", "variable", "i", "j", "value", "filled"}
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
    def model(power, mdot):
        if power == 8e6 and mdot == 0.3:
            raise AlgRuntimeError("diverged")
        return make_case_frame(power, mdot)

    with pytest.warns(UserWarning, match="1 of 8 cases failed"):
        frames = run_cases(model, grid_cases)
    with pytest.warns(UserWarning, match="case 7 .*diverged"):
        s = Sweep(grid_cases, frames, agr)
    assert s.holes == {7: "diverged"}
    assert 7 not in set(s.table.case)


def test_misaligned_lists_raise(grid_cases, agr):
    with pytest.raises(ValueError, match="8 cases.*7 frames"):
        Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases[:-1]], agr)


def test_non_float_case_value_raises(agr):
    with pytest.raises(TypeError, match="case 0.*'tag'.*'up'"):
        Sweep([dict(power=1.0, tag=["up"])], [make_case_frame(1.0, 0.3)], agr)


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


def cases_of(result):
    return sorted(set(result.frame.case))


def test_where_selects_exact_partial_and_no_cases(sweep):
    assert cases_of(sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3})) == [3]
    assert cases_of(sweep.profile("T_cool", where={"power": 4e6})) == [2, 3]
    assert cases_of(sweep.reduce("CHFR", "min", versus="power")) == list(range(8))
    assert cases_of(sweep.reduce("CHFR", "min", versus="power", where={})) == list(range(8))


def test_where_tolerates_float_noise(sweep):
    r = sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3 + 1e-12})
    assert cases_of(r) == [1, 3, 5, 7] and r.fixed == {"mdot": 0.3}


def test_where_refusals(sweep):
    with pytest.raises(ValueError, match="depth.*power, mdot"):
        sweep.profile("T_cool", where={"depth": 1.0})
    with pytest.raises(ValueError, match="mdot=0.35.*0.2, 0.3"):
        sweep.profile("T_cool", where={"mdot": 0.35})
    with pytest.raises(ValueError, match="power is a number; got 'x'"):
        sweep.profile("T_cool", where={"power": "x"})


def test_where_nearest_warns_with_the_case_used(sweep):
    with pytest.warns(UserWarning, match="mdot=0.35.*mdot=0.3"):
        r = sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.35}, nearest=True)
    assert cases_of(r) == [3]


def test_holes_are_listed_on_the_sweep(holed_sweep):
    assert list(holed_sweep.holes) == [7]


def test_quantities_resolve_their_calculation_and_validate(sweep):
    assert sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3}).calculation == "CC"
    assert sweep.field("T", where={"power": 4e6, "mdot": 0.3}).calculation == "plate"
    assert sweep.profile(["T_cool", "CHFR"], of="CC", where={"power": 4e6, "mdot": 0.3}).dims == {"quantity": ["T_cool", "CHFR"]}
    with pytest.raises(ValueError, match="'OSVR'.*CC.*T_cool"):
        sweep.profile("OSVR", of="CC")
    with pytest.raises(ValueError, match="'hot'.*CC, plate"):
        sweep.profile("T_cool", of="hot")


def test_a_channel_profile_sits_on_its_cell_centres(sweep):
    r = sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3})
    np.testing.assert_allclose(r.frame.z, [0.05, 0.15, 0.25, 0.35])


def test_a_plate_field_carries_its_mesh(sweep):
    f = sweep.field("T", where={"power": 4e6, "mdot": 0.3})
    np.testing.assert_allclose(f.xs, [0.5e-3, 1.5e-3, 2.5e-3])
    np.testing.assert_allclose(f.z_bounds, [0.0, 0.5, 1.0])
    assert f.mask.shape == (2, 3)


def test_a_calculation_the_aggregator_lacks_is_named(agr):
    s = Sweep.single(frame([("ghost", "v", 0, j, float(j)) for j in range(3)]), agr)
    with pytest.raises(ValueError, match="'ghost'.*aggregator.*CC, plate"):
        s.profile("v")


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


def test_a_plate_without_bounds_weighs_its_cells_equally(agr):
    class ThinPlate:
        def __init__(self):
            self.z_centers = np.array([0.25, 0.75])
            self.x_centers = np.array([0.5e-3, 1.5e-3, 2.5e-3])

    agr["thin"] = ThinPlate()
    rows = [("thin", "v", i, j, float(3 * i + j * j)) for i in range(2) for j in range(3)]
    s = Sweep.single(frame(rows), agr)
    f = s.field("v")
    assert f.z_bounds is None and f.x_bounds is None and f.mask is None
    np.testing.assert_allclose(s.reduce("v", "mean", axis="x").frame.value, [5.0 / 3.0, 3.0 + 5.0 / 3.0])


def wall_cases():
    return [dict(power=p, wall_law=w) for w in ("regime", "Dittus") for p in (2e6, 4e6, 6e6)]


def wall_sweep(agr):
    cs = wall_cases()
    return Sweep(cs, [make_case_frame(c["power"], 0.3) for c in cs], agr)


def test_string_parameter_levels_keep_first_appearance_order(agr):
    sweep = wall_sweep(agr)
    assert list(sweep.levels("wall_law")) == ["regime", "Dittus"]
    np.testing.assert_allclose(sweep.levels("power"), [2e6, 4e6, 6e6])


def test_mixed_kinds_in_one_parameter_raise(agr):
    cs = wall_cases()
    cs[3]["wall_law"] = 1.0
    with pytest.raises(TypeError, match="parameter 'wall_law' is a number in case 3 and a string in case 0; a parameter is one or the other"):
        Sweep(cs, [make_case_frame(c["power"], 0.3) for c in cs], agr)


def test_where_matches_a_string_exactly(agr):
    sweep = wall_sweep(agr)
    assert cases_of(sweep.reduce("CHFR", "min", versus="power", where={"wall_law": "Dittus"})) == [3, 4, 5]
    with pytest.raises(ValueError, match="'Regime' is not a level of wall_law; the levels are regime, Dittus"):
        sweep.reduce("CHFR", "min", versus="power", where={"wall_law": "Regime"})
    with pytest.raises(ValueError, match="'Regime' is not a level of wall_law"):
        sweep.reduce("CHFR", "min", versus="power", where={"wall_law": "Regime"}, nearest=True)


def test_string_parameter_colours_and_titles(agr):
    sweep = wall_sweep(agr)
    r = sweep.reduce("CHFR", "min", versus="power")
    assert r.dims == {"wall_law": ["regime", "Dittus"]}
    fig, ax = r.plot()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["regime", "Dittus"]
    assert ax.get_legend().get_title().get_text() == "wall_law"
    fig, axes = r.plot(col="wall_law")
    assert axes[0, 0].get_title().startswith("wall_law = regime")
    assert sweep.profile("CHFR", where=dict(power=4e6, wall_law="Dittus")).subtitle == "Power = 4e+06 W, wall_law = Dittus, CC"


def test_string_parameter_refused_as_an_axis(agr):
    sweep = wall_sweep(agr)
    with pytest.raises(ValueError, match="wall_law is categorical; it can colour, style or panel a figure, not be an axis"):
        sweep.reduce("CHFR", "min", versus="wall_law")
    with pytest.raises(ValueError, match="wall_law is categorical"):
        sweep.required("CHFR", equals=1.5, solve_for="wall_law", versus="power")
    with pytest.raises(ValueError, match="wall_law is categorical"):
        sweep.required("CHFR", equals=1.5, solve_for="power", versus="wall_law")


def test_cases_product():
    out = cases(power=[2e6, 4e6], mdot=[0.2, 0.3], Tin=300.0, wall_law="regime")
    assert out[:2] == [dict(power=2e6, mdot=0.2, Tin=300.0, wall_law="regime"), dict(power=2e6, mdot=0.3, Tin=300.0, wall_law="regime")]
    assert len(out) == 4
    with pytest.raises(ValueError, match="power has no levels"):
        cases(power=[])
    with pytest.raises(ValueError, match="cases\\(\\) needs at least one parameter"):
        cases()


def holed(agr, grid_cases, *holes, fill=True, cases_=None):
    cs = cases_ or grid_cases
    frames = [make_case_frame(**c) for c in cs]
    for k in holes:
        frames[k] = None
    with pytest.warns(UserWarning):
        return Sweep(cs, frames, agr, fill_holes=fill)


def test_fill_holes_interpolates_a_bracketed_hole(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning) as record:
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    assert "filled unsolved case(s) by interpolation:\n  case 3 (power=4e+06, mdot=0.3): along power between case 1 and case 5" in [
        str(w.message) for w in record
    ]
    assert sweep.filled == {3: {"power": (1, 5)}} and 3 in sweep.holes
    t = sweep.table
    assert "filled" in t.columns and t.filled.sum() == len(t[t.case == 3])
    lo = t[(t.case == 1) & (t.variable == "CHFR")].sort_values("j").value.to_numpy()
    hi = t[(t.case == 5) & (t.variable == "CHFR")].sort_values("j").value.to_numpy()
    got = t[(t.case == 3) & (t.variable == "CHFR")].sort_values("j").value.to_numpy()
    np.testing.assert_allclose(got, 0.5 * (lo + hi))


def test_fill_holes_averages_two_bracketing_parameters(agr):
    cs = [dict(power=p, mdot=m) for p in (2e6, 4e6, 6e6) for m in (0.2, 0.3, 0.4)]
    sweep = holed(agr, None, 4, cases_=cs)
    assert sweep.filled == {4: {"power": (1, 7), "mdot": (3, 5)}}
    t = sweep.table

    def val(k):
        return t[(t.case == k) & (t.variable == "CHFR")].sort_values("j").value.to_numpy()

    np.testing.assert_allclose(val(4), 0.5 * (0.5 * (val(1) + val(7)) + 0.5 * (val(3) + val(5))))


def test_fill_holes_leaves_edge_and_adjacent_holes(agr, grid_cases):
    sweep = holed(agr, grid_cases, 0)
    assert sweep.filled == {} and 0 in sweep.holes
    sweep = holed(agr, grid_cases, 3, 5)
    assert sweep.filled == {}
    assert not sweep.table.filled.any()


def test_fill_holes_off_by_default_keeps_the_column(agr, grid_cases):
    sweep = holed(agr, grid_cases, 3, fill=False)
    assert sweep.filled == {} and "filled" in sweep.table.columns and not sweep.table.filled.any()
    assert "filled" in Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr).table.columns


def test_repr_lists_what_can_be_asked(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, limits={"CHFR": at_least(1.3)}, fill_holes=True)
    text = repr(sweep)
    assert text.splitlines() == [
        "Sweep: 8 cases, 1 hole (1 filled), no uncertainty",
        "  parameters: power [2e+06, 4e+06, 6e+06, 8e+06], mdot [0.2, 0.3]",
        "  CC: CHFR (profile), T_cool (profile), power (scalar)",
        "  plate: T (field)",
        "  limits: CHFR at least 1.3",
        "  holes: case 3 (power=4e+06, mdot=0.3), filled along power",
    ]


def test_fill_holes_interpolates_the_bands(agr, grid_cases):
    def make(power, mdot):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        return frame(rows, sys=np.full(len(rows), power / 1e7), stat=np.full(len(rows), mdot / 10))

    frames = [make(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    assert sweep.banded and sweep.filled == {3: {"power": (1, 5)}}
    rows = sweep.table[sweep.table.case == 3]
    np.testing.assert_allclose(rows.sys, 0.4)
    np.testing.assert_allclose(rows.stat, 0.03)


def test_repr_counts_filled_holes_only_when_filling(agr, grid_cases):
    sweep = holed(agr, grid_cases, 3, fill=False)
    assert repr(sweep).splitlines()[0] == "Sweep: 8 cases, 1 hole, no uncertainty"


def test_cases_refuses_none_levels():
    with pytest.raises(ValueError, match="x has no levels"):
        cases(x=None)
