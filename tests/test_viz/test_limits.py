import numpy as np
import pytest
from conftest import make_case_frame

from stream.viz import Limit, Sweep, at_least, at_most


def test_at_least_and_at_most_sides_and_reducers():
    lo, hi = at_least(1.3), at_most(900)
    assert lo == Limit(1.3, "above") and hi == Limit(900.0, "below")
    assert lo.reducer == "min" and hi.reducer == "max"
    assert lo.text() == "at least 1.3" and hi.text() == "at most 900"
    np.testing.assert_array_equal(lo.unsafe([1.0, 1.3, 2.0]), [True, False, False])
    np.testing.assert_array_equal(hi.unsafe([800.0, 900.0, 950.0]), [False, False, True])


@pytest.mark.parametrize("bad", ["1.3", True, None])
def test_limit_helpers_take_numbers_only(bad):
    with pytest.raises(TypeError, match="at_least\\(\\) takes a number"):
        at_least(bad)
    with pytest.raises(TypeError, match="at_most\\(\\) takes a number"):
        at_most(bad)


def test_sweep_keeps_limits_read_only(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    sweep = Sweep(grid_cases, frames, agr, limits={"CHFR": at_least(1.3)})
    assert sweep.limits["CHFR"] == at_least(1.3)
    with pytest.raises(TypeError):
        sweep.limits["T_cool"] = at_most(100.0)
    assert Sweep(grid_cases, frames, agr).limits == {}


def test_sweep_rejects_unknown_quantity_and_wrong_type(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    with pytest.raises(ValueError, match="'OSVR' in limits= is not a variable of any calculation; the variables are CHFR, T, T_cool, power"):
        Sweep(grid_cases, frames, agr, limits={"OSVR": at_least(1.1)})
    with pytest.raises(TypeError, match="limits\\['CHFR'\\] must come from at_least\\(\\) or at_most\\(\\), not 1.3"):
        Sweep(grid_cases, frames, agr, limits={"CHFR": 1.3})


def test_results_carry_the_limits_of_their_quantities(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    sweep = Sweep(grid_cases, frames, agr, limits={"CHFR": at_least(1.3)})
    assert sweep.profile("CHFR", where=dict(mdot=0.3)).limits == {"CHFR": at_least(1.3)}
    assert sweep.profile("T_cool", where=dict(mdot=0.3)).limits == {}
    assert sweep.reduce("CHFR", "min", versus="power").limits == {"CHFR": at_least(1.3)}
    assert sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3)).limits == {}
    r = sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot")
    assert r.limits == {"CHFR": at_least(1.3)} and r.safe_side == "below" and r.binding is False and r.y is None


def test_limit_refuses_an_unknown_side():
    with pytest.raises(ValueError, match="side must be 'above' or 'below', not 'abov'"):
        Limit(1.3, "abov")


def test_limits_must_be_a_mapping(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    with pytest.raises(TypeError, match=r"limits= must be a mapping from quantity to a limit from at_least\(\) or at_most\(\), not"):
        Sweep(grid_cases, frames, agr, limits=[at_least(1.3)])
