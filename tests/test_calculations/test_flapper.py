import numpy as np
import pytest
from hypothesis import given, settings

from stream.calculations import Flapper
from stream.calculations.flapper import continuously_differentiable_relaxation
from stream.substances import light_water
from stream.utilities import directed_Tin

from .conftest import medium_floats


@pytest.mark.implementation
@settings(deadline=None)
@given(medium_floats, medium_floats, medium_floats)
def test_fully_open_flapper_acts_as_resistor(mdot, Tin, Tin_minus):
    F = Flapper(open_at_current=1.0, f=2.0, fluid=light_water, area=1, open_rate=1.0)
    F.t_open = 100.0
    result = F.calculate([0, 1.0], mdot=mdot, Tin=Tin, Tin_minus=Tin_minus, t=101.0)
    rho = light_water.density
    T = directed_Tin(Tin, Tin_minus, mdot)
    assert np.allclose(result, [-T, mdot + np.sqrt((2 * rho(T)) / F.f)])


@given(medium_floats, medium_floats)
def test_closed_flapper_zero_flow_residue_is_zero(Tin, Tin_minus):
    F = Flapper(
        open_at_current=1.0,
        f=2.0,
        fluid=light_water,
        area=1,
        open_rate=1.0,
        stop_on_open=True,
    )
    result = F.calculate([0, 0], mdot=0.0, Tin=Tin, Tin_minus=Tin_minus, t=0)

    assert result[F.indices("pressure")] == 0.0


def test_flapper_should_not_continue_at_opening_condition():
    F = Flapper(
        open_at_current=1.0,
        f=2.0,
        fluid=light_water,
        area=1,
        open_rate=1.0,
        stop_on_open=True,
    )
    assert F.should_continue([0, 0], ref_mdot=0.5, t=100.0)
    assert np.isposinf(F.t_open)
    F.change_state([0, 0], ref_mdot=0.5, t=100.0)
    assert not F.should_continue([0, 0], ref_mdot=0.5, t=100.0)
    assert F.t_open == 100.0
    assert not F.should_continue([0, 0], ref_mdot=0.5, t=101.0)


def test_flapper_change_state_is_idempotent_and_margin_is_consumed():
    """change_state latches the opening once and stays latched when re-evaluated
    at the same or a later time, and the event margin is consumed after opening."""
    F = Flapper(open_at_current=1.0, f=2.0, fluid=light_water, area=1, open_rate=1.0)
    assert F.event_margin([0, 0], ref_mdot=2.0) > 0
    F.change_state([0, 0], ref_mdot=0.5, t=100.0)
    assert F.t_open == 100.0
    assert F._latched
    F.change_state([0, 0], ref_mdot=0.5, t=100.0)
    F.change_state([0, 0], ref_mdot=0.05, t=100.0)
    assert F.t_open == 100.0
    assert F.event_margin([0, 0], ref_mdot=0.5) == 1.0


def test_flapper_residual_is_continuous_in_time_across_opening():
    """The temperature residual does not jump at t_open: the inlet temperature is
    blended with the relaxation ramp, so just after opening it matches the closed
    branch even when Tin and Tin_minus differ."""
    F = Flapper(
        open_at_current=1.0,
        f=2.0,
        fluid=light_water,
        area=1,
        open_rate=1.0,
        relaxation=continuously_differentiable_relaxation,
    )
    F.open(10.0)
    Tin, Tin_minus = 100.0, 50.0
    kw = {"mdot": 0.0, "Tin": Tin, "Tin_minus": Tin_minus}
    closed = F.calculate([0.0, -1.0], t=10.0, **kw)
    just_open = F.calculate([0.0, -1.0], t=10.0 + 1e-9, **kw)
    assert abs(just_open[0] - closed[0]) < 1e-3
