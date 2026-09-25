import numpy as np
import pytest
from hypothesis import given, settings

from stream.calculations import Flapper
from stream.calculations.flapper import continuously_differentiable_relaxation
from stream.physical_models.pressure_drop import mdot_by_local_pressure_smooth
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
    # relax=1 at t_open+1/open_rate; the open branch uses the regularized inverse
    # law, so out[1] carries mdot_by_local_pressure_smooth at dp=1.0 (which
    # sits at the default dp_eps=1.0 band), not the exact sqrt law.
    mdot_calc = mdot_by_local_pressure_smooth(1.0, rho(T), F.f, F._A, F.dp_eps)
    assert np.allclose(result, [-T, mdot + mdot_calc])


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
    # Sticky: once opened the run stays stopped at every later time, not
    # only at the exact opening instant (the old `t_open == t` edge-equality).
    assert not F.should_continue([0, 0], ref_mdot=0.5, t=101.0)


def test_flapper_change_state_is_idempotent_and_margin_is_consumed():
    """E2: change_state cleared its flag on every call, so re-evaluating the root
    at t_open made the condition vanish. It must latch once and stay latched, and
    the event margin must be consumed (never re-fire) after opening."""
    F = Flapper(open_at_current=1.0, f=2.0, fluid=light_water, area=1, open_rate=1.0)
    assert F.event_margin([0, 0], ref_mdot=2.0) > 0  # above threshold: positive
    F.change_state([0, 0], ref_mdot=0.5, t=100.0)
    assert F.t_open == 100.0
    assert F._latched
    # Re-evaluate at the same time and later: no un-latch, margin stays consumed.
    F.change_state([0, 0], ref_mdot=0.5, t=100.0)
    F.change_state([0, 0], ref_mdot=0.05, t=100.0)
    assert F.t_open == 100.0
    assert F.event_margin([0, 0], ref_mdot=0.5) == 1.0  # consumed, not re-firing


def test_flapper_residual_is_continuous_in_time_across_opening():
    """E2: out[0] jumped by (Tin - Tin_minus)/2 at t_open (directed_Tin midpoint
    vs pure Tin) with no restart, invalidating IDA's history. Blending the inlet
    temperature with the relaxation ramp makes it continuous in t at t_open."""
    F = Flapper(
        open_at_current=1.0,
        f=2.0,
        fluid=light_water,
        area=1,
        open_rate=1.0,
        relaxation=continuously_differentiable_relaxation,
    )
    F.open(10.0)  # t_open = 10
    Tin, Tin_minus = 100.0, 50.0
    kw = dict(mdot=0.0, Tin=Tin, Tin_minus=Tin_minus)
    closed = F.calculate([0.0, -1.0], t=10.0, **kw)  # at t_open: closed branch
    just_open = F.calculate([0.0, -1.0], t=10.0 + 1e-9, **kw)  # relax ~ 0
    # out[0] must not jump (was a 25 K step for this 50 K delta).
    assert abs(just_open[0] - closed[0]) < 1e-3
