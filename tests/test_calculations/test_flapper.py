from hypothesis import given, settings
import numpy as np
import pytest

from stream.calculations import Flapper
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
    assert F.should_continue([0, 0], ref_mdot=0.5, t=101.0)
