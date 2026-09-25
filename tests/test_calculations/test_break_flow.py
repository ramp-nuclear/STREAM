import cloudpickle
import numpy as np
import pytest

from stream.calculations import Orifice, Tank
from stream.physical_models.pressure_drop import lichtarowicz_cd, mdot_by_local_pressure_smooth
from stream.substances import light_water
from stream.utilities import directed_Tin

AREA = 2e-3
CD = 0.61


def _breaker(**kwargs) -> Orifice:
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=0.5, fixed_temperature=30.0)
    return Orifice(light_water, AREA, CD, closes_below=(tank, 2.0), **kwargs)


def test_sealed_orifice_forces_zero_flow_and_frees_pressure():
    o = Orifice(light_water, AREA, CD)
    out = o.calculate([25.0, -7e4], mdot=0.4, Tin=30.0, Tin_minus=20.0, t=3.0)
    assert out[0] == pytest.approx(25.0 - 30.0)
    assert out[1] == pytest.approx(0.4)
    unaffected = o.calculate([25.0, 5e5], mdot=0.4, Tin=30.0, Tin_minus=20.0, t=3.0)
    assert np.allclose(out, unaffected)


def test_open_orifice_matches_smooth_inverse_law():
    o = Orifice(light_water, AREA, CD)
    o.open(0.0)
    dp, mdot, Tin, Tin_minus = -5e4, 1.5, 40.0, 30.0
    out = o.calculate([0.0, dp], mdot=mdot, Tin=Tin, Tin_minus=Tin_minus, t=10.0)
    Tin_d = directed_Tin(Tin, Tin_minus, mdot)
    law = mdot_by_local_pressure_smooth(dp, light_water.density(Tin_d), 1 / CD**2, AREA, o.dp_eps)
    assert out[1] == pytest.approx(mdot + law)
    assert out[0] == pytest.approx(-Tin_d)


def test_residual_is_continuous_across_opening():
    o = Orifice(light_water, AREA, CD)
    o.open(10.0)
    kw = dict(mdot=0.0, Tin=100.0, Tin_minus=50.0)
    closed = o.calculate([0.0, -1e4], t=10.0, **kw)
    just_open = o.calculate([0.0, -1e4], t=10.0 + 1e-9, **kw)
    assert abs(just_open[0] - closed[0]) < 1e-3
    assert abs(just_open[1] - closed[1]) < 1e-3


def test_closure_latch_freezes_and_ramps_flow_to_zero():
    o = _breaker(close_rate=1.0)
    o.open(0.0)
    assert o.event_margin([0.0, 0.0], level=3.0, t=1.0)[0] > 0
    assert o.event_margin([0.0, 0.0], level=1.5, t=5.0)[0] < 0

    o.change_state([0.0, 0.0], level=1.5, mdot=0.7, Tin=30.0, t=5.0)
    at_close = o.calculate([30.0, -1e4], mdot=0.7, Tin=30.0, t=5.0)
    assert at_close[1] == pytest.approx(0.0)
    half_way = o.calculate([30.0, -1e4], mdot=0.0, Tin=30.0, t=5.5)
    assert half_way[1] == pytest.approx(-0.7 * (1.0 - 0.5))
    done = o.calculate([30.0, -1e4], mdot=0.0, Tin=30.0, t=6.0)
    assert done[1] == pytest.approx(0.0)


def test_closure_is_idempotent_and_margin_is_consumed():
    o = _breaker(close_rate=0.1)
    o.open(0.0)
    o.change_state([0.0, 0.0], level=1.5, mdot=0.7, Tin=30.0, t=5.0)
    o.change_state([0.0, 0.0], level=1.0, mdot=0.2, Tin=30.0, t=7.0)
    assert o.t_close == 5.0
    assert o.m_frozen == pytest.approx(0.7)
    assert o.event_margin([0.0, 0.0], level=1.0, t=7.0)[0] == pytest.approx(1.0)
    at_seven = o.calculate([30.0, -1e4], mdot=0.0, Tin=30.0, t=7.0)
    assert at_seven[1] == pytest.approx(-0.7 * (1.0 - (-2 * 0.2**3 + 3 * 0.2**2)))


def test_flashing_margin_positive_when_subcooled_and_warns_on_crossing():
    o = Orifice(light_water, AREA, CD, cc=0.62)
    o.open(0.0)
    subcooled = o.event_margin([0.0, 0.0], mdot=0.5, Tin=30.0, p_abs=2e5, t=1.0)
    assert subcooled.size == 1 and subcooled[0] > 0

    crossing = dict(mdot=6.0, Tin=99.0, p_abs=1.0e5, t=1.0)
    assert o.event_margin([0.0, 0.0], **crossing)[0] < 0
    with pytest.warns(UserWarning, match="Orifice"):
        o.change_state([0.0, 0.0], **crossing)
    assert o.should_continue([0.0, 0.0], **crossing)
    assert o.event_margin([0.0, 0.0], **crossing)[0] == pytest.approx(1.0)


def test_a_terminal_flashing_sentinel_stops_the_run():
    o = Orifice(light_water, AREA, CD, cc=0.62, flashing_terminal=True)
    o.open(0.0)
    crossing = dict(mdot=6.0, Tin=99.0, p_abs=1.0e5, t=1.0)
    assert o.should_continue([0.0, 0.0], **crossing)
    with pytest.warns(UserWarning):
        o.change_state([0.0, 0.0], **crossing)
    assert not o.should_continue([0.0, 0.0], **crossing)


def test_the_flashing_sentinel_is_disarmed_without_a_routed_pressure():
    o = Orifice(light_water, AREA, CD, cc=0.62)
    o.open(0.0)
    assert o.event_margin([0.0, 0.0], mdot=6.0, Tin=99.0, t=1.0).size == 0


def test_margins_are_ordered_closure_then_flashing():
    o = _breaker(cc=0.62)
    o.open(0.0)
    margins = o.event_margin([0.0, 0.0], level=3.0, mdot=0.5, Tin=30.0, p_abs=2e5, t=1.0)
    assert margins.size == 2
    assert margins[0] == pytest.approx(3.0 - 2.0)


def test_dp_out_is_zero_at_zero_flow():
    o = Orifice(light_water, AREA, CD)
    assert o.dp_out(Tin=30.0, mdot=0.0) == 0.0
    assert o.dp_out(Tin=30.0, mdot=0.0, area_factor=0.3) == 0.0


def test_dp_out_is_zero_at_zero_flow_for_a_re_dependent_cd():
    """A flow-fed discharge coefficient has no Reynolds number to be evaluated at when
    the flow vanishes, which is exactly the state every steady guess starts from."""

    def cd(re):
        return 20.0 / re

    o = Orifice(light_water, AREA, cd)
    assert o.dp_out(Tin=30.0, mdot=0.0) == 0.0


def test_an_open_orifice_with_a_re_dependent_cd_is_finite_at_zero_flow():
    """The throat Reynolds number is floored, so a correlation that diverges at rest is
    still evaluable there, and the coefficient it returns is a small fraction of the
    fully-developed one."""
    at_rest = Orifice(light_water, AREA, lambda re: lichtarowicz_cd(re, 2.0))
    at_rest.open(0.0)
    out = at_rest.calculate([30.0, -5e4], mdot=0.0, Tin=30.0, t=10.0)
    assert np.all(np.isfinite(out))

    developed = Orifice(light_water, AREA, lichtarowicz_cd(1e5, 2.0))
    developed.open(0.0)
    wide_open = developed.calculate([30.0, -5e4], mdot=0.0, Tin=30.0, t=10.0)
    assert abs(out[1]) < 0.02 * abs(wide_open[1])


def test_dp_out_opposes_the_flow_that_drives_it():
    o = Orifice(light_water, AREA, CD)
    forward = o.dp_out(Tin=30.0, mdot=1.5)
    backward = o.dp_out(Tin=30.0, mdot=-1.5)
    assert forward < 0 < backward
    assert forward == pytest.approx(-backward)


def test_re_dependent_cd_is_called_with_flow_reynolds():
    seen = []

    def cd(re):
        seen.append(float(re))
        return CD

    o = Orifice(light_water, AREA, cd)
    o.open(0.0)
    Tin, mdot = 40.0, 1.5
    o.calculate([0.0, -1e4], mdot=mdot, Tin=Tin, t=10.0)
    d_h = np.sqrt(4 * AREA / np.pi)
    assert seen
    assert seen[0] == pytest.approx(abs(mdot) * d_h / (AREA * light_water.viscosity(Tin)))


def test_area_factor_shrinks_the_effective_throat():
    o = Orifice(light_water, AREA, CD)
    o.open(0.0)
    dp, mdot, Tin = -5e4, 1.5, 40.0
    out = o.calculate([0.0, dp], mdot=mdot, Tin=Tin, t=10.0, area_factor=0.25)
    law = mdot_by_local_pressure_smooth(dp, light_water.density(Tin), 1 / CD**2, 0.25 * AREA, o.dp_eps)
    assert out[1] == pytest.approx(mdot + law)


def test_orifice_cloudpickle_round_trip():
    o = _breaker(cc=0.62)
    clone = cloudpickle.loads(cloudpickle.dumps(o))
    assert clone.dp_out(Tin=30.0, mdot=1.0) == pytest.approx(o.dp_out(Tin=30.0, mdot=1.0))
    assert clone.closes_below[1] == 2.0
