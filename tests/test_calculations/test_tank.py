"""Tests for the free-surface inventory calculations."""

import cloudpickle
import numpy as np
import pytest

from stream.calculations import Environment, Tank
from stream.substances import light_water

Z_STOP = 1.0


def _tank(**kw):
    return Tank(light_water, 2.0, 4.0, z_uncovery=Z_STOP, **kw)


def test_pinned_tank_residual_anchors_level():
    t = _tank(fixed_temperature=30.0)
    out = t.calculate(np.array([3.7, 30.0]), mdot={})
    assert out[0] == pytest.approx(3.7 - 4.0)
    assert out[1] == pytest.approx(0.0)
    assert tuple(t.mass_vector) == (False, False)


def test_unpinned_level_rate_is_signed_net_inflow_over_rho_area():
    t = _tank(fixed_temperature=30.0)
    t.unpin()
    head_in, head_out = object(), object()
    t.weights = {head_in: 1.0, head_out: 1.0}
    out = t.calculate(
        np.array([4.0, 30.0]),
        mdot={head_in: 3.0, head_out: 1.0},
        Tin={head_in: 30.0},
        Tin_minus={head_out: 30.0},
    )
    rho = float(light_water.density(30.0))
    assert out[0] == pytest.approx((3.0 - 1.0) / (rho * 2.0))
    assert tuple(t.mass_vector) == (True, False)


def test_energy_rate_uses_inflow_temperature_difference_only():
    t = _tank()
    t.unpin()
    src = object()
    t.weights = {src: 1.0}
    hot, T = 50.0, 30.0
    out = t.calculate(np.array([4.0, T]), mdot={src: 2.0}, Tin={src: hot})
    rho = float(light_water.density(T))
    cp = float(light_water.specific_heat(T))
    expected = 2.0 * cp * (hot - T) / (rho * (2.0 * 4.0) * cp)
    assert out[1] == pytest.approx(expected, rel=1e-6)


def test_outflow_at_tank_temperature_does_not_cool_the_tank():
    t = _tank()
    t.unpin()
    drain = object()
    t.weights = {drain: 1.0}
    out = t.calculate(np.array([4.0, 30.0]), mdot={drain: 2.0}, Tin_minus={drain: 30.0})
    assert out[1] == pytest.approx(0.0, abs=1e-12)


def test_uncovery_event_latches_idempotently_and_stops():
    t = _tank(fixed_temperature=30.0)
    t.unpin()
    y = np.array([0.9, 30.0])
    assert t.event_margin(np.array([4.0, 30.0]))[0] > 0
    t.change_state(y)
    assert not t.should_continue(y)
    t.change_state(y)
    assert t.event_margin(y)[0] == 1.0


def test_marks_are_non_terminal_and_starvation_is_terminal():
    t = _tank(fixed_temperature=30.0, marks=dict(hole=2.0), mdot_starve=0.5)
    t.unpin()
    y = np.array([1.8, 30.0])
    t.change_state(y, ref_mdot=1.0)
    assert t.should_continue(y, ref_mdot=1.0)
    t.change_state(y, ref_mdot=0.1)
    assert not t.should_continue(y, ref_mdot=0.1)


def test_a_routed_reference_current_reaches_calculate_untouched():
    t = _tank(fixed_temperature=30.0, mdot_starve=0.5)
    plain = t.calculate(np.array([3.7, 30.0]), mdot={})
    with_ref = t.calculate(np.array([3.7, 30.0]), mdot={}, ref_mdot=1.5)
    assert with_ref == pytest.approx(plain)


def test_callable_area_without_volume_is_rejected():
    with pytest.raises(Exception, match="volume"):
        Tank(light_water, lambda h: 2.0, 4.0, z_uncovery=1.0)


def test_tank_serves_its_temperature_as_tin():
    t = _tank()
    assert t.indices("Tin") == 1
    assert t.indices("Tin_minus") == 1
    assert t.indices("level") == 0


def test_tank_cloudpickle_round_trip():
    t = _tank(marks=dict(hole=2.0))
    t2 = cloudpickle.loads(cloudpickle.dumps(t))
    assert t2.pinned and t2.level0 == 4.0


def test_margins_follow_the_documented_order():
    t = _tank(fixed_temperature=30.0, marks=dict(weir=3.0, hole=2.0), mdot_starve=0.5)
    margins = t.event_margin(np.array([3.5, 30.0]), ref_mdot=2.0)
    assert margins == pytest.approx([3.5 - Z_STOP, 3.5 - 2.0, 3.5 - 3.0, 2.0 - 0.5])


def test_conical_tank_uses_its_volume_law():
    t = Tank(
        light_water,
        lambda h: 2.0 * h,
        4.0,
        z_uncovery=1.0,
        volume=lambda h: h**2,
        fixed_temperature=None,
    )
    t.unpin()
    src = object()
    t.weights = {src: 1.0}
    out = t.calculate(np.array([4.0, 30.0]), mdot={src: 2.0}, Tin={src: 50.0})
    rho = float(light_water.density(30.0))
    cp = float(light_water.specific_heat(30.0))
    assert out[0] == pytest.approx(2.0 / (rho * 8.0), rel=1e-6)
    assert out[1] == pytest.approx(2.0 * cp * 20.0 / (rho * 16.0 * cp), rel=1e-6)


def test_a_tank_outside_a_flow_graph_is_reported():
    t = _tank()
    problems = list(t.validate_wiring({}, {}))
    assert len(problems) == 1 and "flow graph" in problems[0]


def test_a_tapped_component_wired_to_neither_side_is_reported():
    stray = object()
    problems = list(_tank().validate_wiring(dict(mdot={stray: 0}), {}))
    assert len(problems) == 1 and "inventory" in problems[0]


def test_a_flipped_tap_sign_is_reported():
    inflow, outflow = object(), object()

    class _Taps:
        def surface_taps(self, node):
            return {inflow: 1.0, outflow: -1.0}

    external = dict(
        mdot={inflow: 0, outflow: 1},
        Tin={outflow: 1},
        Tin_minus={inflow: 1},
        ref_mdot={_Taps(): 2},
    )
    problems = list(_tank().validate_wiring(external, {}))
    assert len(problems) == 2 and all("sign" in p for p in problems)


def test_environment_carries_its_ambient_pressure():
    env = Environment(name="env")
    assert env.surface_pressure == pytest.approx(101325.0)
    assert Environment(2e5, name="pressurized").surface_pressure == pytest.approx(2e5)
