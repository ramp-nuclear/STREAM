import numpy as np
import pytest

from stream.calculations import CheckValve


def test_check_valve_forward_is_quadratic_and_reverse_is_stiff():
    v = CheckValve(k_forward=0.25, reverse_ratio=1000.0, eps=0.1)
    assert v.dp_out(mdot=100.0) == pytest.approx(-0.25 * 100.0**2, rel=1e-6)
    reverse = v.dp_out(mdot=-10.0)
    forward = -v.dp_out(mdot=10.0)
    assert reverse > 0
    assert reverse / forward > 500


def test_check_valve_is_smooth_through_zero():
    v = CheckValve(k_forward=0.25)
    m = np.linspace(-0.5, 0.5, 2001)
    dp = np.array([v.dp_out(mdot=x) for x in m])
    slope = np.diff(dp) / np.diff(m)
    assert np.all(np.isfinite(slope))
    assert np.max(np.abs(np.diff(slope))) < 50.0


def test_check_valve_is_at_rest_at_zero_flow():
    v = CheckValve(k_forward=0.25, reverse_ratio=1000.0, eps=0.1)
    assert v.dp_out(mdot=0.0) == 0.0
    assert v.dp_out(mdot=1.0) == -0.25


def test_check_valve_never_pushes_forward_flow():
    v = CheckValve(k_forward=0.25, reverse_ratio=1000.0, eps=0.1)
    assert all(v.dp_out(mdot=x) <= 0.0 for x in np.linspace(0.0, 0.1, 11))


def test_check_valve_has_no_jump_or_kink_at_zero_flow():
    v = CheckValve(k_forward=0.25, reverse_ratio=1000.0, eps=0.1)
    assert abs(v.dp_out(mdot=1e-12) - v.dp_out(mdot=-1e-12)) < 1e-12
    h = 1e-6
    for m in (-h, 0.0, h):
        central = (v.dp_out(mdot=m + h) - v.dp_out(mdot=m - h)) / (2 * h)
        assert abs(central) < 1e-3


def test_check_valve_is_reachable_from_the_calculations_package():
    import stream.calculations
    from stream.calculations.ideal.resistors import CheckValve as defined

    assert stream.calculations.CheckValve is defined
    assert CheckValve is defined
