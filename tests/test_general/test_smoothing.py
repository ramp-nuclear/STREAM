import numpy as np
import pytest

from stream.smoothing import (
    smooth_abs,
    smooth_pos_weight,
    smooth_sign,
    smooth_step,
    soft_pos,
)

EPS = 1e-3


def _max_quotient(f, lo, hi, n):
    """max |Δf/Δx| of a scalar function on a uniform grid of n points."""
    x = np.linspace(lo, hi, n)
    y = np.array([float(f(xi)) for xi in x])
    return np.max(np.abs(np.diff(y) / np.diff(x)))


def test_smooth_abs_limits():
    assert smooth_abs(0.0, EPS) == EPS
    assert np.isclose(smooth_abs(3.0, 4.0), 5.0)
    x = np.linspace(-10, 10, 101)
    s = smooth_abs(x, EPS)
    assert np.all(s >= np.abs(x) - 1e-12)
    assert np.all(s <= np.abs(x) + EPS + 1e-12)


def test_smooth_sign_limits():
    assert smooth_sign(0.0, EPS) == 0.0
    assert np.isclose(smooth_sign(1e6, EPS), 1.0)
    assert np.isclose(smooth_sign(-1e6, EPS), -1.0)
    slope0 = (smooth_sign(1e-9, EPS) - smooth_sign(-1e-9, EPS)) / 2e-9
    assert np.isclose(slope0, 1.0 / EPS, rtol=1e-4)


def test_soft_pos_limits_and_positivity():
    assert np.isclose(soft_pos(0.0, EPS), EPS / 2)
    assert np.isclose(soft_pos(1000.0, EPS), 1000.0, atol=1e-6)
    xs = np.array([-1.0, -1e2, -1e4, -1e6])
    assert np.all(soft_pos(xs, EPS) > 0.0)
    assert np.isclose(soft_pos(-1.0, EPS), EPS**2 / 4.0, rtol=1e-3)


def test_smooth_step_limits_and_monotonic():
    assert smooth_step(-1.0, 0.0, 1.0) == 0.0
    assert smooth_step(2.0, 0.0, 1.0) == 1.0
    assert np.isclose(smooth_step(0.5, 0.0, 1.0), 0.5)
    x = np.linspace(-0.5, 1.5, 401)
    y = smooth_step(x, 0.0, 1.0)
    assert np.all(np.diff(y) >= -1e-15)
    assert np.isclose((smooth_step(1e-6, 0, 1) - smooth_step(0.0, 0, 1)) / 1e-6, 0.0, atol=1e-4)
    assert np.isclose((smooth_step(1.0, 0, 1) - smooth_step(1 - 1e-6, 0, 1)) / 1e-6, 0.0, atol=1e-4)


def test_smooth_pos_weight():
    assert smooth_pos_weight(-1.0, EPS) == 0.0
    assert smooth_pos_weight(1.0, EPS) == 1.0
    assert np.isclose(smooth_pos_weight(0.0, EPS), 0.5)
    assert smooth_pos_weight(-1.0001 * EPS, EPS) == 0.0
    assert smooth_pos_weight(1.0001 * EPS, EPS) == 1.0
    q = _max_quotient(lambda m: smooth_pos_weight(m, EPS), -EPS, EPS, 2001)
    assert q <= 0.75 / EPS * (1 + 1e-3)


def test_evenness_and_oddness():
    x = np.linspace(-10, 10, 201)
    assert np.allclose(smooth_abs(-x, EPS), smooth_abs(x, EPS))
    assert np.allclose(smooth_sign(-x, EPS), -smooth_sign(x, EPS))


@pytest.mark.parametrize(
    "f",
    [smooth_abs, smooth_sign, soft_pos, smooth_pos_weight],
)
def test_scalar_and_array_agree(f):
    xs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    arr = f(xs, EPS)
    scal = np.array([float(f(float(x), EPS)) for x in xs])
    assert np.allclose(arr, scal)


def test_smooth_step_array():
    xs = np.array([-1.0, 0.25, 0.5, 0.75, 2.0])
    arr = smooth_step(xs, 0.0, 1.0)
    scal = np.array([float(smooth_step(float(x), 0.0, 1.0)) for x in xs])
    assert np.allclose(arr, scal)


@pytest.mark.parametrize(
    "f,lo,hi",
    [
        (lambda x: smooth_sign(x, EPS), -3 * EPS, 3 * EPS),
        (lambda x: soft_pos(x, EPS), -3 * EPS, 3 * EPS),
        (lambda x: smooth_pos_weight(x, EPS), -3 * EPS, 3 * EPS),
    ],
)
def test_difference_quotient_stable_under_refinement(f, lo, hi):
    q_coarse = _max_quotient(f, lo, hi, 401)
    q_fine = _max_quotient(f, lo, hi, 4001)
    assert q_fine <= 3.0 * q_coarse + 1e-9
