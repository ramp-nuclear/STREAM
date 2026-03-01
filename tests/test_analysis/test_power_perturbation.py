"""Tests for power perturbations."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from stream.analysis.UQ.power_perturbations import local_power_shift

decent_floats = st.floats(min_value=1e-5, max_value=1e3, allow_nan=False, allow_infinity=False)
shifts = st.floats(min_value=0.0, max_value=0.99)
arrs = arrays(float, st.tuples(st.integers(2, 5), st.integers(1, 5)), elements=decent_floats).filter(
    lambda x: ~np.isclose(np.sum(x), np.max(np.sum(x, axis=1)))
)
percentiles = st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False)


@given(arrs, shifts)
def test_power_shift_has_same_integral(array, shift: float):
    s = local_power_shift(array, 1 + shift)
    s1 = np.sum(array)
    s2 = np.sum(s)
    assert np.isclose(s1, s2), s


@given(decent_floats, shifts, percentiles)
def test_power_shift_for_constant_shape_has_no_effect(const, shift, percentile):
    a = np.ones((5, 1)) * const
    s = local_power_shift(a, 1 + shift, percentile)
    assert np.allclose(a, s)


@given(arrs, shifts)
def test_power_shift_for_0_percentile_has_no_effect(array, shift: float):
    s = local_power_shift(array, 1 + shift, 0)
    assert np.allclose(array, s)


@given(arrs.filter(lambda x: np.any(np.diff(np.sum(x, axis=1)) != 0.0)), shifts)
def test_power_shift_shifts_maximum_correctly(array, shift):
    print(np.diff(np.sum(array, axis=1)))
    s = local_power_shift(array, 1 + shift, 100)
    oldmax = np.max(np.sum(array, axis=1))
    new_max_row = np.max(np.sum(s, axis=1))
    assert np.isclose(new_max_row, oldmax * (1 + shift))


@given(shifts)
def test_power_shift_non_locally(shift):
    a = np.atleast_2d(np.arange(20))
    s = local_power_shift(a, 1 + shift, 60)
    assert np.all((s - a)[:-9, :] > 0.0)
