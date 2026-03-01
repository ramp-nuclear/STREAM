import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from stream.composition import uniform_x_power_shape
from stream.utilities import (
    cosine_shape,
    just,
    offset,
    pair_mean,
    pair_mean_1d,
    cosine_shape_by_zero_endpoints,
)
from .conftest import are_close, non_neg_medium_integers


def test_pair_mean_for_one_simple_case():
    a = np.arange(6)
    are_close(pair_mean(a), a[:-1] + 0.5)
    are_close(pair_mean(a, append=6), a + 0.5)
    are_close(pair_mean(a, prepend=-1), a - 0.5)


decent_floats = st.floats(min_value=-1e20, max_value=1e20, allow_nan=False, allow_infinity=False)
non_small_positive_floats = st.floats(min_value=0.5, max_value=10)
small_floats = st.floats(min_value=-10, max_value=10)
ints = st.integers(min_value=1, max_value=100)
PPF_values = st.floats(min_value=1.0, max_value=np.pi / 2)
illegal_ppf_values = st.one_of(
    st.floats(min_value=0, max_value=0.99),
    st.floats(min_value=np.pi / 2 + 0.01, max_value=3),
)


@settings(deadline=None)
@given(
    arrays(float, st.integers(1, 10), elements=decent_floats),
    decent_floats,
    st.integers(0, 2),
)
def test_pair_mean_1d_is_pair_mean(v, ext, case):
    cas = {0: (None, None), 1: (ext, None), 2: (None, ext)}[case]
    cas_numpy = [v if v is not None else np._NoValue for v in cas]
    keys = ("prepend", "append")
    d_numpy = dict(zip(keys, cas_numpy))
    d_1d = dict(zip(keys, cas))
    assert np.allclose(pair_mean(v, **d_numpy), pair_mean_1d(v, **d_1d))


def test_just():
    f = just(7)
    assert f(1, 2, 3) == 7
    assert f(a=1, b=2, c=3) == 7
    assert f(1, 2, 3, a=1, b=2, c=3) == 7


@given(non_neg_medium_integers, non_neg_medium_integers)
def test_offset(move_by, val):
    assert offset(val, move_by) == val + move_by
    assert offset(slice(val), move_by) == slice(move_by, val + move_by)
    assert offset(slice(0, val), move_by) == slice(move_by, val + move_by)
    assert offset({1: val, 2: 2 * val}, move_by) == {
        1: val + move_by,
        2: 2 * val + move_by,
    }
    assert offset({1: val, 2: slice(val, 2 * val), 3: np.zeros(1)}, move_by) == {
        1: val + move_by,
        2: slice(val + move_by, 2 * val + move_by),
        3: np.full(1, move_by),
    }

    are_close(offset(np.arange(3), move_by), np.arange(move_by, 3 + move_by))


def test_offset_of_an_unknown_type_raises_TypeError():
    with pytest.raises(TypeError):
        offset("hello", 8)


def test_cosine_shape_with_ppf_1_is_uniform():
    x = np.linspace(0, 10)
    shape = cosine_shape(x, ppf=1.0)
    assert np.allclose(np.diff(shape), 0)


@given(small_floats, non_small_positive_floats, PPF_values, st.integers(1, 100))
def test_cosine_shape_achieves_maximum_at_center(start, length, ppf, n):
    x = np.linspace(start, start + length, num=2 * n + 2)
    shape = cosine_shape(x, ppf=ppf)
    assert (np.argmax(shape) == n) or np.isclose(ppf, 1.0), shape


@given(PPF_values)
def test_cosine_shape_max_is_like_ppf_for_high_resolution(ppf):
    """This test ensures that the underlying cosine shape's max to mean ratio
    meets the desired PPF value.

    Since the function returns cell-integrated values, in order to approximate
    the ppf value we need a high enough resolution.

    """
    x = np.linspace(0, 1, 10000)
    shape = cosine_shape(x, ppf=ppf)
    mx = np.max(shape) / np.diff(x)[0]
    mean = np.sum(shape)
    assert np.isclose(mx / mean, ppf)


@given(small_floats, non_small_positive_floats, PPF_values, st.integers(3, 100))
def test_cosine_shape_sums_to_1(start, length, ppf, n):
    x = np.linspace(start, start + length, n)
    shape = cosine_shape(x, ppf)
    intv = np.sum(shape)
    assert np.isclose(intv, 1)


@given(small_floats, non_small_positive_floats, illegal_ppf_values)
def test_cosine_shape_throws_for_illegal_ppf(xi, length, ppf):
    x = np.linspace(xi, xi + length, num=100)
    with pytest.raises(ValueError):
        cosine_shape(x, ppf)


@given(small_floats, *(3 * [non_small_positive_floats]), st.integers(3, 101))
def test_cosine_extrapolation_has_sum_1_over_inner_region(xi, l1, l2, l3, n):
    xii = xi + l1
    xei = xii + l2
    xe = xei + l3
    x = np.linspace(xii, xei, n)
    shape_f = cosine_shape_by_zero_endpoints(xi, xe, x)
    # Safe because shape_f is a function, the type check here is wrong.
    # noinspection PyTypeChecker
    intv = np.sum(shape_f)
    assert np.isclose(intv, 1, rtol=1e-6)


@given(small_floats, non_small_positive_floats, st.integers(3, 101))
def test_cosine_extrapolation_has_maximum_at_midpoint(xi, length, n):
    xe = xi + length
    x = np.linspace(xi, xe, 2 * n + 2)
    shape_f = cosine_shape_by_zero_endpoints(xi, xe, x)
    assert np.argmax(shape_f) == n


@given(small_floats, non_small_positive_floats)
def test_cosine_extrapolation_endpoints_are_zero_for_high_resolution(xi, ll):
    """This test ensures that the underlying cosine function nullifies at its
    extrapolation points.

    The test needs high resolution because the return value is cell-integrated,
    so it never truly nullifies in a cell, unless a cell goes beyond the
    extrapolation points.

    """
    xe = xi + ll
    x = np.linspace(xi, xe, 100_000)
    shape = cosine_shape_by_zero_endpoints(xi, xe, x)
    v1, v2 = np.array([shape[0], shape[-1]]) * shape.size
    assert np.isclose(v1, 0, atol=1e-4)
    assert np.isclose(v2, 0, atol=1e-4)


@given(ints, ints, decent_floats, non_small_positive_floats, non_small_positive_floats)
def test_uniform_x_power_shape_works_with_clad_N_equals_zero(z_N, fuel_N, clad_w, meat_w, meat_h):
    power_shape = uniform_x_power_shape(z_N, fuel_N, 0, clad_w, meat_w, meat_h)
    assert power_shape.shape == (z_N, fuel_N)
    assert np.allclose(np.sum(power_shape, axis=0), 1 / fuel_N)
