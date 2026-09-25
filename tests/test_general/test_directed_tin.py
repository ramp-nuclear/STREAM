"""Acceptance tests for directed_Tin — bounded slope through zero flow, per-call
and late width overrides, compact support, and seam continuity."""

import numpy as np
import pytest

from stream import smoothing
from stream.utilities import directed_Tin

TIN, TIN_MINUS = 80.0, 55.0  # 25 K reversal dT
DT = TIN - TIN_MINUS
EPS = smoothing.DEFAULT_MDOT_EPS  # 1e-3


def _max_central_quotient(f, lo, hi, n):
    x = np.linspace(lo, hi, n)
    y = np.array([float(f(xi)) for xi in x])
    return np.max(np.abs(np.diff(y) / np.diff(x)))


def test_bounded_slope():
    """Max central-FD slope over [-2 eps, 2 eps] <= 0.80 ΔT/eps (analytic max 0.75 ΔT/eps)."""
    q = _max_central_quotient(lambda m: directed_Tin(TIN, TIN_MINUS, m), -2 * EPS, 2 * EPS, 4001)
    assert q <= 0.80 * DT / EPS


def test_late_override_honored():
    """Overriding DEFAULT_MDOT_EPS after a prior call must widen the band."""
    directed_Tin(TIN, TIN_MINUS, 0.5)  # prime the first call
    old = smoothing.DEFAULT_MDOT_EPS
    try:
        smoothing.DEFAULT_MDOT_EPS = 1.0
        v = directed_Tin(TIN, TIN_MINUS, 0.5)
    finally:
        smoothing.DEFAULT_MDOT_EPS = old
    assert np.isclose(v, 76.09375)  # w = 0.84375
    assert v != 80.0


def test_per_call_width():
    assert np.isclose(directed_Tin(TIN, TIN_MINUS, 0.05, mdot_eps=0.1), 76.09375)
    assert directed_Tin(TIN, TIN_MINUS, 0.05) == 80.0  # default eps: outside band


def test_compact_support_exactness():
    """At |mdot| just past eps the blend is bit-exact to the hard selection."""
    assert directed_Tin(TIN, TIN_MINUS, 1.0001 * EPS) == 80.0
    assert directed_Tin(TIN, TIN_MINUS, -1.0001 * EPS) == 55.0


def test_c1_seam_quotient_stable():
    f = lambda m: directed_Tin(TIN, TIN_MINUS, m)
    q_coarse = _max_central_quotient(f, -3 * EPS, 3 * EPS, 601)
    q_fine = _max_central_quotient(f, -3 * EPS, 3 * EPS, 6001)
    assert q_fine <= 3.0 * q_coarse + 1e-9


def test_none_fallbacks_preserved():
    assert directed_Tin(80.0, None, 3.0) == 80.0
    assert directed_Tin(None, 55.0, 3.0) == 55.0
    with pytest.raises(ValueError):
        directed_Tin(None, None, 1.0)


def test_array_inputs():
    out = directed_Tin(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.0)
    assert np.allclose(out, [2.0, 3.0])  # w(0) = 1/2
