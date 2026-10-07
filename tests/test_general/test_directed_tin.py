import numpy as np
import pytest

from stream import smoothing
from stream.utilities import directed_Tin

TIN, TIN_MINUS = 80.0, 55.0
DT = TIN - TIN_MINUS
EPS = smoothing.DEFAULT_MDOT_EPS


def _max_central_quotient(f, lo, hi, n):
    x = np.linspace(lo, hi, n)
    y = np.array([float(f(xi)) for xi in x])
    return np.max(np.abs(np.diff(y) / np.diff(x)))


def test_slope_bounded_across_reversal_band():
    """Max finite-difference slope over [-2 eps, 2 eps] stays below 0.80 ΔT/eps
    (analytic max 0.75 ΔT/eps)."""
    q = _max_central_quotient(lambda m: directed_Tin(TIN, TIN_MINUS, m), -2 * EPS, 2 * EPS, 4001)
    assert q <= 0.80 * DT / EPS


def test_per_call_width():
    assert np.isclose(directed_Tin(TIN, TIN_MINUS, 0.05, mdot_eps=0.1), 76.09375)
    assert directed_Tin(TIN, TIN_MINUS, 0.05) == 80.0


def test_hard_selection_exact_outside_band():
    assert directed_Tin(TIN, TIN_MINUS, 1.0001 * EPS) == 80.0
    assert directed_Tin(TIN, TIN_MINUS, -1.0001 * EPS) == 55.0


def test_difference_quotient_stable_under_refinement():
    def f(m):
        return directed_Tin(TIN, TIN_MINUS, m)

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
    assert np.allclose(out, [2.0, 3.0])
