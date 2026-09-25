"""Acceptance tests for the soft-rectified Junction mixing — continuity through
zero flow, mean anchor at stagnation, bounded sensitivity, and exact
hard-selection at nominal flow."""

import numpy as np
import pytest

from stream import smoothing
from stream.calculations.kirchhoff import Junction

EPS = smoothing.DEFAULT_MDOT_EPS  # 1e-3


class _Comp:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return hash(self.n)

    def __repr__(self):
        return self.n


# 2-edge junction, one loop mdot: A upstream (Tin=80), B downstream (Tin_minus=30)
A, B = _Comp("A"), _Comp("B")
TA, TB = 80.0, 30.0
DT = TA - TB  # 50 K


def _target(j, m, **kw):
    """Implied mixing target = value of `variables` zeroing the residual (= N/D)."""
    return float(j.calculate([0.0], Tin={A: TA}, Tin_minus={B: TB}, mdot={A: m, B: m}, **kw)[0])


def test_no_jump_across_zero():
    j = Junction("J")
    assert abs(_target(j, 1e-12) - _target(j, -1e-12)) < 0.1


def test_stagnation_anchor_is_mean():
    j = Junction("J")
    assert np.isclose(_target(j, 0.0), 55.0, atol=1e-9)  # mean(80, 30), not 0


def test_bounded_sensitivity():
    j = Junction("J")
    m = np.linspace(-10 * EPS, 10 * EPS, 801)
    tgt = np.array([_target(j, mi) for mi in m])
    slope = np.abs(np.diff(tgt) / np.diff(m))
    assert slope.max() <= 1.2 * DT / EPS


def test_nominal_exactness():
    j = Junction("J")
    assert abs(_target(j, 1.0) - TA) < 1e-4  # hard-selection value is 80
    assert abs(_target(j, -1.0) - TB) < 1e-4


def test_weights_match_hard_formula_large_flow():
    """A 3-edge weighted junction at operating flows matches the hard weighted
    formula; the soft tail is negligible far from stagnation."""
    c1, c2, c3 = _Comp("1"), _Comp("2"), _Comp("3")
    T1, T2, T3 = 40.0, 60.0, 90.0
    w = {c2: 2.0}  # non-unit weight on edge 2
    mdot = {c1: 200.0, c2: 400.0, c3: -600.0}
    j = Junction("J", weights=w)
    got = float(j.calculate([0.0], Tin={c1: T1, c2: T2}, Tin_minus={c3: T3}, mdot=mdot)[0])
    inc = 200.0 * 1 + 400.0 * 2 + 600.0 * 1
    expected = (200.0 * 1 * T1 + 400.0 * 2 * T2 + 600.0 * 1 * T3) / inc
    assert np.isclose(got, expected, atol=1e-9)


def test_width_override():
    default = Junction("Jd")
    wide = Junction("Jw", mdot_eps=0.1)
    assert _target(default, 0.05) > 79.9  # default eps: essentially unblended at 0.05
    assert _target(wide, 0.05) < 75.0  # wide band: genuinely blended
