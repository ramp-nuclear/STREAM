import numpy as np

from stream import smoothing
from stream.calculations.kirchhoff import Junction

EPS = smoothing.DEFAULT_MDOT_EPS


class _Comp:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return hash(self.n)

    def __repr__(self):
        return self.n


A, B = _Comp("A"), _Comp("B")
TA, TB = 80.0, 30.0
DT = TA - TB


def _target(j, m, **kw):
    """Mixing target: the value of ``variables`` that zeroes the residual."""
    return float(j.calculate([0.0], Tin={A: TA}, Tin_minus={B: TB}, mdot={A: m, B: m}, **kw)[0])


def test_no_jump_across_zero_flow():
    j = Junction("J")
    assert abs(_target(j, 1e-12) - _target(j, -1e-12)) < 1e-6


def test_stagnation_target_is_mean_of_neighbours():
    j = Junction("J")
    assert np.isclose(_target(j, 0.0), 55.0, atol=1e-9)


def test_bounded_sensitivity_to_flow():
    j = Junction("J")
    m = np.linspace(-10 * EPS, 10 * EPS, 801)
    tgt = np.array([_target(j, mi) for mi in m])
    slope = np.abs(np.diff(tgt) / np.diff(m))
    assert slope.max() <= 1.2 * DT / EPS


def test_nominal_flow_matches_hard_selection():
    j = Junction("J")
    assert abs(_target(j, 1.0) - TA) < 1e-4
    assert abs(_target(j, -1.0) - TB) < 1e-4


def test_weighted_three_edge_junction_at_large_flow():
    c1, c2, c3 = _Comp("1"), _Comp("2"), _Comp("3")
    T_1, T_2, T_3 = 40.0, 60.0, 90.0
    w = {c2: 2.0}
    mdot = {c1: 200.0, c2: 400.0, c3: -600.0}
    j = Junction("J", weights=w)
    got = float(j.calculate([0.0], Tin={c1: T_1, c2: T_2}, Tin_minus={c3: T_3}, mdot=mdot)[0])
    inc = 200.0 * 1 + 400.0 * 2 + 600.0 * 1
    expected = (200.0 * 1 * T_1 + 400.0 * 2 * T_2 + 600.0 * 1 * T_3) / inc
    assert np.isclose(got, expected, atol=1e-9)


def test_width_override():
    default = Junction("Jd")
    wide = Junction("Jw", mdot_eps=0.1)
    assert _target(default, 0.05) > 79.9
    assert _target(wide, 0.05) < 75.0
