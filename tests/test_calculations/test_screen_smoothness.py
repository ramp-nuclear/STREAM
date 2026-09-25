"""Acceptance tests for the continuous Screen loss factor across the Re=50 branch seam."""

import numpy as np

from stream.calculations.ideal.resistors import Screen
from stream.physical_models.dimensionless import Re_mdot
from stream.substances import light_water

CLEAR, TOTAL = 0.3, 1.0
F = CLEAR / TOTAL
BASE = 1.3 * (1 - F) + (1 / F - 1) ** 2  # geometric factor


def test_factor_continuous_at_re50():
    lo = Screen.factor(CLEAR, TOTAL, 50.0 - 1e-9)
    hi = Screen.factor(CLEAR, TOTAL, 50.0 + 1e-9)
    at50 = Screen.factor(CLEAR, TOTAL, 50.0)
    assert abs(hi - lo) <= 1e-6 * at50


def test_anchors():
    # low-Re branch matches the tabulated anchor form exactly
    assert np.isclose(Screen.factor(CLEAR, TOTAL, 25.0), 1.44 * BASE + 22.0 * (1 / 25.0 - 1 / 50.0))
    # high-Re: pure geometric factor
    assert Screen.factor(CLEAR, TOTAL, 1000.0) == BASE
    assert Screen.factor(CLEAR, TOTAL, 2000.0) == BASE
    # continuity anchor at Re=50 equals 1.44*BASE (the np.interp end value)
    assert np.isclose(Screen.factor(CLEAR, TOTAL, 50.0), 1.44 * BASE)
    # exactly-zero flow stays finite
    assert np.isfinite(Screen.factor(CLEAR, TOTAL, 0.0))


def test_dp_out_continuous_across_re50():
    screen = Screen(clear_area=CLEAR, total_area=TOTAL, wire_diameter=1e-3, fluid=light_water)
    Tin = 25.0
    mu = light_water.viscosity(Tin)
    # Re_mdot is linear in mdot, so invert via a unit evaluation to hit Re = 50
    m50 = 50.0 / Re_mdot(1.0, CLEAR, 1e-3, mu)
    assert np.isclose(Re_mdot(m50, CLEAR, 1e-3, mu), 50.0)
    dm = 2e-5 * m50
    dp_lo = screen.dp_out(mdot=m50 - dm, Tin=Tin)
    dp_hi = screen.dp_out(mdot=m50 + dm, Tin=Tin)
    rel_step = abs(dp_hi - dp_lo) / abs(dp_lo)
    assert rel_step <= 1e-3


def test_zero_flow_unchanged():
    screen = Screen(clear_area=CLEAR, total_area=TOTAL, wire_diameter=1e-3, fluid=light_water)
    assert screen.dp_out(mdot=0.0, Tin=50.0) == 0.0
