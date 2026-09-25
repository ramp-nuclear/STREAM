"""Acceptance tests for the regularized Flapper — bounded slopes and continuity
through the sqrt-law and opening-ramp seams."""

import numpy as np

from stream.calculations.flapper import (
    Flapper,
    continuously_differentiable_relaxation,
    legacy_relaxation,
)
from stream.physical_models.pressure_drop import (
    mdot_by_local_pressure,
    mdot_by_local_pressure_smooth,
)
from stream.substances import light_water

F_KW = dict(open_at_current=1.0, f=2.0, fluid=light_water, area=1e-3, open_rate=1.0)
TIN = TIN_MINUS = 20.0
RHO = light_water.density(TIN)
MDOT0 = 0.5


def _open_flapper():
    F = Flapper(**F_KW)
    F.t_open = 0.0
    return F


def _out1(F, dp, t):
    return float(F.calculate([0.0, dp], mdot=MDOT0, Tin=TIN, Tin_minus=TIN_MINUS, t=t)[1])


def test_bounded_ddp_slope():
    """|d(out[1])/d(dp)| stays finite across dp=0."""
    F = _open_flapper()
    t = 100.0  # relax = 1
    dp_eps = F.dp_eps
    bound = 1.05 * 1.0 * np.sqrt(2 * RHO * (1e-3) ** 2 / 2.0) / np.sqrt(dp_eps)
    h = 1e-9
    for dp0 in [0.0, 1e-8, -1e-8, 1e-4, -1e-4]:
        slope = (_out1(F, dp0 + h, t) - _out1(F, dp0 - h, t)) / (2 * h)
        assert abs(slope) <= bound, (dp0, slope, bound)


def test_opening_residual_continuous():
    """Both residual rows agree at t_open and t_open+δ as δ→0 (relax(0)=0)."""
    F = _open_flapper()
    F.t_open = 50.0
    r_at = np.asarray(F.calculate([1.0, 3.0], mdot=MDOT0, Tin=80.0, Tin_minus=30.0, t=50.0))
    r_after = np.asarray(F.calculate([1.0, 3.0], mdot=MDOT0, Tin=80.0, Tin_minus=30.0, t=50.0 + 1e-9))
    assert np.allclose(r_at, r_after, atol=1e-6)


def test_default_relaxation_is_c1():
    assert Flapper(**F_KW).relaxation is continuously_differentiable_relaxation


def test_dout_dt_continuous_at_ramp_end():
    """d(out[1])/dt is continuous across t_open + 1/open_rate with the C1 default relaxation."""
    F = _open_flapper()  # default relaxation
    seam = F.t_open + 1.0 / F.open_rate  # = 1.0
    dp = 5.0
    h = 1e-6
    left = (_out1(F, dp, seam) - _out1(F, dp, seam - h)) / h
    right = (_out1(F, dp, seam + h) - _out1(F, dp, seam)) / h
    assert np.isclose(left, right, atol=1e-3)


def test_legacy_relaxation_kinks_slope():
    """The legacy relaxation kinks d(out[1])/dt at the seam."""
    F = Flapper(**F_KW, relaxation=legacy_relaxation)
    F.t_open = 0.0
    seam = 1.0
    dp = 5.0
    h = 1e-6
    left = (_out1(F, dp, seam) - _out1(F, dp, seam - h)) / h
    right = (_out1(F, dp, seam + h) - _out1(F, dp, seam)) / h
    assert not np.isclose(left, right, atol=1e-3)


def test_exact_law_agreement_and_oddness():
    dp_eps = 1.0
    dp = 100 * dp_eps
    exact = mdot_by_local_pressure(dp, RHO, 2.0, 1e-3)
    smooth = mdot_by_local_pressure_smooth(dp, RHO, 2.0, 1e-3, dp_eps)
    assert abs(smooth - exact) / abs(exact) <= 1e-4
    # oddness and no NaN for dp < 0
    assert np.isclose(
        mdot_by_local_pressure_smooth(-dp, RHO, 2.0, 1e-3, dp_eps),
        -mdot_by_local_pressure_smooth(dp, RHO, 2.0, 1e-3, dp_eps),
    )
    assert np.isfinite(mdot_by_local_pressure_smooth(-5.0, RHO, 2.0, 1e-3, dp_eps))


def test_t_row_continuous_across_topen_at_zero_flow():
    """out[0] (the T-target ramp blend) is continuous across t_open even at mdot=0,
    where directed_Tin is the midpoint."""
    F = _open_flapper()
    F.t_open = 50.0
    r_at = F.calculate([0.0, 0.0], mdot=0.0, Tin=80.0, Tin_minus=30.0, t=50.0)[0]
    r_after = F.calculate([0.0, 0.0], mdot=0.0, Tin=80.0, Tin_minus=30.0, t=50.0 + 1e-9)[0]
    assert np.isclose(r_at, r_after, atol=1e-6)
