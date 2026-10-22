"""Acceptance tests for the monotone forced-plus-natural HTC composition: the forced
value combined with the natural one by the Churchill cube norm at every flow. Every
regime switch is fed by the flow, never by the wall temperature; the laminar law holds
its fully developed value at zero flow; and the per-cell wall balance
h(T_wall)*(T_wall - T_cool) stays monotone so a cell has a unique steady root."""

import numpy as np
import pytest

from stream.physical_models.heat_transfer_coefficient.laminar import Sudo_downflow_ratio
from stream.physical_models.heat_transfer_coefficient.natural_convection import Elenbaas_h_spl
from stream.physical_models.heat_transfer_coefficient.single_phase import regime_dependent_h_spl
from stream.substances import light_water

DH, A = 5e-3, 1e-3
KW = {
    "Dh": DH, "A": A, "re_bounds": (2000.0, 4000.0), "coolant_funcs": light_water,
    "depth": 0.003, "Lh": 0.6, "develop_length": np.array([0.3]), "aspect_ratio": 0.1,
}


def _film(T_cool, T_wall):
    return light_water.to_properties(np.array([(T_cool + T_wall) / 2]))


def _independent_natural(T_cool, T_wall):
    """Elenbaas natural HTC as regime_dependent_h_spl invokes it (bulk coolant)."""
    return Elenbaas_h_spl(
        coolant=light_water.to_properties(np.array([T_cool])),
        T_wall=np.array([T_wall]), T_cool=np.array([T_cool]), depth=0.003, Lh=0.6,
    )[0]


def regime_h(mdot, T_cool, T_wall, **over):
    film = _film(T_cool, T_wall)
    return regime_dependent_h_spl(
        coolant=film, mdot=mdot, T_cool=np.array([T_cool]), T_wall=np.array([T_wall]),
        **(KW | over),
    )[0]


def _zero_natural(**_):
    return np.zeros(1)


def forced_only_h(mdot, T_cool, T_wall):
    return regime_h(mdot, T_cool, T_wall, natural=_zero_natural)


def jumps_through_flow_reversal(h_law, re_max, T_cool=60.0, T_wall=90.0):
    """Neighbour jumps of h on 4000 points of Re in [-re_max, re_max], the jump across zero and h at zero flow."""
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    m = np.linspace(-re_max, re_max, 4000) * A * mu / DH
    jumps = np.abs(np.diff([h_law(mi, T_cool, T_wall) for mi in m]))
    return jumps, jumps[np.searchsorted(m, 0.0) - 1], h_law(0.0, T_cool, T_wall)


def test_h_has_no_jump_through_flow_reversal():
    """On 4000 points of Re from -5 to 5, no neighbour jump in h exceeds 1% of the zero flow h, and the jump
    across zero is at most the larger of twice the median neighbour jump and 0.1% of the zero flow h."""
    jumps, crossing, h0 = jumps_through_flow_reversal(regime_h, 5.0)
    assert np.max(jumps) < 0.01 * h0
    assert crossing <= max(2.0 * np.median(jumps), 1e-3 * h0)


def test_superposition_orders_above_both_ingredients():
    """In the through-flow regime h >= h_forced and h >= h_natural (buoyancy only adds)."""
    T_cool, T_wall = 60.0, 90.0
    for mdot in (5e-4, 5e-3, 5e-2):
        hb = regime_h(mdot, T_cool, T_wall)
        hf = forced_only_h(mdot, T_cool, T_wall)
        hn = _independent_natural(T_cool, T_wall)
        assert hb >= hf and hb >= hn
        assert hb <= (hf**3 + hn**3) ** (1 / 3) * (1 + 1e-9)


def test_zero_flow_holds_the_laminar_value():
    T_cool, T_wall = 60.0, 90.0
    h_still = regime_h(0.0, T_cool, T_wall)
    h_creep = regime_h(1e-9, T_cool, T_wall)
    hf = forced_only_h(0.0, T_cool, T_wall)
    assert hf > 0.0
    assert h_still == pytest.approx(h_creep, rel=1e-9)
    assert hf <= h_still <= hf * 1.05


def test_wall_balance_monotone_at_stagnation_scale():
    """The fold regression: at Re ~ 20 (where the replaced interpolation produced a
    three-root N-curve), F = h*(T_wall - T_cool) must rise strictly with T_wall."""
    T_cool = 60.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    mdot = 20.0 * A * mu / DH  # Re = 20
    dts = np.geomspace(0.01, 40.0, 120)
    F = np.array([regime_h(mdot, T_cool, T_cool + dt) * dt for dt in dts])
    assert np.all(np.diff(F) > 0)


def test_cooled_wall_finite_and_symmetric_natural_term():
    """T_wall < T_cool (negative Gr) keeps a finite, positive coefficient."""
    hb_cold = regime_h(5e-4, 60.0, 30.0)
    assert np.isfinite(hb_cold) and hb_cold > 0


def test_zero_flow_isothermal_wall_gives_the_laminar_value():
    """mdot=0, isothermal wall: the natural term vanishes and the laminar value stands,
    finite and non-negative, no 0/0."""
    h = regime_h(0.0, 60.0, 60.0)
    assert np.isfinite(h)
    assert h == pytest.approx(forced_only_h(0.0, 60.0, 60.0), rel=1e-9)


def regime_h_with_downflow_law(mdot, T_cool, T_wall, g=9.80665):
    return regime_h(mdot, T_cool, T_wall, g=g, downflow="sudo")


def test_without_gravity_the_law_is_the_forced_convection_value():
    for mdot in (0.0, 5e-4, -5e-3):
        assert regime_h(mdot, 60.0, 90.0, g=0.0) == pytest.approx(forced_only_h(mdot, 60.0, 90.0), rel=1e-12)


def test_flipping_gravity_direction_does_not_change_h_without_the_downflow_law():
    for mdot in (5e-4, -5e-4, 5e-2):
        assert regime_h(mdot, 60.0, 90.0, g=9.80665) == pytest.approx(regime_h(mdot, 60.0, 90.0, g=-9.80665), rel=1e-12)


def test_doubling_gravity_raises_h_through_the_natural_term():
    assert regime_h(5e-4, 60.0, 90.0, g=2 * 9.80665) > regime_h(5e-4, 60.0, 90.0)


def test_downflow_law_lowers_h_in_laminar_flow_along_gravity():
    """Flow along gravity has a lower h than flow against it, and flow against it keeps the plain law."""
    T_cool, T_wall = 60.0, 90.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    for re in (50.0, 250.0, 700.0, 1500.0):
        m = re * A * mu / DH
        assert regime_h_with_downflow_law(m, T_cool, T_wall) < regime_h_with_downflow_law(-m, T_cool, T_wall)
        assert regime_h_with_downflow_law(-m, T_cool, T_wall) == pytest.approx(regime_h(-m, T_cool, T_wall), rel=1e-12)


def test_downflow_law_has_no_effect_in_turbulent_flow():
    T_cool, T_wall = 60.0, 90.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    m = 6000.0 * A * mu / DH
    assert regime_h_with_downflow_law(m, T_cool, T_wall) == pytest.approx(
        regime_h_with_downflow_law(-m, T_cool, T_wall), rel=1e-12)


def test_downflow_law_depends_only_on_flow_relative_to_gravity():
    for m in (5e-4, 5e-3):
        assert regime_h_with_downflow_law(m, 60.0, 90.0, g=9.80665) == pytest.approx(
            regime_h_with_downflow_law(-m, 60.0, 90.0, g=-9.80665), rel=1e-12)


def test_downflow_law_is_off_without_gravity():
    assert regime_h_with_downflow_law(5e-4, 60.0, 90.0, g=0.0) == pytest.approx(
        forced_only_h(5e-4, 60.0, 90.0), rel=1e-12)


def test_downflow_law_is_off_at_zero_flow():
    assert regime_h_with_downflow_law(0.0, 60.0, 90.0) == pytest.approx(regime_h(0.0, 60.0, 90.0), rel=1e-12)


def test_wall_heat_flux_rises_with_superheat_under_the_downflow_law():
    """h(dT)*dT rises strictly with dT, so the wall balance h(dT)*dT = q has one root, for heated and cooled walls."""
    T_cool = 60.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    mdot = 250.0 * A * mu / DH
    for dts in (np.geomspace(0.01, 40.0, 120), -np.geomspace(0.01, 40.0, 120)[::-1]):
        F = np.array([regime_h_with_downflow_law(mdot, T_cool, T_cool + dt) * dt for dt in dts])
        assert np.all(np.diff(F) > 0)


def test_downflow_law_h_has_no_jump_through_flow_reversal():
    """On 4000 points of Re from -300 to 300, no neighbour jump in h exceeds 1% of the zero flow h, and the jump
    across zero is at most the larger of twice the median neighbour jump and 0.1% of the zero flow h."""
    jumps, crossing, h0 = jumps_through_flow_reversal(regime_h_with_downflow_law, 300.0)
    assert np.max(jumps) < 0.01 * h0
    assert crossing <= max(2.0 * np.median(jumps), 1e-3 * h0)


def test_downflow_law_applies_per_element_to_array_flow():
    """The first element flows along gravity, so the downflow law lowers its h below the second, which flows
    against it."""
    T = np.array([60.0, 60.0])
    kw = KW | {"develop_length": np.array([0.3, 0.3])}
    h = regime_dependent_h_spl(coolant=light_water.to_properties(np.array([75.0, 75.0])), mdot=np.array([5e-3, -5e-3]),
                               T_cool=T, T_wall=T + 30.0, downflow="sudo", **kw)
    assert h[0] < h[1]


@pytest.mark.parametrize("fade", [(16.0, 2.0), (2.0, 2.0), (0.0, 16.0), (-2.0, 16.0)])
def test_downflow_ratio_rejects_a_misordered_fade(fade):
    with pytest.raises(ValueError, match="fade"):
        Sudo_downflow_ratio(10.0, fade=fade)
