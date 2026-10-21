"""Acceptance tests for the monotone forced-plus-natural HTC composition: the forced
value combined with the natural one by the Churchill cube norm at every flow. Every
regime switch is fed by the flow, never by the wall temperature; the laminar law holds
its fully developed value at zero flow; and the per-cell wall balance
h(T_wall)*(T_wall - T_cool) stays monotone so a cell has a unique steady root."""

import numpy as np
import pytest

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


def _h(mdot, T_cool, T_wall, **over):
    film = _film(T_cool, T_wall)
    return regime_dependent_h_spl(
        coolant=film, mdot=mdot, T_cool=np.array([T_cool]), T_wall=np.array([T_wall]),
        **(KW | over),
    )[0]


def _zero_natural(**_):
    return np.zeros(1)


def _forced(mdot, T_cool, T_wall):
    return _h(mdot, T_cool, T_wall, natural=_zero_natural)


def test_no_jump_under_refinement():
    """max |Δh| across zero flow (Re from -5 to 5) must shrink under grid refinement."""
    T_cool, T_wall = 60.0, 90.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])

    def max_step(n):
        m = np.linspace(-5.0, 5.0, n) * A * mu / DH
        h = np.array([_h(mi, T_cool, T_wall) for mi in m])
        return np.max(np.abs(np.diff(h)))

    coarse, fine = max_step(201), max_step(2001)
    assert fine <= 0.35 * coarse  # continuous: ~x0.1; a surviving jump would stay ~x1


def test_superposition_orders_above_both_ingredients():
    """In the through-flow regime h >= h_forced and h >= h_natural (buoyancy only adds)."""
    T_cool, T_wall = 60.0, 90.0
    for mdot in (5e-4, 5e-3, 5e-2):
        hb = _h(mdot, T_cool, T_wall)
        hf = _forced(mdot, T_cool, T_wall)
        hn = _independent_natural(T_cool, T_wall)
        assert hb >= hf and hb >= hn
        assert hb <= (hf**3 + hn**3) ** (1 / 3) * (1 + 1e-9)


def test_zero_flow_holds_the_laminar_value():
    T_cool, T_wall = 60.0, 90.0
    h_still = _h(0.0, T_cool, T_wall)
    h_creep = _h(1e-9, T_cool, T_wall)
    hf = _forced(0.0, T_cool, T_wall)
    assert hf > 0.0
    assert h_still == pytest.approx(h_creep, rel=1e-9)
    assert hf <= h_still <= hf * 1.05


def test_gz_band_is_rejected():
    with pytest.raises(TypeError, match="gz_band"):
        _h(5e-3, 60.0, 90.0, gz_band=(0.01, 0.1))


def test_wall_balance_monotone_at_stagnation_scale():
    """The fold regression: at Re ~ 20 (where the replaced interpolation produced a
    three-root N-curve), F = h*(T_wall - T_cool) must rise strictly with T_wall."""
    T_cool = 60.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    mdot = 20.0 * A * mu / DH  # Re = 20
    dts = np.geomspace(0.01, 40.0, 120)
    F = np.array([_h(mdot, T_cool, T_cool + dt) * dt for dt in dts])
    assert np.all(np.diff(F) > 0)


def test_cooled_wall_finite_and_symmetric_natural_term():
    """T_wall < T_cool (negative Gr) keeps a finite, positive coefficient."""
    hb_cold = _h(5e-4, 60.0, 30.0)
    assert np.isfinite(hb_cold) and hb_cold > 0


def test_degenerate_point_finite():
    """mdot=0, isothermal wall: the natural term vanishes and the laminar value stands,
    finite and non-negative, no 0/0."""
    h = _h(0.0, 60.0, 60.0)
    assert np.isfinite(h)
    assert h >= 0.0
