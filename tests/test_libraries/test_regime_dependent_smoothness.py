"""Acceptance tests for the monotone forced-plus-natural HTC composition: Churchill
cube-norm superposition with a Graetz-number stagnation handover. The composition must
be continuous, ordered above both of its ingredients in the through-flow regime, hand
over to the pure natural function at stagnation, and keep the per-cell wall balance
h(T_wall)*(T_wall - T_cool) monotone so a cell has a unique steady root."""

import numpy as np

from stream.physical_models.heat_transfer_coefficient.natural_convection import Elenbaas_h_spl
from stream.physical_models.heat_transfer_coefficient.single_phase import regime_dependent_h_spl
from stream.substances import light_water

# MTR-like fixture: Dh = 5 mm, light water
DH, A = 5e-3, 1e-3
KW = dict(
    Dh=DH, A=A, re_bounds=(2000.0, 4000.0), coolant_funcs=light_water,
    depth=0.003, Lh=0.6, develop_length=np.array([0.3]), aspect_ratio=0.1,
)


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
    """Forced part alone: natural contribution zeroed, handover forced off."""
    return _h(mdot, T_cool, T_wall, natural=_zero_natural, gz_band=(1e-12, 2e-12))


def test_no_jump_under_refinement():
    """max |Δh| across the handover band must shrink under grid refinement."""
    T_cool, T_wall = 60.0, 90.0
    mu = float(light_water.viscosity(np.array([T_cool]))[0])
    m_hand = 0.05 * KW["Lh"] / DH * A * mu / DH  # flow at the middle of the default gz_band

    def max_step(n):
        m = np.linspace(0.2 * m_hand, 5.0 * m_hand, n)
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


def test_gz_handover_limits():
    """Degenerate gz_band isolates the two limits: pure natural at w=0, pure
    superposition at w=1."""
    T_cool, T_wall, mdot = 60.0, 90.0, 5e-3
    hn = _independent_natural(T_cool, T_wall)
    h_low = _h(mdot, T_cool, T_wall, gz_band=(1e12, 2e12))  # w_flow == 0 everywhere
    assert np.isclose(h_low, hn)
    hf = _forced(mdot, T_cool, T_wall)
    h_high = _h(mdot, T_cool, T_wall, gz_band=(1e-12, 2e-12))  # w_flow == 1 everywhere
    assert np.isclose(h_high, (hf**3 + hn**3) ** (1 / 3))


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
    """mdot=0, isothermal wall: the handover selects the natural function, whose
    conduction-free limit is ~0 -- finite by construction, no 0/0."""
    h = _h(0.0, 60.0, 60.0)
    assert np.isfinite(h)
    assert h == np.clip(h, 0.0, _independent_natural(60.0, 60.0) + 1e-12)
