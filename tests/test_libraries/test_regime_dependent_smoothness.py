"""Acceptance tests for the C1 forced<->natural HTC blend across the Gr/Re^2 crossover."""

import numpy as np

from stream.physical_models.dimensionless import Gr, Re_mdot
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


def _gr(T_cool, T_wall):
    film = _film(T_cool, T_wall)
    return float(Gr(film.density, film.viscosity, film.thermal_expansion,
                    np.array([T_cool]), np.array([T_wall]), DH)[0])


def _phi_at(mdot, T_cool, T_wall):
    film = _film(T_cool, T_wall)
    re_film = float(Re_mdot(mdot, A, DH, film.viscosity)[0])
    return abs(_gr(T_cool, T_wall)) / max(re_film, 1e-30) ** 2


def _mdot_for_phi(phi, T_cool, T_wall):
    """phi ∝ 1/mdot**2, so invert from the phi=1 crossover flow."""
    film = _film(T_cool, T_wall)
    re_cross = np.sqrt(abs(_gr(T_cool, T_wall)))  # re_film at phi=1
    mdot_cross = re_cross / float(Re_mdot(1.0, A, DH, film.viscosity)[0])
    return mdot_cross / np.sqrt(phi)


def _independent_natural(T_cool, T_wall):
    """Elenbaas natural HTC as regime_dependent_h_spl invokes it (bulk coolant)."""
    return Elenbaas_h_spl(
        coolant=light_water.to_properties(np.array([T_cool])),
        T_wall=np.array([T_wall]), T_cool=np.array([T_cool]), depth=0.003, Lh=0.6,
    )[0]


def _h(mdot, T_cool, T_wall, nat_band=(0.5, 2.0)):
    film = _film(T_cool, T_wall)
    return regime_dependent_h_spl(
        coolant=film, mdot=mdot, T_cool=np.array([T_cool]), T_wall=np.array([T_wall]),
        nat_band=nat_band, **KW,
    )[0]


def _forced(mdot, T_cool, T_wall):
    return _h(mdot, T_cool, T_wall, nat_band=(1e12, 2e12))  # w_nat == 0 everywhere


def _natural(mdot, T_cool, T_wall):
    return _h(mdot, T_cool, T_wall, nat_band=(1e-12, 2e-12))  # w_nat == 1 everywhere


def test_no_jump_under_refinement():
    """max |Δh| across the phi-crossover must shrink under grid refinement."""
    T_cool, T_wall = 60.0, 90.0
    mc = _mdot_for_phi(1.0, T_cool, T_wall)
    lo, hi = 0.3 * mc, 3.0 * mc

    def max_step(n):
        m = np.linspace(lo, hi, n)
        h = np.array([_h(mi, T_cool, T_wall) for mi in m])
        return np.max(np.abs(np.diff(h)))

    coarse, fine = max_step(201), max_step(2001)
    assert fine <= 0.35 * coarse  # continuous: ~x0.1; a surviving jump would stay ~x1


def test_blend_ordering_at_phi_one():
    T_cool, T_wall = 60.0, 90.0
    mc = _mdot_for_phi(1.0, T_cool, T_wall)
    hf, hn, hb = _forced(mc, T_cool, T_wall), _natural(mc, T_cool, T_wall), _h(mc, T_cool, T_wall)
    lo, hi = min(hf, hn), max(hf, hn)
    assert lo < hb < hi


def test_cooled_wall_transitions():
    """T_wall < T_cool (gr<0) must still reach the natural branch."""
    T_cool, T_wall = 60.0, 30.0
    m = _mdot_for_phi(5.0, T_cool, T_wall)  # deep in the natural regime
    assert _phi_at(m, T_cool, T_wall) > 2.0
    hb = _h(m, T_cool, T_wall)
    hn = _independent_natural(T_cool, T_wall)
    hf = _forced(m, T_cool, T_wall)
    assert not np.isclose(hb, hf)
    assert np.isclose(hb, hn)


def test_degenerate_point_finite():
    """mdot=0, isothermal wall: gr=0 -> phi=0 -> w_nat=0 (forced), no 0/0
    mis-selection from the Gr/Re^2 switch; the result stays finite and equals
    the pure-forced value."""
    h = _h(0.0, 60.0, 60.0)
    assert np.isfinite(h)
    assert np.isclose(h, _forced(0.0, 60.0, 60.0))  # switch inert at gr=0


def test_compact_support():
    T_cool, T_wall = 60.0, 90.0
    m_forced = _mdot_for_phi(0.4, T_cool, T_wall)  # phi < 0.5 -> pure forced
    assert np.isclose(_h(m_forced, T_cool, T_wall), _forced(m_forced, T_cool, T_wall))
    m_nat = _mdot_for_phi(2.5, T_cool, T_wall)  # phi > 2.0 -> pure natural
    assert np.isclose(_h(m_nat, T_cool, T_wall), _natural(m_nat, T_cool, T_wall))
