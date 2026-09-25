"""Testing the post-analysis thresholds functions"""

import numpy as np
import pytest

from stream.calculations.channel import ChannelAndContacts
from stream.physical_models.thresholds import (
    Fabrega_CHF,
    Mirshak_CHF,
    Saha_Zuber_OSV,
    Sudo_Kaminaga_CHF,
    Whittle_Forgan_OFI,
    boiling_power,
)
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water

from .conftest import mock_pipe


def test_SK_CHF_is_non_negative_for_specific_cases():
    sat_coolant = light_water.to_properties(np.full(3, 100), np.full(3, 1e5))
    q_chf = Sudo_Kaminaga_CHF(T_bulk=np.full(3, 80), sat_coolant=sat_coolant, mdot=1, pipe=mock_pipe)
    assert all(q_chf > 0)


@pytest.mark.parametrize(
    ("mdot", "T_sat", "Tin", "Dh", "Lh", "G", "PF"),
    [
        (0.6, 110.0, 35.0, 0.005, 0.64, 60, 181177.07961),
        (0.7, 120.0, 40.0, 0.005, 0.64, 70, 225389.25186),
        (0.8, 130.0, 45.0, 0.005, 0.64, 80, 273720.35141),
        (0.9, 140.0, 50.0, 0.005, 0.64, 90, 326232.30036),
    ],
)
def test_WF_OFI_for_precalculated_case(mdot, T_sat, Tin, Dh, Lh, G, PF):
    A = mdot / G
    cp = light_water.specific_heat
    pipe = EffectivePipe(Lh, 1.0, 4 * A / Dh, A)
    assert np.isclose(Whittle_Forgan_OFI(mdot, T_sat, Tin, pipe, cp), PF)


@pytest.mark.parametrize(
    ("T", "p", "u", "Dh", "critical_flux"),
    [
        (10, 1e5, 0.2, 0.005, 4777567),
        (30, 2e5, 0.3, 0.005, 5068894),
        (50, 3e5, 0.4, 0.005, 4885202),
        (70, 4e5, 0.5, 0.005, 4435097),
        (90, 5e5, 0.6, 0.005, 3803118),
    ],
)
def test_Saha_Zuber_OSV_precalculated_case(T, p, u, Dh, critical_flux):
    coolant = light_water.to_properties(T, p)
    saha_z = Saha_Zuber_OSV(T, coolant, u, Dh)
    assert np.isclose(saha_z, critical_flux)


@pytest.mark.parametrize(
    ("T", "p", "u", "critical_flux"),
    [
        (10, 1.0e5, 0.1, 3308127.34419),
        (30, 1.2e5, 0.2, 3197236.81637),
        (50, 1.4e5, 0.3, 3054573.43723),
        (70, 1.6e5, 0.4, 2881215.61229),
        (90, 1.8e5, 0.5, 2677587.38616),
        (110, 2.0e5, 0.6, 2443735.47495),
    ],
)
def test_Mirshak_CHF_precalculated_case(T, p, u, critical_flux):
    T_sat = light_water.sat_temperature(p)
    assert np.isclose(Mirshak_CHF(T, T_sat, p, u), critical_flux)


@pytest.mark.parametrize(
    ("T_sat", "Tin", "Dh", "critical_flux"),
    [
        (110, 35.0, 0.005, 314250.0),
        (120, 36.0, 0.005, 324600.0),
        (130, 37.0, 0.005, 334950.0),
        (140, 38.0, 0.005, 345300.0),
        (150, 39.0, 0.005, 355650.0),
    ],
)
def test_Fabrega_CHF_precalculated_case(T_sat, Tin, Dh, critical_flux):
    assert np.isclose(Fabrega_CHF(Tin, T_sat, Dh), critical_flux)


def test_bpr_finite_nonzero_for_one_sided_channel():
    n = 10
    zbounds = np.linspace(0, 10, n + 1)
    sided_pipe = EffectivePipe.rectangular(10, 0.01, 0.1, 0.1, "left")
    chan = ChannelAndContacts(zbounds, light_water, sided_pipe)
    tin = mdot = 1.0
    saved = chan.save(np.ones(len(chan)), T_left=5 * np.ones(n), Tin=tin, mdot=mdot, p_abs=1e5)
    stat_p = saved["static_pressure"]
    tsat = light_water.sat_temperature(stat_p)
    cpin = light_water.specific_heat(tin)
    bpr: np.ndarray = boiling_power(mdot, tsat, tin, cpin)
    extremes = {np.inf, -np.inf, np.nan, 0.0}
    assert all(isinstance(v, float) for v in bpr)
    assert not any(v in extremes for v in bpr)


def _sk_envelope(T_bulk, sat, mdot, pipe, inlet, outlet):
    """The Sudo-Kaminaga q* envelope with explicitly chosen inlet/outlet cells."""
    from stream.physical_models.thresholds import _SKq1, _SKq2, _SKq3, _SKq4
    from stream.units import g

    drho = sat.density - sat.vapor_density
    hfg, cp, Tsat = sat.latent_heat, sat.specific_heat, sat.sat_temperature
    lamda = np.sqrt(sat.surface_tension / drho / g)
    scale = np.sqrt(lamda * drho * sat.vapor_density * g)
    G_star = mdot / pipe.area / scale
    A_ratio = pipe.area / (sum(pipe.heated_parts) * pipe.length)
    dT_in = (cp / hfg) * (Tsat[inlet] - T_bulk[inlet])
    dT_out = (cp / hfg) * (Tsat[outlet] - T_bulk[outlet])
    q1 = _SKq1(G_star)
    q2 = _SKq2(A_ratio=A_ratio, G_star=G_star, dT_inlet=dT_in)
    q3 = _SKq3(A_ratio=A_ratio, w=pipe.width, lamda=lamda, dT_inlet=dT_in, rho_v=sat.vapor_density, rho_l=sat.density)
    q4 = _SKq4(G_star=G_star, dT_outlet=dT_out)
    if np.all(np.asarray(mdot) >= 0):
        return np.maximum(np.minimum(q2, q4), q3) * hfg * scale
    return np.maximum(np.maximum(np.minimum(q2, q4), q1), q3) * hfg * scale


def test_SK_CHF_reversed_flow_uses_far_end_as_inlet():
    """Under reversed (negative-mdot) flow the coolant enters at the LAST cell,
    so the inlet subcooling must come from cell -1 and the outlet subcooling
    from cell 0 -- not the fixed 0/-1 ends of the forward convention."""
    sat = light_water.to_properties(np.full(4, 100.0), np.full(4, 1e5))
    T_bulk = np.array([40.0, 60.0, 80.0, 95.0])
    got = Sudo_Kaminaga_CHF(T_bulk=T_bulk, sat_coolant=sat, mdot=-1.0, pipe=mock_pipe)
    expected = _sk_envelope(T_bulk, sat, -1.0, mock_pipe, inlet=-1, outlet=0)
    assert np.allclose(got, expected)


def test_SK_CHF_forward_flow_keeps_cell0_as_inlet():
    """Forward flow keeps the historical convention exactly: inlet cell 0,
    outlet cell -1."""
    sat = light_water.to_properties(np.full(4, 100.0), np.full(4, 1e5))
    T_bulk = np.array([40.0, 60.0, 80.0, 95.0])
    got = Sudo_Kaminaga_CHF(T_bulk=T_bulk, sat_coolant=sat, mdot=1.0, pipe=mock_pipe)
    expected = _sk_envelope(T_bulk, sat, 1.0, mock_pipe, inlet=0, outlet=-1)
    assert np.allclose(got, expected)
