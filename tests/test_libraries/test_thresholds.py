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
