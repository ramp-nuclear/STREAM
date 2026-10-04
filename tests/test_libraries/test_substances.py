"""Regression tests for the clamped property fits."""

import numpy as np

from stream.substances.heavy_water import _viscosity as hw_viscosity
from stream.substances.light_water import _specific_heat as lw_specific_heat


def test_light_water_specific_heat_finite_past_pole():
    """cp must stay finite past the pole of the saturated-water fit near 366 C, where
    the radicand goes negative."""
    for T in [300.0, 366.0, 370.0, 400.0, 500.0, 548.0, 560.0, 600.0]:
        cp = lw_specific_heat(T)
        assert np.isfinite(cp) and cp > 0, f"cp({T}) = {cp}"
    assert np.isclose(lw_specific_heat(50.0), 4181.4264285644285)


def test_heavy_water_viscosity_positive_below_pole():
    """Viscosity must stay finite and positive below the fit's pole at 0 F (-17.78 C)
    and its negative branch."""
    for T in [-30.0, -20.0, -17.8, -17.0, 0.0, 3.0]:
        mu = hw_viscosity(T)
        assert np.isfinite(mu) and mu > 0, f"mu({T}) = {mu}"
    assert np.isclose(hw_viscosity(50.0), 0.0006441125212510078)
    assert np.isclose(hw_viscosity(100.0), 0.0003301433604774831)
