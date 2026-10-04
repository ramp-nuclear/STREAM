"""Regression tests for the clamped property fits."""

import numpy as np

from stream.substances.light_water import _specific_heat as lw_specific_heat


def test_light_water_specific_heat_finite_past_pole():
    """cp must stay finite past the pole of the saturated-water fit near 366 C, where
    the radicand goes negative."""
    for T in [300.0, 366.0, 370.0, 400.0, 500.0, 548.0, 560.0, 600.0]:
        cp = lw_specific_heat(T)
        assert np.isfinite(cp) and cp > 0, f"cp({T}) = {cp}"
    assert np.isclose(lw_specific_heat(50.0), 4181.4264285644285)


