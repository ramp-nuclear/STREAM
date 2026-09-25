"""Property-fit domain hardening and validity-as-data tests."""

import numpy as np

from stream.substances import heavy_water, light_water
from stream.substances.heavy_water import _viscosity as hw_viscosity
from stream.substances.light_water import _specific_heat as lw_specific_heat
from stream.substances.liquid import LiquidFuncs
from stream.substances.mocks import constant_LiquidFuncs, mock_liquid_funcs


def test_light_water_specific_heat_finite_past_pole():
    """cp must stay finite past the ~366 C denominator pole of the saturated-water
    fit, where sqrt of a negative radicand otherwise returns NaN."""
    for T in [300.0, 366.0, 370.0, 400.0, 500.0, 548.0, 560.0, 600.0]:
        cp = lw_specific_heat(T)
        assert np.isfinite(cp) and cp > 0, f"cp({T}) = {cp}"
    # In-range reference value.
    assert np.isclose(lw_specific_heat(50.0), 4181.4264285644285)


def test_heavy_water_viscosity_positive_below_pole():
    """Viscosity must stay finite and positive below the TF=0 pole (T=-17.78 C) and
    its negative branch."""
    for T in [-30.0, -20.0, -17.8, -17.0, 0.0, 3.0]:
        mu = hw_viscosity(T)
        assert np.isfinite(mu) and mu > 0, f"mu({T}) = {mu}"
    # In-range reference values.
    assert np.isclose(hw_viscosity(50.0), 0.0006441125212510078)
    assert np.isclose(hw_viscosity(100.0), 0.0003301433604774831)


# --- validity as data -----------------------------------------


def test_validity_is_the_last_field_and_optional():
    """The new ``validity`` field is last and defaults to None, so every positional
    LiquidFuncs construction site stays valid."""
    fields = list(LiquidFuncs.__dataclass_fields__)
    assert fields[-1] == "validity"
    assert LiquidFuncs.__dataclass_fields__["validity"].default is None


def test_real_fluids_declare_their_liquid_range():
    """light/heavy water carry a single liquid-validity range; the clamp boundaries
    (light cp at 350 C, heavy viscosity at 3.8 C) fall on the declared bounds."""
    assert light_water.validity == (0.1, 350.0)
    assert heavy_water.validity == (3.8, 300.0)


def test_mock_liquid_funcs_carries_no_validity():
    """The mock is built via uniform_dataclass (which fills every field); validity
    must still come out None, not the np.ones_like filler, and the mock still works."""
    assert mock_liquid_funcs.validity is None
    assert mock_liquid_funcs.density(50.0) == 1.0  # both paths still evaluate


def test_constant_liquidfuncs_carries_source_validity():
    """constant_LiquidFuncs skips validity in its per-property comprehension (a Liquid
    has no such field) and carries the source fluid's range through."""
    clf = constant_LiquidFuncs(light_water, T=20.0, p=1e5)
    assert clf.validity == light_water.validity
    assert bool(clf.density(20.0) == light_water.density(20.0))
    # A mock source (validity None) stays None through the constant construction.
    clf_none = constant_LiquidFuncs(mock_liquid_funcs, T=20.0, p=1e5)
    assert clf_none.validity is None
