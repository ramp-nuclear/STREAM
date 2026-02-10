"""
Testing the Clad-Coolant contact term functions
"""
from functools import partial

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, sampled_from, tuples

from stream import EffectivePipe
# noinspection PyProtectedMember
from stream.calculations.heat_diffusion import _fill
from stream.physical_models.heat_transfer_coefficient import (
    Bergles_Rohsenhow_partial_SCB, Bergles_Rohsenow_T_ONB, Dittus_Boelter_h_spl, film_temperature,
    regime_dependent_h_spl, regime_dependent_q_scb, wall_heat_transfer_coeff, wall_temperature,
    )
from stream.physical_models.heat_transfer_coefficient.single_phase import maximal_h_spl
from stream.substances import heavy_water, light_water
from stream.substances.mocks import mock_liquid_funcs
from stream.utilities import just, lin_interp
from .conftest import are_close, pos_floats, pos_medium_floats, mock_pipe


@pytest.mark.parametrize(("pressure", "q_spl", "T_wall_inc"),
                         [(1e5, 0, heavy_water.sat_temperature(1e5)),
                          (2e5, 2.5e5, 126.4446),
                          (3e5, 3e6, 147.8462),
                          (1e5, 5e6, 128.6497)])
def test_BR_T_wall_inc_for_precalculated_case(pressure, q_spl, T_wall_inc):
    Tsat = heavy_water.sat_temperature(pressure)
    assert np.isclose(Bergles_Rohsenow_T_ONB(pressure, q_spl, Tsat), T_wall_inc)


@pytest.mark.parametrize(("q_spl", "q_scb", "q_scb_inc", "multiply"),
                         [(1e8, 1e9, 1e8, 9.055385138137417),
                          (5e8, 9e9, 2e8, 17.628386199536244),
                          (5e6, 9e8, 2e7, 1.760028408861630e+02),
                          ])
def test_BR_partial_scb_for_precalculated_case(q_spl, q_scb, q_scb_inc, multiply):
    assert np.isclose((Bergles_Rohsenhow_partial_SCB(q_spl, q_scb, q_scb_inc)), multiply)


ar = partial(_fill, shape=1)


@settings(deadline=None)
@given(*(_h_inputs := (
    floats(min_value=1.0, max_value=120.0, allow_infinity=False, allow_nan=False),
    floats(min_value=0.025e5, max_value=85e5, allow_infinity=False, allow_nan=False),
    floats(min_value=1.0, max_value=5.0, allow_infinity=False, allow_nan=False,
           exclude_min=True),
    pos_medium_floats,
    sampled_from(["H2O", "D2O"]),
)))
def test_htc_is_geq_to_Dittus_Boelter(T, p, T_clad_factor, mdot, liq):
    T, p, T_clad = ar(T), ar(p), ar(T * T_clad_factor)
    liquid = dict(H2O=light_water, D2O=heavy_water)[liq]

    coolant = liquid.to_properties(film_temperature(T, T_clad), p)
    h0 = Dittus_Boelter_h_spl(coolant=coolant, mdot=mdot, A=mock_pipe.area,
                              Dh=mock_pipe.hydraulic_diameter)
    h = wall_heat_transfer_coeff(T_wall=T_clad, T_cool=T,
                                 mdot=mdot, pressure=p,
                                 coolant_funcs=liquid,
                                 Dh=mock_pipe.hydraulic_diameter,
                                 A=mock_pipe.area)
    T_wall: np.array = wall_temperature(T_cool=T, T_clad=T_clad, h_cool=h0, h_clad=ar(2.))

    assert np.all(T_clad >= T_wall) or np.allclose(T_clad, T_wall)
    assert np.all(h >= h0)


@settings(deadline=None)
@given(*_h_inputs)
def test_maximal_htc_is_overshadowed_by_infinity(T, p, T_clad_factor, mdot, liq):
    T, p, T_clad = ar(T), ar(p), ar(T * T_clad_factor)
    liquid = dict(H2O=light_water, D2O=heavy_water)[liq]

    coolant = liquid.to_properties(film_temperature(T, T_clad), p)
    h = maximal_h_spl([Dittus_Boelter_h_spl, just(np.inf), just(-5)])(
        coolant=coolant, mdot=mdot, A=mock_pipe.area, pressure=1e5,
        Dh=mock_pipe.hydraulic_diameter, T_cool=T, T_wall=T_clad, coolant_funcs=liquid)
    assert h == np.inf


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (dict(T_cool=2, T_clad=2, h_cool=1, h_clad=1), 2),
        (dict(T_cool=1, T_clad=2, h_cool=1, h_clad=1), 1.5),
        (dict(T_cool=ar(1), T_clad=ar(2), h_cool=ar(1), h_clad=ar(1)), 1.5),
        (dict(T_cool=1, T_clad=2, h_cool=np.inf, h_clad=1), 1),
        (dict(T_cool=1, T_clad=2, h_cool=1, h_clad=np.inf), 2),
        (dict(T_cool=1, T_clad=2, h_cool=1, h_clad=ar(0)), 1),
        (dict(T_cool=1, T_clad=2, h_cool=ar(0), h_clad=ar(1)), 2),
    ],
)
def test_wall_temperature_for_precalculated_case(input, expected):
    result = wall_temperature(**input)
    are_close(result, expected)


@settings(deadline=None)
@given(arrays(float, integers(1, 20), elements=pos_medium_floats))
def test_regime_dependent_q_scb_is_zero_for_Twall_equals_Tsat(re):
    """Both Bergles-Rosenhow and McAdams are zero at Twall=Tsat"""
    re_bounds = 2000, 4000
    T_wall = light_water.sat_temperature(p := np.full_like(re, 1e5))
    q = regime_dependent_q_scb(T_wall, light_water.to_properties(T_wall, p), re, re_bounds)
    assert np.allclose(q, 0)


@settings(deadline=None)
@given(tuples(pos_floats, pos_floats).filter(lambda x: x[0] != x[1]),
       pos_medium_floats,
       pos_medium_floats,
       pos_medium_floats,
       )
def test_regime_dependent_h_spl_assigns_regimes_correctly(re, lam, turb, md):
    """In this test, the reynolds number is set to be just mdot,
    thus it is directly compared with the re regime bounds. The laminar and turbulent
    spl values are set to be simply given values, and are compared with the
    interpolation scheme"""
    re_bounds = (min(re), max(re))
    h = regime_dependent_h_spl(
        mock_liquid_funcs.to_properties(np.array([100.]), 1e5),
        mdot=md,
        Dh=1.,
        A=1.,
        T_cool=np.array([50.]),
        T_wall=np.array([50.]),
        re_bounds=re_bounds,
        laminar=just(np.atleast_1d(lam)),
        natural=just(np.atleast_1d(lam)),
        turbulent=just(np.atleast_1d(turb)),
        coolant_funcs=mock_liquid_funcs
    )
    if md < re_bounds[0]:
        assert np.allclose(h, lam)
    elif md > re_bounds[1]:
        assert np.allclose(h, turb)
    else:
        assert np.allclose(h, lin_interp(*re_bounds, lam, turb, md))
