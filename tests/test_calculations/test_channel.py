"""Tests for the Channel class.

"""
from functools import partial

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, sampled_from

from stream import Aggregator, EffectivePipe
from stream.calculations.channel import (
    Channel,
    ChannelAndContacts,
    coolant_first_order_upwind_dTdt,
)
from stream.state import State
from stream.substances import heavy_water, light_water
from stream.substances.mocks import mock_liquid_funcs
from stream.utilities import just
from .conftest import are_close, pos_medium_floats

mock_eff_pipe = EffectivePipe(
    length=1, heated_perimeter=1 / 4, wet_perimeter=1 / 4, area=1
)


@pytest.mark.slow
@settings(deadline=None)
@given(
    *[floats(10, 120, allow_nan=False, allow_infinity=False)] * 2,
    integers(10, 40),
    integers(100, 200)
)
def test_channel_w_no_heat_flux_maintains_temperature_profile(T0, Tin, cells,
                                                              time_points):
    """
    With no heat flux, temperatures should flow through the channel without
    changing. Special consideration is given to the Courant (CFL) condition.
    """
    channel_length = 1.0
    dx = channel_length / cells
    boundaries = np.linspace(0, channel_length, cells + 1)

    t_final = 1.0
    dt = t_final / time_points
    time = np.linspace(0, t_final, time_points)

    C = Channel(z_boundaries=boundaries, fluid=mock_liquid_funcs, pipe=mock_eff_pipe)

    # Maximally allowed velocity through CFL:
    velocity = (dx / dt)
    mdot = velocity * (C.fluid.density(Tin) * C.pipe.area)
    agr = Aggregator.from_decoupled(
        C,
        funcs={
            C: dict(
                mdot=mdot,
                T_left=T0, T_right=T0,
                h_left=0.0, h_right=0.0,
                p_abs=1e5,
                Tin=lambda t: -Tin * t * velocity,
            )
        },
    )
    solution = agr.solve({C.name: dict(T_cool=Tin * C.bounds[:-1], pressure=7.8)}, time)
    initial_state: State = agr.save(solution[0])
    state = agr.save(solution[-1], strict=True)

    initial_Tc = initial_state[C.name]["T_cool"]
    final_Tc = state[C.name]["T_cool"]

    # Since temperatures are linear (I.C. & B.C),
    # the difference should remain constant
    calculated_diff = initial_Tc - final_Tc
    analytical_diff = Tin * velocity * t_final - Tin * dx
    assert np.allclose(calculated_diff, analytical_diff, rtol=1e-4)


def test_specific_initialized_Channel_for_precalculated_values():
    """Channel instance is created, its variables, mass vector are correct"""
    C = Channel(np.arange(2), mock_liquid_funcs, mock_eff_pipe)
    assert C.indices("pressure") == 1
    assert len(C) == 2
    assert C.variables == {"T_cool": slice(0, 1), "pressure": 1}
    Tc_ids = C.indices("T_cool")
    assert np.all(C.mass_vector[Tc_ids] == 1.0)
    assert np.all(C.mass_vector[C.indices("pressure")] == 0.0)


def test_specific_initialized_ChannelAndContacts_for_precalculated_values():
    """ChannelAndContacts instance is created, its variables, mass vector are
    correct"""
    C = ChannelAndContacts(np.arange(2), mock_liquid_funcs, mock_eff_pipe)
    assert C.indices("pressure") == 1
    assert len(C) == 4
    assert C.variables == {
        "T_cool": slice(0, 1),
        "pressure": 1,
        "h_left": slice(2, 3),
        "h_right": slice(3, 4),
    }
    Tc_ids = C.indices("T_cool")
    assert np.all(C.mass_vector[Tc_ids] == 1.0)
    assert np.all(C.mass_vector[C.indices("pressure")] == 0.0)
    assert np.all(C.mass_vector[C.indices("h_left")] == 0.0)
    assert np.all(C.mass_vector[C.indices("h_right")] == 0.0)


mock_pipe = EffectivePipe(
    length=10, heated_perimeter=np.pi, wet_perimeter=np.pi, area=4 * np.pi
)


@settings(deadline=None)
@given(
    integers(1, 10),
    floats(min_value=1., max_value=120., allow_nan=False, allow_infinity=False),
    sampled_from([light_water, heavy_water]),
)
def test_first_order_upwind_dTdt_is_zero_for_zero_heat_flux(size, T0, fluid):
    """
    If all temperatures are constant, then we should be in stable state
    """
    T = np.full(size, T0)
    dTdt = coolant_first_order_upwind_dTdt(
        T=T,
        Tin=T0,
        q_left=0., q_right=0.,
        mdot=3,
        fluid=fluid,
        pipe=mock_pipe,
        dz=np.ones(size),
    )
    assert np.allclose(dTdt, 0, atol=1e-5)


@settings(deadline=None)
@given(
    integers(1, 30),
    pos_medium_floats,
)
def test_first_order_upwind_is_constant_for_linear_temperature_no_heat_flux(size, mdot):
    """
    If the temperature is rising linearly, dT/dt should be constant if nothing
    else changes
    """
    Tin = 0.0
    T = np.arange(1.0, size + 1)
    dTdt = coolant_first_order_upwind_dTdt(
        T=T,
        Tin=Tin,
        q_left=0., q_right=0.,
        mdot=mdot,
        fluid=mock_liquid_funcs,
        pipe=mock_pipe,
        dz=np.ones(size),
    )
    diff = np.diff(dTdt)
    assert np.allclose(diff, np.zeros_like(diff))
    # All of dTdt should be equal anyway:
    assert dTdt[0] == -mdot / mock_pipe.area


@settings(deadline=None)
@given(
    integers(1, 30),
    floats(1.0, 120.0, allow_nan=False, allow_infinity=False),
    floats(1.0, 120.0, allow_nan=False, allow_infinity=False),
    floats(1e3, 1e4, allow_nan=False, allow_infinity=False),
    sampled_from([light_water, heavy_water]),
)
def test_channel_with_zero_flow_constant_h_reaches_wall_temperature(cells, T0, Tf, h_wall,
                                                                    fluid):
    """
    The channel is subjected to constant wall temperatures. Flow is constantly zero,
    so the system should gradually adjust to the wall temperature.
    The heat transfer coefficient is set to a constant value.
    """

    channel_length = 1.2
    boundaries = np.linspace(0, channel_length, cells + 1)

    C = ChannelAndContacts(boundaries, fluid, mock_eff_pipe, h_wall_func=just(h_wall))

    agr = Aggregator.from_decoupled(
        C, funcs={C: dict(mdot=0.0, T_left=Tf, T_right=Tf, p_abs=1e5, Tin=T0)}
    )
    steady = agr.solve_steady(
        {C.name: dict(T_cool=T0, pressure=1e5, h_left=h_wall, h_right=h_wall)}
    )
    solution = agr.solve(steady, np.linspace(0, 10, 2))
    state = agr.save(solution.data[-1], strict=True)
    are_close(state[C.name]["T_cool"], Tf)


@given(arrays(float, 5, elements=pos_medium_floats), pos_medium_floats)
def test_first_order_upwind_inverts_correctly_on_flow_reversal_no_heat_flux(T, Tin):
    """
    For an isolated system, inlet temperature determines channel profile.
    If the flow is reversed, the profile should move in the other direction.
    """
    cool_dt = partial(
        coolant_first_order_upwind_dTdt,
        T=T,
        Tin=Tin,
        q_left=0., q_right=0.,
        fluid=mock_liquid_funcs,
        pipe=EffectivePipe(1.0, 1.0, 1.0, 1.0),
        dz=np.ones_like(T),
    )
    dTdt = cool_dt(mdot=1.0)
    are_close(dTdt, -np.diff(T, prepend=Tin))

    dTdt = cool_dt(mdot=-1.0)
    are_close(dTdt, -np.diff(T[::-1], prepend=Tin)[::-1])


def test_no_exceptions_raised_through_external_flow_inversion_on_Channel():
    """Inverting current in channel raises no error"""
    C = ChannelAndContacts(np.linspace(0, 1, 11), light_water,
                           EffectivePipe.rectangular(1, 0.1, 0.1, 0.1),
                           h_wall_func=just(5e3))
    mdot0 = 3.0

    def decreasing_mdot(t):
        return mdot0 - t

    agr = Aggregator.from_decoupled(
        C,
        funcs={C: dict(mdot=decreasing_mdot, T_left=80, T_right=80, Tin=80,
                       p_abs=1e5)},
    )

    initial = {C.name: dict(T_cool=80, h_left=5e3, h_right=5e3, pressure=-3e5)}
    agr.solve(initial, time=np.linspace(0, 6))
