import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from stream.calculations import (
    Resistor,
    Pump,
    PointKinetics,
    PointKineticsWInput,
    Inertia,
    KirchhoffWDerivatives,
    Junction,
)
from stream.calculations.ideal.ideal import LumpedComponent
from stream.composition import Calculation_factory, FlowGraph, flow_edge
from stream.composition.subsystems import (
    point_kinetics_steady_state,
    symmetric_plate_steady_state,
)
from stream.units import pcm
from stream.substances import light_water
from stream.utilities import just
from .conftest import MTR_fuel_and_channel


@pytest.mark.slow
@settings(deadline=None)
@given(Tin=st.floats(0, 100), iterations=st.integers(1, 20))
def test_symmetric_plate_steady_state_has_zero_diff_in_power(Tin, iterations):
    f, c = MTR_fuel_and_channel(z_N=25, fuel_N=8, clad_N=2)

    state = symmetric_plate_steady_state(
        c=c,
        f=f,
        mdot=0.6,
        p_abs=2e5,
        power=1e4,
        Tin=Tin,
        initial_guess_iterations=iterations,
    )

    assert np.allclose(np.diff(state[c.name]["T_cool"], n=2), 0.0, atol=1e-5)


@pytest.mark.slow
@settings(deadline=None)
@given(mdot=st.floats(1e-4, 1))
def test_symmetric_plate_steady_state_has_zero_diff_in_low_mdot(mdot):
    f, c = MTR_fuel_and_channel(z_N=25, fuel_N=8, clad_N=2)

    state = symmetric_plate_steady_state(
        c=c,
        f=f,
        mdot=mdot,
        p_abs=2e5,
        power=1e-6,
        Tin=35.0,
        initial_guess_iterations=20,
    )

    assert np.allclose(np.diff(state[c.name]["T_cool"], n=2), 0.0, atol=1e-5)


def test_symmetric_plate_steady_state_accepts_negative_flow_rate():
    """Tests whether the function accepts negative flow rate"""
    mdot = -1
    f, c = MTR_fuel_and_channel(z_N=5, fuel_N=2, clad_N=2)
    symmetric_plate_steady_state(c=c, f=f, mdot=mdot, p_abs=2e5, power=1e-6, Tin=35.0)


def test_symmetric_plate_steady_state_solves_negative_flow_rate():
    """Tests whether the function solves negative flow rate correctly"""
    mdot = -1.0
    power = 1e5
    Tin = 35.0
    f, c = MTR_fuel_and_channel(z_N=5, fuel_N=2, clad_N=2)
    solution = symmetric_plate_steady_state(c=c, f=f, mdot=mdot, p_abs=2e5, power=power, Tin=Tin)
    T_top_computed = solution[c.name]["T_cool"][0]
    T_top_estimated = power / light_water.specific_heat(Tin) + Tin
    assert np.isclose(T_top_computed, T_top_estimated, rtol=1.0e-2)


lambdas = st.floats(1e-3, 10.0, allow_nan=False)
betas = st.floats(700 * pcm * 10.0, 700 * pcm * 1000.0, allow_nan=False)
lifetimes = st.floats(1e-6, 1e-4, allow_nan=False)
specific_dicts = st.fixed_dictionaries(
    dict(
        generation_time=lifetimes,
        delayed_neutron_fractions=arrays(np.float64, 6, elements=betas, unique=True),
        delayed_groups_decay_rates=arrays(np.float64, 6, elements=lambdas, unique=True),
    )
)
powers = st.floats(min_value=1.0, max_value=3e7, allow_nan=False, allow_infinity=False)


@given(power=powers, kwargs=specific_dicts)
def test_point_kinetics_steady_state_follows_analytic_formula(power, kwargs):
    pk = PointKinetics(**kwargs)
    state = point_kinetics_steady_state(pk, power=power)
    assert np.allclose(pk.calculate(pk.load(state[pk.name]), T=None, t=0.0) / power, 0.0, atol=1e-8)


low_fraction = st.floats(0, 0.7, allow_nan=False)


@given(power=powers, kwargs=specific_dicts, ex_fraction=low_fraction)
def test_point_kinetics_w_input_steady_state_follows_analytic_formula(power, kwargs, ex_fraction):
    pk = PointKineticsWInput(**kwargs, temp_worth={}, ref_temp={})
    state = point_kinetics_steady_state(pk, power, power_input=ex_fraction * power)
    assert np.allclose(
        pk.calculate(pk.load(state[pk.name]), T={}, t=0.0, power_input=ex_fraction * power)
        / ((1 + ex_fraction) * power),
        0.0,
        atol=1e-8,
    )


def test_hydraulic_steady_state_is_a_root_for_a_single_loop_case():
    A, B = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((A, B), r := Resistor(1.0), i := Inertia(1.0)),
        flow_edge((B, A), p := Pump(pressure=1.0)),
        inertial_comps=[i],
        k_constructor=KirchhoffWDerivatives,
        reference_node=(A, 3),
        abs_pressure_comps=[r, p],
    )
    s = fg.guess_steady_state({r: 1.0, p: 1.0}, 10)
    y = fg.aggregator.load(s)
    assert np.allclose(fg.aggregator.compute(y), 0.0)


def test_hydraulic_steady_state_is_a_root_for_a_simple_parallel_case():
    A, B = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((A, B), r := Resistor(1.0, "r1"), i := Inertia(1.0)),
        flow_edge((A, B), r2 := Resistor(2.0, "r2")),
        flow_edge((B, A), p := Pump(pressure=1.0)),
        inertial_comps=[i],
        k_constructor=KirchhoffWDerivatives,
        reference_node=(A, 3),
        abs_pressure_comps=[r, p],
    )
    s = fg.guess_steady_state({r: 1.0, p: 1.5, r2: 0.5}, 10)
    y = fg.aggregator.load(s)
    assert np.allclose(fg.aggregator.compute(y), 0.0)


_FakeC = Calculation_factory(just(0.0), [False, False], dict(Tin=0, pressure=1))
_FakeC.indices = LumpedComponent.indices
fake = _FakeC("fake")


def test_hydraulic_steady_state_assumes_0_pressure_drop_for_unsupported_calculations():
    a, b = Junction("A"), Junction("B")

    fg = FlowGraph(
        flow_edge((a, b), r := Resistor(1.0), fake),
        flow_edge((b, a), p := Pump(pressure=1.0)),
        k_constructor=KirchhoffWDerivatives,
        reference_node=(a, 3),
    )
    s = fg.guess_steady_state({r: 1.0, p: 1.0}, 10)
    assert s["fake"]["pressure"] == 0.0
    y = fg.aggregator.load(s)
    assert np.allclose(fg.aggregator.compute(y), 0.0)


def test_hydraulic_steady_state_uses_strategy_when_provided():
    a, b = Junction("A"), Junction("B")

    fg = FlowGraph(
        flow_edge((a, b), r := Resistor(1.0), fake),
        flow_edge((b, a), p := Pump(pressure=1.0)),
        k_constructor=KirchhoffWDerivatives,
        reference_node=(a, 3),
    )
    s = fg.guess_steady_state({r: 1.0, p: 1.0}, 10, {fake: lambda mdot, T: mdot + T})
    assert s["fake"]["pressure"] == 11.0
