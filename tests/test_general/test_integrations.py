"""
Testing global, or integrative, arrangements.
"""
from functools import partial

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from networkx import DiGraph
from scipy.constants import g

from stream.aggregator import vars_, Aggregator, CalculationGraph
from stream.calculations import (
    Channel, ChannelAndContacts, Flapper, Friction, Fuel, Gravity,
    HeatExchanger, Inertia, Junction, Kirchhoff, KirchhoffWDerivatives,
    LocalPressureDrop, PointKinetics, Pump, Resistor, VolumetricFlowResistor
    )
from stream.calculations.ideal.ideal import LumpedComponent
from stream.calculations.flapper import continuously_differentiable_relaxation as cdr
from stream.composition import guess_hydraulic_steady_state
from stream.composition.cycle import (
    flow_edge, flow_graph, flow_graph_to_agr_and_k as agr_k, kirchhoffify,
    )
from stream.composition.mtr_geometry import symmetric_plate
from stream.jacobians import ALG_jacobian, DAE_jacobian
from stream.physical_models.pressure_drop import pressure_diff, friction_factor
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.substances.mocks import mock_liquid_funcs, mock_solid
from stream.units import cm, hour
from stream.utilities import just, summed, identity, strictly_monotonous as mono

from .conftest import are_close, pos_medium_floats, MTR_fuel_and_channel


@settings(deadline=None)
@given(*3 * [pos_medium_floats])
def test_pump_resistor_in_series_follows_analytic_solution(T, dp, r):
    """
    A simple circuit of an ideal pump and an ideal resistor.
    """
    P = Pump(pressure=dp)
    R = Resistor(resistance=r)

    fg = flow_graph(
        flow_edge(("A", "B"), P),
        flow_edge(("B", "A"), R),
        )
    agr, K = agr_k(fg, {P.name: dict(Tin=T), R.name: dict(Tin=T)})

    calculation = agr.solve_steady(
        State.merge(
            {K.name: {K.component_edge(P): dp / r,
                      K.component_edge(R): dp / r}},
            State.uniform((R, P), T),
            {R.name: dict(pressure=-dp), P.name: dict(pressure=dp)}
            )
        )

    assert np.allclose(calculation, [T, dp, T, -dp, dp / r, dp / r])


@settings(deadline=None)
@given(*3 * [pos_medium_floats])
def test_parallel_resistors_with_pump_against_analytic_solution(p, r1, r2):
    """
    A simple circuit of an ideal pump and two ideal resistors in parallel.
    """
    P = Pump(pressure=p)
    R1 = Resistor(resistance=r1, name='R1')
    R2 = Resistor(resistance=r2, name='R2')
    J0 = Junction(name='J0')
    J1 = Junction(name='J1')

    fg = flow_graph(
        flow_edge((J0, J1), R1),
        flow_edge((J0, J1), R2),
        flow_edge((J1, J0), P),
        )

    agr, K = agr_k(fg)
    calculation = agr.solve_steady(
        State.merge(
            {K.name: {K.component_edge(P): p * (r1 + r2) / (r1 * r2),
                      K.component_edge(R1): p / r1,
                      K.component_edge(R2): p / r2}},
            State.uniform((J0, J1, R1, R2, P), 1.0),
            {R1.name: dict(pressure=-p),
             R2.name: dict(pressure=-p),
             P.name: dict(pressure=p)}
            )
        )

    state = agr.save(calculation)

    total_flow = state[K.name][K.component_edge(P)]
    total_resistance = (r1 * r2) / (r1 + r2)
    assert np.isclose(total_flow, p / total_resistance)


@settings(deadline=None)
@given(integers(1, 100), *2 * [floats(1e-2, 1e3, allow_nan=False, allow_infinity=False)])
def test_resistors_in_series_against_analytic_solution(N, pressure, total_r):
    resistance = total_r / N
    P = Pump(pressure=pressure)
    J0 = Junction(name='J0')
    J1 = Junction(name='J1')

    Rs = [Resistor(resistance, name=str(i)) for i in range(N)]

    fg = flow_graph(
        flow_edge((J0, J1), *Rs),
        flow_edge((J1, J0), P)
        )
    agr, K = agr_k(fg, funcs={r: dict(Tin=1.0) for r in Rs})
    agr: Aggregator
    T = 1.0
    calculation = agr.solve_steady(np.full(len(agr), T), jac=ALG_jacobian(agr))
    state = agr.save(calculation)
    for R in Rs:
        assert np.isclose(state[R.name]["Tin"], T)
        assert np.isclose(state[R.name]["pressure"], -pressure / N)


@settings(deadline=None)
@given(*3 * [floats(1e-1, 1e2, allow_nan=False, allow_infinity=False)])
def test_channel_stable_state_with_uniform_heating_increases_linearly(P, mdot, T0):
    """
    For uniform heat flux, the temperature should rise linearly in the
    simple channel model (T flow).
    Each fuel cell receives :param P power.
    """
    channel_length = 1.0
    cells = 10
    boundaries = np.linspace(0, channel_length, cells + 1)
    x_bounds = np.array([0,1])

    power_shape = np.full((cells, len(x_bounds)-1), 1 / cells)
    pipe = EffectivePipe(length=channel_length, heated_perimeter=1,
                         wet_perimeter=1, area=1, heated_parts=(0, 1))
    C = Channel(z_boundaries=boundaries, fluid=mock_liquid_funcs, pipe=pipe)
    F = Fuel(
        z_boundaries=boundaries,
        x_boundaries=x_bounds,
        material=mock_solid,
        y_length=1.0,
        power_shape=power_shape,
        )

    agr = Aggregator(
        graph=DiGraph([
            (C, F, vars_('T_left')),
            (F, C, vars_('T_right'))]),
        funcs={
            C: dict(mdot=mdot, Tin=T0, p_abs=1e5, h_right=(h := 1.)),
            F: dict(power=P, h_left=h)
            })
    y0 = np.full(agr.vector_length, T0 + 1)
    y0[agr.var_index(C, "pressure")] = np.sum(C.pressure(T=T0, Tw=T0, mdot=mdot))
    state = agr.save(agr.solve_steady(y0, jac=ALG_jacobian(agr)))
    Tc_calculated = state[C.name]["T_cool"]
    Tw_calculated = state[F.name]["T_wall_left"]
    Tf_calculated = np.squeeze(state[F.name]["T"])
    cp = C.fluid.specific_heat(Tc_calculated)
    Tc_analytical = T0 + np.cumsum(P * power_shape.flatten() / (cp * mdot))
    h_fw = 2 * mock_solid.conductivity / F.dx
    # Tf_analytical = Tw_calculated + P * power_shape / h_fw
    Tw_analytical = (Tc_analytical * h + Tf_calculated * h_fw) / (h + h_fw)
    are_close(Tc_calculated, Tc_analytical)
    are_close(Tw_calculated, Tw_analytical)
    # are_close(Tf_calculated, Tf_analytical) TODO: Complete for fuel temps.


def test_channel_point_kinetics():
    """
    Checking compatibility between channels and Point Kinetics
    through a simple, analytical case.
    """
    T0 = 40
    channels_num = 10
    channel_length = 1.2
    cells = 7
    boundaries = np.linspace(0, channel_length, cells + 1)
    x_bounds = np.array([0,1])

    power_shape = np.full((cells, len(x_bounds)-1), 1 / cells)

    pipe = EffectivePipe(
        length=channel_length, heated_perimeter=1, wet_perimeter=1, area=1
        )
    channel_input = dict(
        z_boundaries=boundaries,
        fluid=mock_liquid_funcs,
        pipe=pipe,
        h_wall_func=just(1.0),
        )
    fuel_input = dict(
        z_boundaries=boundaries,
        x_boundaries=x_bounds,
        material=mock_solid,
        y_length=1.0,
        power_shape=power_shape,
        )
    channels = [ChannelAndContacts(**channel_input, name=f'CC{i}')
                for i in range(channels_num)]
    fuels = [Fuel(**fuel_input, name=f'F{i}') for i in range(channels_num)]

    alpha_Tc = {ch: np.random.rand() * np.ones(cells) for ch in channels}
    alpha_Tf = {f: np.random.rand() * np.ones(cells) for f in fuels}
    temp_worth = alpha_Tf | alpha_Tc
    lambdak = np.array([55.72, 22.72, 6.22, 2.3, 0.618, 0.23])
    betak = 700e-5 * np.ones_like(lambdak) / len(lambdak)
    PK = PointKinetics(
        generation_time=1,
        delayed_groups_decay_rates=lambdak,
        delayed_neutron_fractions=betak,
        temp_worth=temp_worth,
        ref_temp=dict.fromkeys(temp_worth, T0),
        )
    power_control = CalculationGraph.from_decoupled(PK, funcs={PK: dict(t=identity)})

    rods = summed(symmetric_plate(
        channel, fuel, funcs={channel: dict(mdot=1, Tin=T0 - 10, p_abs=1e5)})
                  for fuel, channel in zip(fuels, channels))
    core = rods + power_control
    initial_state = {}
    for channel, fuel in zip(channels, fuels):
        core.graph.add_edge(channel, PK, variables=("T",))
        core.graph.add_edge(fuel, PK, variables=("T",))
        core.graph.add_edge(PK, fuel, variables=("power",))
        initial_state[fuel.name] = dict(T=T0, T_wall_left=T0, T_wall_right=T0)
        initial_state[channel.name] = dict(
            T_cool=T0, h_left=1.0, h_right=1.0, pressure=0.0)
    initial_state[PK.name] = dict(power=1, ck=np.ones_like(lambdak))
    agr: Aggregator = core.to_aggregator()
    solution = agr.solve_steady(initial_state, jac=ALG_jacobian(agr))
    steady_state = agr.save(solution, strict=True)
    # Channels temperatures should rise linearly along the channel
    # at steady state
    for channel in channels:
        Tc = steady_state[channel.name]["T_cool"]
        assert np.all(np.diff(Tc) > 0)
        assert np.allclose(np.diff(np.diff(Tc)), 0)


def test_kirchhoff_with_decaying_pump_eventually_flips_flow_direction_gravity():
    """
    Gravity induces pressure differences. p1, p2 designate the uphill
    and downhill gravity differences, in a very crude model.
    """
    p0 = 4000
    high_T = 60.0
    low_T = 20.0

    def _tran_p(time): return p0 * np.exp(-time)

    g1 = Gravity(fluid=light_water, disposition=1.0, name='G1')
    g2 = Gravity(fluid=light_water, disposition=-1.0, name='G2')
    pt = Pump()
    r = Resistor(resistance=1e5)

    G = DiGraph()
    G.add_nodes_from([pt, r, g1, g2])
    agr_input = CalculationGraph(G, {pt: dict(Tin=low_T, pressure=_tran_p),
                                     g1: dict(Tin=high_T),
                                     g2: dict(Tin=low_T),
                                     r: dict(Tin=low_T)})

    k = Kirchhoff(flow_graph(
        flow_edge(('A', 'B'), r, g1, g2),
        flow_edge(('B', 'A'), pt),
        ))
    agr = kirchhoffify(agr_input, k).to_aggregator()

    initial = agr.solve_steady(np.zeros(len(agr)))
    delta_rho = light_water.density(low_T) - light_water.density(high_T)
    are_close(agr.save(initial)[r.name]["pressure"], g * delta_rho - p0)
    t = np.linspace(0, 10, 10)
    transient = agr.solve(initial, time=t)
    mdot = agr.at_times(transient, k, k.component_edge(r))
    assert mdot[-1] < 0  # Flow reversal due to temperature differences


def test_Tin_jumps_at_resistor_between_two_hxs_at_flow_reversal():
    r"""
    Circuit::

        HX1 -->-- R -->-- HX2
         |                 |
         ---<--- P ---<----

    When flow direction is positive, the resistor's temperature
    should be that of HX1. If it is reversed, the temperature should be
    that of HX2.
    """
    HX1 = HeatExchanger(outlet=20.0, name='HX1')
    HX2 = HeatExchanger(outlet=60.0, name='HX2')
    R = Resistor(resistance=1.0)
    P = Pump(pressure=1.0)
    J0 = Junction(name='J0')
    J1 = Junction(name='J1')

    fg = flow_graph(
        flow_edge((J0, J1), HX1, R, HX2),
        flow_edge((J1, J0), P),
        )

    agr, K = agr_k(fg)

    sol = agr.solve_steady(np.ones(len(agr)))
    are_close(agr.save(sol)[R.name]["Tin"], HX1.T)

    # Let's flip
    P.p *= -1

    sol = agr.solve_steady(np.ones(len(agr)))
    are_close(agr.save(sol)[R.name]["Tin"], HX2.T)


def test_power_is_negligible_for_negative_Tfuel_feedback_and_ref_temp_is_boundary_conditions():
    """
    This test checks whether negative feedback may be established between a
    Fuel calculation and Point Kinetics. In this setup, a fuel plate is tied to
    a thermal bath of temperature T0, which is also the critical reference
    temperature. Thus, at steady state the power should be close to zero and
    all temperatures should be T0.
    """
    T0 = 35.0

    z_N, fuel_N, clad_N = 10, 5, 2

    F, _ = MTR_fuel_and_channel(z_N, fuel_N, clad_N)
    F.power_shape = np.ones(z_N * fuel_N) / (z_N * fuel_N)
    PK = PointKinetics(
        generation_time=1e-2,
        delayed_neutron_fractions=np.array([0.007]),
        delayed_groups_decay_rates=np.array([1.0]),
        temp_worth={F: np.full(z_N * fuel_N, 1e-1)},
        ref_temp={F: T0},
        )

    agr = Aggregator(
        graph=DiGraph([(F, PK, vars_("T")), (PK, F, vars_("power"))]),
        funcs={F: dict(T_left=T0, T_right=T0), PK: dict(t=identity)},
        )

    y0 = np.zeros(len(agr))
    y0[agr.sections[F]] = 2 * T0
    y0[agr.var_index(PK, "ck")] = 1e3
    y0[agr.var_index(PK, "power")] = 1e5

    sol = agr.solve_steady(y0)
    state = agr.save(sol)
    assert state[PK.name]["power"] < 1e-3
    assert np.allclose(state[F.name]["T"], T0)


def test_power_is_negligible_for_negative_Tcool_feedback_and_ref_temp_is_inlet():
    """
    This test checks whether negative feedback may be established between an MTR
    Channel calculation and Point Kinetics. In this setup, a fuel plate
    and channel represent the core. T0 is the critical reference
    temperature of the coolant. Thus, at steady state the power should be
    close to zero and all temperatures should be close to T0.
    """
    T0 = 35.0
    z_N, fuel_N, clad_N = 10, 5, 2
    F, C = MTR_fuel_and_channel(z_N, fuel_N, clad_N)
    PK = PointKinetics(
        generation_time=1e-2,
        delayed_neutron_fractions=np.array([0.007]),
        delayed_groups_decay_rates=np.array([1.0]),
        temp_worth={C: np.full(z_N, 1e-1)},
        ref_temp={C: T0},
        )

    power_stuff = CalculationGraph(
        graph=DiGraph([(PK, F, vars_("power")), (C, PK, vars_("T"))]),
        funcs={C: dict(p_abs=2e5, Tin=T0, mdot=0.1), PK: dict(t=identity)},
        )

    agr = (symmetric_plate(C, F) + power_stuff).to_aggregator()

    y0 = np.zeros(len(agr))
    y0[agr.sections[F]] = 2 * T0
    y0[agr.var_index(PK, "ck")] = 1e3
    y0[agr.var_index(PK, "power")] = 1e5
    y0[agr.var_index(C, "h_left")] = 0.9e5
    y0[agr.var_index(C, "h_right")] = 0.9e5
    y0[agr.var_index(C, "T_cool")] = 2 * T0
    y0[agr.var_index(C, "pressure")] = -1e4

    sol = agr.solve_steady(y0)
    state = agr.save(sol)
    assert state[PK.name]["power"] < 1e-3
    assert np.allclose(state[C.name]["T_cool"], T0)


@settings(deadline=None)
@given(*2 * [integers(1, 10)])
def test_inertia_through_RL_circuit_follows_analytic_solution(r, inertia):
    r"""

    Parameters
    ----------
    r: float
        Resistance
    inertia: float
        Moment of inertia
    """
    R = Resistor(resistance=r)
    L = Inertia(inertia=inertia)

    fg = flow_graph(
        flow_edge(("A", "B"), L),
        flow_edge(("B", "A"), R),
        )

    AGR, KD = agr_k(
        fg,
        {R: dict(Tin=0.0), L: dict(Tin=0.0)},
        inertial_comps=(L,),
        k_constructor=KirchhoffWDerivatives,
        )

    steady = AGR.solve_steady(guess=np.full(len(AGR), 0.0))
    steady[AGR.var_index(KD, KD.component_edge(L))] = 1.0
    steady[AGR.var_index(KD, KD.component_edge(R))] = 1.0
    time = np.linspace(0, 1.0)
    sol = AGR.solve(y0=steady, time=time, jacfn=DAE_jacobian(AGR))
    states = [AGR.save(sol[i], t) for i, t in enumerate(time)]
    mdot = [state[KD.name][KD.component_edge(R)] for state in states]

    are_close(mdot, np.exp(-(r / inertia) * time), rtol=1e-4)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_kirchhoff_significance_in_two_in_series_resistors(r1, r2, p, signify):
    R1 = Resistor(resistance=r1, name='R1')
    R2 = Resistor(resistance=r2, name='R2')
    P = Pump(pressure=p)

    fg = flow_graph(
        flow_edge(("A", "B"), R1, signify=signify),
        flow_edge(("B", "A"), R2, P),
        )

    AGR, K = agr_k(fg)

    y0 = np.full(len(AGR), 0.0)
    y0[AGR.var_index(K, K.component_edge(R1))] = m1 = p / (r1 + signify * r2)
    y0[AGR.var_index(K, K.component_edge(R2))] = m2 = m1 * signify
    y0[AGR.var_index(R1, "pressure")] = - m1 * r1
    y0[AGR.var_index(R2, "pressure")] = - m2 * r2
    y0[AGR.var_index(P, "pressure")] = p

    assert np.allclose(AGR.compute(y0), 0)
    state: State = AGR.save(AGR.solve_steady(guess=y0))

    mdot1 = state[K.name][K.component_edge(R1)]
    mdot2 = state[K.name][K.component_edge(R2)]

    assert np.isclose(mdot1 * signify, mdot2)
    assert np.isclose(mdot1, m1)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats, integers(1, 10))
def test_kirchhoff_significance_for_many_parallel_edges(r1, r2, p, signify):
    r"""Looking at :func:`test_kirchhoff_significance_in_two_in_series_resistors`,
    for integer values of ``signify`` there should be an equivalent system in which
    ``R1`` is duplicated ``signify`` times.
    """

    R1s = [Resistor(resistance=r1, name=f'R1-{i}') for i in range(signify)]
    R2 = Resistor(resistance=r2, name='R2')
    P = Pump(pressure=p)
    J0, J1 = Junction(name='J0'), Junction(name='J1')
    fg = flow_graph(
        *(flow_edge((J0, J1), R1) for R1 in R1s),
        flow_edge((J1, J0), R2, P),
        )

    AGR, K = agr_k(fg)

    y0 = np.full(len(AGR), 0.0)
    m1 = p / (r1 + signify * r2)
    for R1 in R1s:
        y0[AGR.var_index(K, K.component_edge(R1))] = m1
        y0[AGR.var_index(R1, "pressure")] = - m1 * r1
    y0[AGR.var_index(K, K.component_edge(P))] = m2 = m1 * signify
    y0[AGR.var_index(R2, "pressure")] = - m2 * r2
    y0[AGR.var_index(P, "pressure")] = p

    state: State = AGR.save(AGR.solve_steady(guess=y0))

    for R1 in R1s:
        assert np.isclose(state[K.name][K.component_edge(R1)], m1)

    mdot1 = state[K.name][K.component_edge(R1s[0])]
    mdot2 = state[K.name][K.component_edge(R2)]

    assert np.isclose(mdot2, m2)
    assert np.isclose(mdot1, m1)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats)
def test_pump_and_current_source(p, mdot):
    fg = flow_graph(
        flow_edge(("A", "B"), P := Pump(pressure=p, name='P1')),
        flow_edge(("B", "A"), I := Pump(mdot0=mdot, name='P2')),
        )

    AGR, K = agr_k(fg)
    y0 = np.full(len(AGR), 0.0)
    y0[AGR.var_index(K, K.component_edge(P))] = mdot
    y0[AGR.var_index(K, K.component_edge(I))] = mdot
    y0[AGR.var_index(I, "pressure")] = -p
    y0[AGR.var_index(P, "pressure")] = p

    assert np.allclose(AGR.compute(y0), 0)


def test_flapper_opens_with_ref_mdot():
    mdot0 = 1.0
    p = 1.0
    F = Flapper(
        open_at_current=0.1 * mdot0,
        f=1.0,
        fluid=mock_liquid_funcs,
        area=1.0,
        open_rate=1e1,
        )
    R = Resistor(resistance=p / mdot0)

    J0, J1 = Junction(name='J0'), Junction(name='J1')

    fg = flow_graph(
        flow_edge((J0, J1), P := Pump()),
        flow_edge((J1, J0), R, ref_mdot_for=(F,)),
        flow_edge((J1, J0), F),
        )

    AGR, K = agr_k(
        fg,
        funcs={
            P: dict(pressure=lambda t: p * np.exp(-t), Tin=0.0),
            R: dict(Tin=0.0),
            F: dict(Tin=0.0, t=identity),
            },
        ref_mdots=(F,),
        )

    assert AGR.external[F]["ref_mdot"][K] == AGR.external[R]["mdot"][K]

    y0 = np.full(len(AGR), 0.0)
    y0[AGR.var_index(K, K.component_edge(P))] = mdot0
    y0[AGR.var_index(K, K.component_edge(R))] = mdot0
    y0[AGR.var_index(K, K.component_edge(F))] = 0.0
    y0[AGR.var_index(R, "pressure")] = -p
    y0[AGR.var_index(F, "pressure")] = -p
    y0[AGR.var_index(P, "pressure")] = p

    assert np.allclose(AGR.compute(y0), 0)
    assert np.isinf(F.t_open)
    time = np.linspace(0, 5, 100)
    transient = AGR.solve(
        y0=AGR.solve_steady(y0), time=time, eq_type="DAE", max_step_size=1e-3
        )

    assert np.isclose(F.t_open, np.log(10.0), rtol=1e-3)
    mdot_F = AGR.at_times(transient, K, K.component_edge(F))
    assert np.allclose(mdot_F[(closed := time <= np.log(10))], 0)
    assert not np.allclose(mdot_F[~closed], 0)


def test_flapper_and_pump():
    mdot0 = 1.0
    p = 1.0
    F = Flapper(
        open_at_current=0.1 * mdot0,
        f=1.0,
        fluid=mock_liquid_funcs,
        area=1.0,
        open_rate=1e1,
        )
    F.open(2.5)  # Sets the Flapper to open at a specific time

    fg = flow_graph(
        flow_edge(("A", "B"), P := Pump()),
        flow_edge(("B", "A"), F),
        )

    AGR, K = agr_k(
        fg,
        funcs={P: dict(pressure=lambda t: p * np.exp(- t), Tin=0.),
               F: dict(Tin=0., t=identity, ref_mdot=np.inf)},
        )

    y0 = np.full(len(AGR), 0.0)
    y0[AGR.var_index(K, K.component_edge(P))] = 0.0
    y0[AGR.var_index(K, K.component_edge(F))] = 0.0
    y0[AGR.var_index(F, "pressure")] = -p
    y0[AGR.var_index(P, "pressure")] = p

    assert np.allclose(AGR.compute(y0), 0)
    time = np.linspace(0, 5, 100)
    sol = AGR.solve(y0=y0, time=time, eq_type="DAE")
    assert np.all(
        sol[np.argwhere(time >= F.t_open), AGR.var_index(K, K.component_edge(P))] != 0.)


def test_pump_coastdown_allows_channels_to_reverse_flow_direction():
    """
    Let us assume the following system: two channels standing upright where
    one is hotter than the other, and a pump pumping water from the cold
    channel to the hot channel. As the pump coasts down, the gravitational
    effect becomes more prominent until the flow is reversed.
    """
    D = 10 * cm
    mdot0 = 1.0
    T_cold = 20.0
    T_hot = 80.0
    pipe = EffectivePipe.circular(length=1.0, diameter=D)
    z_bounds = np.linspace(0, pipe.length, 10)
    dp_func = partial(pressure_diff,
                      f=friction_factor("regime_dependent", re_bounds=(2000, 5000), k_R=1.))
    cold = Channel(
        z_boundaries=z_bounds,
        fluid=light_water,
        pipe=pipe,
        pressure_func=partial(dp_func, g=-g),
        name='ColdC'
        )
    hot = Channel(
        z_boundaries=z_bounds,
        fluid=light_water,
        pipe=pipe,
        pressure_func=partial(dp_func, g=g),
        name='HotC'
        )

    fg = flow_graph(
        flow_edge(("A", "B"), P := Pump(mdot0=mdot0), hot), flow_edge(("B", "A"), cold)
        )

    AGR, K = agr_k(
        fg, funcs={
            cold: dict(Tin=T_cold, Tin_minus=T_cold),
            hot: dict(Tin=T_hot, Tin_minus=T_hot),
            })

    initial_guess = State.merge(
        State.uniform([cold], T_cold, "T_cool"),
        State.uniform([hot], T_hot, "T_cool"),
        State.uniform([K], mdot0),
        State.uniform([cold, hot], -1e3, "pressure"),
        {P.name: dict(pressure=1e3, Tin=T_cold)},
        )

    steady = AGR.solve_steady(initial_guess)
    time = np.linspace(0, 0.05, 1000)

    P.mdot0 = None
    AGR.funcs[P] = dict(
        pressure=lambda t: steady[AGR.var_index(P, 'pressure')] * np.exp(-t),
        Tin=T_cold, Tin_minus=T_cold)
    transient = AGR.solve(steady, time=time, jacfn=DAE_jacobian(AGR))
    assert transient

    mdot = AGR.at_times(transient, K, K.component_edge(P))
    assert np.allclose(AGR.at_times(transient, cold, "T_cool"), T_cold)
    assert np.allclose(AGR.at_times(transient, hot, "T_cool"), T_hot)
    assert np.all(np.diff(mdot) < 0)
    assert mdot[0] > 0 > mdot[-1]

    t_zero = time[np.argmin(np.abs(mdot))]
    delta_rho = light_water.density(T_cold) - light_water.density(T_hot)
    gravity_pressure_difference = pipe.length * g * delta_rho
    assert np.isclose(AGR.funcs[P]["pressure"](t_zero), gravity_pressure_difference)


def test_inertia_with_friction_in_PCS_coastdown():
    r"""This test is equivalent
    to :ref:`Primary Cooling System Coastdown`.
    """
    dp0 = 1.6e5
    pump = Pump(pressure=dp0)
    inertia = 8e3
    flywheel = Inertia(inertia=inertia)
    T = 20.0
    Q0 = 2000 / hour
    rho0 = light_water.density(T)
    mdot0 = Q0 * rho0
    f = 2 * rho0 * dp0 / (mdot0 ** 2)

    resistor = Friction(f=f, fluid=light_water, length=1.,
                        hydraulic_diameter=1., area=1.)
    fg = flow_graph(flow_edge(('A', 'B'), pump, flywheel),
                    flow_edge(('B', 'A'), resistor))

    agr, K = agr_k(fg, inertial_comps=[flywheel],
                   k_constructor=KirchhoffWDerivatives,
                   funcs={resistor: dict(Tin=T)})
    AB = K.component_edge(pump)
    BA = K.component_edge(resistor)
    ab2 = f"(A -> B, mdot2 0)"
    ba2 = f"(B -> A, mdot2 0)"
    assert ab2 in K.variables, (ab2, list(K.variables.keys()))
    assert ba2 in K.variables, (ba2, list(K.variables.keys()))

    steady = agr.save(agr.solve_steady({
        K.name: {AB: mdot0, ab2: 0., BA: mdot0, ba2: 0.},
        pump.name: dict(Tin=T, pressure=dp0),
        resistor.name: dict(Tin=T, pressure=-dp0),
        flywheel.name: dict(Tin=T, pressure=0.)
        }))

    pump.p = 0.0  # Immediate Shutdown
    t = np.linspace(0, 300, 100)
    transient = agr.solve(steady, time=t)

    mdotc = agr.at_times(transient, K, AB)
    alpha = np.abs(resistor.dp_out(mdot=1.0, Tin=T) / inertia)
    mdota = mdot0 / (1 + alpha * mdot0 * t)

    assert np.allclose(mdotc, mdota)


def test_inertia_with_flapper_in_PCS_coastdown():
    r"""This test is conceptually equivalent
    to :ref:`Primary Cooling System Coastdown`, only this time another branch is added,
    hosting a :class:`.Flapper`. The flapper is set to open at a specific time.

    Eventually, flow between the open flapper and the resistor should be equal.
    """
    k = 1
    flapper = Flapper(open_at_current=0., f=2*k, area=1.0, open_rate=1.0,
                      relaxation=cdr, name="F", fluid=mock_liquid_funcs)
    flapper.open(t_open := 100.0)

    T = 20.0
    flywheel = Inertia(inertia=1e3)
    pump = Pump(mdot0=(mdot0 := 1.0))
    R = VolumetricFlowResistor(k=k, density_func=just(1.0), name='R')
    A = Junction("A")
    B = Junction("B")

    fg = flow_graph(flow_edge((A, B), pump, flywheel),
                    flow_edge((B, A), R),
                    flow_edge((B, A), flapper))

    agr, K = agr_k(fg, inertial_comps=[flywheel], k_constructor=KirchhoffWDerivatives,
                   funcs={R: dict(Tin=T), flapper: dict(t=identity, ref_mdot=np.inf)})

    flows = {pump: mdot0, R: mdot0, flapper: 0.0}
    guess = guess_hydraulic_steady_state(K, flows, T)
    steady = agr.save(agr.solve_steady(guess))

    pump.p = 0.0  # Immediate Shutdown
    pump.mdot0 = None
    t = mono(np.linspace(0, 300, 100), np.linspace(t_open - 2, t_open + 5, 100))

    atol = np.full(len(agr), 1e-12)
    atol[agr.sections[K]][K.variables_by_type["mdot2"]] = 1e-5

    transient = agr.solve(steady, time=t, atol=atol)

    mdot_r = agr.at_times(transient, K, K.component_edge(R))[-1]
    mdot_f = agr.at_times(transient, K, K.component_edge(flapper))[-1]

    assert np.isclose(mdot_r, mdot_f)


sensible_resistors = floats(allow_infinity=False, allow_nan=False, max_value=30,
                            min_value=0.01)


def test_inertia_with_transistor_in_PCS_coastdown():
    r"""This test is conceptually equivalent
    to :ref:`Primary Cooling System Coastdown`. This time another branch is added,
    hosting a time dependent parabolic resistor which starts at a large resistance,
    and decreases drastically at some point in time.
    We only check for convergence.

    Absolute tolerance (atol) in :math:`\ddot{m}` eqs. should be noted.
    """
    k1 = k_final = 1
    k2 = 1e7
    t_open = 100.0
    t_final = 300.0

    class Transistor(LumpedComponent):
        name = "Transistor"

        def dp_out(self, *, mdot, t=0, **_):
            t = float(t)
            if t <= t_open:
                k = k2
            else:
                k = (k2 - k_final) * np.exp(-50 * (t - t_open) / t_final) + k_final
            return - k * mdot ** 2

    transistor = Transistor()

    T = 20.0
    flywheel = Inertia(inertia=1e3)
    pump = Pump(mdot0=(mdot0 := 1.0))
    R = VolumetricFlowResistor(k=k1, density_func=just(1.0), name='R')
    A = Junction("A")
    B = Junction("B")

    fg = flow_graph(flow_edge((A, B), pump, flywheel),
                    flow_edge((B, A), R),
                    flow_edge((B, A), transistor))

    agr, K = agr_k(fg, inertial_comps=[flywheel], k_constructor=KirchhoffWDerivatives,
                   funcs={R: dict(Tin=T), transistor: dict(t=identity)})

    def sr(a, b): 
        return 1 + np.sqrt(a/b)
    
    flows = {pump: mdot0, R: mdot0 / sr(k1, k2), transistor: mdot0 / sr(k2, k1)}
    guess = guess_hydraulic_steady_state(K, flows, T)
    steady = agr.save(agr.solve_steady(guess))

    total_k = steady[pump.name]["pressure"] / mdot0 ** 2
    assert np.isclose(total_k, k1*k2 / (np.sqrt(k1) + np.sqrt(k2)) ** 2)

    pump.p = 0.0  # Immediate Shutdown
    pump.mdot0 = None
    time = np.linspace(0, t_final, 101)
    y0 = agr.load(steady)
    atol = np.full(len(agr), 1e-12)
    atol[agr.sections[K]][K.variables_by_type["mdot2"]] = 1e-5

    assert agr.solve(y0, time=time, atol=atol)


@settings(deadline=None)
@given(sensible_resistors, sensible_resistors)
def test_inertia_with_two_parallel_resistors(k1, k2):
    r"""This test is equivalent conceptually to :ref:`Primary Cooling System Coastdown`,
    but with two resistors in parallel instead of one. This test was added due to
    prove :class:`.KirchhoffWDerivatives`'s ability to handle more complex situations.
    """
    T = 20.0
    flywheel = Inertia(inertia=(inertia := 1e3))
    pump = Pump(mdot0=(mdot0 := 1.0))
    R1 = VolumetricFlowResistor(k=k1, density_func=just(1.0), name='R1')
    R2 = VolumetricFlowResistor(k=k2, density_func=just(1.0), name='R2')
    A = Junction("A")
    B = Junction("B")

    fg = flow_graph(flow_edge((A, B), pump, flywheel),
                    flow_edge((B, A), R1),
                    flow_edge((B, A), R2))

    agr, K = agr_k(fg, inertial_comps=[flywheel], k_constructor=KirchhoffWDerivatives,
                   funcs={R1: dict(Tin=T)})

    def sr(a, b): 
        return 1 + np.sqrt(a/b)
    
    flows = {pump: mdot0, R1: mdot0 / sr(k1, k2), R2: mdot0 / sr(k2, k1)}
    guess = guess_hydraulic_steady_state(K, flows, T)
    guess[pump.name]["pressure"] = k1 * flows[R1] ** 2
    steady = agr.save(agr.solve_steady(guess, jac=ALG_jacobian(agr)))

    total_k = steady[pump.name]["pressure"] / mdot0 ** 2
    assert np.isclose(total_k, k1*k2 / (np.sqrt(k1) + np.sqrt(k2)) ** 2)

    pump.p = 0.0  # Immediate Shutdown
    pump.mdot0 = None
    t = np.linspace(0, 100, 100)
    transient = agr.solve(steady, time=t, rtol=1e-6, atol=1e-7)

    mdotc = agr.at_times(transient, K, K.component_edge(pump))
    alpha = total_k / inertia
    mdota = mdot0 / (1 + alpha * mdot0 * t)

    assert np.allclose(mdotc, mdota)


def test_local_pressure_with_flow_reversal():
    """Testing a simple circuit in which a local pressure drop experiences flow
    reversal
    """
    pump = Pump()
    A1, A2 = 1.0, 2.0
    r = LocalPressureDrop(fluid=light_water, A1=A1, A2=A2)

    fg = flow_graph(flow_edge(('A', 'B'), pump),
                    flow_edge(('B', 'A'), r))

    mdot0 = 3.0
    Tin = 20.0
    agr, K = agr_k(
        fg, funcs={pump: dict(mdot0=lambda t: mdot0 - t, Tin=Tin)}
        )
    dp0 = np.abs(r.dp_out(Tin=Tin, mdot=mdot0))

    steady = State.merge(
        State.uniform(agr.graph, Tin, "Tin"),
        State.uniform([K], mdot0),
        {pump.name: dict(pressure=dp0), r.name: dict(pressure=-dp0)},
        )
    assert np.allclose(agr.compute(agr.load(steady)), 0.0)

    time = np.linspace(0, 6, 7)
    transient = agr.solve(steady, time)
    mdot = agr.at_times(transient, K, K.component_edge(pump))
    assert np.all(mdot[:4] >= 0)
    assert np.all(mdot[3:] <= 1e-5)
