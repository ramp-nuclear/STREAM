import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from stream.calculations import (
    ChannelAndContacts,
    Gravity,
    Inertia,
    Junction,
    KirchhoffWDerivatives,
    PointKinetics,
    PointKineticsWInput,
    Pump,
    Resistor,
)
from stream.aggregator import Aggregator
from stream.calculations.ideal.ideal import LumpedComponent
from stream.composition import Calculation_factory, FlowGraph, flow_edge, seed_steady_state
from stream.composition.subsystems import (
    guess_hydraulic_steady_state,
    point_kinetics_steady_state,
    symmetric_plate_steady_state,
)
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water
from stream.units import mm, pcm
from stream.utilities import just

from .conftest import MTR_fuel_and_channel
from .test_natural_convection import NC_MDOT, _build


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
    # atol 1e-7: at the smallest generation_time (Λ=1e-6) the PK terms are O(1/Λ)≈1e6, so the
    # steady-state residual floors at ~1e-8 float roundoff — well below any physics threshold.
    assert np.allclose(pk.calculate(pk.load(state[pk.name]), T=None, t=0.0) / power, 0.0, atol=1e-7)


low_fraction = st.floats(0, 0.7, allow_nan=False)


@given(power=powers, kwargs=specific_dicts, ex_fraction=low_fraction)
def test_point_kinetics_w_input_steady_state_follows_analytic_formula(power, kwargs, ex_fraction):
    pk = PointKineticsWInput(**kwargs, temp_worth={}, ref_temp={})
    state = point_kinetics_steady_state(pk, power, power_input=ex_fraction * power)
    assert np.allclose(
        pk.calculate(pk.load(state[pk.name]), T={}, t=0.0, power_input=ex_fraction * power)
        / ((1 + ex_fraction) * power),
        0.0,
        atol=1e-7,  # same Λ=1e-6 roundoff floor as above
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


# Zero residual, but as a full length-2 vector: a scalar 0.0 would rely on the
# silent broadcast that compute() now rejects.
_FakeC = Calculation_factory(just(np.zeros(2)), [False, False], dict(Tin=0, pressure=1))
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


def test_hydraulic_guess_includes_channel_htc_so_it_loads():
    """A ChannelAndContacts owns algebraic h_left/h_right variables; the hydraulic
    guess must supply them so the returned State loads instead of a bare KeyError."""
    zb = np.linspace(0, 1, 6)
    pipe = EffectivePipe.rectangular(length=1, edge1=2 * mm, edge2=70 * mm, heated_edge=70 * mm)
    c = ChannelAndContacts(z_boundaries=zb, fluid=light_water, pipe=pipe)
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), c, r := Resistor(1.0)),
        flow_edge((b, a), p := Pump(pressure=1.0)),
        reference_node=(a, 1e5),
        abs_pressure_comps=[c],
    )
    s = fg.guess_steady_state({c: 1.0, r: 1.0, p: 1.0}, 40.0)
    assert {"h_left", "h_right"} <= set(s[c.name].keys())
    y = fg.aggregator.load(s)  # must not raise KeyError
    assert len(y) == len(fg.aggregator)


def test_reversed_flow_temperature_guess_uses_suffix_cumsum(monkeypatch):
    """For mdot<0 the coolant-temperature guess must follow the reverse-flow energy
    balance (suffix cumsum), not a reversed prefix cumsum — the two agree only for an
    axially symmetric power shape, which is why CI never caught it."""
    z_N = 20
    zb = np.linspace(0, 1, z_N + 1)
    zc = 0.5 * (zb[:-1] + zb[1:])
    f, c = MTR_fuel_and_channel(z_N=z_N, fuel_N=8, clad_N=2, z_weight=np.exp(-6 * zc))

    # Intercept the guess before the solver runs: return it verbatim so save() exposes it.
    captured = {}

    def capture_guess(self, y0, **_):
        captured["y0"] = y0
        return y0

    monkeypatch.setattr(Aggregator, "solve_steady", capture_guess)

    mdot, power, Tin, p_abs = -0.6, 2e5, 35.0, 2e5
    state = symmetric_plate_steady_state(
        c=c, f=f, mdot=mdot, p_abs=p_abs, power=power, Tin=Tin, initial_guess_iterations=1
    )
    tc0 = np.asarray(state[c.name]["T_cool"])

    cp = c.fluid.specific_heat(Tin)
    power_mat = np.zeros(f.shape)
    power_mat[f.meat == 1] = power * f.power_shape
    dT = np.sum(power_mat, 1) / (abs(mdot) * cp)
    expected = Tin + np.cumsum(dT[::-1])[::-1]  # reverse-flow energy balance
    reversed_prefix = (Tin + np.cumsum(dT))[::-1]  # the old, wrong guess

    assert np.allclose(tc0, expected)
    assert not np.allclose(tc0, reversed_prefix)


# Deliberately overrides edge-routed Tin via funcs (fixed-boundary idiom) -> shadow warning.
@pytest.mark.filterwarnings("ignore:funcs for .* shadow edge-routed")
def test_open_flapper_gets_a_physically_consistent_dp_guess():
    """A closed Flapper's dp is undetermined (0.0), but an OPEN Flapper carrying known
    flow has a computable dp; the guess must use it instead of the DPCalculation 0.0."""
    from stream.calculations import Flapper
    from stream.calculations.flapper import continuously_differentiable_relaxation as cdr
    from stream.calculations.ideal.resistors import VolumetricFlowResistor
    from stream.composition import guess_hydraulic_steady_state
    from stream.composition.cycle import flow_edge, flow_graph
    from stream.composition.cycle import flow_graph_to_agr_and_k as agr_k
    from stream.physical_models.pressure_drop import local_pressure_by_mdot
    from stream.substances.mocks import mock_liquid_funcs
    from stream.utilities import identity, just

    kf = 1.0
    flapper = Flapper(
        open_at_current=0.0, f=2 * kf, area=1.0, open_rate=1.0, relaxation=cdr, name="F", fluid=mock_liquid_funcs
    )
    T = 20.0
    flywheel = Inertia(inertia=1e3)
    pump = Pump(mdot0=1.0)
    R = VolumetricFlowResistor(k=kf, density_func=just(1.0), name="R")
    a, b = Junction("A"), Junction("B")
    fg = flow_graph(
        flow_edge((a, b), pump, flywheel),
        flow_edge((b, a), R),
        flow_edge((b, a), flapper),
    )
    _, K = agr_k(
        fg,
        inertial_comps=[flywheel],
        k_constructor=KirchhoffWDerivatives,
        funcs={R: dict(Tin=T), flapper: dict(t=identity, ref_mdot=np.inf)},
    )
    flows = {pump: 1.0, R: 0.5, flapper: 0.5}
    dp_true = -local_pressure_by_mdot(0.5, mock_liquid_funcs.density(T), flapper.f, flapper._A)
    assert dp_true != 0.0

    # Closed flapper: dp not physically determined -> 0.0.
    assert guess_hydraulic_steady_state(K, flows, T)["F"]["pressure"] == 0.0
    # Open flapper: guess must reflect the open-state resistance.
    flapper.open(0.0)
    assert np.isclose(guess_hydraulic_steady_state(K, flows, T)["F"]["pressure"], dp_true)


def _gravity_loop():
    from stream.calculations import Gravity, HeatExchanger
    from stream.substances import light_water

    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=0.5, name="Pump")
    hx = HeatExchanger(outlet=40.0, name="HX")
    hot = Gravity(fluid=light_water, disposition=1.0, name="HotLeg")
    cold = Gravity(fluid=light_water, disposition=-1.0, name="ColdLeg")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), hot),
        flow_edge((j_bot, j_top), pump, hx, cold),
        reference_node=(j_top, 2e5),
    )
    return fg, dict(j_top=j_top, j_bot=j_bot, pump=pump, hx=hx, hot=hot, cold=cold)


def test_hydraulic_guess_scalar_temperature_matches_uniform_mapping():
    fg, r = _gravity_loop()
    k = fg.kirchhoff
    mdots = {c: 0.5 for c in k.components}
    scalar = guess_hydraulic_steady_state(k, mdots, 40.0)
    mapping = guess_hydraulic_steady_state(k, mdots, {c: 40.0 for c in list(k.components) + [r["j_top"], r["j_bot"]]})
    for name in scalar:
        for var in scalar[name]:
            assert np.allclose(scalar[name][var], mapping[name][var])


def test_hydraulic_guess_mapping_gives_gravity_legs_their_own_density():
    fg, r = _gravity_loop()
    k = fg.kirchhoff
    mdots = {c: 0.5 for c in k.components}
    temps = {c: 40.0 for c in list(k.components) + [r["j_top"], r["j_bot"]]}
    temps[r["hot"]] = 90.0
    state = guess_hydraulic_steady_state(k, mdots, temps)
    assert state[r["hot"].name]["Tin"] == 90.0
    assert state[r["cold"].name]["Tin"] == 40.0
    assert abs(state[r["hot"].name]["pressure"]) < abs(state[r["cold"].name]["pressure"])


def test_hydraulic_guess_mapping_puts_a_profile_into_the_channel():
    f, c = MTR_fuel_and_channel(z_N=5, fuel_N=2, clad_N=2)
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=0.5, name="Pump")
    fg = FlowGraph(flow_edge((j_top, j_bot), c), flow_edge((j_bot, j_top), pump), reference_node=(j_top, 2e5))
    k = fg.kirchhoff
    profile = np.linspace(40.0, 60.0, c.n)
    state = guess_hydraulic_steady_state(k, {c: 0.5, pump: 0.5}, {c: profile, pump: 40.0, j_top: 40.0, j_bot: 60.0})
    assert np.allclose(state[c.name]["T_cool"], profile)
    assert state[j_bot.name]["Tin"] == 60.0


def test_hydraulic_guess_mapping_missing_a_calculation_names_it():
    fg, r = _gravity_loop()
    k = fg.kirchhoff
    mdots = {c: 0.5 for c in k.components}
    with pytest.raises(KeyError, match="ColdLeg"):
        guess_hydraulic_steady_state(k, mdots, {c: 40.0 for c in k.components if c is not r["cold"]})


def test_hydraulic_seed_stores_scalar_pressures_for_gravity():
    agr, fg, _, _, pump = _build(0.0, "constant")
    k = fg.kirchhoff
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state = seed_steady_state(agr, k, flows={pump: NC_MDOT})
        agr.load(state)
    gravities = [c for c in k.components if isinstance(c, Gravity)]
    assert gravities
    for g in gravities:
        assert np.ndim(state[g.name]["pressure"]) == 0
