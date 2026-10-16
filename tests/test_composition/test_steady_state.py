"""Steady-state guesses for a flow network wall-coupled to fuel plates."""

import warnings

import numpy as np
import pytest
from networkx import DiGraph

from stream.aggregator import CalculationGraph, vars_
from stream.calculations import (
    ChannelAndContacts,
    Flapper,
    Gravity,
    HeatExchanger,
    Junction,
    PointKinetics,
    Pump,
    Resistor,
)
from stream.composition import FlowGraph, chain_fuels_channels, flow_edge, symmetric_plate
from stream.composition.steady_state import (
    Blocks,
    MissingPowerError,
    _complete_flows,
    _fuel_powers,
    _known_flows,
    _side_powers,
    _walk_temperatures,
    decompose,
    guess_steady_state,
    seed_steady_state,
    solve_block,
)
from stream.composition.subsystems import MissingFlowError, symmetric_plate_seed, symmetric_plate_steady_state
from stream.pipe_geometry import EffectivePipe
from stream.solvers import AlgRuntimeError
from stream.state import State
from stream.substances import light_water
from stream.units import mm, pcm
from stream.utilities import identity

from .conftest import MTR_fuel_and_channel
from .test_natural_convection import NC_MDOT
from .test_natural_convection import _build as _nc_build

TIN, P_REF, POWER, MDOT = 40.0, 2e5, 1000.0, 0.5


def _pair(fuel_name: str, channel_name: str):
    fuel, channel = MTR_fuel_and_channel(z_N=10, fuel_N=4, clad_N=2)
    fuel.name, channel.name = fuel_name, channel_name
    return fuel, channel


def _forced_plate_loop(power=POWER, pump=None):
    fuel, channel = _pair("Fuel", "Channel")
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = pump or Pump(mdot0=MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), pump, hx),
        abs_pressure_comps=[channel],
        reference_node=(j_top, P_REF),
    )
    agr = fg.aggregator + symmetric_plate(channel, fuel, funcs={fuel: dict(power=power)}).to_aggregator()
    return agr, fg, dict(channel=channel, fuel=fuel, pump=pump, hx=hx, j_top=j_top, j_bot=j_bot)


def _chain_loop():
    fuel, c1 = _pair("Fuel", "C1")
    _, c2 = _pair("Unused", "C2")
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=2 * MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), c1),
        flow_edge((j_top, j_bot), c2),
        flow_edge((j_bot, j_top), pump, hx),
        abs_pressure_comps=[c1, c2],
        reference_node=(j_top, P_REF),
    )
    thermal = chain_fuels_channels([c1, c2], [fuel])
    thermal.funcs = {fuel: dict(power=POWER)}
    agr = fg.aggregator + thermal.to_aggregator()
    return agr, fg, dict(c1=c1, c2=c2, fuel=fuel, pump=pump, hx=hx)


def _resistor_loop(dp=1e4, r=2e4):
    pump = Pump(pressure=dp, name="Pump")
    res = Resistor(resistance=r, name="R")
    fg = FlowGraph(flow_edge(("A", "B"), pump), flow_edge(("B", "A"), res))
    return fg.aggregator, fg, dict(pump=pump, res=res)


def _parallel_core():
    f1, c1 = _pair("F1", "C1")
    f2, _ = _pair("F2", "Unused")
    c2 = ChannelAndContacts(
        z_boundaries=c1.bounds,
        fluid=light_water,
        pipe=EffectivePipe.rectangular(length=1, edge1=3 * mm, edge2=70 * mm, heated_edge=70 * mm),
        name="C2",
    )
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=2 * MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), c1),
        flow_edge((j_top, j_bot), c2),
        flow_edge((j_bot, j_top), pump, hx),
        abs_pressure_comps=[c1, c2],
        reference_node=(j_top, P_REF),
    )
    agr = (
        fg.aggregator
        + symmetric_plate(c1, f1, funcs={f1: dict(power=POWER)}).to_aggregator()
        + symmetric_plate(c2, f2, funcs={f2: dict(power=2 * POWER)}).to_aggregator()
    )
    return agr, fg, dict(c1=c1, c2=c2, f1=f1, f2=f2, pump=pump, hx=hx)


def _forced_and_passive_loops():
    fuel, channel = _pair("Fuel", "Channel")
    passive_fuel, passive_channel = _pair("PassiveFuel", "PassiveChannel")
    j_top, j_bot, j_side = Junction(name="J_top"), Junction(name="J_bot"), Junction(name="J_side")
    pump = Pump(mdot0=MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), pump, hx),
        flow_edge((j_top, j_side), passive_channel),
        flow_edge((j_side, j_top), Resistor(resistance=2e4, name="R")),
        abs_pressure_comps=[channel, passive_channel],
        reference_node=(j_top, P_REF),
    )
    agr = (
        fg.aggregator
        + symmetric_plate(channel, fuel, funcs={fuel: dict(power=POWER)}).to_aggregator()
        + symmetric_plate(passive_channel, passive_fuel, funcs={passive_fuel: dict(power=POWER)}).to_aggregator()
    )
    return agr, fg, dict(channel=channel, passive_channel=passive_channel, fuel=fuel, passive_fuel=passive_fuel, pump=pump)


def _chain_with_outer_plates():
    f0, c0 = _pair("F0", "C0")
    f1, c1 = _pair("F1", "C1")
    f2, _ = _pair("F2", "Unused")
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=2 * MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), c0),
        flow_edge((j_top, j_bot), c1),
        flow_edge((j_bot, j_top), pump, hx),
        abs_pressure_comps=[c0, c1],
        reference_node=(j_top, P_REF),
    )
    thermal = chain_fuels_channels([c0, c1], [f0, f1, f2])
    thermal.funcs = {f: dict(power=POWER) for f in (f0, f1, f2)}
    agr = fg.aggregator + thermal.to_aggregator()
    return agr, fg, dict(c0=c0, c1=c1, f0=f0, f1=f1, f2=f2, pump=pump, hx=hx)


def test_decompose_symmetric_plate_loop():
    agr, fg, r = _forced_plate_loop()
    blocks = decompose(agr, fg.kirchhoff)
    assert isinstance(blocks, Blocks)
    assert blocks.hydraulic == {fg.kirchhoff, r["channel"], r["pump"], r["hx"], r["j_top"], r["j_bot"]}
    assert blocks.thermal == (frozenset({r["channel"], r["fuel"]}),)
    assert blocks.kinetics is None
    assert blocks.unassigned == ()


def test_decompose_chain_puts_both_channels_in_one_cluster():
    agr, _, r = _chain_loop()
    blocks = decompose(agr)
    assert blocks.thermal == (frozenset({r["c1"], r["c2"], r["fuel"]}),)
    assert r["c1"] in blocks.hydraulic and r["c2"] in blocks.hydraulic


def test_decompose_parallel_core_finds_two_clusters():
    agr, _, r = _parallel_core()
    blocks = decompose(agr)
    assert set(blocks.thermal) == {frozenset({r["c1"], r["f1"]}), frozenset({r["c2"], r["f2"]})}


def test_decompose_loop_without_fuel_has_no_clusters():
    agr, _, _ = _resistor_loop()
    blocks = decompose(agr)
    assert blocks.thermal == ()
    assert blocks.hydraulic == set(agr.graph.nodes)


def _point_kinetics(name="PK"):
    return PointKinetics(
        generation_time=1e-5,
        delayed_neutron_fractions=np.full(6, 700 * pcm / 6),
        delayed_groups_decay_rates=np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]),
        name=name,
    )


def test_decompose_reports_kinetics_and_strays():
    agr, _, r = _forced_plate_loop()
    pk = _point_kinetics()
    stray = HeatExchanger(outlet=TIN, name="Stray")
    extra = CalculationGraph(DiGraph([(pk, r["fuel"], vars_("power"))]))
    extra.graph.add_node(stray)
    full = agr + extra.to_aggregator()
    blocks = decompose(full)
    assert blocks.kinetics == pk
    assert blocks.unassigned == (stray,)


def test_decompose_rejects_a_kirchhoff_not_in_the_graph():
    agr, _, _ = _forced_plate_loop()
    _, other, _ = _resistor_loop()
    with pytest.raises(ValueError):
        decompose(agr, other.kirchhoff)


def test_decompose_rejects_two_kirchhoffs():
    agr, _, _ = _forced_plate_loop()
    other = FlowGraph(
        flow_edge(("A", "B"), Pump(pressure=1e4, name="OtherPump")),
        flow_edge(("B", "A"), Resistor(resistance=2e4, name="OtherR")),
    )
    other.kirchhoff.name = "OtherKirchhoff"
    with pytest.raises(ValueError, match="OtherKirchhoff"):
        decompose(agr + other.aggregator)


def test_decompose_rejects_two_kinetics_nodes():
    agr, _, r = _parallel_core()
    pks = [_point_kinetics(name) for name in ("PK1", "PK2")]
    extra = CalculationGraph(DiGraph([(pks[0], r["f1"], vars_("power")), (pks[1], r["f2"], vars_("power"))]))
    with pytest.raises(ValueError, match="PK1"):
        decompose(agr + extra.to_aggregator())


def _merged_guess(fg, r, mdot=MDOT, power=POWER):
    return State.merge(
        fg.guess_steady_state({r["channel"]: mdot, r["pump"]: mdot}, TIN),
        symmetric_plate_steady_state(r["channel"], r["fuel"], mdot=mdot, p_abs=P_REF, power=power, Tin=TIN),
    )


def test_solve_block_of_the_plate_reproduces_the_plate_seeder():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    plate = symmetric_plate_steady_state(r["channel"], r["fuel"], mdot=MDOT, p_abs=P_REF, power=POWER, Tin=TIN)
    cold = State.merge(guess, {r["fuel"].name: dict(T=np.full(r["fuel"].shape, TIN), T_wall_left=np.full(r["fuel"].m, TIN), T_wall_right=np.full(r["fuel"].m, TIN))})
    block = solve_block(agr, [r["channel"], r["fuel"]], cold)
    assert set(block) == {r["channel"].name, r["fuel"].name}
    assert np.allclose(block[r["fuel"].name]["T_wall_left"], plate[r["fuel"].name]["T_wall_left"], atol=1e-3)
    assert np.allclose(block[r["channel"].name]["T_cool"], plate[r["channel"].name]["T_cool"], atol=1e-3)


def test_solve_block_of_the_hydraulics_freezes_the_walls():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    blocks = decompose(agr)
    block = solve_block(agr, blocks.hydraulic, guess)
    k = fg.kirchhoff
    assert k.name in block and r["fuel"].name not in block
    assert block[k.name][k.component_edge(r["channel"])] == pytest.approx(MDOT)
    assert block[r["j_bot"].name]["Tin"] > TIN


def test_solve_block_rejects_a_variable_split_across_the_cut():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    with pytest.raises(ValueError, match="inside and outside"):
        solve_block(agr, [fg.kirchhoff, r["channel"]], guess)


def test_solve_block_rejects_a_junction_on_the_boundary():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    with pytest.raises(ValueError, match="Junction"):
        solve_block(agr, [r["j_bot"]], guess)


def test_solve_block_fails_at_load_when_the_state_is_partial():
    agr, fg, r = _forced_plate_loop()
    guess = fg.guess_steady_state({r["channel"]: MDOT, r["pump"]: MDOT}, TIN)
    with pytest.raises(KeyError, match="Fuel"):
        solve_block(agr, [r["channel"], r["fuel"]], guess)


def test_solve_block_keeps_the_blocks_own_func_over_the_frozen_boundary_value():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    agr.funcs[r["channel"]] = dict(Tin=TIN + 20)
    block = solve_block(agr, [r["channel"]], guess)
    assert set(block) == {r["channel"].name}
    assert guess[r["channel"].name]["T_cool"].max() < TIN + 1
    assert block[r["channel"].name]["T_cool"].min() > TIN + 5


def test_known_flows_reads_the_pump_flow():
    agr, fg, r = _forced_plate_loop()
    k = fg.kirchhoff
    known = _known_flows(k, {}, 0.0)
    assert known == {k.component_edge(r["pump"]): MDOT}


def test_known_flows_user_value_and_pump_disagree():
    agr, fg, r = _forced_plate_loop()
    k = fg.kirchhoff
    with pytest.raises(ValueError, match="disagree"):
        _known_flows(k, {r["pump"]: 0.1}, 0.0)


def test_complete_flows_splits_identical_parallel_edges_equally():
    agr, fg, r = _chain_loop()
    k = fg.kirchhoff
    flows = _complete_flows(k, _known_flows(k, {}, 0.0))
    assert flows[k.component_edge(r["c1"])] == pytest.approx(MDOT)
    assert flows[k.component_edge(r["c2"])] == pytest.approx(MDOT)
    assert flows[k.component_edge(r["pump"])] == pytest.approx(2 * MDOT)


def test_complete_flows_gives_weighted_parallel_edges_the_same_per_element_flow():
    pump = Pump(mdot0=4.0, name="Pump")
    many, one = Resistor(resistance=1e4, name="Many"), Resistor(resistance=1e4, name="One")
    a, b = Junction(name="A"), Junction(name="B")
    fg = FlowGraph(
        flow_edge((a, b), many, signify=3.0),
        flow_edge((a, b), one),
        flow_edge((b, a), pump),
    )
    k = fg.kirchhoff
    flows = _complete_flows(k, _known_flows(k, {}, 0.0))
    assert flows[k.component_edge(many)] == pytest.approx(1.0)
    assert flows[k.component_edge(one)] == pytest.approx(1.0)
    x = np.array([flows[name] for name in list(k.variables)[: k.edges_count]])
    assert np.max(np.abs(k._kcl @ x)) < 1e-12


def test_complete_flows_keeps_a_closed_flapper_leg_at_zero():
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=MDOT, name="Pump")
    core = Resistor(resistance=2e4, name="Core")
    flapper = Flapper(open_at_current=0.0, f=1.0, fluid=light_water, area=1e-3, open_rate=1.0, name="Flapper")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), core),
        flow_edge((j_top, j_bot), flapper),
        flow_edge((j_bot, j_top), pump),
        funcs={flapper: dict(t=identity, ref_mdot=np.inf)},
    )
    k = fg.kirchhoff
    assert np.isposinf(flapper.t_open)
    known = _known_flows(k, {}, 0.0)
    assert known[k.component_edge(flapper)] == 0.0
    flows = _complete_flows(k, known)
    assert flows[k.component_edge(flapper)] == 0.0
    assert flows[k.component_edge(core)] == pytest.approx(MDOT)
    residual = flapper.calculate(np.array([TIN, 0.0]), mdot=flows[k.component_edge(flapper)], Tin=TIN, t=0.0)
    assert residual[1] == 0.0

    flapper.open(0.0)
    open_flows = _complete_flows(k, _known_flows(k, {}, 1.0))
    assert open_flows[k.component_edge(flapper)] == pytest.approx(MDOT / 2)


def test_seed_pump_free_heated_loop_asks_for_a_signed_flow():
    agr, _, r = _forced_plate_loop(pump=Pump(pressure=0.0, name="Pump"))
    with pytest.raises(MissingFlowError, match="Channel"):
        seed_steady_state(agr)


def test_seed_rejects_a_stagnant_heated_leg_completed_by_least_squares():
    agr, fg, r = _forced_and_passive_loops()
    k = fg.kirchhoff
    assert _complete_flows(k, _known_flows(k, {}, 0.0))[k.component_edge(r["passive_channel"])] == 0.0
    with pytest.raises(MissingFlowError, match="PassiveChannel"):
        seed_steady_state(agr, k)


def test_fuel_power_comes_from_the_func_and_powers_overrides_it():
    agr, _, r = _forced_plate_loop()
    blocks = decompose(agr)
    assert _fuel_powers(agr, blocks, {}, 0.0) == {r["fuel"]: POWER}
    assert _fuel_powers(agr, blocks, {r["fuel"]: 3.0}, 0.0) == {r["fuel"]: 3.0}


def test_fuel_power_is_read_at_time_t():
    agr, _, r = _forced_plate_loop(power=lambda t: POWER * (1.0 + t))
    blocks = decompose(agr)
    assert _fuel_powers(agr, blocks, {}, 1.0) == {r["fuel"]: 2 * POWER}


def _kinetics_loop():
    pk = _point_kinetics()
    fuel, channel = _pair("Fuel", "Channel")
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), Pump(mdot0=MDOT, name="Pump"), HeatExchanger(outlet=TIN, name="HX")),
        abs_pressure_comps=[channel],
        reference_node=(j_top, P_REF),
    )
    thermal = symmetric_plate(channel, fuel) + CalculationGraph(DiGraph([(pk, fuel, vars_("power"))]))
    return fg.aggregator + thermal.to_aggregator(), fg, dict(pk=pk, fuel=fuel, channel=channel)


def test_kinetics_driven_fuel_without_powers_is_an_error():
    full, _, r = _kinetics_loop()
    blocks = decompose(full)
    with pytest.raises(MissingPowerError, match="Fuel"):
        _fuel_powers(full, blocks, {}, 0.0)
    assert _fuel_powers(full, blocks, {r["fuel"]: POWER}, 0.0) == {r["fuel"]: POWER}


def test_side_powers_symmetric_plate_heats_both_sides_with_half_each():
    agr, _, r = _forced_plate_loop()
    blocks = decompose(agr)
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})[r["channel"]]
    assert np.allclose(sides.left, sides.right)
    assert np.sum(sides.total) == pytest.approx(POWER)


def test_side_powers_chain_splits_the_fuel_between_its_two_channels():
    agr, _, r = _chain_loop()
    blocks = decompose(agr)
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})
    assert np.sum(sides[r["c1"]].right) == pytest.approx(POWER / 2)
    assert np.all(sides[r["c1"]].left == 0.0)
    assert np.sum(sides[r["c2"]].left) == pytest.approx(POWER / 2)
    assert np.all(sides[r["c2"]].right == 0.0)


def test_walk_heated_channel_outlet_is_inlet_plus_power_over_flow_cp():
    agr, fg, r = _forced_plate_loop()
    k = fg.kirchhoff
    blocks = decompose(agr)
    flows = {k.component_edge(c): MDOT for c in k.components}
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})
    channel_powers = {c: s.total for c, s in sides.items()}
    outlets, profiles, nodes = _walk_temperatures(agr, k, flows, channel_powers, None, 0.0)
    cp = light_water.specific_heat(TIN)
    assert profiles[r["channel"]][0] > TIN
    assert outlets[r["channel"]] == pytest.approx(TIN + POWER / (MDOT * cp), rel=1e-6)
    assert nodes[r["j_bot"]] == pytest.approx(outlets[r["channel"]])
    assert outlets[r["hx"]] == TIN
    assert nodes[r["j_top"]] == pytest.approx(TIN)


def test_walk_honours_a_pinned_reverse_inlet():
    fuel, channel = _pair("Fuel", "Channel")
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(mdot0=-MDOT, name="Pump")
    hx = HeatExchanger(outlet=TIN, name="HX")
    grav = Gravity(fluid=light_water, disposition=-1.0, name="Grav")
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), pump, grav, hx),
        abs_pressure_comps=[channel],
        reference_node=(j_top, P_REF),
        funcs={grav: dict(Tin_minus=15.0)},
    )
    agr = fg.aggregator + symmetric_plate(channel, fuel, funcs={fuel: dict(power=POWER)}).to_aggregator()
    k = fg.kirchhoff
    flows = {k.component_edge(c): -MDOT for c in k.components}
    outlets, _, _ = _walk_temperatures(agr, k, flows, {channel: np.full(channel.n, POWER / channel.n)}, None, 0.0)
    assert outlets[grav] == 15.0


def test_walk_without_a_sink_needs_a_temperature():
    agr, fg, r = _resistor_loop()
    k = fg.kirchhoff
    flows = {k.component_edge(c): 0.5 for c in k.components}
    with pytest.raises(ValueError, match="temperature"):
        _walk_temperatures(agr, k, flows, {}, None, 0.0)
    with pytest.warns(UserWarning, match="sink"):
        outlets, _, _ = _walk_temperatures(agr, k, flows, {}, 33.0, 0.0)
    assert outlets[r["res"]] == 33.0


def test_seed_of_a_symmetric_plate_equals_the_plate_seed():
    agr, _, r = _forced_plate_loop()
    seed = seed_steady_state(agr)
    channel, fuel = r["channel"].name, r["fuel"].name
    plate = symmetric_plate_seed(r["channel"], r["fuel"], mdot=MDOT, p_abs=P_REF, power=POWER, Tin=TIN)
    assert np.allclose(seed[channel]["T_cool"], plate[channel]["T_cool"])
    assert np.allclose(seed[channel]["h_left"], plate[channel]["h_left"], rtol=1e-6)
    assert np.allclose(seed[fuel]["T_wall_left"], plate[fuel]["T_wall_left"], rtol=1e-6)
    assert np.allclose(np.asarray(seed[fuel]["T"]).ravel(), np.asarray(plate[fuel]["T"]).ravel(), rtol=1e-6)
    solved = symmetric_plate_steady_state(r["channel"], r["fuel"], mdot=MDOT, p_abs=P_REF, power=POWER, Tin=TIN)
    assert all(
        np.allclose(np.asarray(seed[name][var]).ravel(), np.asarray(solved[name][var]).ravel(), rtol=1e-3)
        for name, var in ((channel, "T_cool"), (fuel, "T_wall_left"), (fuel, "T"))
    )


def test_seed_loads_and_seeds_every_calculation():
    for build in (_forced_plate_loop, _chain_loop, _parallel_core):
        agr, _, _ = build()
        seed = seed_steady_state(agr)
        y = agr.load(seed)
        assert np.all(np.isfinite(y))


def test_seed_chain_gives_each_channel_side_its_own_wall():
    agr, _, r = _chain_loop()
    seed = seed_steady_state(agr)
    c1, fuel = r["c1"], r["fuel"]
    assert np.allclose(seed[fuel.name]["T_wall_left"], seed[fuel.name]["T_wall_right"])
    assert np.all(seed[fuel.name]["T_wall_left"] > seed[c1.name]["T_cool"])


def test_seed_gives_an_outer_plate_the_same_wall_on_both_sides():
    agr, _, r = _chain_with_outer_plates()
    seed = seed_steady_state(agr)
    outer, interior = seed[r["f0"].name], seed[r["f1"].name]
    assert np.allclose(outer["T_wall_left"], outer["T_wall_right"])
    assert np.all(outer["T_wall_left"] > seed[r["c0"].name]["T_cool"])
    assert np.all(outer["T_wall_left"] > interior["T_wall_left"])


def test_seed_isothermal_resistor_loop_is_a_root():
    dp, r_ = 1e4, 2e4
    agr, _, r = _resistor_loop(dp, r_)
    with pytest.warns(UserWarning, match="sink"):
        seed = seed_steady_state(agr, flows={r["pump"]: dp / r_}, temperature=25.0)
    assert np.allclose(agr.compute(agr.load(seed)), 0.0, atol=1e-9)


def test_seed_reports_unassigned_and_merges_overrides_last():
    agr, _, r = _forced_plate_loop()
    stray = HeatExchanger(outlet=TIN, name="Stray")
    extra = CalculationGraph(DiGraph())
    extra.graph.add_node(stray)
    full = agr + extra.to_aggregator()
    with pytest.warns(UserWarning, match="Stray"):
        seed = seed_steady_state(full)
    assert stray.name not in seed
    with pytest.raises(KeyError, match="Stray"):
        full.load(seed)
    seed = seed_steady_state(full, overrides={stray.name: dict(Tin=TIN, pressure=0.0), r["hx"].name: dict(Tin=99.0)})
    assert seed[r["hx"].name]["Tin"] == 99.0
    full.load(seed)


def test_seed_does_not_warn_about_a_node_the_overrides_cover():
    agr, _, r = _forced_plate_loop()
    stray = HeatExchanger(outlet=TIN, name="Stray")
    extra = CalculationGraph(DiGraph())
    extra.graph.add_node(stray)
    full = agr + extra.to_aggregator()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seed = seed_steady_state(full, overrides={stray.name: dict(Tin=TIN, pressure=0.0)})
    full.load(seed)


def test_seed_kinetics_from_powers():
    full, _, r = _kinetics_loop()
    seed = seed_steady_state(full, powers={r["fuel"]: POWER})
    assert seed[r["pk"].name]["power"] == POWER
    full.load(seed)


def _standalone_plate(channel_funcs=None):
    fuel, channel = _pair("Fuel", "Channel")
    if channel_funcs is None:
        channel_funcs = {"mdot": MDOT, "p_abs": P_REF, "Tin": TIN, "Tin_minus": TIN}
    agr = symmetric_plate(channel, fuel, {channel: channel_funcs, fuel: {"power": POWER}}).to_aggregator()
    return agr, {"channel": channel, "fuel": fuel}


def test_decompose_a_plate_without_a_kirchhoff_has_an_empty_hydraulic_block():
    agr, r = _standalone_plate()
    blocks = decompose(agr)
    assert blocks.hydraulic == frozenset()
    assert blocks.thermal == (frozenset({r["channel"], r["fuel"]}),)
    assert blocks.unassigned == ()


def test_seed_of_a_plate_without_a_kirchhoff_equals_the_plate_seed():
    agr, r = _standalone_plate()
    seed = seed_steady_state(agr)
    plate = symmetric_plate_seed(r["channel"], r["fuel"], mdot=MDOT, p_abs=P_REF, power=POWER, Tin=TIN)
    assert set(seed) == set(plate)
    for name, variables in plate.items():
        for var, value in variables.items():
            assert np.allclose(np.asarray(seed[name][var]).ravel(), np.asarray(value).ravel()), (name, var)


def test_guess_of_a_plate_without_a_kirchhoff_converges_without_globalization():
    agr, _ = _standalone_plate()
    y = _solve_without_globalization(agr, guess_steady_state(agr))
    assert np.linalg.norm(agr.compute(y)) < 1e-6


def test_seed_of_a_channel_without_a_flow_asks_for_one():
    agr, _ = _standalone_plate({"p_abs": P_REF, "Tin": TIN, "Tin_minus": TIN})
    with pytest.raises(MissingFlowError, match="mdot"):
        seed_steady_state(agr)


def test_seed_of_a_kinetics_driven_plate_without_a_kirchhoff():
    fuel, channel = _pair("Fuel", "Channel")
    pk = _point_kinetics()
    funcs = {channel: {"mdot": MDOT, "p_abs": P_REF, "Tin": TIN, "Tin_minus": TIN}}
    thermal = symmetric_plate(channel, fuel, funcs) + CalculationGraph(DiGraph([(pk, fuel, vars_("power"))]))
    seed = seed_steady_state(thermal.to_aggregator(), powers={fuel: POWER})
    assert seed[pk.name]["power"] == POWER


def _solve_without_globalization(agr, guess):
    return agr.solve_steady(guess, globalize=False)


def test_guess_with_zero_sweeps_returns_the_seed():
    agr, _, _ = _forced_plate_loop()
    seed = seed_steady_state(agr)
    guess = guess_steady_state(agr, sweeps=0)
    assert np.allclose(agr.load(seed), agr.load(guess))


def test_guess_forced_plate_loop_converges_without_globalization():
    agr, _, r = _forced_plate_loop()
    guess = guess_steady_state(agr)
    y = _solve_without_globalization(agr, guess)
    st = agr.save(y)
    cp = light_water.specific_heat(TIN)
    assert st[r["channel"].name]["T_cool"][-1] == pytest.approx(TIN + POWER / (MDOT * cp), rel=1e-2)
    assert np.linalg.norm(agr.compute(agr.load(guess))) < np.linalg.norm(agr.compute(agr.load(seed_steady_state(agr))))


def test_guess_refines_a_blocks_own_func_at_time_t():
    agr, _, r = _forced_plate_loop(power=lambda t: POWER * (1.0 + t))
    guess = guess_steady_state(agr, t=1.0)
    cp = light_water.specific_heat(TIN)
    rise = guess[r["channel"].name]["T_cool"][-1] - TIN
    assert rise == pytest.approx(2 * POWER / (MDOT * cp), rel=1e-2)


def test_guess_chain_loop_converges_without_globalization():
    agr, _, _ = _chain_loop()
    y = _solve_without_globalization(agr, guess_steady_state(agr))
    assert np.linalg.norm(agr.compute(y)) < 1e-6


@pytest.mark.slow
def test_guess_natural_convection_loop_converges_without_globalization():
    agr, fg, channel, fuel, pump = _nc_build(0.0, "regime_dependent")
    guess = guess_steady_state(agr, flows={channel: -0.01})
    y = _solve_without_globalization(agr, guess)
    K = fg.kirchhoff
    mdot = float(np.asarray(y)[agr.sections[K]][K.variables_by_type["mdot"]][0])
    assert mdot == pytest.approx(NC_MDOT, abs=1e-4)


def test_guess_warns_and_keeps_the_seed_when_a_block_fails(monkeypatch):
    import stream.composition.steady_state as module

    agr, fg, r = _forced_plate_loop()
    seed = seed_steady_state(agr)
    real = module.solve_block

    def failing(agr_, nodes, state, t=0.0, **options):
        if fg.kirchhoff in set(nodes):
            raise AlgRuntimeError("hydraulic block refused")
        return real(agr_, nodes, state, t=t, **options)

    monkeypatch.setattr(module, "solve_block", failing)
    with pytest.warns(UserWarning, match="hydraulic"):
        guess = guess_steady_state(agr)
    k = fg.kirchhoff
    assert guess[k.name][k.component_edge(r["channel"])] == seed[k.name][k.component_edge(r["channel"])]
    assert not np.allclose(guess[r["fuel"].name]["T"], seed[r["fuel"].name]["T"])
    agr.load(guess)


def test_guess_parallel_core_needs_only_the_pump_flow():
    agr, fg, r = _parallel_core()
    k = fg.kirchhoff
    seed = seed_steady_state(agr)
    assert seed[k.name][k.component_edge(r["c1"])] == pytest.approx(seed[k.name][k.component_edge(r["c2"])])
    guess = guess_steady_state(agr, k)
    st = agr.save(_solve_without_globalization(agr, guess))
    assert st[k.name][k.component_edge(r["c2"])] > st[k.name][k.component_edge(r["c1"])]
    assert st[k.name][k.component_edge(r["c1"])] + st[k.name][k.component_edge(r["c2"])] == pytest.approx(2 * MDOT)
