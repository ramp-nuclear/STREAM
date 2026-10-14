"""Steady-state guesses for a flow network wall-coupled to fuel plates."""

import numpy as np
import pytest
from networkx import DiGraph

from stream.aggregator import CalculationGraph, vars_
from stream.calculations import (
    ChannelAndContacts,
    HeatExchanger,
    Junction,
    PointKinetics,
    Pump,
    Resistor,
)
from stream.composition import FlowGraph, chain_fuels_channels, flow_edge, symmetric_plate
from stream.composition.steady_state import (
    Blocks,
    decompose,
    solve_block,
)
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.units import mm, pcm

from .conftest import MTR_fuel_and_channel

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
    assert blocks.kinetics == ()
    assert blocks.unassigned == ()


def test_decompose_chain_puts_both_channels_in_one_cluster():
    agr, fg, r = _chain_loop()
    blocks = decompose(agr, fg.kirchhoff)
    assert blocks.thermal == (frozenset({r["c1"], r["c2"], r["fuel"]}),)
    assert r["c1"] in blocks.hydraulic and r["c2"] in blocks.hydraulic


def test_decompose_parallel_core_finds_two_clusters():
    agr, fg, r = _parallel_core()
    blocks = decompose(agr, fg.kirchhoff)
    assert set(blocks.thermal) == {frozenset({r["c1"], r["f1"]}), frozenset({r["c2"], r["f2"]})}


def test_decompose_loop_without_fuel_has_no_clusters():
    agr, fg, _ = _resistor_loop()
    blocks = decompose(agr, fg.kirchhoff)
    assert blocks.thermal == ()
    assert blocks.hydraulic == set(agr.graph.nodes)


def test_decompose_reports_kinetics_and_strays():
    agr, fg, r = _forced_plate_loop()
    pk = PointKinetics(
        generation_time=1e-5,
        delayed_neutron_fractions=np.full(6, 700 * pcm / 6),
        delayed_groups_decay_rates=np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]),
        name="PK",
    )
    stray = HeatExchanger(outlet=TIN, name="Stray")
    extra = CalculationGraph(DiGraph([(pk, r["fuel"], vars_("power"))]))
    extra.graph.add_node(stray)
    full = agr + extra.to_aggregator()
    blocks = decompose(full, fg.kirchhoff)
    assert blocks.kinetics == (pk,)
    assert blocks.unassigned == (stray,)


def test_decompose_rejects_a_kirchhoff_not_in_the_graph():
    agr, _, _ = _forced_plate_loop()
    _, other, _ = _resistor_loop()
    with pytest.raises(ValueError):
        decompose(agr, other.kirchhoff)


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
    blocks = decompose(agr, fg.kirchhoff)
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
