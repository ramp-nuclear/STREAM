"""Steady-state guesses for a flow network wall-coupled to fuel plates."""

import numpy as np
import pytest
from networkx import DiGraph

from stream.aggregator import CalculationGraph, vars_
from stream.calculations import (
    ChannelAndContacts,
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
    decompose,
    guess_steady_state,
    seed_steady_state,
    solve_block,
)
from stream.composition.subsystems import MissingFlowError, symmetric_plate_steady_state
from stream.errors import StreamConstructionError
from stream.pipe_geometry import EffectivePipe
from stream.solvers import AlgRuntimeError
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
    with pytest.warns(UserWarning):
        full = agr + extra.to_aggregator()
    blocks = decompose(full, fg.kirchhoff)
    assert blocks.kinetics == (pk,)
    assert blocks.unassigned == (stray,)


def test_decompose_rejects_a_kirchhoff_not_in_the_graph():
    agr, _, _ = _forced_plate_loop()
    _, other, _ = _resistor_loop()
    with pytest.raises(StreamConstructionError):
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
    with pytest.raises(StreamConstructionError, match="inside and outside"):
        solve_block(agr, [fg.kirchhoff, r["channel"]], guess)


def test_solve_block_rejects_a_junction_on_the_boundary():
    agr, fg, r = _forced_plate_loop()
    guess = _merged_guess(fg, r)
    with pytest.raises(StreamConstructionError, match="Junction"):
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


import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

from stream.composition.steady_state import _complete_flows, _known_flows  # noqa: E402

_CASE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "lofa" / "case.py"


def _load_case():
    spec = importlib.util.spec_from_file_location("lofa_benchmark_case", _CASE_PATH)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


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


def test_complete_flows_keeps_a_closed_flapper_leg_at_zero():
    case = _load_case()
    agr, k, refs = case.build()
    known = _known_flows(k, {refs["pump"]: case.MDOT0}, 0.0)
    assert known[k.component_edge(refs["flapper"])] == 0.0
    flows = _complete_flows(k, known)
    assert flows[k.component_edge(refs["flapper"])] == 0.0
    assert flows[k.component_edge(refs["channel"])] == pytest.approx(case.MDOT0)


def test_seed_pump_free_heated_loop_asks_for_a_signed_flow():
    agr, fg, r = _forced_plate_loop(pump=Pump(pressure=0.0, name="Pump"))
    with pytest.raises(MissingFlowError, match="Channel"):
        seed_steady_state(agr, fg.kirchhoff)


def test_seed_rejects_a_stagnant_heated_leg_completed_by_least_squares():
    agr, fg, r = _forced_and_passive_loops()
    k = fg.kirchhoff
    assert _complete_flows(k, _known_flows(k, {}, 0.0))[k.component_edge(r["passive_channel"])] == 0.0
    with pytest.raises(MissingFlowError, match="PassiveChannel"):
        seed_steady_state(agr, k)


from stream.composition.steady_state import _fuel_powers, _side_powers, _walk_temperatures  # noqa: E402


def test_fuel_power_comes_from_the_func_and_powers_overrides_it():
    agr, fg, r = _forced_plate_loop()
    blocks = decompose(agr, fg.kirchhoff)
    assert _fuel_powers(agr, blocks, {}, 0.0) == {r["fuel"]: POWER}
    assert _fuel_powers(agr, blocks, {r["fuel"]: 3.0}, 0.0) == {r["fuel"]: 3.0}


def test_fuel_power_is_read_at_time_t():
    agr, fg, r = _forced_plate_loop(power=lambda t: POWER * (1.0 + t))
    blocks = decompose(agr, fg.kirchhoff)
    assert _fuel_powers(agr, blocks, {}, 1.0) == {r["fuel"]: 2 * POWER}


def _kinetics_loop():
    pk = PointKinetics(
        generation_time=1e-5,
        delayed_neutron_fractions=np.full(6, 700 * pcm / 6),
        delayed_groups_decay_rates=np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]),
        name="PK",
    )
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
    full, fg, r = _kinetics_loop()
    blocks = decompose(full, fg.kirchhoff)
    with pytest.raises(MissingPowerError, match="Fuel"):
        _fuel_powers(full, blocks, {}, 0.0)
    assert _fuel_powers(full, blocks, {r["fuel"]: POWER}, 0.0) == {r["fuel"]: POWER}


def test_side_powers_symmetric_plate_heats_both_sides_with_half_each():
    agr, fg, r = _forced_plate_loop()
    blocks = decompose(agr, fg.kirchhoff)
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})[r["channel"]]
    assert np.allclose(sides["left"], sides["right"])
    assert np.sum(sides["left"] + sides["right"]) == pytest.approx(POWER)


def test_side_powers_chain_splits_the_fuel_between_its_two_channels():
    agr, fg, r = _chain_loop()
    blocks = decompose(agr, fg.kirchhoff)
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})
    assert np.sum(sides[r["c1"]]["right"]) == pytest.approx(POWER / 2)
    assert np.all(sides[r["c1"]]["left"] == 0.0)
    assert np.sum(sides[r["c2"]]["left"]) == pytest.approx(POWER / 2)
    assert np.all(sides[r["c2"]]["right"] == 0.0)


def test_walk_heated_channel_outlet_is_inlet_plus_power_over_flow_cp():
    agr, fg, r = _forced_plate_loop()
    k = fg.kirchhoff
    blocks = decompose(agr, k)
    flows = {k.component_edge(c): MDOT for c in k.components}
    sides = _side_powers(agr, blocks, {r["fuel"]: POWER})
    channel_powers = {c: s["left"] + s["right"] for c, s in sides.items()}
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
    with pytest.warns(UserWarning, match="shadow"):
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


def _plate_seed(monkeypatch, channel, fuel, mdot, power):
    from stream.aggregator import Aggregator

    monkeypatch.setattr(Aggregator, "solve_steady", lambda self, y0, **_: np.asarray(y0, dtype=float))
    return symmetric_plate_steady_state(channel, fuel, mdot=mdot, p_abs=P_REF, power=power, Tin=TIN)


def test_seed_symmetric_plate_matches_the_plate_seeder(monkeypatch):
    agr, fg, r = _forced_plate_loop()
    seed = seed_steady_state(agr, fg.kirchhoff)
    plate = _plate_seed(monkeypatch, r["channel"], r["fuel"], MDOT, POWER)
    assert np.allclose(seed[r["channel"].name]["T_cool"], plate[r["channel"].name]["T_cool"])
    assert np.allclose(seed[r["channel"].name]["h_left"], plate[r["channel"].name]["h_left"], rtol=1e-6)
    assert np.allclose(seed[r["fuel"].name]["T_wall_left"], plate[r["fuel"].name]["T_wall_left"], rtol=1e-6)
    assert np.allclose(np.asarray(seed[r["fuel"].name]["T"]).ravel(), np.asarray(plate[r["fuel"].name]["T"]).ravel(), rtol=1e-6)


def test_seed_loads_and_seeds_every_calculation():
    for build in (_forced_plate_loop, _chain_loop, _parallel_core):
        agr, fg, _ = build()
        seed = seed_steady_state(agr, fg.kirchhoff)
        y = agr.load(seed)
        assert np.all(np.isfinite(y))


def test_seed_chain_gives_each_channel_side_its_own_wall():
    agr, fg, r = _chain_loop()
    seed = seed_steady_state(agr, fg.kirchhoff)
    c1, fuel = r["c1"], r["fuel"]
    assert np.allclose(seed[fuel.name]["T_wall_left"], seed[fuel.name]["T_wall_right"])
    assert np.all(seed[fuel.name]["T_wall_left"] > seed[c1.name]["T_cool"])


def test_seed_gives_an_outer_plate_the_same_wall_on_both_sides():
    agr, fg, r = _chain_with_outer_plates()
    seed = seed_steady_state(agr, fg.kirchhoff)
    outer, interior = seed[r["f0"].name], seed[r["f1"].name]
    assert np.allclose(outer["T_wall_left"], outer["T_wall_right"])
    assert np.all(outer["T_wall_left"] > seed[r["c0"].name]["T_cool"])
    assert np.all(outer["T_wall_left"] > interior["T_wall_left"])


def test_seed_isothermal_resistor_loop_is_a_root():
    dp, r_ = 1e4, 2e4
    agr, fg, r = _resistor_loop(dp, r_)
    with pytest.warns(UserWarning, match="sink"):
        seed = seed_steady_state(agr, fg.kirchhoff, flows={r["pump"]: dp / r_}, temperature=25.0)
    assert np.allclose(agr.compute(agr.load(seed)), 0.0, atol=1e-9)


def test_seed_reports_unassigned_and_merges_overrides_last():
    agr, fg, r = _forced_plate_loop()
    stray = HeatExchanger(outlet=TIN, name="Stray")
    extra = CalculationGraph(DiGraph())
    extra.graph.add_node(stray)
    full = agr + extra.to_aggregator()
    with pytest.warns(UserWarning, match="Stray"):
        seed = seed_steady_state(full, fg.kirchhoff)
    assert stray.name not in seed
    with pytest.raises(KeyError, match="Stray"):
        full.load(seed)
    with pytest.warns(UserWarning, match="Stray"):
        seed = seed_steady_state(full, fg.kirchhoff, overrides={stray.name: dict(Tin=TIN, pressure=0.0), r["hx"].name: dict(Tin=99.0)})
    assert seed[r["hx"].name]["Tin"] == 99.0
    full.load(seed)


def test_seed_kinetics_from_powers():
    full, fg, r = _kinetics_loop()
    seed = seed_steady_state(full, fg.kirchhoff, powers={r["fuel"]: POWER})
    assert seed[r["pk"].name]["power"] == POWER
    full.load(seed)


from .test_natural_convection import NC_MDOT, _build as _nc_build  # noqa: E402


def _first_rung(agr, guess):
    return agr.solve_steady(guess, globalize=False)


def test_guess_refine_false_returns_the_seed():
    agr, fg, _ = _forced_plate_loop()
    seed = seed_steady_state(agr, fg.kirchhoff)
    guess = guess_steady_state(agr, fg.kirchhoff, refine=False)
    assert np.allclose(agr.load(seed), agr.load(guess))


def test_guess_forced_plate_loop_converges_on_the_first_rung():
    agr, fg, r = _forced_plate_loop()
    guess = guess_steady_state(agr, fg.kirchhoff)
    y = _first_rung(agr, guess)
    st = agr.save(y)
    cp = light_water.specific_heat(TIN)
    assert st[r["channel"].name]["T_cool"][-1] == pytest.approx(TIN + POWER / (MDOT * cp), rel=1e-2)
    assert np.linalg.norm(agr.compute(agr.load(guess))) < np.linalg.norm(agr.compute(agr.load(seed_steady_state(agr, fg.kirchhoff))))


def test_guess_refines_a_blocks_own_func_at_time_t():
    agr, fg, r = _forced_plate_loop(power=lambda t: POWER * (1.0 + t))
    guess = guess_steady_state(agr, fg.kirchhoff, t=1.0)
    cp = light_water.specific_heat(TIN)
    rise = guess[r["channel"].name]["T_cool"][-1] - TIN
    assert rise == pytest.approx(2 * POWER / (MDOT * cp), rel=1e-2)


def test_guess_chain_loop_converges_on_the_first_rung():
    agr, fg, _ = _chain_loop()
    y = _first_rung(agr, guess_steady_state(agr, fg.kirchhoff))
    assert np.linalg.norm(agr.compute(y)) < 1e-6


@pytest.mark.slow
def test_guess_natural_convection_loop_converges_on_the_first_rung():
    agr, fg, channel, fuel, pump = _nc_build(0.0, "regime_dependent")
    guess = guess_steady_state(agr, fg.kirchhoff, flows={channel: -0.01})
    y = _first_rung(agr, guess)
    K = fg.kirchhoff
    mdot = float(np.asarray(y)[agr.sections[K]][K.variables_by_type["mdot"]][0])
    assert mdot == pytest.approx(NC_MDOT, abs=1e-4)


def test_guess_warns_and_keeps_the_seed_when_a_block_fails(monkeypatch):
    import stream.composition.steady_state as module

    agr, fg, r = _forced_plate_loop()
    seed = seed_steady_state(agr, fg.kirchhoff)
    real = module.solve_block

    def failing(agr_, nodes, state, t=0.0, **options):
        if fg.kirchhoff in set(nodes):
            raise AlgRuntimeError("hydraulic block refused")
        return real(agr_, nodes, state, t=t, **options)

    monkeypatch.setattr(module, "solve_block", failing)
    with pytest.warns(UserWarning, match="hydraulic"):
        guess = guess_steady_state(agr, fg.kirchhoff)
    k = fg.kirchhoff
    assert guess[k.name][k.component_edge(r["channel"])] == seed[k.name][k.component_edge(r["channel"])]
    assert not np.allclose(guess[r["fuel"].name]["T"], seed[r["fuel"].name]["T"])
    agr.load(guess)


def test_guess_rejects_zero_sweeps():
    agr, fg, _ = _forced_plate_loop()
    with pytest.raises(ValueError):
        guess_steady_state(agr, fg.kirchhoff, sweeps=0)


def test_guess_parallel_core_needs_only_the_pump_flow():
    agr, fg, r = _parallel_core()
    k = fg.kirchhoff
    seed = seed_steady_state(agr, k)
    assert seed[k.name][k.component_edge(r["c1"])] == pytest.approx(seed[k.name][k.component_edge(r["c2"])])
    guess = guess_steady_state(agr, k)
    st = agr.save(_first_rung(agr, guess))
    assert st[k.name][k.component_edge(r["c2"])] > st[k.name][k.component_edge(r["c1"])]
    assert st[k.name][k.component_edge(r["c1"])] + st[k.name][k.component_edge(r["c2"])] == pytest.approx(2 * MDOT)
