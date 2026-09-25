"""Free-surface systems: a pool draining through a leg into the ambient."""

import numpy as np
import pytest

from stream.calculations import Environment, Junction, LevelHead, Orifice, Pump, Resistor, Tank
from stream.composition import FlowGraph, break_to_ambient, flow_edge, pool
from stream.composition.subsystems import loc_steady_state
from stream.errors import StreamConstructionError
from stream.substances import light_water
from stream.units import g
from stream.utilities import identity


def _draining_pool(make_leg, **tank_kwargs):
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0, name="pool", **tank_kwargs)
    env = Environment(name="ambient")
    mid = Junction(name="mid")
    head = LevelHead(light_water, 0.0, 4.0, name="head")
    leg = make_leg(tank)
    fg = FlowGraph(
        flow_edge((tank, mid), head),
        flow_edge((mid, env), leg, ref_mdot_for=(tank,)),
        surface_nodes={tank: None, env: None},
    )
    return tank, mid, head, leg, fg


def _hole(_tank) -> Resistor:
    return Resistor(1.0, name="hole")


def _thermally_uniform_state(agr, tank, mid) -> np.ndarray:
    y = np.zeros(len(agr))
    y[agr.var_index(tank, "T")] = 30.0
    y[agr.var_index(mid, "Tin")] = 30.0
    return y


def test_levelhead_receives_routed_level():
    tank, mid, head, _, fg = _draining_pool(_hole)
    agr = fg.aggregator
    y = _thermally_uniform_state(agr, tank, mid)

    at_empty = agr._op("calculate", y, 0.0, head).copy()
    y[agr.var_index(tank, "level")] = 3.0
    at_three = agr._op("calculate", y, 0.0, head)

    rho = light_water.density(30.0)
    assert at_three[1] - at_empty[1] == pytest.approx(-rho * g * 3.0)


def test_an_unrouted_levelhead_would_have_held_its_initial_level():
    """The routed level is the tank's, not the ``level0`` the component was built with."""
    tank, mid, head, _, fg = _draining_pool(_hole)
    agr = fg.aggregator
    y = _thermally_uniform_state(agr, tank, mid)
    y[agr.var_index(tank, "level")] = 3.0

    routed = agr._op("calculate", y, 0.0, head)[1]
    assert routed != pytest.approx(-head.dp_out(Tin=30.0))
    assert routed == pytest.approx(-head.dp_out(Tin=30.0, level=3.0))


def test_breaker_orifice_receives_routed_level():
    def breaker_for(tank) -> Orifice:
        return Orifice(light_water, 2e-3, 0.61, closes_below=(tank, 2.0), name="breaker")

    tank, mid, _, breaker, fg = _draining_pool(breaker_for)
    agr = fg.aggregator
    y = _thermally_uniform_state(agr, tank, mid)

    y[agr.var_index(tank, "level")] = 3.0
    assert agr._op("event_margin", y, 0.0, breaker)[0] == pytest.approx(1.0)
    y[agr.var_index(tank, "level")] = 1.5
    assert agr._op("event_margin", y, 0.0, breaker)[0] == pytest.approx(-0.5)


def test_starvation_watches_the_marked_edge_not_the_adjacent_one():
    tank, mid, head, hole, fg = _draining_pool(_hole, mdot_starve=0.1)
    agr, k = fg.aggregator, fg.kirchhoff
    y = _thermally_uniform_state(agr, tank, mid)
    y[agr.var_index(tank, "level")] = 3.0
    adjacent = agr.var_index(k, k.component_edge(head))
    marked = agr.var_index(k, k.component_edge(hole))

    y[adjacent], y[marked] = 5.0, 0.25
    assert agr._op("event_margin", y, 0.0, tank)[-1] == pytest.approx(0.25 - 0.1)

    y[adjacent], y[marked] = 0.05, 2.0
    assert agr._op("event_margin", y, 0.0, tank)[-1] == pytest.approx(2.0 - 0.1)


def test_a_closure_that_watches_a_levelless_node_is_rejected():
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0, name="pool")
    env = Environment(name="ambient")
    mid = Junction(name="mid")
    breaker = Orifice(light_water, 2e-3, 0.61, closes_below=(env, 2.0), name="breaker")
    with pytest.raises(StreamConstructionError, match="closes_below"):
        FlowGraph(
            flow_edge((tank, mid), LevelHead(light_water, 0.0, 4.0, name="head")),
            flow_edge((mid, env), breaker),
            surface_nodes={tank: None, env: None},
        )


def _circulating_pool(break_area=5e-4):
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0, name="pool")
    env = Environment(name="ambient")
    j_top, j_bot = Junction(name="j_top"), Junction(name="j_bot")
    pump = Pump(pressure=5e3, name="pump")
    core = Resistor(5e3, name="core")
    breach = Orifice(light_water, break_area, 0.61, dp_eps=1e-3, name="breach")
    fg = FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_bot: 0.0}),
        flow_edge((j_top, j_bot), pump, core),
        break_to_ambient(j_bot, breach, env),
        surface_nodes={tank: None, env: None},
        funcs={breach: dict(t=identity)},
    )
    return fg, tank, breach, pump, core


def test_pool_heads_are_signed_as_the_flow_solver_demands():
    fg, tank, _, _, _ = _circulating_pool()
    supply = fg.aggregator["head_pool_j_top"]
    ret = fg.aggregator["head_j_bot_pool"]
    assert (supply.sign, ret.sign) == (1.0, -1.0)


def test_a_hand_flipped_head_sign_is_rejected():
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0, name="pool")
    env = Environment(name="ambient")
    j = Junction(name="j")
    with pytest.raises(StreamConstructionError, match="sign"):
        FlowGraph(
            flow_edge((tank, j), LevelHead(light_water, 0.0, 4.0, sign=-1.0, name="head")),
            flow_edge((j, env), Resistor(1.0, name="hole")),
            surface_nodes={tank: None, env: None},
        )


def test_the_seeder_needs_no_entry_for_a_sealed_break():
    fg, tank, breach, pump, core = _circulating_pool()
    agr, k = fg.aggregator, fg.kirchhoff
    supply, ret = fg.aggregator["head_pool_j_top"], fg.aggregator["head_j_bot_pool"]

    guess = loc_steady_state(k, {supply: 1.0, ret: 1.0, pump: 1.0, core: 1.0}, 30.0)

    assert guess["pool"] == dict(level=4.0, T=30.0)
    assert guess["Kirchhoff"][k.component_edge(breach)] == 0.0
    agr.load(guess)


def test_the_seeded_intact_system_solves_to_a_standing_steady_state():
    fg, tank, breach, pump, core = _circulating_pool()
    agr, k = fg.aggregator, fg.kirchhoff
    supply, ret = fg.aggregator["head_pool_j_top"], fg.aggregator["head_j_bot_pool"]

    guess = loc_steady_state(k, {supply: 1.0, ret: 1.0, pump: 1.0, core: 1.0}, 30.0)
    vec = agr.solve_steady(guess)
    state = agr.save(vec)

    assert np.abs(agr.compute(vec, 0.0)).max() < 1e-6 * max(1.0, np.abs(vec).max())
    assert state["Kirchhoff"][k.component_edge(core)] == pytest.approx(1.0, rel=1e-9)
    assert state["Kirchhoff"][k.component_edge(breach)] == pytest.approx(0.0, abs=1e-12)
    assert state["pool"]["level"] == pytest.approx(4.0)
