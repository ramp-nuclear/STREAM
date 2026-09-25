from functools import partial

import pytest

from stream.calculations import Gravity, Junction, LevelHead, Pump, Resistor, Tank
from stream.composition import FlowGraph, flow_edge, pool
from stream.composition.subsystems import GravityMismatchError
from stream.substances import light_water
from stream.units import standard_acceleration as g

G = partial(Gravity, light_water)


def test_simplest_case_where_gravity_is_quite_alright():
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), G(1.0 + 1e-10, name="g1")),
        flow_edge((b, a), G(-1.0, name="g2")),
    )
    fg.check_gravity_mismatch()


def test_simplest_case_where_gravity_is_not_alright():
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), G(3.0, name="g1")),
        flow_edge((a, b), G(3.0, name="g3")),
        flow_edge((b, a), G(-1.0, name="g2")),
    )
    dp = 2.0
    with pytest.raises(GravityMismatchError, match=f"{dp}"):
        fg.check_gravity_mismatch(head=light_water.density(10) * g)


def test_gravity_checker_works_with_pump():
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), G(1.0, name="g1")),
        flow_edge((b, a), G(-1.0, name="g2"), Pump(pressure=5)),
    )
    fg.check_gravity_mismatch()


def test_gravity_checker_without_gravity():
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), Resistor(1.0, name="r1")),
        flow_edge((b, a), Resistor(1.0, name="r2"), Pump(pressure=5)),
    )
    fg.check_gravity_mismatch()


def test_level_head_reversed_flow_temperature_source_is_reported():
    a, b = Junction("A"), Junction("B")
    fg = FlowGraph(
        flow_edge((a, b), LevelHead(light_water, 0.0, 1.0, name="pool_head")),
        flow_edge((b, a), G(-1.0, name="g2"), Pump()),
    )
    with pytest.warns(UserWarning, match=r"LevelHead 'pool_head'.*'B'"):
        fg.check_gravity_mismatch()


def _pool_loop(return_elevation):
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0, name="pool")
    j_top, j_bot = Junction("j_top"), Junction("j_bot")
    return FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_bot: return_elevation}),
        flow_edge((j_top, j_bot), Resistor(1.0, name="loop")),
        surface_nodes={tank: None},
    )


def test_a_pool_feeding_and_receiving_at_one_elevation_closes_its_loop():
    _pool_loop(0.0).check_gravity_mismatch()


def test_a_pool_return_connected_at_the_wrong_elevation_is_caught():
    with pytest.raises(GravityMismatchError):
        _pool_loop(1.0).check_gravity_mismatch(head=light_water.density(10) * g)
