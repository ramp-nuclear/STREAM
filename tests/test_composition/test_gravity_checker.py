from functools import partial

import pytest

from stream.calculations import Gravity, Junction, Pump, Resistor
from stream.composition import FlowGraph, flow_edge
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
