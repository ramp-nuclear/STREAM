"""Tests for ``scaled_step`` and ``Aggregator.scaled_atol``.

The fixture is a compact FlowGraph loop (two junctions, a pump and a resistor
closed by a Kirchhoff node).
"""

import numpy as np
import pytest

from stream.calculations import Junction, Pump, Resistor
from stream.composition import FlowGraph, flow_edge
from stream.jacobians import scaled_step
from stream.scales import DEFAULT_SCALES, scale_vector


@pytest.fixture
def system():
    """A minimal aggregator (Kirchhoff node + ideal comps)."""
    a, b = Junction("A"), Junction("B")
    pump = Pump(pressure=5.0, name="pump")
    res = Resistor(1.0, name="res")
    fg = FlowGraph(
        flow_edge((a, b), pump),
        flow_edge((b, a), res),
        abs_pressure_comps=[pump, res],
        reference_node=(a, 2e5),
    )
    return fg.aggregator


def test_scaled_step_uses_the_larger_of_value_and_nominal_scale():
    """The step is sqrt(eps) * max(|y|, typ), whether or not ydot is passed."""
    typ = np.array([1.0, 1e5, 1e-3])
    y = np.array([10.0, 0.0, -2.0])
    step = scaled_step(typ)
    expected = np.sqrt(np.finfo(float).eps) * np.array([10.0, 1e5, 2.0])
    np.testing.assert_array_equal(step(y), expected)
    np.testing.assert_array_equal(step(y, np.zeros(3)), expected)


def test_scaled_atol_forms(system):
    """scaled_atol returns rel * typ for the None/dict/array scales forms."""
    agr = system
    n = len(agr)

    np.testing.assert_array_equal(agr.scaled_atol(), 1e-6 * scale_vector(agr))
    np.testing.assert_array_equal(agr.scaled_atol(rel=1e-3), 1e-3 * scale_vector(agr))

    registry = dict(DEFAULT_SCALES, Tin=7.0)
    np.testing.assert_array_equal(
        agr.scaled_atol(rel=1e-4, scales=registry),
        1e-4 * scale_vector(agr, registry=registry),
    )

    typ = np.arange(1.0, n + 1.0)
    np.testing.assert_array_equal(agr.scaled_atol(rel=2.0, scales=typ), 2.0 * typ)


def test_scaled_atol_array_length_validated(system):
    """A wrong-length scales array is rejected."""
    agr = system
    with pytest.raises(ValueError):
        agr.scaled_atol(scales=np.ones(len(agr) + 1))

