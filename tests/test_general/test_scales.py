"""Tests for the nominal-scale registry (``stream.scales``).

The fixture builds a compact FlowGraph loop (two junctions, a pump and a
resistor closed by a Kirchhoff node) so every branch of ``scale_vector`` is
exercised: owned names routed by ``variables`` and the Kirchhoff family routed
by ``variables_by_type``.
"""

import logging

import numpy as np
import pytest

from stream.calculations import Junction, Pump, Resistor
from stream.composition import FlowGraph, flow_edge
from stream.scales import DEFAULT_SCALES, scale_vector
from stream.utilities import offset


@pytest.fixture
def system():
    """A minimal aggregator holding a Kirchhoff node plus ideal components."""
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


def _owned_indices(node, section, place) -> np.ndarray:
    """Global positions a local ``place`` occupies once shifted by the section."""
    return np.atleast_1d(np.arange(section.stop)[offset(place, section.start)])


def _expected(agr, registry) -> np.ndarray:
    """Independent oracle: rebuild the aligned scale vector from the sections."""
    expected = np.ones(len(agr))
    for node, section in agr.sections.items():
        places = getattr(node, "variables_by_type", None) or node.variables
        for name, place in places.items():
            expected[offset(place, section.start)] = registry.get(name, 1.0)
    return expected


def test_length_matches_state_vector(system):
    assert scale_vector(system).shape == (len(system),)


def test_full_coverage_partitions_the_state_vector(system):
    """Every node's section is entirely assigned; positions tile [0, N) once."""
    typ = scale_vector(system)
    owned: list[int] = []
    for node, section in system.sections.items():
        places = getattr(node, "variables_by_type", None) or node.variables
        node_idx = np.concatenate([_owned_indices(node, section, p) for p in places.values()])
        assert len(node_idx) == len(node)  # coverage per node == len(node)
        owned.extend(node_idx.tolist())
    assert sorted(owned) == list(range(len(system)))  # exact partition, no gaps/overlaps
    np.testing.assert_array_equal(typ, _expected(system, DEFAULT_SCALES))


def test_kirchhoff_owned_vars_routed_by_type(system):
    typ = scale_vector(system)
    (kirchhoff, section) = next(
        (n, s) for n, s in system.sections.items() if hasattr(n, "variables_by_type")
    )
    mdot_idx = _owned_indices(kirchhoff, section, kirchhoff.variables_by_type["mdot"])
    abs_idx = _owned_indices(kirchhoff, section, kirchhoff.variables_by_type["abs_pressure"])
    assert abs_idx.size > 0  # the loop actually carries absolute pressures
    assert np.all(typ[mdot_idx] == DEFAULT_SCALES["mdot"])       # == 1.0
    assert np.all(typ[abs_idx] == DEFAULT_SCALES["abs_pressure"])  # == 1e5


def test_override_wins_for_exactly_its_positions(system):
    base = scale_vector(system)
    typ = scale_vector(system, overrides={("pump", "pressure"): 42.0})
    place = next(
        offset(node.variables["pressure"], section.start)
        for node, section in system.sections.items()
        if getattr(node, "name", None) == "pump"
    )
    assert typ[place] == 42.0
    changed = np.where(typ != base)[0]
    assert changed.tolist() == [place]


def test_unknown_name_defaults_to_one_and_logs(system, caplog):
    """A variable absent from the registry falls back to 1.0 and is reported."""
    registry = {k: v for k, v in DEFAULT_SCALES.items() if k != "Tin"}
    with caplog.at_level(logging.INFO, logger="stream.scales"):
        typ = scale_vector(system, registry=registry)
    tin_idx = next(
        offset(node.variables["Tin"], section.start)
        for node, section in system.sections.items()
        if getattr(node, "name", None) == "A"
    )
    assert typ[tin_idx] == 1.0
    records = [r for r in caplog.records if r.name == "stream.scales"]
    assert records and any("Tin" in r.getMessage() for r in records)


def test_custom_registry_is_honored(system):
    registry = dict(DEFAULT_SCALES, Tin=7.0)
    typ = scale_vector(system, registry=registry)
    np.testing.assert_array_equal(typ, _expected(system, registry))
