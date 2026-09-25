"""Composition-layer validation, each with firing and non-firing sides:

  - Kirchhoff rejects duplicate ``str(node)`` junction names.
  - kirchhoffify wires all components on ``None``, none on ``[]``, exactly the listed ones on a list.
  - flow_graph_to_aggregator dissolves adjacent SISO string junctions, leaving no phantom node.
  - check_gravity_mismatch warns naming the gravity on a single-HX natural-convection loop,
    stays silent on a two-HX sandwich and on a forced-flow pump, and raises TypeError on a
    non-mapping strategy.
  - symmetric_plate_steady_state raises on mdot=0.
  - EffectivePipe / ResistorFromKnownPoint raise (not assert) their preconditions.
  - ChannelHeatFlux.save defaults q_left/q_right.
  - State.uniform copies values without aliasing.
  - MissingFlowError names the components, not only the edge string.
"""

import warnings

import numpy as np
import pytest
from networkx import MultiDiGraph

from stream.calculations import (
    Gravity,
    HeatExchanger,
    Junction,
    Kirchhoff,
    Pump,
    Resistor,
)
from stream.calculations.channel import ChannelHeatFlux
from stream.composition import Calculation_factory, FlowGraph, check_gravity_mismatch, flow_edge
from stream.composition.constructors import ResistorFromKnownPoint
from stream.composition.cycle import flow_graph_to_aggregator, kirchhoffify
from stream.composition.subsystems import (
    GravityMismatchError,
    MissingFlowError,
    guess_hydraulic_steady_state,
    symmetric_plate_steady_state,
)
from stream.errors import StreamConstructionError, StreamError
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water

from .conftest import MTR_fuel_and_channel


def test_kirchhoff_rejects_duplicate_junction_names():
    a1, a2, b = Junction(name="A"), Junction(name="A"), Junction(name="B")
    g = MultiDiGraph(
        [
            flow_edge((a1, b), Pump(pressure=1.0, name="P1")),
            flow_edge((b, a1), Resistor(1.0, name="R1")),
            flow_edge((a2, b), Pump(pressure=2.0, name="P2")),
            flow_edge((b, a2), Resistor(2.0, name="R2")),
        ]
    )
    with pytest.raises(StreamConstructionError, match="stringify uniquely"):
        Kirchhoff(g)


def test_kirchhoff_accepts_unique_junction_names():
    a, b = Junction(name="A"), Junction(name="B")
    k = Kirchhoff(MultiDiGraph([flow_edge((a, b), Pump(pressure=1.0, name="P")), flow_edge((b, a), Resistor(1.0, name="R"))]))
    assert k.edges_count == len(k.variables)  # no silent collapse


def _kirchhoffify_wired(hydraulic_comps):
    j0, j1 = Junction(name="A"), Junction(name="B")
    p, r = Pump(pressure=1.0, name="P"), Resistor(1.0, name="R")
    fg = MultiDiGraph([flow_edge((j0, j1), p), flow_edge((j1, j0), r)])
    k = Kirchhoff(fg)
    a = kirchhoffify(flow_graph_to_aggregator(fg), k, hydraulic_comps=hydraulic_comps)
    return {str(v) for (u, v) in a.graph.edges() if u is k}, p, r


def test_kirchhoffify_none_wires_all_components():
    wired, p, r = _kirchhoffify_wired(None)
    assert {"P", "R"} <= wired


def test_kirchhoffify_empty_list_wires_no_components():
    wired, p, r = _kirchhoffify_wired([])
    assert "P" not in wired and "R" not in wired  # junctions may still be wired, components are not


def test_kirchhoffify_explicit_list_wires_exactly_those():
    j0, j1 = Junction(name="A"), Junction(name="B")
    p, r = Pump(pressure=1.0, name="P"), Resistor(1.0, name="R")
    fg = MultiDiGraph([flow_edge((j0, j1), p), flow_edge((j1, j0), r)])
    k = Kirchhoff(fg)
    a = kirchhoffify(flow_graph_to_aggregator(fg), k, hydraulic_comps=[p])
    wired = {str(v) for (u, v) in a.graph.edges() if u is k}
    assert "P" in wired and "R" not in wired


def test_comps_free_string_junctions_leave_no_phantom():
    jA = Junction(name="A")
    p, r = Pump(pressure=1.0, name="P"), Resistor(1.0, name="R")
    # A -P-> "x", "x" --(no comps)--> "y", "y" -R-> A : x, y are adjacent SISO string junctions.
    g = MultiDiGraph([flow_edge((jA, "x"), p), flow_edge(("x", "y")), flow_edge(("y", jA), r)])
    agr = flow_graph_to_aggregator(g)
    from stream.calculation import Calculation

    assert all(isinstance(n, Calculation) for n in agr.graph.nodes)


def test_single_siso_junction_still_dissolves():
    jA, jB = Junction(name="A"), Junction(name="B")
    p, r = Pump(pressure=1.0, name="P"), Resistor(1.0, name="R")
    g = MultiDiGraph([flow_edge((jA, "mid"), p), flow_edge(("mid", jB), r), flow_edge((jB, jA), Resistor(2.0, name="R2"))])
    agr = flow_graph_to_aggregator(g)
    assert "mid" not in {str(n) for n in agr.graph.nodes}


def _nc_gravity_loop(second_hx: bool, pump_pressure: float = 0.0):
    """A Gravity on a pump-driven leg with one (broken) or two (sandwich)
    HeatExchangers around it; a Resistor closes the loop."""
    jt, jb = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=pump_pressure, name="Pump")
    grav = Gravity(fluid=light_water, disposition=-0.7, name="Grav")
    hx1 = HeatExchanger(outlet=40.0, name="HX1")
    comps = (pump, hx1, grav, HeatExchanger(outlet=40.0, name="HX2")) if second_hx else (pump, hx1, grav)
    return FlowGraph(flow_edge((jt, jb), Resistor(1.0, name="R")), flow_edge((jb, jt), *comps))


def _gravity_warnings(fg):
    """Gravity-naming warnings from check_gravity_mismatch. The static zero-flow check may raise
    GravityMismatchError first (a single unbalanced gravity leg); the topology warning fires
    before it, which is what this collects."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_gravity_mismatch(fg.kirchhoff)
        except GravityMismatchError:
            pass
    return [str(w.message) for w in caught if "Grav" in str(w.message)]


def test_warns_naming_gravity_on_single_hx_nc_loop():
    warns = _gravity_warnings(_nc_gravity_loop(second_hx=False))
    assert any("reversed-flow density temperature from 'J_top'" in m for m in warns), warns


def test_silent_on_two_hx_sandwich():
    assert not _gravity_warnings(_nc_gravity_loop(second_hx=True))


def test_silent_when_pump_is_forced():
    # A forced pump (nonzero head) is not NC-intent, so the gate is off: no warning even single-HX.
    assert not _gravity_warnings(_nc_gravity_loop(second_hx=False, pump_pressure=5.0))


def test_check_gravity_mismatch_rejects_callable_strategy():
    j0, j1 = Junction(name="A"), Junction(name="B")
    k = Kirchhoff(MultiDiGraph([flow_edge((j0, j1), Pump(pressure=1.0, name="P")), flow_edge((j1, j0), Resistor(1.0, name="R"))]))
    with pytest.raises(TypeError, match="HydraulicStrategyMap"):
        check_gravity_mismatch(k, strategy=lambda m, T: 0.0)


def test_symmetric_plate_steady_state_rejects_zero_mdot():
    f, c = MTR_fuel_and_channel(z_N=5, fuel_N=2, clad_N=2)
    with pytest.raises(ValueError, match="mass flow must be nonzero"):
        symmetric_plate_steady_state(c=c, f=f, mdot=0.0, p_abs=2e5, power=1.0, Tin=40.0)


def test_symmetric_plate_steady_state_accepts_small_signed_mdot():
    f, c = MTR_fuel_and_channel(z_N=5, fuel_N=2, clad_N=2)
    # A tiny signed mdot is fine (no ZeroDivision); it should return a State.
    assert symmetric_plate_steady_state(c=c, f=f, mdot=-0.01, p_abs=2e5, power=1e-6, Tin=35.0)


def test_effective_pipe_bad_partition_raises_value_error_citing_default_tolerance():
    # A raised ValueError survives python -O, unlike a bare assert; a >~1e-5 mismatch.
    with pytest.raises(ValueError, match="rtol=1e-05, atol=1e-08"):
        EffectivePipe(length=0.5, heated_perimeter=0.07, wet_perimeter=0.144, area=1.4e-4, heated_parts=(0.035, 0.05))


def test_effective_pipe_good_partition_constructs():
    EffectivePipe(length=0.5, heated_perimeter=0.07, wet_perimeter=0.144, area=1.4e-4, heated_parts=(0.035, 0.035))


def test_resistor_from_known_point_raises_value_error_not_assertion():
    # python -O would strip a bare assert; these are raised.
    with pytest.raises(ValueError):
        ResistorFromKnownPoint(behavior="constant")
    with pytest.raises(ValueError):
        ResistorFromKnownPoint(mdot=1.0, behavior="linear")


def test_channel_heat_flux_save_defaults_q():
    pipe = EffectivePipe.rectangular(length=0.5, edge1=0.07, edge2=0.002, heated_edge=0.07)
    ch = ChannelHeatFlux(z_boundaries=np.linspace(0, 0.5, 4), fluid=light_water, pipe=pipe)
    y = np.concatenate([np.full(ch.n, 40.0), [0.0]])
    saved = ch.save(y, Tin=40.0, mdot=0.1)  # no q wired -> defaults 0.0, matches calculate
    assert saved is not None


def test_state_uniform_copies_value_no_aliasing():
    A = Calculation_factory(lambda y: y, [True] * 2, {"T": slice(0, 2)})("A")
    B = Calculation_factory(lambda y: y, [True] * 2, {"T": slice(0, 2)})("B")
    value = np.array([300.0, 300.0])
    st = State.uniform([A, B], value)
    assert np.array_equal(st["A"]["T"], value) and np.array_equal(st["B"]["T"], value)
    # Distinct objects: editing one variable must not touch the others.
    assert st["A"]["T"] is not st["B"]["T"]
    assert st["A"]["T"] is not value
    st["A"]["T"][0] = 999.0
    assert st["B"]["T"][0] == 300.0


def test_state_uniform_scalar_value_unchanged():
    A = Calculation_factory(lambda y: y, [True], {"T": 0})("A")
    st = State.uniform([A], 42.0)
    assert st["A"]["T"] == 42.0


def test_missing_flow_error_names_components():
    j0, j1 = Junction(name="A"), Junction(name="B")
    p = Pump(pressure=1.0, name="P")
    r = Resistor(1.0, name="R")
    k = Kirchhoff(MultiDiGraph([flow_edge((j0, j1), p), flow_edge((j1, j0), r)]))
    with pytest.raises(MissingFlowError) as exc:
        guess_hydraulic_steady_state(k, {p: 1.0}, 40.0)  # R's flow omitted
    assert "R" in str(exc.value) and "Missing flow data" in str(exc.value)


def test_stream_error_catches_construction_error():
    a1, a2, b = Junction(name="A"), Junction(name="A"), Junction(name="B")
    g = MultiDiGraph(
        [
            flow_edge((a1, b), Pump(pressure=1.0, name="P1")),
            flow_edge((b, a1), Resistor(1.0, name="R1")),
            flow_edge((a2, b), Pump(pressure=2.0, name="P2")),
            flow_edge((b, a2), Resistor(2.0, name="R2")),
        ]
    )
    with pytest.raises(StreamError):
        Kirchhoff(g)
