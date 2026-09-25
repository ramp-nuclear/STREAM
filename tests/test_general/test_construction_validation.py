"""Construction-time wiring validation.

Each check has a firing and a non-firing side: a malformed graph raises
``StreamConstructionError`` (or warns via ``pytest.warns`` with a silent
happy-path counterpart), while a well-formed one constructs quietly. Covers the
``validate_wiring`` hooks (PointKinetics feedback/source wiring,
PointKineticsWInput power input, frozen-time controller), cross-node error
collection, ``StreamError`` catching a construction error, and a full
fuel+channel composition constructing silently.

A wall-less ``ChannelAndContacts`` still constructs, because the wall check is
not a construction-time raise (it would fire on the legitimate wall-less
intermediate ``FlowGraph`` aggregator).
"""

import warnings

import numpy as np
import pytest
from networkx import DiGraph

from stream import Aggregator
from stream.aggregator import CalculationGraph
from stream.calculations import Junction, PointKinetics, PointKineticsWInput
from stream.calculations.channel import ChannelAndContacts
from stream.calculations.point_kinetics import ReactivityController
from stream.composition import Calculation_factory, symmetric_plate
from stream.errors import StreamConstructionError, StreamError
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water

from .conftest import MTR_fuel_and_channel


def _calc(calculate, mass_vector, variables, name):
    """A minimal Calculation from a lambda (see :func:`Calculation_factory`)."""
    return Calculation_factory(calculate=calculate, mass_vector=mass_vector, variables=variables)(name)


def _pk(name="PK", **kw):
    return PointKinetics(1e-4, np.array([0.0065]), np.array([0.08]), name=name, **kw)


def _silent(build):
    """Assert ``build()`` constructs without emitting any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build()


def test_empty_graph_raises():
    with pytest.raises(StreamConstructionError, match="no Calculations"):
        Aggregator(DiGraph())


def test_nonempty_graph_constructs():
    _silent(lambda: Aggregator.from_decoupled(_calc(lambda y: -y, [True], {"v": 0}, "A")))


def test_string_node_raises():
    g = DiGraph()
    g.add_node("phantom_junction")
    with pytest.raises(StreamConstructionError, match="not a Calculation"):
        Aggregator(g)


def test_calculation_node_ok():
    _silent(lambda: Aggregator.from_decoupled(_calc(lambda y: -y, [True], {"v": 0}, "A")))


def test_funcs_orphan_key_raises_naming_object():
    A = _calc(lambda y, *, k: -y + k, [True], {"v": 0}, "A")
    orphan = _calc(lambda y, *, k: -y + k, [True], {"v": 0}, "A_orphan")
    with pytest.raises(StreamConstructionError, match="A_orphan") as ei:
        Aggregator.from_decoupled(A, funcs={orphan: {"k": lambda t: 5.0}})
    assert "not a Calculation in this Aggregator's graph" in str(ei.value)


def test_funcs_graph_key_ok():
    A = _calc(lambda y, *, k: -y + k, [True], {"v": 0}, "A")
    _silent(lambda: Aggregator.from_decoupled(A, funcs={A: {"k": lambda t: 5.0}}))


def test_no_kwargs_consumer_rejects_wired_variable():
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y: -2.0 * y, [True], {"v": 0}, "B")  # no 'v' param, no **kwargs
    with pytest.raises(StreamConstructionError, match="cannot accept wired variable"):
        Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("v",)))


def test_kwargs_consumer_is_unverifiable_and_constructs():
    # A **kwargs consumer cannot be checked, so wiring an unused variable constructs silently.
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y, **_: -2.0 * y, [True], {"v": 0}, "B")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("v",))))


def test_named_param_consumer_ok():
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y, *, v: -2.0 * y + v, [True], {"v": 0}, "B")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("v",))))


def test_source_lacks_variable_raises_naming_edge_and_available():
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y, **_: -y, [True], {"v": 0}, "B")
    with pytest.raises(StreamConstructionError) as ei:
        Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("temperature",)))
    msg = str(ei.value)
    assert "A -> B" in msg and "'temperature'" in msg and "'v'" in msg
    assert isinstance(ei.value.__cause__, KeyError)


def test_source_owns_variable_ok():
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y, **_: -y, [True], {"v": 0}, "B")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("v",))))


def test_self_edge_raises():
    A = _calc(lambda y, *, v: -y + v, [True], {"v": 0}, "A")
    g = DiGraph()
    g.add_node(A)
    g.add_edge(A, A, variables=("v",))
    with pytest.raises(StreamConstructionError, match="Self-edge"):
        Aggregator(g)


def test_distinct_edge_ok():
    A = _calc(lambda y: -y, [True], {"v": 0}, "A")
    B = _calc(lambda y, *, v: -y + v, [True], {"v": 0}, "B")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(A), Aggregator.from_decoupled(B), (A, B, ("v",))))


def _shadow_graph():
    A = _calc(lambda y: -y, [True], {"T": 0}, "A")
    B = _calc(lambda y, *, T: T - y, [False], {"x": 0}, "B")
    cg = CalculationGraph.connect(
        CalculationGraph.from_decoupled(A), CalculationGraph.from_decoupled(B), (A, B, ("T",))
    )
    return cg, A, B


def test_funcs_shadow_edge_warns():
    cg, A, B = _shadow_graph()
    cg.funcs = {B: {"T": lambda t: 999.0}}
    with pytest.warns(UserWarning, match="shadow"):
        cg.to_aggregator()


def test_no_shadow_is_silent():
    cg, A, B = _shadow_graph()
    _silent(cg.to_aggregator)


def test_tin_without_tin_minus_warns():
    S = _calc(lambda y: -y, [True], {"Tin": 0}, "S")
    C = _calc(lambda y, *, Tin, Tin_minus=None, **_: -y, [True], {"c": 0}, "C")
    with pytest.warns(UserWarning, match="Tin_minus"):
        Aggregator.connect(Aggregator.from_decoupled(S), Aggregator.from_decoupled(C), (S, C, ("Tin",)))


def test_both_wired_is_silent():
    S = _calc(lambda y: -y, [True], {"Tin": 0}, "S")
    S2 = _calc(lambda y: -y, [True], {"Tin_minus": 0}, "S2")
    C = _calc(lambda y, *, Tin, Tin_minus=None, **_: -y, [True], {"c": 0}, "C")
    _silent(
        lambda: Aggregator.connect(
            Aggregator.from_decoupled(S) + Aggregator.from_decoupled(S2),
            Aggregator.from_decoupled(C),
            (S, C, ("Tin",)),
            (S2, C, ("Tin_minus",)),
        )
    )


def test_junction_is_excluded_from_warning():
    # A Junction legally aggregates absent sides, so a Tin-only supplier must not warn.
    S = _calc(lambda y: -y, [True], {"Tin": 0}, "S")
    J = Junction(name="J")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(S), Aggregator.from_decoupled(J), (S, J, ("Tin",))))


def test_pk_temp_worth_without_T_raises_naming_node_and_fix():
    channel = object()
    pk = _pk(temp_worth={channel: 1e-4}, ref_temp={channel: 40.0})
    with pytest.raises(StreamConstructionError) as ei:
        Aggregator.from_decoupled(pk)
    msg = str(ei.value)
    assert "'PK'" in msg and "temp_worth" in msg and "'T' edge" in msg


def test_pk_feedback_with_T_wired_ok():
    channel = _calc(lambda y: -y, [True], {"T": 0}, "chan")
    pk = _pk(temp_worth={channel: 1e-4}, ref_temp={channel: 40.0})
    # Compose via CalculationGraph so the aggregator is validated once with the T edge present.
    cg = CalculationGraph.connect(
        CalculationGraph.from_decoupled(channel), CalculationGraph.from_decoupled(pk), (channel, pk, ("T",))
    )
    _silent(cg.to_aggregator)


def test_pk_double_source_raises_naming_both_suppliers():
    pk = _pk()
    s1 = _calc(lambda y: -y, [True], {"source": 0}, "S1")
    s2 = _calc(lambda y: -y, [True], {"source": 0}, "S2")
    cg = CalculationGraph.connect(
        CalculationGraph.from_decoupled(pk),
        CalculationGraph.from_decoupled(s1) + CalculationGraph.from_decoupled(s2),
        (s1, pk, ("source",)),
        (s2, pk, ("source",)),
    )
    with pytest.raises(StreamConstructionError) as ei:
        cg.to_aggregator()
    msg = str(ei.value)
    assert "S1" in msg and "S2" in msg and "source" in msg


def test_pk_single_source_ok():
    pk = _pk()
    s1 = _calc(lambda y: -y, [True], {"source": 0}, "S1")
    _silent(lambda: Aggregator.connect(Aggregator.from_decoupled(s1), Aggregator.from_decoupled(pk), (s1, pk, ("source",))))


def test_pk_time_dependent_controller_frozen_t_warns():
    pk = _pk(controls=ReactivityController(input_reactivity=lambda s, ts, t: -1e-3 * t))
    with pytest.warns(UserWarning, match="frozen"):
        Aggregator.from_decoupled(pk, funcs={pk: {"t": 0}})


def test_pk_time_dependent_controller_callable_t_is_silent():
    pk = _pk(controls=ReactivityController(input_reactivity=lambda s, ts, t: -1e-3 * t))
    _silent(lambda: Aggregator.from_decoupled(pk, funcs={pk: {"t": lambda t: t}}))


def test_pk_static_controller_constant_t_no_warn():
    # A controller with no ramp/trip/state-machine is not time-dependent, so a constant funcs t must not warn.
    pk = _pk()
    _silent(lambda: Aggregator.from_decoupled(pk, funcs={pk: {"t": 0}}))


def test_pkwinput_without_power_input_raises():
    pk = PointKineticsWInput(1e-4, np.array([0.0065]), np.array([0.08]), name="PKW")
    with pytest.raises(StreamConstructionError) as ei:
        Aggregator.from_decoupled(pk)
    assert "'PKW'" in str(ei.value) and "power_input" in str(ei.value)


def test_pkwinput_with_power_input_ok():
    pk = PointKineticsWInput(1e-4, np.array([0.0065]), np.array([0.08]), name="PKW")
    src = _calc(lambda y: -y, [True], {"power_input": 0}, "decay")
    cg = CalculationGraph.connect(
        CalculationGraph.from_decoupled(src), CalculationGraph.from_decoupled(pk), (src, pk, ("power_input",))
    )
    _silent(cg.to_aggregator)


def test_collects_multiple_nodes_into_one_error():
    channel = object()
    pk1 = _pk(name="PK1", temp_worth={channel: 1e-4}, ref_temp={channel: 40.0})
    pk2 = PointKineticsWInput(1e-4, np.array([0.0065]), np.array([0.08]), name="PK2")
    with pytest.raises(StreamConstructionError) as ei:
        Aggregator.from_decoupled(pk1, pk2)
    msg = str(ei.value)
    assert "PK1" in msg and "PK2" in msg


def test_stream_error_catches_construction_error():
    try:
        Aggregator(DiGraph())
    except StreamError as caught:
        assert isinstance(caught, StreamConstructionError)
    else:
        pytest.fail("expected a StreamConstructionError")


def test_wall_less_channel_does_not_raise_at_construction():
    """A wall-less ChannelAndContacts constructs, because the wall check is not a
    construction-time raise (it would fire on the legitimate wall-less intermediate
    FlowGraph aggregator)."""
    pipe = EffectivePipe.rectangular(length=0.5, edge1=0.07, edge2=0.002, heated_edge=0.07)
    cc = ChannelAndContacts(z_boundaries=np.linspace(0, 0.5, 4), fluid=light_water, pipe=pipe, name="CC")
    Aggregator.from_decoupled(cc)


def test_real_symmetric_plate_composition_constructs_without_warning():
    fuel, channel = MTR_fuel_and_channel(z_N=5, fuel_N=4, clad_N=2)
    _silent(lambda: symmetric_plate(channel, fuel, funcs={fuel: dict(power=1000.0)}).to_aggregator())
