from contextlib import nullcontext
from itertools import count
from typing import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, nothing, one_of, text
from networkx import DiGraph
from networkx.utils import graphs_equal

from stream.aggregator import (
    CONSTRAINT,
    VARS,
    Aggregator,
    CalculationGraph,
    NonUniqueCalculationNameError,
    add_variables,
    create_constraints,
    vars_,
)
from stream.calculation import Calculation, unpacked
from stream.composition import Calculation_factory
from stream.jacobians import _associated_calculations
from stream.solvers import TransientRuntimeError, differential_algebraic
from stream.errors import StreamConstructionError
from stream.units import Place
from stream.utilities import ignore_warnings, mutually_exclusive

from .conftest import are_close, medium_floats
from .test_calculation import Addition, add, divide, multiply


@pytest.fixture(scope="module")
def mock_agr():
    return Aggregator(DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))]))


def test_example_aggregator_has_known_shape(mock_agr):
    """
    Creating simple aggregator input
    """
    assert mock_agr.vector_length == 2
    assert np.allclose(mock_agr.mass, np.array((0, 0)))
    assert mock_agr.external == {add: {"x": {multiply: 1}}, multiply: {"y": {add: 0}}}
    assert mock_agr.funcs == {}


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_save(mock_agr, y, x):
    assert mock_agr.save(solution=(y, x)) == {
        add.name: {"y": y},
        multiply.name: {"x": x},
    }


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_collision_of_calculations_raises_by_example(y, x):
    with pytest.raises(NonUniqueCalculationNameError):
        Aggregator.from_decoupled(Addition("A"), Addition("A"))


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_load(mock_agr, y, x):
    assert np.allclose(mock_agr.load({add.name: {"y": y}, multiply.name: {"x": x}}), (y, x))


def test_load_reverses_save_by_example(mock_agr):
    vec = np.array([1, 2])
    assert np.allclose(vec, mock_agr.load(mock_agr.save(vec)))


@given(medium_floats, medium_floats)
def test_compute_of_a_graph_vs_known_implementation(mock_agr, y, x):
    """Tests data transfer through the Aggregator"""
    assert np.allclose(mock_agr.compute(np.array([y, x])), [y + x, y * x])


def test_composition_of_specific_agrs_yields_known_agr():
    g_a = DiGraph()
    g_a.add_edge(1, 2, data="hello")
    g_a.add_edge(2, 3, data="hi")

    g_b = DiGraph()
    g_b.add_edge(3, 4, data="welcome")
    # noinspection PyTypeChecker
    a = CalculationGraph(g_a, {1: {"unit": lambda x: x}})
    # noinspection PyTypeChecker
    b = CalculationGraph(g_b, {4: {"one": lambda x: 1}})
    edge = (2, 4, "var")

    # noinspection PyTypeChecker
    c = CalculationGraph.connect(a, b, edge)

    assert list(c.graph.edges(data=True)) == [
        (1, 2, dict(data="hello")),
        (2, 3, dict(data="hi")),
        (2, 4, dict(variables="var")),
        (3, 4, dict(data="welcome")),
    ]
    assert list(c.funcs.keys()) == [1, 4]


def test_aggregator_input_works_as_expected():
    DoNothing = Calculation_factory(
        calculate=lambda y, *, var=None: y + var if var is not None else y,
        mass_vector=np.zeros(5),
        variables={"var": slice(0, 5)},
    )

    a = DoNothing(name="a")
    b = DoNothing(name="b")
    assert np.all(a.calculate(np.arange(6)) == np.arange(6))
    agr = Aggregator(DiGraph([(a, b, vars_("var")), (b, a, vars_("var"))]))
    assert np.all(agr.compute(np.arange(10)) == np.tile(np.arange(5.0, 15.0, 2), 2))


def test_ida_root_functions():
    last_call = []

    def F(y, t):
        last_call[:] = (y, t)
        return y

    options = dict(rtol=1e-9)

    out, _ = differential_algebraic(
        F=F,
        mass=np.ones(1),
        y0=np.ones(1),
        time=np.arange(10),
        yp0=np.ones(1),
        R=lambda y, t: np.asarray(y < 1000),
        nr_rootfns=1,
        **options,
    )
    assert last_call[1] > np.log(1000)
    assert last_call[0] > 1000
    are_close(np.squeeze(out), np.exp(np.arange(7)))


def test_agr_input_connect():
    g_a = DiGraph([(1, 2, vars_("hi"))])
    g_b = DiGraph([(1, 2, vars_("hello"))])
    # funcs is dict[Calculation, dict[Name, FunctionOfTime]]; inner values are dicts.
    a = CalculationGraph(g_a, {1: {"a": 2}, 2: {"a": 3}})
    b = CalculationGraph(g_b, {1: {"a": 3}})

    c = a + b
    # the shared (1, 2) edge now unions a's and b's routings instead of dropping a's
    assert list(c.graph.edges(data=True)) == [(1, 2, vars_("hi", "hello"))]
    assert c.funcs == {1: {"a": 3}, 2: {"a": 3}}
    # noinspection PyTypeChecker
    d = CalculationGraph.connect(a, b, (1, 2, ("welcome",)))
    assert list(d.graph.edges(data=True)) == [(1, 2, vars_("hi", "hello", "welcome"))]
    assert d.funcs == {1: {"a": 3}, 2: {"a": 3}}


def _decay_aggregator():
    decay = Calculation_factory(calculate=lambda y: -y, mass_vector=[True], variables={"v": 0})("A")
    g = DiGraph()
    g.add_node(decay)
    return Aggregator(g)


def test_load_reconstructs_a_timeseries_with_integer_time_keys():
    """K4: load() detected a timeseries via isinstance(key, float), which is False
    for the np.int64 keys that saving an integer-time solve produces, so a genuine
    timeseries was misrouted to _vector_from_state and raised a cryptic KeyError."""
    agr = _decay_aggregator()
    ts = agr.save(agr.solve(np.array([1.0]), time=[0, 1, 2], eq_type="ODE"))
    assert all(not isinstance(k, str) for k in ts)  # numeric (np.int64) time keys

    sol = agr.load(ts)  # must reconstruct a Solution, not raise KeyError
    assert np.array_equal(sol.time, [0, 1, 2])
    assert sol.data.shape[0] == 3


def test_to_dataframe_detects_integer_keyed_timeseries():
    """K4: the duplicate float-key check in state.to_dataframe misparsed an
    int-keyed timeseries as a single State (no 'time' column)."""
    from stream.state import to_dataframe

    agr = _decay_aggregator()
    ts = agr.save(agr.solve(np.array([1.0]), time=[0, 1, 2], eq_type="ODE"))
    assert "time" in to_dataframe(ts).columns


def test_connect_deep_merges_funcs_for_a_shared_calculation():
    """K2: connecting two graphs that both carry funcs for the SAME calculation
    must union the inner name bindings; the shallow af | bf let the second inner
    dict wholesale replace the first, silently dropping non-clashing bindings."""
    a = CalculationGraph(DiGraph([(1, 2, vars_("x"))]), {1: {"pressure": lambda t: t, "Tin": 300.0}})
    b = CalculationGraph(DiGraph([(1, 2, vars_("y"))]), {1: {"Tin": 300.0, "mdot": 1.0}})

    c = a + b
    assert set(c.funcs[1]) == {"pressure", "Tin", "mdot"}


def test_connect_merges_edge_variables_when_both_graphs_share_an_edge():
    """K3: compose() lets b's edge data replace a's, dropping a's variable
    routings for a shared edge. connect must union (deduped) both edges' vars."""
    g1 = CalculationGraph(DiGraph([(1, 2, vars_("T_left", "h_left", "T_right", "h_right"))]))
    g2 = CalculationGraph(DiGraph([(1, 2, vars_("T_left"))]))
    expected = {"T_left", "h_left", "T_right", "h_right"}
    assert set((g1 + g2).graph.edges[1, 2][VARS]) == expected
    assert set((g2 + g1).graph.edges[1, 2][VARS]) == expected


def test_connect_dedups_repeated_explicit_edge_variables():
    """K3: the explicit-edges merge path must also dedup a name it already routes."""
    g1 = CalculationGraph(DiGraph([(1, 2, vars_("T_left", "h_left"))]))
    empty = CalculationGraph(DiGraph())
    m = CalculationGraph.connect(g1, empty, (1, 2, ("T_left",)))
    assert m.graph.edges[1, 2][VARS] == ("T_left", "h_left")


def test_ida_continuous_mode():
    class StubbornCalc(Calculation):
        c = count()
        i = 0
        name = "Stubborn"

        @unpacked
        def calculate(self, y):
            return np.asarray(y)

        @property
        def mass_vector(self) -> Sequence[bool]:
            return (True,)

        @property
        def variables(self) -> dict[str, Place]:
            return dict(y=1)

        @unpacked
        def should_continue(self, y, **kwargs):
            return bool(self.i % 40)

        @unpacked
        def change_state(self, y, **kwargs):
            self.i = next(self.c)

    agr = Aggregator.from_decoupled(StubbornCalc())
    sol = agr.solve(
        y0=np.ones(1),
        time=(t := np.linspace(0, 10, 100)),
        continuous=True,
        eq_type="DAE",
    )
    assert np.allclose(sol[:, 0], np.exp(t), rtol=1e-4), sol[:, 0] - np.exp(t)


def test_associated_calculations_for_a_known_example(mock_agr):
    assoc = _associated_calculations(mock_agr)
    assert assoc == {0: [add, multiply], 1: [multiply, add]}


justx = Calculation_factory(calculate=lambda x, *, y: x, mass_vector=[False], variables={"x": 0})("justx")
justy = Calculation_factory(calculate=lambda y, *, x: y, mass_vector=[False], variables={"y": 0})("justy")


@pytest.mark.parametrize(
    ["graph", "expectation"],
    [
        (
            DiGraph([(justx, justy, vars_("x")), (justy, justx, vars_("y"))]),
            nullcontext(),
        ),
        (
            DiGraph([(justx, justy, vars_("missing_variable")), (justy, justx, vars_("y"))]),
            pytest.raises(KeyError, match="missing_variable"),
        ),
        (
            DiGraph([(justx, justy, vars_("x")), (justy, justx, vars_("missing_variable"))]),
            pytest.raises(KeyError, match="missing_variable"),
        ),
    ],
)
def test_agr_identifies_missing_variables_in_indices_for_known_examples(graph, expectation):
    with expectation:
        Aggregator(graph)


@given(s=text(), n=one_of(text(), nothing()))
def test_add_variables_accepts_added_variables_correctly(s, n):
    mock_graph = DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))])
    original_variables = list(mock_graph[add][multiply]["variables"])
    added_variables = [s, n] if n != s else [s]
    new_variables = [x for x in added_variables if x not in original_variables]
    add_variables(mock_graph, add, multiply, s, n)
    assert mock_graph[add][multiply]["variables"] == tuple(original_variables + new_variables)


def test_add_variables_creates_new_edge_if_referenced_edge_doesnt_exist():
    mock_graph = DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))])
    add_variables(mock_graph, add, divide, "w")
    assert (add, divide) in mock_graph.edges()


@given(text())
def test_add_variables_is_idempotent(s):
    mock_graph = DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))])
    add_variables(mock_graph, add, multiply, s)
    graph_prior = mock_graph.copy()
    add_variables(mock_graph, add, multiply, s)
    assert graphs_equal(graph_prior, mock_graph)


def test_create_constraints_for_a_known_example():
    calc = Calculation_factory(
        lambda v, **_: v - np.array([-1, 0, 1]),
        [False] * 3,
        dict(v_neg=0, v_zero=1, v_pos=2),
    )()
    agr = Aggregator.from_decoupled(calc)
    assert np.all(
        create_constraints(agr, negative=["v_neg"], positive=["v_pos"])
        == np.array([c.value for c in [CONSTRAINT.negative, CONSTRAINT.none, CONSTRAINT.positive]])
    )


def test_mutually_exclusive_handles_unequal_length_categories():
    """mutually_exclusive must accept differently-sized categories, not only the
    accidental equal-length case that flattens into a 2-D array."""
    assert mutually_exclusive(["mdot_a", "mdot_b"], ["h"])
    assert not mutually_exclusive(["mdot_a", "mdot_b"], ["mdot_b"])


def test_create_constraints_with_unequal_category_sizes():
    """The documented API — categories of different sizes — must not crash the
    mutual-exclusivity assertion."""
    calc = Calculation_factory(
        lambda v, **_: v - np.zeros(3),
        [False] * 3,
        dict(mdot_a=0, mdot_b=1, h=2),
    )()
    agr = Aggregator.from_decoupled(calc)
    result = create_constraints(agr, non_negative=["mdot_a", "mdot_b"], positive=["h"])
    assert np.all(
        result
        == np.array([c.value for c in [CONSTRAINT.non_negative, CONSTRAINT.non_negative, CONSTRAINT.positive]])
    )


def test_create_constraints_with_bad_name_errors_well():
    calc = Calculation_factory(
        lambda v, **_: v - np.array([-1, 0, 1]),
        [False] * 3,
        dict(v_neg=0, v_zero=1, v_pos=2),
    )()
    agr = Aggregator.from_decoupled(calc)
    with pytest.raises(KeyError, match="moo. Must be one of"):
        create_constraints(agr, moo=["v_neg"], positive=["v_pos"])


# --- reserved solver options on the event-DAE path ---


class _EventNode:
    """A minimal DAE node with a localizable event margin, so solve() takes the
    rootfn path that manages nr_rootfns/rootfn itself."""

    name = "trip"
    variables = {"x": 0}
    mass_vector = np.array([True])

    def __len__(self):
        return 1

    def __hash__(self):
        return hash(self.name)

    def calculate(self, y, **_):
        return np.array([-y[0]])

    def indices(self, v, asking=None):
        return self.variables[v]

    def load(self, s):
        return np.array([s["x"]])

    def save(self, y, t=0):
        return {"x": y[0]}

    strict_save = save

    def event_margin(self, y, **_):
        return np.array([y[0] - 0.5])

    def has_event(self):
        return True

    def should_continue(self, y, **_):
        return True

    def change_state(self, y, **_):
        pass


@pytest.mark.parametrize("key", ["nr_rootfns", "rootfn"])
def test_event_dae_rejects_reserved_solver_option(key):
    """A user-supplied nr_rootfns/rootfn on the event-DAE path is managed by the
    Aggregator; it must raise a named StreamConstructionError, not a duplicate-keyword
    TypeError (nor a silent rootfn override)."""
    g = DiGraph()
    g.add_node(_EventNode())
    agr = Aggregator(g)
    with ignore_warnings(UserWarning):
        with pytest.raises(StreamConstructionError) as exc:
            agr.solve(np.array([1.0]), np.linspace(0.0, 1.0, 11), eq_type="DAE", **{key: 1})
    assert key in str(exc.value)
    assert not isinstance(exc.value, TypeError)  # the collision no longer surfaces raw


def test_marginless_dae_solve_does_not_trip_reserved_guard():
    """The reserved-option guard is event-path-only; a system with no event margins
    passes the same option straight through to IDA and solves normally."""
    Decay = Calculation_factory(calculate=lambda y: -y, mass_vector=[True], variables=dict(y=0))
    agr = Aggregator.from_decoupled(Decay())
    with ignore_warnings(UserWarning):
        sol = agr.solve(np.array([1.0]), np.linspace(0.0, 1.0, 5), eq_type="DAE", nr_rootfns=1)
    assert sol.data.shape[0] == len(sol.time)


def test_polling_driver_attaches_pre_failure_trajectory():
    """A later-segment failure in _integrate_with_polling must carry the accumulated
    pre-failure trajectory, not just the tiny failed segment: the pre-failure horizon
    is attached on e.t/e.y via the shared _merge_failure_trajectory."""

    class Decay:
        """A marginless event node (overrides change_state, exposes no event_margin) that
        latches a blow-up once it decays below 0.4 — routing through the polling driver."""

        name = "decay"
        variables = {"y": 0}
        mass_vector = np.array([True])

        def __init__(self):
            self._blown = False

        def __len__(self):
            return 1

        def __hash__(self):
            return hash(self.name)

        def calculate(self, y, **_):
            return np.array([1e8 * y[0] ** 2]) if self._blown else np.array([-y[0]])

        def indices(self, v, asking=None):
            return self.variables[v]

        def load(self, s):
            return np.array([s["y"]])

        def save(self, y, t=0):
            return {"y": y[0]}

        strict_save = save

        def change_state(self, y, **_):
            if y[0] < 0.4:
                self._blown = True

        def should_continue(self, y, **_):
            return True

        def event_margin(self, y, **_):
            return np.empty(0)  # no localizable margin -> forces the polling driver

        def has_event(self):
            return True

    g = DiGraph()
    g.add_node(Decay())
    agr = Aggregator(g)
    time = np.linspace(0.0, 5.0, 26)  # decays through 0.4 around t~0.9, then the restart blows up
    with ignore_warnings(RuntimeWarning):
        with ignore_warnings(UserWarning):
            with pytest.raises(TransientRuntimeError) as exc:
                agr.solve(np.array([1.0]), time, eq_type="ODE")
    reached = np.atleast_1d(exc.value.t)
    assert len(reached) > 2  # the pre-failure horizon, not just the tiny failed restart segment
    assert np.all(np.diff(reached) > 0)  # strictly monotone
