from itertools import count
from typing import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats, text, nothing, one_of
from networkx import DiGraph
from networkx.utils import graphs_equal

from stream import Aggregator, CalculationGraph, Calculation, unpacked
# noinspection PyProtectedMember
from stream.aggregator import vars_, NonUniqueCalculationNameError, add_variables
from stream.composition import Calculation_factory
from stream.jacobians import _associated_calculations
from stream.solvers import differential_algebraic
from stream.units import Place
from .conftest import are_close, medium_floats
from .test_calculation import add, multiply, divide, Addition


@pytest.fixture(scope="module")
def mock_agr():
    return Aggregator(DiGraph([
        (add, multiply, vars_('y')),
        (multiply, add, vars_('x'))
    ]))


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
    assert mock_agr.save(solution=(y, x)) == {add.name: {"y": y}, multiply.name: {"x": x}}


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_collision_of_calculations_raises_by_example(y, x):
    with pytest.raises(NonUniqueCalculationNameError):
        Aggregator.from_decoupled(Addition("A"), Addition("A"))


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_load(mock_agr, y, x):
    assert np.allclose(mock_agr.load({add.name: {"y": y},
                                      multiply.name: {"x": x}}), (y, x))


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
        variables={"var": slice(0, 5)}
    )

    a = DoNothing(name='a')
    b = DoNothing(name='b')
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
        F=F, mass=np.ones(1), y0=np.ones(1), time=np.arange(10),
        yp0=np.ones(1), R=lambda y, t: np.asarray(y < 1000), nr_rootfns=1, **options)
    assert last_call[1] > np.log(1000)
    assert last_call[0] > 1000
    are_close(np.squeeze(out), np.exp(np.arange(7)))


def test_agr_input_connect():
    g_a = DiGraph([(1, 2, vars_("hi"))])
    g_b = DiGraph([(1, 2, vars_("hello"))])
    # noinspection PyTypeChecker
    a = CalculationGraph(g_a, {1: 2, 2: 3})
    # noinspection PyTypeChecker
    b = CalculationGraph(g_b, {1: 3})

    c = a + b
    assert list(c.graph.edges(data=True)) == list(b.graph.edges(data=True))
    assert c.funcs == {1: 3, 2: 3}
    # noinspection PyTypeChecker
    d = CalculationGraph.connect(a, b, (1, 2, ("welcome",)))
    assert list(d.graph.edges(data=True)) == [(1, 2, vars_("hello", "welcome"))]
    assert d.funcs == {1: 3, 2: 3}


def test_ida_continuous_mode():
    class StubbornCalc(Calculation):
        c = count()
        i = 0
        name = 'Stubborn'

        @unpacked
        def calculate(self, y): return np.asarray(y)

        @property
        def mass_vector(self) -> Sequence[bool]: return True,

        @property
        def variables(self) -> dict[str, Place]: return dict(y=1)

        @unpacked
        def should_continue(self, y, **kwargs):
            return bool(self.i % 40)

        @unpacked
        def change_state(self, y, **kwargs):
            self.i = next(self.c)

    agr = Aggregator.from_decoupled(StubbornCalc())
    sol = agr.solve(y0=np.ones(1), time=(t := np.linspace(0, 10, 100)),
                    continuous=True, eq_type='DAE')
    assert np.allclose(sol[:, 0], np.exp(t), rtol=1e-4), sol[:, 0] - np.exp(t)


def test_associated_calculations_for_a_known_example(mock_agr):
    assoc = _associated_calculations(mock_agr)
    assert assoc == {0: [add, multiply], 1: [multiply, add]}


@given(s=text(), n=one_of(text(), nothing()))
def test_add_variables_accepts_added_variables_correctly(s, n):
    mock_graph = DiGraph([(add, multiply, vars_('y')), (multiply, add, vars_('x'))])
    original_variables = list(mock_graph[add][multiply]["variables"])
    added_variables = [s, n] if n != s else [s]
    new_variables = [x for x in added_variables if x not in original_variables]
    add_variables(mock_graph, add, multiply, s, n)
    assert mock_graph[add][multiply]["variables"] == tuple(original_variables + new_variables)


def test_add_variables_creates_new_edge_if_referenced_edge_doesnt_exist():
    mock_graph = DiGraph([(add, multiply, vars_('y')), (multiply, add, vars_('x'))])
    add_variables(mock_graph, add, divide, "w")
    assert (add, divide) in mock_graph.edges()


@given(text())
def test_add_variables_is_idempotent(s):
    mock_graph = DiGraph([(add, multiply, vars_('y')), (multiply, add, vars_('x'))])
    add_variables(mock_graph, add, multiply, s)
    graph_prior = mock_graph.copy()
    add_variables(mock_graph, add, multiply, s)
    assert graphs_equal(graph_prior, mock_graph)
