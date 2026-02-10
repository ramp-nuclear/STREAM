"""
Oh, how the turns have tabled...
"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from networkx import MultiDiGraph

from stream.aggregator import Aggregator
from stream.calculations.kirchhoff import (
    build_kvl_matrix, Junction, Kirchhoff, KirchhoffWDerivatives,
    to_graph_for_cycles, to_str)
from .conftest import are_close, pos_medium_floats


def test_a_multigraph_to_graph_for_cycles():
    g = MultiDiGraph()
    g.add_edge("A", "B", name="hi")
    g.add_edge("A", "B", name="hello")
    g.add_edge("A", "B", name="greetings")

    h = to_graph_for_cycles(g)

    assert [data["name"] for data in h.adj["A"].values()] == ["hi", "hello", "greetings"]
    assert list(h.adj["B"].values()) == [{}, {}, {}]


def test_build_kvl_matrix_from_a_multigraph():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=(1,))
    g.add_edge("A", "B", comps=(2, 3))
    g.add_edge("B", "A", comps=(4,))

    kvl = build_kvl_matrix(g, {i + 1: i for i in range(4)})
    are_close(kvl @ np.array([8.0, -3.0, 11.0, -8.0]), np.zeros(2))


def test_junction_mixing_a_given_set_of_currents():
    """
    The temperature a Junction defines is just the weighted sum of all incoming
    temperatures with the corresponding mass current.
    """
    J = Junction()

    mdot = {1: 1.0, 2: 2.0, 3: -3.0, 4: -4.0}
    Tin_plus = {1: 10.0, 2: 0}
    Tin_minus = {3: 5, 4: 17.0}
    kwargs = dict(mdot=mdot, Tin=Tin_plus, Tin_minus=Tin_minus)
    are_close(J.calculate([0], **kwargs), (10 * 1 + 4 * 17 + 3 * 5) / 10)

    mdot[3] = 3
    are_close(J.calculate([0], **kwargs), (10 * 1 + 4 * 17) / 7)


@pytest.fixture(scope="module")
def mock_graph(J) -> MultiDiGraph:
    J0, J1 = J
    g = MultiDiGraph()
    g.add_edge(J0, J1, comps=("A", "B", "C"))
    g.add_edge(J0, J1, comps=("D", "E"))
    g.add_edge(J1, J0, comps=("F",))
    return g


@pytest.fixture(scope='module')
def J(): return Junction(name='J0'), Junction(name='J1')


@pytest.fixture(scope='module')
def K(mock_graph): return Kirchhoff(mock_graph)


Edge = list[str]


def _make_cycle(values: list[Edge]) -> MultiDiGraph:
    g = MultiDiGraph()
    for i, val in enumerate(values):
        g.add_edge(str(i), str((i+1) % len(values)), 0, comps=val)
    return g


def _exclusive(edgelist: list[Edge]) -> bool:
    return len(set(sum(edgelist, start=[]))) == sum(len(v) for v in edgelist)


graph_sizes = st.shared(st.integers(2, 20), key='size')
names = st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=5)
complists = st.lists(names, unique=True, min_size=1, max_size=4)
edges = st.shared(graph_sizes.flatmap(lambda n: st.lists(complists,
                                                         min_size=n+1,
                                                         max_size=n+1))
                  .filter(_exclusive),
                  key='edges')
comps = edges.map(lambda v: sum(v, start=[]))
graphs = edges.map(_make_cycle)
subsets = comps.flatmap(lambda x: st.lists(st.sampled_from(x), unique=True))
reference_nodes = st.tuples(graphs.flatmap(lambda x: st.sampled_from(list(x.nodes))),
                            pos_medium_floats)
kirchhoffs = st.builds(lambda x, y, z: Kirchhoff(x, *y, reference_node=z),
                       graphs, subsets, reference_nodes)
k_derivs = st.builds(lambda x, y, z: KirchhoffWDerivatives(x, *y, reference_node=z),
                     graphs, subsets, reference_nodes)


@given(st.one_of(kirchhoffs, k_derivs))
def test_kirchhoff_variable_names_are_strings(k):
    assert all(isinstance(key, str) for key in k.variables.keys())


def test_kvl_matrix_works_on_mock_graph(K):
    expected_kvl = np.array([[1., 1., 1., 0., 0., 1.],
                             [1., 1., 1., -1., -1., 0.]])
    test_dp = np.array([1, 1, 1, 1.5, 1.5, -3])
    are_close(K._kvl @ test_dp, expected_kvl @ test_dp)


def test_kcl_matrix_works_on_mock_graph(K, J):
    expected_kcl = np.array([[-1, -1, 1], [1, 1, -1]])
    test_mdot = np.array([2, 2, -4])
    are_close(K._kcl @ test_mdot, expected_kcl @ test_mdot)


def test_kirchhoff_indexing_works_on_mock_graph(K, J):
    J0, J1 = J

    assert K.indices("mdot", J0) == dict(A=0, D=1, F=2)
    assert K.indices("mdot", J1) == dict(C=0, E=1, F=2)
    assert K.component_edge("A") == to_str((J0, J1, 0))
    assert K.component_edge("E") == to_str((J0, J1, 1))
    assert K.component_edge("F") == to_str((J1, J0, 0))
    assert K.variables_by_type == dict(mdot=slice(0, 3), abs_pressure=slice(3, 3))


def test_kirchhoff_calculate_works_for_mock_graph(K):
    res = K.calculate(
        [1, 2, 3], pressure=dict(A=-1.0, B=-1.0, C=-1.0, D=-2.0, E=-1, F=3.0)
    )
    are_close(res, 0.0)


def test_kirchoff_w_mdot2_accepts_a_graph_and_has_correct_length():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=(123,))
    g.add_edge("B", "A", comps=(345,))
    KD = KirchhoffWDerivatives(g)
    assert len(KD) == 4


def test_Kirchoff_kcl_matrix_fits_known_example_with_weights():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=(123,), signify=50)
    g.add_edge("B", "A", comps=(345,))
    K = Kirchhoff(g)
    assert np.all(K._kcl.toarray() == np.array([[-50.0, 1.0], [50.0, -1.0]]))


def test_Kirchoff_supplies_correct_absolute_pressures_for_one_example(mock_graph, J):
    J0, J1 = J
    p0 = 1.5
    abs_comps = ("A", "B", "F")
    K = Kirchhoff(mock_graph, *abs_comps, reference_node=(J0, p0))
    assert K.ref_node is J0
    assert K.ref_pressure == p0
    assert len(K) == 6

    for c in abs_comps:
        # noinspection PyTypeChecker
        assert K.indices("p_abs", asking=c) == K.variables[to_str(("p_abs", c))]

    are_close(K._abs_matrix @ np.arange(6), np.array([0, 0, 0 + 1 + 2]))

    # noinspection SpellCheckingInspection
    comps = "ABCDEF"
    dps = [1, 3, -4, 5, 7, 8]
    expected = p0 + np.array([0, dps[0], sum(dps[:3])])
    res = K.calculate([1, 2, 3, 0, 0, 0], pressure=dict(zip(comps, dps)))
    abs_pressures = res[K.edges_count:]
    are_close(abs_pressures, expected)


def test_kirchoff_save_fits_known_value_for_one_example(K):
    assert K.save([1, 2, 3]) == {"(J0 -> J1, 0)": 1,
                                 "(J0 -> J1, 1)": 2,
                                 "(J1 -> J0, 0)": 3}


triplets = st.tuples(pos_medium_floats, pos_medium_floats, pos_medium_floats)


@given(triplets)
def test_agr_of_kirchhoff_load_reverses_save_by_example(K, tpl):
    g = MultiDiGraph()
    g.add_node(K)
    agr = Aggregator(g)
    assert np.allclose(agr.load(agr.save(tpl)), tpl)
