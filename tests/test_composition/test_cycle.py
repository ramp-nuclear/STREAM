from networkx import DiGraph, MultiDiGraph
import pytest

from stream import CalculationGraph
from stream.aggregator import VARS
from stream.calculations import Junction, Kirchhoff, KirchhoffWDerivatives
from stream.composition.cycle import (
    flow_edge, flow_graph, flow_graph_to_aggregator, in_parallel, in_series,
    kirchhoffify)


def test_in_series_with_no_arguments_gives_an_empty_graph_and_funcs():
    agr = in_series()
    assert not agr.graph and not agr.funcs


def test_in_series_two_components_creates_a_graph_with_known_edges():
    # noinspection PyTypeChecker
    agr = in_series(1, 2)
    assert agr.graph[1][2][VARS] == ("Tin",)
    assert agr.graph[2][1][VARS] == ("Tin_minus",)

    # noinspection PyTypeChecker
    agr = in_series(1, 2, cyclic=True)
    assert agr.graph[1][2][VARS] == ("Tin", "Tin_minus")
    assert agr.graph[2][1][VARS] == ("Tin", "Tin_minus")


def test_in_series_cyclic_for_3_calculations_has_the_right_edges():
    # noinspection PyTypeChecker
    agr = in_series(1, 2, 3, cyclic=True)
    assert agr.graph[1][2][VARS] == ("Tin",)
    assert agr.graph[2][3][VARS] == ("Tin",)
    assert agr.graph[3][1][VARS] == ("Tin",)
    assert agr.graph[1][3][VARS] == ("Tin_minus",)


def test_in_parallel_of_with_no_comps_gives_empty_graph_and_funcs():
    # noinspection PyTypeChecker
    agr = in_parallel(1, 2)
    assert not agr.graph and not agr.funcs


def test_in_parallel_for_two_paths_with_a_comp_each_has_correct_edges():
    # noinspection PyTypeChecker
    agr = in_parallel("A", "B", 1, 2)
    assert agr.graph["A"][1][VARS] == ("Tin",)
    assert agr.graph["A"][2][VARS] == ("Tin",)
    assert agr.graph[1]["B"][VARS] == ("Tin",)
    assert agr.graph[2]["B"][VARS] == ("Tin",)
    assert agr.graph["B"][1][VARS] == ("Tin_minus",)
    assert agr.graph["B"][2][VARS] == ("Tin_minus",)
    assert agr.graph[1]["A"][VARS] == ("Tin_minus",)
    assert agr.graph[2]["A"][VARS] == ("Tin_minus",)


def test_kirchhoffify_of_empty_graphs_makes_empty_graph_and_funcs():
    a = CalculationGraph(DiGraph())
    k = Kirchhoff(MultiDiGraph())
    b = kirchhoffify(a, k)
    assert not b.graph and not b.funcs


def test_kirchhoffify_w_2_hydraulic_comps_and_empty_graph_sets_comps_edges_correctly():
    a = CalculationGraph(DiGraph())
    k = Kirchhoff(MultiDiGraph())
    # noinspection PyTypeChecker
    b = kirchhoffify(a, k, hydraulic_comps=[1, 2])
    assert b.graph[k][1][VARS] == ("mdot",)
    assert b.graph[k][2][VARS] == ("mdot",)
    assert b.graph[1][k][VARS] == ("pressure",)
    assert b.graph[2][k][VARS] == ("pressure",)


def test_kirchhoffify_given_inertial_components_and_regular_kirchoff_throws():
    a = CalculationGraph(DiGraph())
    k = Kirchhoff(MultiDiGraph())
    expected_message = (
        "<class 'stream.calculations.kirchhoff.Kirchhoff'> "
        "does not handle inertial components, as it does "
        "not index mdot2"
    )
    with pytest.raises(AssertionError, match=expected_message):
        # noinspection PyTypeChecker
        kirchhoffify(a, k, inertial_comps=[1, 2])


def test_kirchhoffify_w_inertial_comps_adds_mdot_and_mdot2():
    a = CalculationGraph(DiGraph())
    k = KirchhoffWDerivatives(MultiDiGraph())
    # noinspection PyTypeChecker
    b = kirchhoffify(a, k, inertial_comps=[1, 2])
    assert b.graph[k][1][VARS] == ("mdot", "mdot2")
    assert b.graph[k][2][VARS] == ("mdot", "mdot2")


def test_kirchhoffify_one_junction_connects_k_to_junction_but_not_vice_versa():
    a = CalculationGraph(DiGraph())
    g = MultiDiGraph()
    g.add_node(j := Junction())
    k = Kirchhoff(g)
    b = kirchhoffify(a, k)
    assert b.graph[k][j][VARS] == ("mdot",)
    assert not b.graph[j]


def test_kirchhoffify_with_ref_mdots_gives_them_the_right_comps_by_example():
    a = CalculationGraph(DiGraph())
    k = Kirchhoff(MultiDiGraph())
    # noinspection PyTypeChecker
    b = kirchhoffify(a, k, hydraulic_comps=[1, 2], ref_mdots=[1, 3])
    assert b.graph[k][1][VARS] == ("mdot", "ref_mdot")
    assert b.graph[k][2][VARS] == ("mdot",)
    assert b.graph[1][k][VARS] == ("pressure",)
    assert b.graph[2][k][VARS] == ("pressure",)
    assert b.graph[k][3][VARS] == ("ref_mdot",)


def test_flow_edge_example():
    # noinspection PyTypeChecker
    assert (1, 2, {"comps": (0,), "signify": 1.0}) == flow_edge((1, 2), 0)


def test_flow_graph_to_agr_for_2_junctions_2_comps_has_correct_edges():
    # noinspection PyTypeChecker
    fg = flow_graph(flow_edge((J0 := Junction(), J1 := Junction()), 1, 2))
    a = flow_graph_to_aggregator(fg)

    assert a.graph[J0][1][VARS] == ("Tin",)
    assert a.graph[1][2][VARS] == ("Tin",)
    assert a.graph[2][J1][VARS] == ("Tin",)
    assert a.graph[J1][2][VARS] == ("Tin_minus",)
    assert a.graph[2][1][VARS] == ("Tin_minus",)
    assert a.graph[1][J0][VARS] == ("Tin_minus",)


def test_flow_graph_to_agr_for_2_junction_and_a_virtual_junction_has_correct_edges():
    # noinspection PyTypeChecker
    fg = flow_graph(
        flow_edge((J0 := Junction(), "A"), 1), flow_edge(("A", J1 := Junction()), 2)
    )
    a = flow_graph_to_aggregator(fg)

    assert a.graph[J0][1][VARS] == ("Tin",)
    assert a.graph[1][2][VARS] == ("Tin",)
    assert a.graph[2][J1][VARS] == ("Tin",)
    assert a.graph[J1][2][VARS] == ("Tin_minus",)
    assert a.graph[2][1][VARS] == ("Tin_minus",)
    assert a.graph[1][J0][VARS] == ("Tin_minus",)
