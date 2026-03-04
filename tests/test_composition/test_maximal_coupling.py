from stream.aggregator import VARS
from stream.calculations import HeatExchanger, Pump
from stream.composition.maximal_coupling import maximally_coupled
from tests.test_general.test_aggregator import add


def test_maximally_coupled_creates_a_correct_graph():
    hx, p = HeatExchanger(25.0), Pump()

    pump_extras = {"pressure", "mdot0"}
    agr = maximally_coupled(hx, p, exclude=lambda x: x in pump_extras)
    assert agr.graph[hx][p][VARS] == ("Tin", "Tin_minus")
    assert agr.graph[p][hx][VARS] == ("Tin", "Tin_minus")

    agr = maximally_coupled(hx, p, exclude=lambda v: v in pump_extras | {"Tin_minus"})
    assert agr.graph[hx][p][VARS] == ("Tin",)
    assert agr.graph[p][hx][VARS] == ("Tin",)


def test_maximally_coupled_for_calcs_with_no_variable_overlap_is_edgeless_graph():
    agr = maximally_coupled(Pump(), add)
    assert len(agr.graph.nodes) == 2
    assert len(agr.graph.edges) == 0


def test_maximally_coupled_for_calcs_with_only_some_variable_overlap_is_only_partially_connected():

    hx, p = HeatExchanger(25.0), Pump()
    agr = maximally_coupled(p, add, hx)
    assert len(agr.graph.nodes) == 3
    assert len(agr.graph.edges) == 2
