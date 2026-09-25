"""
Oh, how the turns have tabled...
"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from networkx import MultiDiGraph

from stream.aggregator import Aggregator
from stream.calculations.ideal.ideal import LumpedComponent
from stream.calculations.ideal.resistors import LevelHead, Resistor
from stream.calculations.kirchhoff import (
    Junction,
    Kirchhoff,
    KirchhoffWDerivatives,
    build_kcl_matrix,
    build_kvl_matrix,
    build_signed_paths,
    to_graph_for_cycles,
    to_str,
)
from stream.calculations.tank import Environment, Tank
from stream.composition.cycle import flow_edge, flow_graph
from stream.errors import StreamConstructionError, StreamError
from stream.substances import light_water
from stream.units import g as gravity

from .conftest import are_close, pos_medium_floats


def test_a_multigraph_to_graph_for_cycles():
    g = MultiDiGraph()
    g.add_edge("A", "B", name="hi")
    g.add_edge("A", "B", name="hello")
    g.add_edge("A", "B", name="greetings")

    h = to_graph_for_cycles(g)

    assert [data["name"] for data in h.adj["A"].values()] == [
        "hi",
        "hello",
        "greetings",
    ]
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


def test_junction_mixing_with_only_incoming_edges():
    """A dead-end junction fed only by incoming edges gets no Tin_minus edge; the
    missing mapping must count as empty rather than crashing."""
    J = Junction()
    mdot = {1: 1.0, 2: 2.0}
    Tin = {1: 10.0, 2: 0.0}
    are_close(J.calculate([0], mdot=mdot, Tin=Tin), (10 * 1 + 0 * 2) / 3)


def test_junction_mixing_with_only_outgoing_edges():
    """The symmetric dead-end: only outgoing edges, so Tin is absent."""
    J = Junction()
    mdot = {3: -3.0, 4: -4.0}
    Tin_minus = {3: 5.0, 4: 17.0}
    are_close(J.calculate([0], mdot=mdot, Tin_minus=Tin_minus), (3 * 5 + 4 * 17) / 7)


@pytest.fixture(scope="module")
def mock_graph(J) -> MultiDiGraph:
    J0, J1 = J
    g = MultiDiGraph()
    g.add_edge(J0, J1, comps=("A", "B", "C"))
    g.add_edge(J0, J1, comps=("D", "E"))
    g.add_edge(J1, J0, comps=("F",))
    return g


@pytest.fixture(scope="module")
def J():
    return Junction(name="J0"), Junction(name="J1")


@pytest.fixture(scope="module")
def K(mock_graph):
    return Kirchhoff(mock_graph)


Edge = list[str]


def _make_cycle(values: list[Edge]) -> MultiDiGraph:
    g = MultiDiGraph()
    for i, val in enumerate(values):
        g.add_edge(str(i), str((i + 1) % len(values)), 0, comps=val)
    return g


def _exclusive(edgelist: list[Edge]) -> bool:
    return len(set(sum(edgelist, start=[]))) == sum(len(v) for v in edgelist)


graph_sizes = st.shared(st.integers(2, 20), key="size")
names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5)
complists = st.lists(names, unique=True, min_size=1, max_size=4)
edges = st.shared(
    graph_sizes.flatmap(lambda n: st.lists(complists, min_size=n + 1, max_size=n + 1)).filter(_exclusive),
    key="edges",
)
comps = edges.map(lambda v: sum(v, start=[]))
graphs = edges.map(_make_cycle)
subsets = comps.flatmap(lambda x: st.lists(st.sampled_from(x), unique=True))
reference_nodes = st.tuples(graphs.flatmap(lambda x: st.sampled_from(list(x.nodes))), pos_medium_floats)
kirchhoffs = st.builds(lambda x, y, z: Kirchhoff(x, *y, reference_node=z), graphs, subsets, reference_nodes)
k_derivs = st.builds(
    lambda x, y, z: KirchhoffWDerivatives(x, *y, reference_node=z),
    graphs,
    subsets,
    reference_nodes,
)


@given(st.one_of(kirchhoffs, k_derivs))
def test_kirchhoff_variable_names_are_strings(k):
    assert all(isinstance(key, str) for key in k.variables.keys())


def test_kvl_matrix_works_on_mock_graph(K):
    expected_kvl = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, -1.0, -1.0, 0.0]])
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
    res = K.calculate([1, 2, 3], pressure=dict(A=-1.0, B=-1.0, C=-1.0, D=-2.0, E=-1, F=3.0))
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
    abs_pressures = res[K.edges_count :]
    are_close(abs_pressures, expected)


def test_kirchoff_save_fits_known_value_for_one_example(K):
    assert K.save([1, 2, 3]) == {
        "(J0 -> J1, 0)": 1,
        "(J0 -> J1, 1)": 2,
        "(J1 -> J0, 0)": 3,
    }


triplets = st.tuples(pos_medium_floats, pos_medium_floats, pos_medium_floats)


@given(triplets)
def test_agr_of_kirchhoff_load_reverses_save_by_example(K, tpl):
    g = MultiDiGraph()
    g.add_node(K)
    agr = Aggregator(g)
    assert np.allclose(agr.load(agr.save(tpl)), tpl)


def _inertia_loop_kirchhoff(constructor):
    from stream.calculations.ideal.inertia import Inertia

    inertia = Inertia(inertia=100.0, name="Inertia")
    J0, J1 = Junction(name="J0"), Junction(name="J1")
    g = MultiDiGraph()
    g.add_edge(J0, J1, comps=(inertia,))
    g.add_edge(J1, J0, comps=("pump",))
    return constructor(g), inertia


def test_plain_kirchhoff_indices_rejects_unknown_variable_names():
    """K7: indices() fell through to the mdot place for ANY name, so a misrouted
    'mdot2' (or a typo) silently resolved to mdot — turning an inertia into an
    Ohmic resistor. Unknown names must raise KeyError."""
    k, inertia = _inertia_loop_kirchhoff(Kirchhoff)
    assert isinstance(k.indices("mdot", asking=inertia), (int, np.integer))  # served name works
    with pytest.raises(KeyError):
        k.indices("mdot2", asking=inertia)
    with pytest.raises(KeyError):
        k.indices("typo", asking=inertia)


def test_kirchhoff_w_derivatives_still_serves_mdot2():
    """K7: the whitelist must leave KirchhoffWDerivatives, which does serve mdot2,
    working — while still rejecting genuinely unknown names."""
    k, inertia = _inertia_loop_kirchhoff(KirchhoffWDerivatives)
    assert k.indices("mdot2", asking=inertia) != k.indices("mdot", asking=inertia)
    with pytest.raises(KeyError):
        k.indices("typo", asking=inertia)


def test_kirchhoff_rejects_a_disconnected_flow_graph():
    """Two hydraulically independent loops in one graph make the residual over-
    determined; Kirchhoff must reject it clearly, not crash later in compute."""
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("P1",))
    g.add_edge("B", "A", comps=("R1",))
    g.add_edge("C", "D", comps=("P2",))
    g.add_edge("D", "C", comps=("R2",))
    with pytest.raises(ValueError, match="connect"):
        Kirchhoff(g)


def test_kirchhoff_rejects_a_self_loop_edge():
    """A single edge closed on one junction generates no KVL row and a broken junction
    mdot map; Kirchhoff must reject it with a clear message rather than crash later."""
    g = MultiDiGraph()
    g.add_edge("J", "J", comps=("pump", "res"))
    with pytest.raises(ValueError, match="[Ss]elf-loop"):
        Kirchhoff(g)


def test_kirchhoff_rejects_a_reused_component():
    """Reusing one component object on two edges collapses the component index map,
    causing a bare IndexError or silent mdot aliasing; reject it clearly."""
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("X",))
    g.add_edge("B", "A", comps=("X", "C2"))
    with pytest.raises(ValueError, match="X"):
        Kirchhoff(g)

    # The same component twice on a single edge is likewise rejected.
    g2 = MultiDiGraph()
    g2.add_edge("A", "B", comps=("X", "X"))
    g2.add_edge("B", "A", comps=("C2",))
    with pytest.raises(ValueError, match="X"):
        Kirchhoff(g2)


def test_kirchhoff_reports_when_reference_cannot_reach_abs_pressure_target():
    """Anchoring the reference at a dead-end pressurizer node leaves it unable to reach
    abs-pressure targets along flow orientations; report it clearly rather than crashing
    with a raw NetworkXNoPath."""
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("core",))
    g.add_edge("B", "A", comps=("pump",))
    g.add_edge("A", "P", comps=("surge_line",))  # dead-end pressurizer branch
    with pytest.raises(ValueError, match="reach"):
        Kirchhoff(g, "core", reference_node=("P", 1.55e7))


def _line_graph():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("c1",), signify=1.0)
    g.add_edge("C", "B", comps=("c2",), signify=1.0)
    return g


def test_signed_paths_flip_sign_against_orientation():
    g = _line_graph()
    order = dict(c1=0, c2=1)
    m = build_signed_paths(g, order, "A", lambda c: None, "C").toarray()
    assert m.tolist() == [[1.0, -1.0]]


def test_signed_paths_component_target_stops_mid_edge():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("c1", "c2", "c3"), signify=1.0)
    order = dict(c1=0, c2=1, c3=2)
    edge_of = {"c1": ("A", "B", 0), "c2": ("A", "B", 0), "c3": ("A", "B", 0)}
    m = build_signed_paths(g, order, "A", edge_of.__getitem__, "c2").toarray()
    assert m.tolist() == [[1.0, 1.0, 0.0]]


def test_signed_paths_pick_the_traversed_parallel_edge():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("c1",), signify=1.0)
    g.add_edge("A", "B", comps=("c2", "c3"), signify=1.0)
    order = dict(c1=0, c2=1, c3=2)
    edge_of = {"c1": ("A", "B", 0), "c2": ("A", "B", 1), "c3": ("A", "B", 1)}
    m = build_signed_paths(g, order, "A", edge_of.__getitem__, "c3").toarray()
    assert m[0, 0] == 0.0 and m[0, 1] == 1.0 and m[0, 2] == 1.0


def test_closed_network_residual_layout_is_unchanged():
    j1, j2 = Junction(name="a"), Junction(name="b")
    r1, r2 = Resistor(1.0, name="r1"), Resistor(2.0, name="r2")
    g = flow_graph(flow_edge((j1, j2), r1), flow_edge((j2, j1), r2))
    k = Kirchhoff(g)
    out = k.calculate(np.array([2.0, 3.0]), pressure={r1: 5.0, r2: 7.0})
    kcl = (build_kcl_matrix(g) @ np.array([2.0, 3.0]))[:-1]
    assert out.shape == (2,)
    assert out[0] == pytest.approx(kcl[0])


def test_signed_paths_stop_at_the_inlet_of_a_component_target():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("c1", "c2"), signify=1.0)
    g.add_edge("C", "B", comps=("c3",), signify=1.0)
    order = dict(c1=0, c2=1, c3=2)
    edge_of = {"c1": ("A", "B", 0), "c2": ("A", "B", 0), "c3": ("C", "B", 0)}
    forward = build_signed_paths(g, order, "A", edge_of.__getitem__, "c2", to_inlet=True).toarray()
    assert forward.tolist() == [[1.0, 0.0, 0.0]]

    backward = build_signed_paths(g, order, "C", edge_of.__getitem__, "c2", to_inlet=True).toarray()
    assert backward.tolist() == [[0.0, -1.0, 1.0]]


class _Head(LumpedComponent):
    """A hydrostatic head fed by the level of the surface it hangs from."""

    def __init__(self, fluid, z_connection, level0, sign=1.0, name="head"):
        self.name = name
        self._rho = fluid.density
        self.z = z_connection
        self.level0 = level0
        self.sign = sign

    def dp_out(self, *, Tin, level=None, **_):
        return self.sign * self._rho(Tin) * gravity * ((self.level0 if level is None else level) - self.z)


def _open_line():
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0)
    env = Environment(name="env")
    j = Junction(name="mid")
    head = _Head(light_water, 0.0, 4.0, name="head")
    hole = Resistor(1.0, name="hole")
    g = flow_graph(flow_edge((tank, j), head), flow_edge((j, env), hole))
    return tank, env, g, head, hole


def test_surface_network_is_square_and_keeps_interior_conservation():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, surface_nodes={tank: None, env: None})
    out = k.calculate(np.array([1.0, 3.0]), pressure={head: 10.0, hole: -10.0})
    assert out.shape == (2,)
    assert out[0] == pytest.approx(1.0 - 3.0)
    assert out[1] == pytest.approx(0.0)


def test_surface_pressure_difference_enters_the_path_row():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, surface_nodes={tank: 201325.0, env: 101325.0})
    out = k.calculate(np.array([1.0, 1.0]), pressure={head: 0.0, hole: 0.0})
    assert out[1] == pytest.approx(0.0 - (101325.0 - 201325.0))


def test_surface_taps_carry_incidence_signs():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, surface_nodes={tank: None, env: None})
    assert k.surface_taps(tank) == {head: -1.0}
    assert k.surface_taps(env) == {hole: 1.0}


def test_non_inventory_surface_node_is_rejected():
    j1, j2 = Junction(name="a"), Junction(name="b")
    g = flow_graph(flow_edge((j1, j2), Resistor(1.0, name="r")))
    with pytest.raises(Exception, match="surface"):
        Kirchhoff(g, surface_nodes={j1: 101325.0})


def test_reference_defaults_to_first_surface_and_pabs_traverses_heads():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, hole, surface_nodes={tank: 101325.0, env: None})
    p = {head: 100.0, hole: -40.0}
    v = np.array([1.0, 1.0, 0.0])
    out = k.calculate(v, pressure=p)
    assert out[-1] == pytest.approx(101325.0 + 100.0 - 0.0)


def test_derivative_kirchhoff_accepts_surfaces():
    tank, env, g, head, hole = _open_line()
    k = KirchhoffWDerivatives(g, surface_nodes={tank: None, env: None})
    assert len(k) == 4
    out = k.calculate(np.array([1.0, 3.0, 0.1, 0.2]), pressure={head: 10.0, hole: -10.0})
    assert out[0] == pytest.approx(0.1) and out[1] == pytest.approx(0.2)


def test_a_surface_pressure_schedule_is_read_at_the_given_time():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, surface_nodes={tank: lambda t: 101325.0 + 1000.0 * t, env: 101325.0})
    out = k.calculate(np.array([1.0, 1.0]), pressure={head: 0.0, hole: 0.0}, t=2.0)
    assert out[1] == pytest.approx(2000.0)


def test_a_surface_pressure_schedule_without_a_time_is_reported():
    tank, env, g, head, hole = _open_line()
    k = Kirchhoff(g, surface_nodes={tank: lambda t: 101325.0, env: 101325.0})
    with pytest.raises(StreamError, match="t="):
        k.calculate(np.array([1.0, 1.0]), pressure={head: 0.0, hole: 0.0})


def test_a_scheduled_first_surface_cannot_ground_absolute_pressure():
    tank, env, g, head, hole = _open_line()
    with pytest.raises(StreamConstructionError, match="reference_node"):
        Kirchhoff(g, hole, surface_nodes={tank: lambda t: 101325.0, env: None})


def test_a_surface_node_outside_the_graph_is_rejected():
    tank, env, g, head, hole = _open_line()
    stray = Environment(name="stray")
    with pytest.raises(StreamConstructionError, match="surface"):
        Kirchhoff(g, surface_nodes={tank: None, stray: None})


def test_a_mis_signed_level_head_at_a_surface_is_rejected():
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0)
    env = Environment(name="env")
    j = Junction(name="mid")
    head = LevelHead(light_water, 0.0, 4.0, sign=-1.0, name="head")
    hole = Resistor(1.0, name="hole")
    g = flow_graph(flow_edge((tank, j), head), flow_edge((j, env), hole))
    with pytest.raises(StreamConstructionError, match=r"sign -1\b.*sign \+1"):
        Kirchhoff(g, surface_nodes={tank: None, env: None})


def _marked_open_line():
    tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0)
    env = Environment(name="env")
    j = Junction(name="mid")
    head = _Head(light_water, 0.0, 4.0, name="head")
    hole = Resistor(1.0, name="hole")
    g = flow_graph(flow_edge((tank, j), head), flow_edge((j, env), hole, ref_mdot_for=(tank,)))
    return tank, env, g, head, hole


def test_a_surface_node_asking_for_a_reference_current_gets_the_marked_edge():
    tank, env, g, head, hole = _marked_open_line()
    k = Kirchhoff(g, surface_nodes={tank: None, env: None})
    assert k.indices("ref_mdot", asking=tank) == k.variables[k.component_edge(hole)]
    assert k.indices("mdot", asking=tank) == {head: k.variables[k.component_edge(head)]}


def test_a_surface_node_asking_for_absolute_pressure_is_not_served_a_flow_map():
    tank, env, g, head, hole = _marked_open_line()
    k = Kirchhoff(g, hole, surface_nodes={tank: None, env: None})
    with pytest.raises(KeyError):
        k.indices("p_abs", asking=tank)
