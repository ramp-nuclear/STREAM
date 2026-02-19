"""
In the case of incompressible flow, Kirchhoff's rules dictate flow behavior.
This calculation receives a graph containing calculations representing real
system objects as nodes, and connected by edges depicting the connected cycles.

.. attention::
    This module uses the assumption that "regular" Calculations lie on the
    edges of a flow map (or electric circuit), whose nodes are
    Junction Calculations.

.. note::
    Virtual (SISO) junctions may be created as needed, and may be any hashable.

"""
import logging
from itertools import chain, count, takewhile
from typing import Any, Callable, Hashable, Sequence, Iterable

import networkx as nx
import numpy as np
from cytoolz import keymap
from networkx import Graph, MultiDiGraph, MultiGraph
from networkx.utils import pairwise
from scipy.sparse import csr_matrix, dok_matrix

from stream import Calculation
from stream.units import Array1D, Celsius, KgPerS, Name, Pascal, Place
from stream.utilities import concat, STREAM_DEBUG


COMPS = "comps"
logger = logging.getLogger("stream.kirchhoff")


class Kirchhoff(Calculation):
    """Dictates flow in a given circuit for an incompressible liquid"""

    def __init__(self, graph: MultiDiGraph, *abs_pressure_comps: Hashable,
                 reference_node: tuple[Hashable, Pascal] = None,
                 name: str = 'Kirchhoff'):
        r"""
        Parameters
        ----------
        graph: MultiDiGraph
            Containing calculations representing real system objects as comps in edges, whose nodes are Junctions
            (If multiple entries are involved) or junction-names which are strings.
        abs_pressure_comps: Sequence[Hashable]
            If a component needs to know the absolute pressure, such information may be provided
            (abs. pressure at its set upwind end) by the Kirchhoff calculation. Such components must be given here.
        reference_node: tuple[Hashable, Pascal]
            For the absolute pressure to be computed, a reference pressure ("ground") must be provided.
            This reference must be given at some node of the graph.
        name: str or None
            Calculation's name


        Notes
        -----
        This object should be treated as frozen once initialized, because it performs some
        pre-computation during initialization.

        Also notice that the flow graph cannot be the same graph received by the Aggregator, for two reasons:

        1. Kirchhoff is itself a calculation which should be present in The Aggregator graph, which is thus incomplete.

        2. While this graph requires the calculations to be contained on the
           edges, the Aggregator requires calculations as nodes to pass
           variables to each other through the edges.
           In the final Aggregator graph, Kirchhoff will itself point to all hydraulics-related nodes.

        See Also
        --------
        Junction
        """
        self.name = name
        self.g = graph

        if reference_node and reference_node[0] not in graph:
            raise KeyError(f"The reference node {reference_node} wasn't in the graph")
        if difference := (set(abs_pressure_comps) - set(self._edge_components)):
            raise KeyError("Some of the absolute pressure components were not in the "
                           f"graph: {difference}")

        self.components = dict(zip(self._edge_components, count()))
        self._var_book = {comp: i for i, comp in self._edge_num_components}
        self._edge_book = dict(zip(graph.edges(keys=True), count()))

        self._save_names = {str(node): node for node in graph.nodes}

        self.nodes_count = self.g.number_of_nodes()
        self._n = self.edges_count = self.g.number_of_edges()

        self._edge_order = tuple(graph.edges)
        self._kcl = build_kcl_matrix(graph)
        self._kvl = build_kvl_matrix(graph, self.components)
        self.abs_pressure_comps = abs_pressure_comps or ()
        self.ref_node, self.ref_pressure = reference_node or (None, [])
        self._abs_pressure_book = {('p_abs', comp): i + self._n for i, comp
                                   in enumerate(self.abs_pressure_comps)}
        self.ref_mdots = {asks: self._edge_book[(u, v, k)]
                          for u, v, k, asking in
                          graph.edges(data='ref_mdot_for', keys=True)
                          if asking is not None for asks in asking}

        self._abs_matrix = build_paths(
            graph, self.components, self.ref_node, self._component_edge,
            *abs_pressure_comps
            )

        self._vars = keymap(to_str, self._edge_book | self._abs_pressure_book)

        logger.log(
            STREAM_DEBUG,
            f"New Kirchhoff with {self.nodes_count} nodes and {self.edges_count} edges"
            )

    @property
    def _edge_components(self) -> Iterable:
        """Iterable of the components on the edges in the graph.

        """
        yield from (v[1] for v in self._edge_num_components)

    @property
    def _edge_num_components(self) -> Iterable[tuple[int, Any]]:
        """Iterable of the edge numbers and their components.

        """
        view = self.g.edges(data=COMPS, keys=True)
        for i, (_, _, _, comps) in enumerate(view):
            for comp in comps:
                yield i, comp

    # noinspection PyMethodOverriding
    def calculate(self, variables: Sequence[KgPerS], *,
                  pressure) -> Array1D:
        pressure = np.fromiter((pressure[comp] for comp in self.components),
                               dtype=float, count=len(self.components))
        abs_eqs = (self.ref_pressure + self._abs_matrix @ pressure
                   - np.asarray(variables[self._n:]))

        return concat(
            (self._kcl @ np.asarray(variables[:self._n]))[:-1],
            self._kvl @ pressure,
            abs_eqs)

    def indices(self, variable: Name, asking=None
                ) -> Place | dict[Calculation, Place]:
        r"""For a given variable name, return its position in the vector.

        - If the calculation asking is :class:`.Junction` which is present at
          a node, then the :math:`\dot{m}` places of the closest components
          on the connected edges are returned.
        - If the variable requested is ``p_abs``, then the absolute pressure
          place is returned for the asking calculation.

        Parameters
        ----------
        variable: Name
            Name of requested variable
        asking: Calculation or None
            What calculation is asking for the indices?

        Returns
        -------
        indices: Place or dict[Calculation, Place]
            The place in which the calculation uses the variable, or a dictionary with variable names related to this
            name and their places.
        """
        if isinstance(asking, Junction):
            return _comps_closest(asking, self.g, self._var_book)
        if variable == "p_abs":
            return self._abs_pressure_book[("p_abs", asking)]
        if variable == "ref_mdot":
            return self.ref_mdots[asking]
        return self._var_book[asking]

    @property
    def mass_vector(self) -> Sequence[bool]:
        return np.zeros(len(self), dtype=bool)

    def __len__(self) -> int:
        """
        All edges in graph are needed for KCL, and Kirchhoff calculates E = N + L - 1 equations, where
        E = Number of Edges
        N = Number of Nodes
        L = Number of Loops
        Returns E
        """
        return self.edges_count + len(self.abs_pressure_comps)

    @property
    def variables(self) -> dict[str, Place]:
        """
        The variables contained herein are currents at all edges and absolute pressures.
        The currents are represented by a str of a  three-tuple: (u, v, k), where
        (u, v) are the junctions (tail, head), and k the key depicting the specific
        edge of (u, v), by default integers in order of creation.
        Absolute pressures are two-tuples of 'p_abs' and the component name.
        """
        return self._vars

    def component_edge(self, component: Hashable) -> str:
        r"""Returns the edge in which ``component`` is embedded

        Parameters
        ----------
        component: Hashable
            Requested component

        Returns
        -------
        str
            Edge in the flow graph, see :meth:`variables`
        """
        return to_str(self._component_edge(component))

    def _component_edge(self, component: Hashable) -> tuple[Any, Any, int]:
        return tuple(self._edge_book.keys())[self._var_book[component]]

    @property
    def variables_by_type(self) -> dict[str, Place]:
        return dict(mdot=slice(0, self._n), abs_pressure=slice(self._n, len(self)))

    def loop_components(self, i: int) -> list[Calculation]:
        """Return loop ``i``'s components"""
        return [c for c, j in self.components.items() if self._kvl[i, j] != 0]

    def kvl_errors(self, component_dps: Pascal) -> Pascal:
        r"""Returns loop summations (KVL) of pressure drops :math:`\sum_\text{loop}\Delta p`

        Parameters
        ----------
        component_dps : Pascal
            A vector of pressure drops per component, ordered as ``self.components``
        """
        return (self._kvl @ component_dps).flatten()


class _VirtualNode:
    pass


def to_graph_for_cycles(g: MultiGraph) -> Graph:
    r"""
    A MultiGraph is not supported for the NetworkX cycle_basis function. Since
    it is also the easiest interpretation of a circuit scheme, the need for
    a transformation arises. The idea is that any MultiEdge may be split
    without affecting the overall cycles:

    Say this is the graph and transformation::

        A ----(0)----- B    -->    A ---- i ----- B
        |_____(1)______|    -->    |_____ j ______|

    On the left, a MultiEdge, which has been split on the right into
    two additional nodes, ``i, j``. The returned cycle would be
    ``[A, i, B, j]``. Removing those new nodes yields the desired ``[A, B]``.
    Note that ``i, j`` are _VirtualNodes.

    Parameters
    ----------
    g: MultiGraph

    Returns
    -------
    graph: Graph
        object whose MultiEdges have been transformed as explained above
    """
    m = Graph()
    for (u, v, k, data) in g.edges(data=True, keys=True):
        m.add_edge(u, vn := _VirtualNode(), **data)
        m.add_edge(vn, v)
    return m


def build_kvl_matrix(g: MultiDiGraph, comps_order) -> csr_matrix:
    r"""
    Writing the KVL (Kirchhoff Voltage Law) equations such that the
    equations are written as:

    .. math:: \mathbf{M}\vec{\Delta p} = 0

    Where M denotes which loops contain which nodes, pressure
    is the vector of all pressures at nodes (dimension N). L equations are
    given by KVL.

    Notes
    -----
    An explanation of methodology is due. Say a multigraph has nodes junctions
    A, B, and calculations C1, C2 which lie on the edges
    s.t.::

        A ->- [C1] ->- B ->- [C2] ->- A

    Now, there is one loop, thus :math:`\Delta p_{C1}= - \Delta p_{C2}`, Where
    the pressure difference is given in regard to the `positive flow
    direction` (here, left-to-right).

    A simple graph is produced, writing::

        A ->- [C1] ->- i ->- B ->- [C2] ->- j ->- A

    Where :math:`i,j` are _VirtualNodes. The edges [u, v, k]
    (reads "an edge from node u to node v containing data k") and resultant
    values in the [loops, components] matrix are then:

    - [A, i, C1] - the sign of [loop 0, C1] will be +1
    - [i, B]
    - [B, j, C2] - the sign of [loop 0, C2] will be +1
    - [j, A]

    The resultant eq. is indeed :math:`\Delta p_{C1}+\Delta p_{C2}=0`

    Examples
    --------
    Let's see the above scenario in the "wild":

    >>> g = MultiDiGraph()
    >>> g.add_edge('A', 'B', comps=('C1',))
    0
    >>> g.add_edge('B', 'A', comps=('C2',))
    0
    >>> kvl = build_kvl_matrix(g, dict(C1=0, C2=1)).toarray()
    >>> kvl *= kvl[0, 0] # The answer is given up to a sign change
    >>> kvl
    array([[1, 1]])

    If the opposite positive direction is chosen for the second edge,
    the value is flipped properly:

    >>> g = MultiDiGraph()
    >>> g.add_edge('A', 'B', comps=('C1',))
    0
    >>> g.add_edge('A', 'B', comps=('C2',))
    1
    >>> kvl = build_kvl_matrix(g, dict(C1=0, C2=1)).toarray()
    >>> kvl *= kvl[0, 0] # The answer is given up to a sign change
    >>> kvl
    array([[ 1, -1]])


    See Also
    --------
    to_graph_for_cycles
    """
    h = to_graph_for_cycles(g)
    loops = nx.cycle_basis(h)
    mat = dok_matrix((len(loops), len(comps_order)), dtype=int)
    for i, loop in enumerate(loops):
        for edge in pairwise(loop, cyclic=True):
            comps = h.edges[edge].get(COMPS, ())
            sign = isinstance(edge[0], _VirtualNode) - isinstance(edge[1], _VirtualNode)
            for comp in comps:
                mat[i, comps_order[comp]] = sign
    return mat.tocsr()


def build_kcl_matrix(g: MultiDiGraph) -> csr_matrix:
    r"""
    Writing the KCL (Kirchhoff Current Law) equations such that the equations are written as:

    .. math:: \mathbf{M}\vec{\dot{m}} = 0

    Where M denotes which edges connect to which node (+1 for incoming, -1 for outgoing,
    also called the incidence matrix), :math:`\vec{\dot{m}}` is the vector of all flows in edges (dimension E).
    N equations are given by KCL.

    Notes
    -----
    The incidence matrix is weighted by the ``signify`` keyword in each edge.
    """
    return nx.incidence_matrix(g, oriented=True, weight="signify")


def build_paths(g: MultiDiGraph, comps_order, source, component_edge,
                *targets) -> csr_matrix:
    r"""Building a matrix containing the path (by component) from a source node to target *components*.

    """
    m = dok_matrix((len(targets), len(comps_order)))

    for i, target in enumerate(targets):
        if target in g:
            path = nx.dijkstra_path(g, source=source, target=target)
            comps_in_edge = ()
        else:
            u, v, k = edge = component_edge(target)
            # First, build the path to the u-node
            path = nx.dijkstra_path(g, source=source, target=u)
            # Then, from u along edge, add components until target is reached
            comps_in_edge = takewhile(lambda c: c is not target, g.edges[edge][COMPS])

        comps_from_path = (
            comp for edge in pairwise(path) for comp in g.edges[(*edge, 0)][COMPS]
            )

        comps = chain(comps_from_path, comps_in_edge)
        m[i, [comps_order[comp] for comp in comps]] = 1
    return m.tocsr()


class Junction(Calculation):
    """
    A junction calculation should be used anywhere several hydraulic inputs and outputs meet.
    Temperature is mixed according to incoming mass currents.
    """

    _counter = count()

    def __init__(self, name: str = None, weights: dict[Calculation, float] = None):
        r"""

        Parameters
        ----------
        name : str
         Calculation's name
        weights : dict[Calculation, float] or None
         Calculations which lie on edges whose weight (also known as the ``signify``
         keyword, see :func:`~stream.composition.cycle.flow_edge`) differs from 1 must
         be specified so that the correct total incoming mass flows are considered.
        """
        self._i = next(Junction._counter)
        self.weights = weights if weights is not None else {}
        self.name = name or f"J{self._i}"

    def indices(self, variable: Name, asking=None) -> Place:
        allowed = {"Tin", "Tin_minus"}
        if variable not in allowed:
            raise KeyError(f"Junction indices raise for everything but {allowed}, which does not include the variable {variable} or the asking {asking}")
        return 0

    @property
    def mass_vector(self) -> Sequence[bool]: return False,

    def __len__(self) -> int: return 1

    @property
    def variables(self) -> dict[str, Place]: return dict(Tin=0)

    # noinspection PyMethodOverriding
    def calculate(self, variables: Sequence[Celsius],
                  *, Tin, Tin_minus=None, mdot) -> Array1D:
        r"""Computes divergence from total mixing of temperatures in junction,
        defined as

        .. math::
            T = \frac{\sum\dot{m}_\text{in}T_\text{in}}{\sum\dot{m}_\text{in}}

        Parameters
        ----------
        variables: Sequence[Celsius]
            Inlet temperature variable
        Tin, Tin_minus: dict[Calculation, Celsius]
            Mapping from components on connected edges to inlet temperatures,
            depending on :math:`\text{sign}(\dot{m})`
        mdot: dict[Calculation, KgPerS]
            Mapping from components on connected edges to associated mass
            currents

        Returns
        -------
        out: Array1D
            Divergence of ``variables`` from the computed total mixing
        """

        def _f(_g: Callable[[Calculation, float, float], float]) -> float:
            return (sum(_g(k, v, T) for k, T in Tin.items()
                        if (v := mdot[k]) >= 0)
                    - sum(_g(k, v, T) for k, T in Tin_minus.items()
                          if (v := mdot[k]) < 0)
                    )

        incoming_mdot = _f(lambda k, md, T: md * self.weights.get(k, 1.))
        weighted_Tin = _f(lambda k, md, T: md * T * self.weights.get(k, 1.))
        weighted_Tin /= incoming_mdot or 1.0

        return weighted_Tin - np.asarray(variables)


def to_str(key: str | tuple) -> str:
    """Idempotent stringification of keys in a CalcState's keys.

    Parameters
    ----------
    key: Named | tuple[Named, Named, int] | tuple[Named, Named, int, str]
        Either something that can be str-ed or a triplet from Kirchoff, who always has to be special.

    Returns
    -------
    str

    """
    if isinstance(key, tuple):
        match key:
            case var, node:
                return f"({var} of {node})"
            case a, b, i:
                return f'({a} -> {b}, {i})'
            case a, b, i, "mdot2":
                return f'({a} -> {b}, mdot2 {i})'
            case _:
                raise ValueError(f"The tuple has to be of a very specific form. Not {key}")
    else:
        return str(key)


def _comps_closest(j: Junction, g: MultiDiGraph, var_book) -> dict:
    def _filter(u, v, _):
        return (u is j) or (v is j)

    sub = nx.subgraph_view(g, filter_edge=_filter)
    return {(comp := comps[0 if u is j else -1]): var_book[comp]
            for u, v, comps in sub.edges(data=COMPS)}


class KirchhoffWDerivatives(Kirchhoff):
    r"""A Kirchhoff Calculation containing :math:`\ddot{m}`

    While Kirchhoff contains the mass current :math:`\dot{m}`, this subclass
    adds also the next derivative :math:`\ddot{m}`, which is useful for
    calculations containing flow inertia terms.

    The Kirchhoff equations are now solved along with
    :math:`\frac{d\dot{m_i}}{dt} = \ddot{m_i}`
    Where indices[:math:`\ddot{m_i}`] = indices[:math:`\dot{m_i}`] + :math:`n`.

    See Also
    --------
    ~stream.calculations.inertia.Inertia
    """

    def __init__(self, graph: MultiDiGraph, *abs_pressure_comps: Hashable,
                 reference_node: tuple[Hashable, Pascal] = None,
                 name: str = 'Kirchhoff'):
        super().__init__(graph, *abs_pressure_comps,
                         reference_node=reference_node, name=name)
        self._n_ = super().__len__()

    def indices(self, variable: Name, asking=None) -> Place:
        if variable == "mdot2":
            return super().indices("mdot", asking=asking) + self._n_
        return super().indices(variable, asking=asking)

    def __len__(self): return self._n_ + self._n

    @property
    def mass_vector(self) -> Sequence[bool]:
        M = np.zeros(len(self), dtype=bool)
        M[:self._n] = True
        return M

    @property
    def variables(self) -> dict[str, Place]:
        mdots = self._edge_book
        mdots2 = {(*k, "mdot2"): v + self._n_ for k, v in mdots.items()}
        return super().variables | keymap(to_str, mdots2)

    def calculate(self, variables: Sequence[float], *,
                  pressure) -> Array1D:
        super_vars = variables[:self._n_]
        mdots2 = variables[-self._n:]
        algebraic = super().calculate(super_vars, pressure=pressure)
        return concat(mdots2, algebraic)

    @property
    def variables_by_type(self) -> dict[str, Place]:
        return dict(
            mdot=slice(0, self._n),
            abs_pressure=slice(self._n, self._n_),
            mdot2=slice(self._n_, len(self)),
            )
