"""
Utilizing the incompressible scheme, this module contains functions for
constructing coolant cycles and adding Kirchhoff constraints to its flow
"""

import logging
from functools import partial, wraps
from typing import Any, Hashable, Sequence, Type

from cytoolz import keymap
from networkx import DiGraph, MultiDiGraph
from networkx.utils import pairwise

from stream import State
from stream.aggregator import (
    VARS,
    Aggregator,
    BaseAgr,
    CalculationGraph,
    ExternalFunctions,
    add_variables,
    vars_,
)
from stream.calculation import Calculation
from stream.calculations import Junction, Kirchhoff, KirchhoffWDerivatives
from stream.calculations.kirchhoff import COMPS
from stream.composition import guess_hydraulic_steady_state
from stream.composition.subsystems import HydraulicStrategyMap, check_gravity_mismatch
from stream.units import Celsius, KgPerS, Pascal
from stream.utilities import summed

__all__ = [
    "flow_edge",
    "flow_graph",
    "flow_graph_to_aggregator",
    "flow_graph_to_agr_and_k",
    "in_parallel",
    "in_series",
    "kirchhoffify",
    "FlowGraph",
]

logger = logging.getLogger("stream.cycle")
MIS_MSG = "all calculations in a `FlowGraph` must have ('Tin', 'Tin_minus', 'pressure') in `indices`."


def _indices_missing(calc, *variables):
    """Returns which variables are missing from the given variables names"""
    missing = []
    if isinstance(calc, Calculation):
        for var in variables:
            try:
                calc.indices(var)
            except KeyError:
                missing.append(var)
    return missing


def _check_missing_tin(f):
    """Decorator for functions that we want to ensure check for missing Tin variables in its positional arguments"""

    @wraps(f)
    def _wrap(*components, **kw):
        for comp in components:
            if missing := _indices_missing(comp, "Tin", "Tin_minus"):
                raise KeyError(f"{comp} is missing {missing} in its 'indices' method and {MIS_MSG}")
        return f(*components, **kw)

    return _wrap


@_check_missing_tin
def in_series(
    *components: Calculation,
    cyclic: bool = False,
    funcs: ExternalFunctions | None = None,
) -> CalculationGraph:
    """
    Construct a serial part of the hydraulic (incompressible) cycle

    Parameters
    ----------
    components: Iterable[Calculation]
        Components (calculations) to be connected
    cyclic: bool
        Should the series be connected in a loop?
    funcs: ExternalFunctions | None
        External functions relating to the components

    Returns
    -------
    agr: CalculationGraph
        Graph whose nodes are serially linked components with edges containing variables = Tin, Tin_minus.
    """

    if len(components) == 2 and cyclic:
        comp1, comp2 = components
        return CalculationGraph(
            DiGraph(
                [
                    (comp1, comp2, vars_("Tin", "Tin_minus")),
                    (comp2, comp1, vars_("Tin", "Tin_minus")),
                ]
            ),
            funcs=funcs,
        )

    G = DiGraph()
    for comp1, comp2 in pairwise(components, cyclic=cyclic):
        G.add_edge(comp1, comp2, variables=("Tin",))
        G.add_edge(comp2, comp1, variables=("Tin_minus",))
    return CalculationGraph(G, funcs=funcs)


@_check_missing_tin
def in_parallel(
    start_comp: Calculation,
    end_comp: Calculation,
    *parallel: Calculation,
    funcs: ExternalFunctions | None = None,
) -> CalculationGraph:
    r"""Construct a parallel part of the hydraulic (incompressible) cycle

    Parameters
    ----------
    start_comp: Calculation
        Component from which all parallel components are fed.
    end_comp: Calculation
        Component into which all parallel components feed.
    parallel: Iterable[Calculation]
        Parallel components
    funcs: ExternalFunctions | None
        External functions relating to the components

    Returns
    -------
    agr: CalculationGraph
        whose graph nodes are linked components
        in parallel between start_comp and end_comp (included), with edges
        containing variables = Tin.

    Notes
    -----
    ``start_comp`` is connected through ``parallel`` to ``end_comp``.
    if no parallel components are given, the returned ``CalculationGraph`` is simply empty.

    """
    G = DiGraph()
    for comp in parallel:
        G.add_edge(start_comp, comp, variables=("Tin",))
        G.add_edge(comp, start_comp, variables=("Tin_minus",))
        G.add_edge(comp, end_comp, variables=("Tin",))
        G.add_edge(end_comp, comp, variables=("Tin_minus",))

    return CalculationGraph(G, funcs=funcs)


def kirchhoffify(
    agr: BaseAgr,
    k: Kirchhoff,
    hydraulic_comps: Sequence[Calculation] = None,
    inertial_comps: Sequence[Calculation] = None,
    ref_mdots: Sequence[Calculation] = None,
    abs_pressure_comps: Sequence[Calculation] = None,
) -> CalculationGraph:
    r"""Given a subset of the given CalculationGraph calculations which adhere to Kirchhoff's laws, construct a
    CalculationGraph which contains Kirchhoff linked properly.

    Parameters
    ----------
    agr: BaseAgr
    k: Kirchhoff
        Calculation which already contains:
    hydraulic_comps: Sequence[Calculation]
        a subset of calculations in agr which interact with Kirchhoff. If it empty, the calculations contained in k's
        flow graph are used.
    inertial_comps: Sequence[Calculation]
        a subset of calculations in agr which require :math:`\ddot{m}` from kirchhoff.
        This requires ``k`` to be KirchhoffWDerivatives.
    ref_mdots: Sequence[Calculation]
        Calculations for which a reference current is desired.
        This list must be a subset of a list already known to ``k`` from its flow-graph.
    abs_pressure_comps: Sequence[Hashable] or None
        Calculations for which the absolute pressure should be calculated.

    Returns
    -------
    agr: CalculationGraph
    """

    a = CalculationGraph(DiGraph(agr.graph), agr.funcs)
    add = partial(add_variables, a.graph)

    for component in hydraulic_comps or k.components.keys():
        add(k, component, "mdot")
        if _indices_missing(component, "pressure"):
            raise KeyError(f"{component} is missing 'pressure' in its 'indices' method and {MIS_MSG}")
        add(component, k, "pressure")

    for component in filter(lambda _n: isinstance(_n, Junction), k.g.nodes):
        add(k, component, "mdot")

    if inertial_comps is not None:
        assert isinstance(k, KirchhoffWDerivatives), (
            f"{type(k)} does not handle inertial components, as it does not index mdot2"
        )
        for component in inertial_comps:
            add(k, component, "mdot", "mdot2")

    if ref_mdots is not None:
        for ref in ref_mdots:
            add(k, ref, "ref_mdot")

    if abs_pressure_comps is not None:
        for component in abs_pressure_comps:
            add(k, component, "p_abs")

    return a


FlowEdge = tuple[Hashable, Hashable, dict[str, Any]]


def flow_graph(*edges: FlowEdge) -> MultiDiGraph:
    r"""A more declarative way to construct a flow graph, to be used in a :class:`~.Kirchhoff` type constructor.

    Parameters
    ----------
    edges: FlowEdge
        These are ``(u, v, d)`` of the graph. Since this is a MultiDiGraph, ``(u, v)`` may be repeated,
        and would mostly be of type :class:`~.Junction` (but any hashable is okay).
        The data ``d`` holds a list of components (under the ``comps`` key) on each edge, its "significance"
        (under ``signify``) and any other datum.

    Returns
    -------
    flow_graph: MultiDiGraph
    """
    return MultiDiGraph(edges)


def flow_edge(
    edge: tuple[Hashable, Hashable],
    *components: Calculation,
    signify: float = 1.0,
    **kwargs,
) -> FlowEdge:
    r"""A tool to make input for :func:`flow_graph` more streamlined.

    .. warning:: Junction weights are **edited** to reflect ``signify`` weighting

    Parameters
    ----------
    edge: tuple[Hashable, Hashable]
        The edge (u, v) to be created.
    components: Sequence[Calculation]
        Components on that edge, sequentially from `u` to `v`.
    signify: float
        The weight this edge carries (or how many equivalent parallel edges it `signifies`.

    Returns
    -------
    edge: tuple[hashable, hashable, dict[str, Any]]
        u, v, edge_data
    """
    src, dst = edge[0], edge[1]

    if isinstance(src, Junction):
        src.weights[components[0]] = signify
    if isinstance(dst, Junction):
        dst.weights[components[-1]] = signify

    return edge[0], edge[1], dict(comps=components, signify=signify) | kwargs


def flow_graph_to_aggregator(f_graph: MultiDiGraph, funcs: ExternalFunctions | None = None) -> CalculationGraph:
    r"""Create a CalculationGraph which stems from a flow_graph.
    Essentially, this function creates CalculationGraphs by using :func:`in_series` for each edge (including the nodes),
    and adding them together.

    Parameters
    ----------
    f_graph: MultiDiGraph
        Flow graph. See :class:`~.Kirchhoff`.
    funcs: ExternalFunctions or None
        External functions which are inserted into the resultant CalculationGraph

    Returns
    -------
    agr: CalculationGraph
        An "Hydraulic" Aggregator input.
    """
    edges = f_graph.edges(data=COMPS, keys=True)
    agr = summed(in_series(u, *comps, v) for u, v, _, comps in edges)
    g = DiGraph(agr.graph)
    # Non-Calculation Single Input Single Output (SISO) junctions are allowed,
    # Here we deal with them, since they can't go into the Aggregator as nodes.
    siso = filter(lambda n: not isinstance(n, Calculation), f_graph.nodes)
    for junction in siso:
        u, v = None, None
        for node, data in (adj := g.adj[junction]).items():
            u = node if "Tin_minus" in data[VARS] else u
            v = node if "Tin" in data[VARS] else v
        assert u and v
        agr += in_series(u, v, cyclic=(u, v) in agr.graph.edges)
        agr.graph.remove_node(junction)
        assert len(adj) == 2, "Virtual nodes cannot be connected by more than 2 edges"
    agr.funcs = funcs
    return agr


def flow_graph_to_agr_and_k(
    f_graph: MultiDiGraph,
    funcs: ExternalFunctions | None = None,
    reference_node: tuple[Hashable, Pascal] = None,
    abs_pressure_comps: Sequence[Hashable] = None,
    inertial_comps: Sequence[Calculation] = None,
    ref_mdots: Sequence[Calculation] = None,
    k_constructor: Type[Kirchhoff] = Kirchhoff,
) -> tuple[Aggregator, Kirchhoff]:
    r"""Create an Aggregator and Kirchhoff objects from a flow graph.

    Essentially, it's just a merger of :func:`flow_graph_to_aggregator` and :func:`kirchhoffify`.

    Parameters
    ----------
    f_graph: MultiDiGraph
        Flow graph.
    funcs: ExternalFunctions or None
        External Aggregator functions
    reference_node: tuple[Hashable, Pascal] or None
        A reference ("ground") node and its absolute pressure
    abs_pressure_comps: Sequence[Hashable] or None
        Calculations for which the absolute pressure should be calculated
    inertial_comps: Sequence[Calculation] or None
        A subset of calculations in agr which require :math:`\ddot{m}` from kirchhoff.
        This requires ``k`` to be KirchhoffWDerivatives.
    ref_mdots: Sequence[Calculation] or None
        Calculations for which a reference current is desired.
        This list must be a subset of a list already known to ``k`` from its flow-graph
    k_constructor: Type[Kirchhoff]
        A Kirchhoff constructor

    Returns
    -------
    agr, k: Aggregator, Kirchhoff
    """
    abs_comps = a if (a := abs_pressure_comps) is not None else ()
    K = k_constructor(f_graph, *abs_comps, reference_node=reference_node)
    agr = flow_graph_to_aggregator(f_graph, funcs=funcs)
    agr = kirchhoffify(
        agr,
        K,
        inertial_comps=inertial_comps,
        ref_mdots=ref_mdots or K.ref_mdots.keys(),
        abs_pressure_comps=abs_comps,
    )
    return Aggregator.from_CalculationGraph(agr), K


class FlowGraph:
    r"""A container for the Aggregator and Kirchhoff objects generated from a flow graph."""

    def __init__(
        self,
        *edges,
        funcs: ExternalFunctions | None = None,
        reference_node: tuple[Hashable, Pascal] = None,
        abs_pressure_comps: Sequence[Hashable] = None,
        inertial_comps: Sequence[Calculation] = None,
        ref_mdots: Sequence[Calculation] = None,
        k_constructor: Type[Kirchhoff] = Kirchhoff,
    ):
        r"""
        Parameters
        ----------
        edges: FlowEdge
            These are ``(u, v, d)`` of the graph. Since this is a MultiDiGraph, ``(u, v)`` may be repeated,
            and would mostly be of type :class:`~.Junction` (but any hashable is okay).
            The data ``d`` holds a list of components (under the ``comps`` key) on each edge,
            its "significance" (under ``signify``) and any other datum.
        funcs: ExternalFunctions or None
            External Aggregator functions
        reference_node: tuple[Hashable, Pascal] or None
            A reference ("ground") node and its absolute pressure
        abs_pressure_comps: Sequence[Hashable] or None
            Nodes for which the absolute pressure should be calculated
        inertial_comps: Sequence[Calculation] or None
            A subset of calculations in agr which require :math:`\ddot{m}` from kirchhoff.
            This requires ``k`` to be KirchhoffWDerivatives.
        ref_mdots: Sequence[Calculation] or None
            Calculations for which a reference current is desired.
            This list must be a subset of a list already known to ``k`` from its flow-graph
        k_constructor: Type[Kirchhoff]
            A Kirchhoff constructor
        """
        f_graph = MultiDiGraph(edges)
        self.aggregator, self.kirchhoff = flow_graph_to_agr_and_k(
            f_graph,
            funcs,
            reference_node,
            abs_pressure_comps,
            inertial_comps,
            ref_mdots,
            k_constructor,
        )

    def guess_steady_state(
        self,
        mdots: dict[Calculation | str, KgPerS],
        temperature: Celsius,
        strategy: HydraulicStrategyMap | None = None,
    ) -> State:
        r"""A guess for a :class:`.Kirchhoff` derived system, in which the mass flow rates are assumed to be known

        Parameters
        ----------
        mdots : dict[Calculation | str, KgPerS]
            Known mass flow rates :math:`\dot{m}` for components in the hydraulic system. Supported Calculations are
            :class:`.DPCalculation` and :class:`.Channel`.
        temperature : Celsius
            Assumed temperature for hydraulic calculations
        strategy : HydraulicStrategyMap | None
            For unknown calculations, pressure drop functions :math:`\Delta p(\dot{m}, T)` may be provided.
            These are used when the Calculation isn't identified as known types or protocols, and failing that,
            the guess is ``0.0``.

        Returns
        -------
        State
            A guess in which pressures are computed from the known flow rates, and the flow rates themselves.
        """
        md = keymap(lambda x: self.aggregator[x] if isinstance(x, str) else x, mdots)
        return guess_hydraulic_steady_state(self.kirchhoff, md, temperature, strategy)

    def check_gravity_mismatch(
        self,
        temperature: Celsius = 10.0,
        strategy: HydraulicStrategyMap | None = None,
        tol: float = 1e-5,
        head: Pascal = 1.0,
    ) -> None:
        r"""A wrapper for :func:`~.check_gravity_mismatch`, see therein for more information"""
        return check_gravity_mismatch(self.kirchhoff, temperature, strategy, tol, head)
