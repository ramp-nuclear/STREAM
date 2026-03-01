from collections import defaultdict
from typing import Protocol, Iterable, Sequence

from more_itertools import unique_everseen
from networkx import DiGraph

from stream.calculation import Calculation
from stream.units import Name, FunctionOfTime, Place
from stream.utilities import uppercase_numeric_only, offset

VARS = "variables"
ExternalFunctions = dict[Calculation, dict[Name, FunctionOfTime]]


class BaseAgr(Protocol):
    """A protocol for data objects similar to the :class:`~.stream.aggregator.Aggregator`
    class.
    These can be used in most places where aggregator creation tools are necessary.

    """

    graph: DiGraph
    funcs: ExternalFunctions | None

    @classmethod
    def connect(
        cls,
        a: "BaseAgr",
        b: "BaseAgr",
        *edges: tuple[Calculation, Calculation, Iterable[Name]],
    ) -> "BaseAgr":
        """

        Parameters
        ----------
        a: BaseAgr
            First object to connect
        b: BaseAgr
            Second object to connect
        edges: tuple[Calculation, Calculation, Iterable[Name]]
            Edges to connect between the two graphs of the objects with connection data.

        """
        raise NotImplementedError

    def __add__(self, other: "BaseAgr") -> "BaseAgr":
        raise NotImplementedError


def non_unique_calculations(g: DiGraph) -> dict[str, list[Calculation]]:
    """Returns which calculations in the graph that are not uniquely named.

    Parameters
    ----------
    g: DiGraph

    Returns
    -------
    dict[str, list[Calculation]]
        A dictionary whose keys are non-unique names and its values are the non-unique
        calculations

    """
    d = defaultdict(list)
    for calc in g.nodes:
        d[calc.name].append(calc)
    return {key: val for key, val in d.items() if len(val) > 1}


def draw_aggregator(graph: DiGraph, node_options=None, edge_options=None):
    r"""
    Draw an Aggregator's graph. This is useful for presentation and
    system design.

    Creates a Matplotlib figure, so ``plt.show()`` should be called.

    Parameters
    ----------
    graph: DiGraph
        The Aggregator's graph to be drawn
    node_options: dict
        Options to change default behavior.
    edge_options: dict
        Options to change default edge label behavior.

    See Also
    --------
    networkx.draw_networkx,
    networkx.draw_networkx_edge_labels

    """
    node_options = node_options or {}
    edge_options = edge_options or {}
    from matplotlib import rc_context
    import networkx as nx

    # noinspection SpellCheckingInspection
    with rc_context({r"text.usetex": False}):
        labels = {n: uppercase_numeric_only(type(n).__name__) for n in graph}
        default = dict(
            pos=nx.circular_layout(graph),
            labels=labels,
            node_size=1200,
            node_color="k",
            font_color="w",
            font_family="serif",
            font_size=12,
            node_shape="s",
        )
        kwargs = default | node_options
        nx.draw_networkx(graph, **kwargs)

        edge_labels = {(u, v): d for u, v, d in graph.edges(data=VARS)}
        default = dict(label_pos=0.2, edge_labels=edge_labels, font_size=10)
        edge_kwargs = default | edge_options
        nx.draw_networkx_edge_labels(graph, kwargs["pos"], **edge_kwargs)


def partition(nodes: Sequence[Calculation]) -> tuple[dict[Calculation, slice], int]:
    """
    Go over all nodes in the graph, partitioning the vector.

    Returns
    -------
    Partitions, length of the aggregator vector
    """
    index = 0
    sections = {node: slice(index, index := index + len(node)) for node in nodes}
    return sections, index


def map_externals(
    edges: Iterable[tuple[Calculation, Calculation, Iterable[str]]],
    sections: dict[Calculation, slice],
) -> dict[Calculation, dict[str, dict[Calculation, Place]]]:
    """
    Go over every edge in the graph, assigning every calculation with its
    outer variables.
    """
    external = {}
    for u, v, var_names in edges:
        position = sections[u].start
        for name in var_names:
            places = offset(u.indices(name, asking=v), position)
            external.setdefault(v, {}).setdefault(name, {})
            if isinstance(places, dict):
                external[v][name].update(places)
            else:
                external[v][name][u] = places
    return external


def vars_(*v):
    return {VARS: tuple(v)}


def add_variables(graph: DiGraph, source_calc: Calculation, target_calc: Calculation, *added_vars: str) -> None:
    r"""Add variables to the edge (source_calc,target_calc) in a graph.
    If the edge (source_calc,target_calc) doesn't exist, create it.

    Parameters
    ----------
    graph: DiGraph
        The calculation graph.
    source_calc: Calculation
        The source node.
    target_calc: Calculation
        The target node.
    *added_vars: str
        Variables to be added to the edge (source_calc, target_calc) in the graph.

    Returns
    -------
    None.

    """

    if (source_calc, target_calc) in graph.edges:
        edge_vars = list(graph[source_calc][target_calc][VARS])
        graph[source_calc][target_calc][VARS] = tuple(unique_everseen(edge_vars + list(added_vars)))
    else:
        graph.add_edge(source_calc, target_calc, variables=tuple(unique_everseen(added_vars)))
