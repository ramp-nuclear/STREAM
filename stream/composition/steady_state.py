"""Steady-state initial guesses for a flow network wall-coupled to fuel plates."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from networkx import Graph, connected_components

from stream.aggregator import VARS, Aggregator
from stream.calculation import Calculation
from stream.calculations import (
    Fuel,
    Junction,
    Kirchhoff,
    PointKinetics,
)
from stream.state import DictState, State
from stream.units import Second

__all__ = [
    "Blocks",
    "decompose",
    "solve_block",
]

THERMAL_VARIABLES = frozenset(
    {"T_left", "T_right", "T_top", "T_bottom", "h_left", "h_right", "h_top", "h_bottom"}
)


@dataclass(frozen=True)
class Blocks:
    """The partition :func:`decompose` finds.

    Attributes
    ----------
    hydraulic: frozenset[Calculation]
        The Kirchhoff, every component on its flow graph, and every junction.
    thermal: tuple[frozenset[Calculation], ...]
        One set per wall-coupled cluster of fuels and channels. Channels belong to
        their cluster and to the hydraulic block.
    kinetics: tuple[Calculation, ...]
        Point-kinetics nodes that supply ``power`` to a fuel.
    unassigned: tuple[Calculation, ...]
        Every other node; the seed leaves these to the caller.
    """

    hydraulic: frozenset[Calculation]
    thermal: tuple[frozenset[Calculation], ...]
    kinetics: tuple[Calculation, ...]
    unassigned: tuple[Calculation, ...]


def decompose(agr: Aggregator, k: Kirchhoff) -> Blocks:
    """Split an aggregator into its hydraulic block and its thermal clusters.

    Thermal clusters are the connected components of the edges that route only wall
    variables (``T_left``, ``h_left`` and their family). Kinetics nodes are those
    that route ``power`` into a fuel.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff
        The flow solver that sits in ``agr``.

    Returns
    -------
    Blocks

    Raises
    ------
    ValueError
        If ``k`` is not a node of ``agr``.
    """
    if k not in agr.graph:
        raise ValueError(
            f"{k} is not a Calculation of this Aggregator; pass the Kirchhoff that sits in agr.graph."
        )
    order = {node: i for i, node in enumerate(agr.graph)}
    hydraulic = {k, *k.components, *(n for n in k.g.nodes if isinstance(n, Calculation))}
    wall_edges = [(u, v) for u, v, names in agr.graph.edges(data=VARS) if names and set(names) <= THERMAL_VARIABLES]
    clusters = sorted((frozenset(c) for c in connected_components(Graph(wall_edges))), key=lambda c: min(order[n] for n in c))
    kinetics = [
        u
        for u, v, names in agr.graph.edges(data=VARS)
        if isinstance(u, PointKinetics) and isinstance(v, Fuel) and "power" in names
    ]
    kinetics = sorted(set(kinetics), key=order.__getitem__)
    assigned = hydraulic | set().union(*clusters) | set(kinetics)
    unassigned = tuple(n for n in agr.graph if n not in assigned)
    return Blocks(frozenset(hydraulic), tuple(clusters), tuple(kinetics), unassigned)


def _frozen_inputs(agr: Aggregator, block: set, y: np.ndarray, t: Second) -> dict:
    funcs = {v: {name: _at(f, t) for name, f in agr.funcs[v].items()} for v in block if v in agr.funcs}
    for v in block:
        for name, sources in agr.external.get(v, {}).items():
            outside = {u: place for u, place in sources.items() if u not in block}
            if not outside:
                continue
            if name in funcs.get(v, {}):
                continue
            if len(outside) != len(sources):
                raise ValueError(
                    f"{v} receives {name!r} from both inside and outside the block "
                    f"({sorted(str(u) for u in sources)}); a block must take each variable "
                    f"wholly from inside or wholly from outside."
                )
            if isinstance(v, Junction):
                raise ValueError(
                    f"{v} is a Junction on the block boundary; junctions mix their inputs by "
                    f"source and cannot be frozen. Put the junction and its neighbours in the same block."
                )
            values = [np.atleast_1d(np.asarray(y[place], dtype=float)) for place in outside.values()]
            frozen = np.concatenate(values)
            funcs.setdefault(v, {})[name] = frozen.item() if frozen.size == 1 else frozen
    return funcs


def solve_block(
    agr: Aggregator,
    nodes: Iterable[Calculation],
    state: DictState | np.ndarray,
    t: Second = 0.0,
    **solver_options,
) -> State:
    """Solve a subset of an aggregator's calculations with the rest of the system frozen.

    Every input a block calculation receives from outside the block is read from
    ``state`` and handed to a throwaway sub-aggregator as a constant func; the block's
    own funcs are kept and take precedence. The sub-aggregator is solved with
    :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    nodes: Iterable[Calculation]
        The calculations to solve. Every other calculation is frozen.
    state: DictState or Array1D
        The current state of the whole system; the block starts from its own part of it.
    t: Second
        Time at which the block's own funcs and the frozen boundary values are evaluated,
        and the time stamp of the returned State.
    solver_options:
        Forwarded to :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Returns
    -------
    State
        The solved block, calculations of the block only.

    Raises
    ------
    ValueError
        If a block calculation takes one variable partly from inside and partly from
        outside the block, or a Junction sits on the boundary.
    KeyError
        If ``state`` lacks a calculation of ``agr``.
    ~stream.solvers.AlgRuntimeError
        If the block does not solve.
    """
    block = set(nodes)
    y = np.asarray(state, dtype=float) if isinstance(state, np.ndarray) else agr.load(state)
    sub = Aggregator(agr.graph.subgraph(block).copy(), _frozen_inputs(agr, block, y, t))
    y0 = np.empty(len(sub))
    for node, section in sub.sections.items():
        y0[section] = y[agr.sections[node]]
    return sub.save(sub.solve_steady(y0, **solver_options), t=t)


def _at(f, t: Second):
    return f(t) if callable(f) else f

