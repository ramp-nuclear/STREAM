"""Steady-state initial guesses for a flow network wall-coupled to fuel plates.

The tools here serve systems built as a :class:`~stream.composition.cycle.FlowGraph`
hydraulic aggregator plus thermal aggregators of :class:`~stream.calculations.Fuel`
plates and :class:`~stream.calculations.ChannelAndContacts` channels. They read the
aggregator's own wiring, so any plate layout made with the helpers in
:mod:`stream.composition.mtr_geometry`, or by hand with the same variable names, is
handled the same way.
"""

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from networkx import Graph, connected_components

from stream import smoothing
from stream.aggregator import VARS, Aggregator
from stream.calculation import Calculation
from stream.calculations import (
    Channel,
    ChannelAndContacts,
    Flapper,
    Fuel,
    HeatExchanger,
    Junction,
    Kirchhoff,
    Orifice,
    PointKinetics,
    PointKineticsWInput,
    Pump,
    Tank,
)
from stream.calculations.kirchhoff import COMPS
from stream.composition.subsystems import (
    HydraulicStrategyMap,
    MissingFlowError,
    guess_hydraulic_steady_state,
    point_kinetics_steady_state,
)
from stream.errors import StreamConstructionError, StreamError
from stream.smoothing import soft_pos
from stream.solvers import AlgRuntimeError
from stream.state import DictState, State
from stream.units import Celsius, KgPerS, Second, Watt

__all__ = [
    "Blocks",
    "MissingPowerError",
    "decompose",
    "guess_steady_state",
    "seed_steady_state",
    "solve_block",
]

THERMAL_VARIABLES = frozenset(
    {"T_left", "T_right", "T_top", "T_bottom", "h_left", "h_right", "h_top", "h_bottom"}
)


class MissingPowerError(StreamError):
    """A fuel's steady power is unknown: it has no ``power`` func and no ``powers`` entry."""


@dataclass(frozen=True)
class Blocks:
    """The partition :func:`decompose` finds.

    Attributes
    ----------
    hydraulic: frozenset[Calculation]
        The Kirchhoff, every component on its flow graph, and every junction and tank.
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
    StreamConstructionError
        If ``k`` is not a node of ``agr``.
    """
    if k not in agr.graph:
        raise StreamConstructionError(
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
                raise StreamConstructionError(
                    f"{v} receives {name!r} from both inside and outside the block "
                    f"({sorted(str(u) for u in sources)}); a block must take each variable "
                    f"wholly from inside or wholly from outside."
                )
            if isinstance(v, Junction):
                raise StreamConstructionError(
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
    StreamConstructionError
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


def _func(agr: Aggregator, node: Calculation, name: str, t: Second):
    fns = agr.funcs.get(node, {})
    return _at(fns[name], t) if name in fns else None


def _resolve(agr: Aggregator, mapping: Mapping | None) -> dict:
    return {(agr[key] if isinstance(key, str) else key): value for key, value in (mapping or {}).items()}


def _known_flows(k: Kirchhoff, flows: dict, t: Second) -> dict[str, float]:
    known: dict[str, tuple] = {}

    def put(comp, value):
        edge = k.component_edge(comp)
        if edge in known and not np.isclose(known[edge][1], value):
            raise ValueError(
                f"Known flows on edge {edge} disagree: {known[edge][0]} gives {known[edge][1]}, "
                f"{comp} gives {value}."
            )
        known.setdefault(edge, (comp, float(value)))

    for comp, value in flows.items():
        put(comp, value)
    for comp in k.components:
        match comp:
            case Pump() if comp.p is None and comp.mdot0 is not None:
                put(comp, comp.mdot0)
            case Flapper() if t <= comp.t_open:
                put(comp, 0.0)
            case Orifice() if comp._closed or t <= comp.t_break:
                put(comp, 0.0)
    return {edge: value for edge, (_, value) in known.items()}


def _complete_flows(k: Kirchhoff, known: dict[str, float]) -> dict[str, float]:
    names = list(k.variables)[: k.edges_count]
    rows = k._keep_rows if k.surface_nodes else slice(0, -1)
    A = k._kcl.toarray()[rows]
    x = np.zeros(len(names))
    is_known = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        if name in known:
            x[i], is_known[i] = known[name], True
    if not is_known.all():
        rhs = -A[:, is_known] @ x[is_known]
        x[~is_known] = np.linalg.lstsq(A[:, ~is_known], rhs, rcond=None)[0]
        x[np.abs(x) < 1e-12 * max(1.0, np.abs(x).max())] = 0.0
    return dict(zip(names, x))


def _check_heated_legs(k: Kirchhoff, blocks: Blocks, flows: dict[str, float]) -> None:
    heated = [c for cluster in blocks.thermal for c in cluster if isinstance(c, Channel)]
    stalled = sorted(c.name for c in heated if flows[k.component_edge(c)] == 0.0)
    if stalled:
        raise MissingFlowError(
            f"Heated channels {stalled} have no flow: nothing on their loop fixes it. Pass a signed "
            f"flow for one of them, e.g. flows={{{stalled[0]!r}: 0.01}} (negative for reversed flow)."
        )


def _row_power(fuel: Fuel, power: Watt) -> np.ndarray:
    power_mat = np.zeros(fuel.shape)
    power_mat[fuel.meat == 1] = power * fuel.power_shape
    return power_mat.sum(axis=1)


def _fuel_powers(agr: Aggregator, blocks: Blocks, powers: dict, t: Second) -> dict:
    out = {}
    for fuel in (n for cluster in blocks.thermal for n in cluster if isinstance(n, Fuel)):
        if fuel in powers:
            out[fuel] = float(powers[fuel])
            continue
        value = _func(agr, fuel, "power", t)
        if value is None:
            raise MissingPowerError(
                f"{fuel} has no 'power' func and no entry in powers; its steady power cannot be "
                f"read from the system (a kinetics node supplies it at run time). Pass "
                f"powers={{{fuel.name!r}: <Watt>}}."
            )
        out[fuel] = float(np.asarray(value).item())
    return out


def _side_powers(agr: Aggregator, blocks: Blocks, fuel_powers: dict) -> dict:
    sides = {
        n: dict(left=np.zeros(n.n), right=np.zeros(n.n))
        for cluster in blocks.thermal
        for n in cluster
        if isinstance(n, Channel)
    }
    wired = {}
    for u, v, names in agr.graph.edges(data=VARS):
        if isinstance(u, Fuel) and isinstance(v, Channel):
            for side in ("left", "right"):
                if f"T_{side}" in names:
                    wired.setdefault(u, []).append((v, side))
    for fuel, targets in wired.items():
        rows = _row_power(fuel, fuel_powers[fuel]) / len(targets)
        for channel, side in targets:
            if len(rows) == channel.n:
                sides[channel][side] += rows
            else:
                warnings.warn(
                    f"{fuel} has {len(rows)} rows but {channel} has {channel.n} cells; its power is "
                    f"spread uniformly over the channel for the seed."
                )
                sides[channel][side] += np.full(channel.n, rows.sum() / channel.n)
    return sides


def _outlet(comp: Calculation, inlet: float, mdot: float, row_power) -> tuple[float, np.ndarray | None]:
    if isinstance(comp, Channel):
        if row_power is None:
            profile = np.full(comp.n, inlet)
        else:
            dT = np.asarray(row_power, dtype=float) / (abs(mdot) * comp.fluid.specific_heat(inlet))
            profile = inlet + (np.cumsum(dT) if mdot >= 0 else np.cumsum(dT[::-1])[::-1])
        return float(profile[-1] if mdot >= 0 else profile[0]), profile
    T_out = getattr(comp, "T_out", None)
    if T_out is None:
        return inlet, None
    return float(np.asarray(T_out(Tin=inlet, mdot=mdot)).item()), None


def _mixing_eps(node) -> float:
    eps = getattr(node, "mdot_eps", None)
    return eps if eps is not None else smoothing.DEFAULT_MDOT_EPS


def _walk_temperatures(agr: Aggregator, k: Kirchhoff, flows: dict, channel_powers: dict, temperature, t: Second):
    sources = [comp.T for comp in k.components if isinstance(comp, HeatExchanger)]
    for comp in k.components:
        for name in ("Tin", "Tin_minus"):
            if (pinned := _func(agr, comp, name, t)) is not None:
                sources.append(float(pinned))
    sources += [n.fixed_temperature for n in k.g.nodes if isinstance(n, Tank) and n.fixed_temperature is not None]
    if not sources and temperature is None:
        raise ValueError(
            "The flow network has no temperature source (no HeatExchanger, no fixed-temperature Tank, "
            "no pinned Tin func); pass temperature=<Celsius>."
        )
    if not sources:
        warnings.warn("The flow network has no temperature sink; every temperature is seeded at the reference temperature.")
    T0 = float(temperature) if temperature is not None else float(np.mean(sources))
    nodes = dict.fromkeys(k.g.nodes, T0)
    outlets = dict.fromkeys(k.components, T0)
    profiles = {}
    edges = list(k.g.edges(data=COMPS))
    for _ in range(50):
        change = 0.0
        for u, v, comps in edges:
            mdot = flows[k.component_edge(comps[0])]
            series = [u, *comps, v] if mdot >= 0 else [v, *comps[::-1], u]
            T = nodes[series[0]]
            for comp in series[1:-1]:
                pinned = _func(agr, comp, "Tin" if mdot >= 0 else "Tin_minus", t)
                if pinned is not None:
                    T = float(pinned)
                T, profile = _outlet(comp, T, mdot, channel_powers.get(comp))
                if profile is not None:
                    profiles[comp] = profile
                change = max(change, abs(T - outlets[comp]))
                outlets[comp] = T
        for node in k.g.nodes:
            if isinstance(node, Tank) and node.fixed_temperature is not None:
                nodes[node] = float(node.fixed_temperature)
                continue
            eps = _mixing_eps(node)
            weights = getattr(node, "weights", {})
            N = D = 0.0
            for _, _, comps in k.g.in_edges(node, data=COMPS):
                mdot = flows[k.component_edge(comps[0])]
                w = weights.get(comps[-1], 1.0) * soft_pos(mdot, eps)
                N, D = N + w * outlets[comps[-1]], D + w
            for _, _, comps in k.g.out_edges(node, data=COMPS):
                mdot = flows[k.component_edge(comps[0])]
                w = weights.get(comps[0], 1.0) * soft_pos(-mdot, eps)
                N, D = N + w * outlets[comps[0]], D + w
            mixed = N / D if D > 0 else nodes[node]
            change = max(change, abs(mixed - nodes[node]))
            nodes[node] = mixed
        if change < 1e-6:
            break
    return outlets, profiles, nodes


def _abs_pressure(k: Kirchhoff, hydraulic: State, channel: Channel) -> float:
    key = f"(p_abs of {channel.name})"
    if key in hydraulic[k.name]:
        return float(hydraulic[k.name][key])
    return float(k.ref_pressure) if np.ndim(k.ref_pressure) == 0 else 1e5


def _cluster_seed(agr: Aggregator, k: Kirchhoff, cluster, hydraulic: State, profiles: dict, flows: dict, sides: dict, T0: float) -> State:
    state = {}
    walls = {}
    for c in (n for n in cluster if isinstance(n, ChannelAndContacts)):
        mdot, tc, p_abs = flows[k.component_edge(c)], profiles[c], _abs_pressure(k, hydraulic, c)
        h_floor = c.fluid.conductivity(tc) / c.pipe.hydraulic_diameter
        h_sides = {}
        for side, width in zip(("left", "right"), c.pipe.heated_parts):
            q = sides[c][side] / (c.dz * width) if width > 0 else np.zeros(c.n)
            tw = tc.copy()
            for _ in range(2):
                h = np.maximum(c.h_wall(T_wall=tw, T_cool=tc, mdot=mdot, pressure=p_abs), h_floor)
                tw = tc + q / h
            walls[(c, side)], h_sides[side] = tw, h
        state[c.name] = dict(h_left=h_sides["left"], h_right=h_sides["right"])
    for f in (n for n in cluster if isinstance(n, Fuel)):
        fed = {}
        for u, v, names in agr.graph.in_edges(f, data=VARS):
            if not isinstance(u, ChannelAndContacts):
                continue
            for side, opposite in (("left", "right"), ("right", "left")):
                if f"T_{side}" in names:
                    tw = walls[(u, opposite)]
                    fed[side] = tw if len(tw) == f.m else np.full(f.m, float(np.mean(tw)))
        left = fed.get("left", fed.get("right", np.full(f.m, T0)))
        right = fed.get("right", left)
        interior = np.tile(((left + right) / 2.0)[:, None], (1, f.n))
        state[f.name] = dict(T=interior, T_wall_left=left, T_wall_right=right)
    return State(state)


def _kinetics_seed(agr: Aggregator, blocks: Blocks, fuel_powers: dict, t: Second) -> State:
    state = {}
    for pk in blocks.kinetics:
        fed = [v for u, v, names in agr.graph.out_edges(pk, data=VARS) if isinstance(v, Fuel) and "power" in names]
        power = sum(fuel_powers[f] for f in fed)
        power_input = _func(agr, pk, "power_input", t) if isinstance(pk, PointKineticsWInput) else None
        state |= point_kinetics_steady_state(pk, power, power_input)
    return State(state)


def _tank_seed(k: Kirchhoff, nodes: dict) -> State:
    return State(
        {
            n.name: dict(level=n.level0, T=nodes[n] if n.fixed_temperature is None else n.fixed_temperature)
            for n in k.g.nodes
            if isinstance(n, Tank)
        }
    )


def seed_steady_state(
    agr: Aggregator,
    k: Kirchhoff,
    *,
    flows: Mapping[Calculation | str, KgPerS] | None = None,
    temperature: Celsius | None = None,
    powers: Mapping[Calculation | str, Watt] | None = None,
    overrides: DictState | None = None,
    t: Second = 0.0,
    strategy: HydraulicStrategyMap | None = None,
) -> State:
    """A closed-form steady-state guess for a flow network wall-coupled to fuel plates.

    Nothing is solved. Flows come from what is known (``flows``, a pump's fixed flow,
    a closed flapper, a sealed orifice) and a minimum-norm completion of the current
    law; temperatures are walked around the network in flow direction with each
    heated channel adding its power over flow times heat capacity; pressures come from
    :func:`~stream.composition.subsystems.guess_hydraulic_steady_state` at those
    temperatures; walls sit at coolant plus flux over a floored heat transfer
    coefficient, and fuel interiors are tiled from their walls. Funcs win over walked
    values, as they do in the Aggregator.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff
        The flow solver that sits in ``agr``.
    flows: Mapping[Calculation | str, KgPerS] or None
        Known flows by component or name. Needed only where the network cannot fix a
        flow itself: a head-driven pump, or a pump-free loop (pass a small signed flow).
    temperature: Celsius or None
        Reference temperature. Required when the network has no heat exchanger, no
        fixed-temperature tank and no pinned inlet; otherwise it only sets the walk's start.
    powers: Mapping[Calculation | str, Watt] or None
        Fuel powers by fuel or name. Override a fuel's ``power`` func; required for a
        fuel driven by a kinetics node.
    overrides: DictState or None
        Merged last, for calculations the seed does not recognise.
    t: Second
        Time at which funcs are evaluated.
    strategy: HydraulicStrategyMap or None
        Per-component pressure-drop laws for components the hydraulic guesser does not know.

    Returns
    -------
    State

    Raises
    ------
    ~stream.composition.subsystems.MissingFlowError
        A heated channel whose flow nothing fixes.
    MissingPowerError
        A fuel whose steady power is unknown.
    ValueError
        Two known flows on one edge disagree, or no temperature source and no ``temperature``.
    """
    blocks = decompose(agr, k)
    flows = _complete_flows(k, _known_flows(k, _resolve(agr, flows), t))
    _check_heated_legs(k, blocks, flows)
    fuel_powers = _fuel_powers(agr, blocks, _resolve(agr, powers), t)
    sides = _side_powers(agr, blocks, fuel_powers)
    channel_powers = {c: s["left"] + s["right"] for c, s in sides.items()}
    outlets, profiles, nodes = _walk_temperatures(agr, k, flows, channel_powers, temperature, t)
    temperatures = {c: profiles.get(c, outlets[c]) for c in k.components}
    temperatures |= {n: nodes[n] for n in k.g.nodes if isinstance(n, Calculation)}
    hydraulic = guess_hydraulic_steady_state(k, {c: flows[k.component_edge(c)] for c in k.components}, temperatures, strategy)
    T0 = float(np.mean(list(nodes.values())))
    for c in k.components:
        if isinstance(c, Channel) and c not in profiles:
            profiles[c] = np.full(c.n, outlets[c])
    clusters = [_cluster_seed(agr, k, cluster, hydraulic, profiles, flows, sides, T0) for cluster in blocks.thermal]
    if blocks.unassigned:
        names = sorted(getattr(n, "name", str(n)) for n in blocks.unassigned)
        warnings.warn(
            f"The seed does not recognise {names} and leaves them out; merge their values through overrides."
        )
    return State.merge(hydraulic, *clusters, _kinetics_seed(agr, blocks, fuel_powers, t), _tank_seed(k, nodes), overrides or {})


def guess_steady_state(
    agr: Aggregator,
    k: Kirchhoff,
    *,
    flows: Mapping[Calculation | str, KgPerS] | None = None,
    temperature: Celsius | None = None,
    powers: Mapping[Calculation | str, Watt] | None = None,
    overrides: DictState | None = None,
    t: Second = 0.0,
    strategy: HydraulicStrategyMap | None = None,
    refine: bool = True,
    sweeps: int = 2,
    **solver_options,
) -> State:
    """A steady-state guess for a flow network wall-coupled to fuel plates.

    Seeds with :func:`seed_steady_state`, then, unless ``refine`` is off, sweeps the
    system block by block with :func:`solve_block`: every thermal cluster with its
    hydraulics frozen, then the hydraulic block with its walls frozen, ``sweeps``
    times, and the clusters once more so the returned walls match the returned flows.
    A block whose solve raises :class:`~stream.solvers.AlgRuntimeError` keeps its seed
    values and is named in a warning.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff
        The flow solver that sits in ``agr``.
    flows, temperature, powers, overrides, t, strategy:
        As in :func:`seed_steady_state`.
    refine: bool
        Sweep the blocks after seeding. Default ``True``.
    sweeps: int
        Number of cluster-then-hydraulic sweeps. One is enough for forced flow.
    solver_options:
        Forwarded to every block's :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Returns
    -------
    State
        Loadable into ``agr`` and ready for :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Raises
    ------
    ValueError
        If ``sweeps`` is less than one.
    """
    if sweeps < 1:
        raise ValueError(f"sweeps must be at least 1, was {sweeps}")
    blocks = decompose(agr, k)
    state = seed_steady_state(
        agr, k, flows=flows, temperature=temperature, powers=powers, overrides=overrides, t=t, strategy=strategy
    )
    if not refine:
        return state
    agr.load(state)

    def attempt(nodes, label: str) -> None:
        nonlocal state
        try:
            state = State.merge(state, solve_block(agr, nodes, state, t=t, **solver_options))
        except AlgRuntimeError as err:
            names = sorted(getattr(n, "name", str(n)) for n in nodes)
            warnings.warn(
                f"The {label} block {names} did not solve from the seed and keeps its seed values: {err}"
            )

    for _ in range(sweeps):
        for i, cluster in enumerate(blocks.thermal):
            attempt(cluster, f"thermal cluster {i}")
        attempt(blocks.hydraulic, "hydraulic")
    for i, cluster in enumerate(blocks.thermal):
        attempt(cluster, f"thermal cluster {i}")
    return state
