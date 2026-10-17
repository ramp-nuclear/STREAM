"""Steady-state initial guesses for a flow network wall-coupled to fuel plates.

One call, :func:`guess_steady_state`, produces a guess for the whole system that
:meth:`~stream.aggregator.Aggregator.solve_steady` can start from. It works in three
stages:

1. :func:`decompose` splits the aggregator into blocks: the hydraulic block (the
   :class:`~stream.calculations.Kirchhoff` and everything on its flow graph), one
   thermal cluster per group of wall-coupled :class:`~stream.calculations.Fuel` plates
   and :class:`~stream.calculations.ChannelAndContacts` channels, and the point-kinetics
   node that supplies fuel power.
2. :func:`seed_steady_state` seeds every block in closed form, without solving: flows
   from what is known and the current law, temperatures by walking the network in flow
   direction with each heated channel adding its power, pressures from
   :func:`~stream.composition.subsystems.guess_hydraulic_steady_state`, walls and fuel
   interiors from the local heat flux, and kinetics from
   :func:`~stream.composition.subsystems.point_kinetics_steady_state`.
3. The seed is refined by solving one block at a time with the rest of the system
   frozen (:func:`solve_block`): the thermal clusters, then the hydraulics, a few times
   over.

Use this module for a system that couples a flow network to heated plates, where the
flows set the temperatures and the temperatures feed back into the pressure drops.
:meth:`~stream.composition.cycle.FlowGraph.guess_steady_state` covers the hydraulic part
alone and needs every flow and one temperature from the caller; the helpers in
:mod:`stream.composition.subsystems` cover a single plate or a single kinetics node.
:func:`guess_steady_state` combines them, completes the flows it is not given, and
reconciles the blocks with each other.

For a loop of one heated plate, where ``channel`` is a
:class:`~stream.calculations.ChannelAndContacts`, ``fuel`` its
:class:`~stream.calculations.Fuel` and ``top``, ``bottom`` two
:class:`~stream.calculations.Junction` nodes:

.. code-block:: python

    fg = FlowGraph(
        flow_edge((top, bottom), channel),
        flow_edge((bottom, top), Pump(mdot0=0.5, name="Pump"), HeatExchanger(outlet=40.0, name="HX")),
        abs_pressure_comps=[channel],
        reference_node=(top, 2e5),
    )
    agr = fg.aggregator + symmetric_plate(channel, fuel, {fuel: {"power": 1e3}}).to_aggregator()
    y = agr.solve_steady(guess_steady_state(agr))

Limitations: an aggregator holds at most one Kirchhoff (with any number of loops on it)
and at most one kinetics node. Fuels are heated through the ``T_left`` and ``T_right``
walls of a :class:`~stream.calculations.Channel` only. A fuel driven by a kinetics node
needs its power in ``powers``. A heated channel on a loop that nothing drives needs a
signed flow in ``flows``. Calculations the seed does not recognise go through
``overrides``.
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
    PointKinetics,
    PointKineticsWInput,
    Pump,
)
from stream.calculations.kirchhoff import COMPS
from stream.composition.subsystems import (
    HydraulicStrategyMap,
    MissingFlowError,
    guess_hydraulic_steady_state,
    point_kinetics_steady_state,
)
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


class MissingPowerError(ValueError):
    """A fuel's steady power is unknown: it has no ``power`` func and no ``powers`` entry."""


@dataclass(frozen=True)
class SidePowers:
    """Power per cell a channel receives through its left and right walls."""

    left: np.ndarray
    right: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.left + self.right


@dataclass(frozen=True)
class Blocks:
    """The partition :func:`decompose` finds.

    Attributes
    ----------
    hydraulic: frozenset[Calculation]
        The Kirchhoff, every component on its flow graph, and every junction. Empty
        when the system has no Kirchhoff.
    thermal: tuple[frozenset[Calculation], ...]
        One set per wall-coupled cluster of fuels and channels. Channels belong to
        their cluster and to the hydraulic block.
    kinetics: PointKinetics or None
        The point-kinetics node that supplies ``power`` to the fuels, if any. Its seed
        power is the sum of the steady powers of the fuels it feeds.
    unassigned: tuple[Calculation, ...]
        Every other node; the seed leaves these to the caller.
    """

    hydraulic: frozenset[Calculation]
    thermal: tuple[frozenset[Calculation], ...]
    kinetics: PointKinetics | None
    unassigned: tuple[Calculation, ...]


def _find_kirchhoff(agr: Aggregator, k: Kirchhoff | None) -> Kirchhoff | None:
    if k is not None:
        if k not in agr.graph:
            raise ValueError(
                f"{k} is not a Calculation of this Aggregator; pass the Kirchhoff that sits in agr.graph."
            )
        return k
    found = [n for n in agr.graph if isinstance(n, Kirchhoff)]
    if len(found) > 1:
        raise ValueError(
            f"The Aggregator holds {len(found)} Kirchhoffs {sorted(n.name for n in found)}; pass the one to use as k."
        )
    return found[0] if found else None


def decompose(agr: Aggregator, k: Kirchhoff | None = None) -> Blocks:
    """Split an aggregator into its hydraulic block and its thermal clusters.

    Thermal clusters are the connected components of the edges that route only wall
    variables (``T_left``, ``h_left`` and their family). The kinetics node is the one
    that routes ``power`` into a fuel.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff or None
        The flow solver that sits in ``agr``. If None, the Kirchhoff of ``agr`` is used,
        and a system without one has an empty hydraulic block.

    Returns
    -------
    Blocks

    Raises
    ------
    ValueError
        If ``k`` is not a node of ``agr``, ``k`` is None and ``agr`` holds more than one
        Kirchhoff, or more than one kinetics node routes ``power`` into a fuel.
    """
    k = _find_kirchhoff(agr, k)
    order = {node: i for i, node in enumerate(agr.graph)}
    hydraulic = set() if k is None else {k, *k.components, *(n for n in k.g.nodes if isinstance(n, Calculation))}
    wall_edges = [(u, v) for u, v, names in agr.graph.edges(data=VARS) if names and set(names) <= THERMAL_VARIABLES]
    clusters = sorted((frozenset(c) for c in connected_components(Graph(wall_edges))), key=lambda c: min(order[n] for n in c))
    kinetics = {
        u
        for u, v, names in agr.graph.edges(data=VARS)
        if isinstance(u, PointKinetics) and isinstance(v, Fuel) and "power" in names
    }
    if len(kinetics) > 1:
        raise ValueError(
            f"More than one kinetics node routes power into a fuel: {sorted(n.name for n in kinetics)}; "
            f"the seed handles one."
        )
    assigned = hydraulic | set().union(*clusters) | kinetics
    unassigned = tuple(n for n in agr.graph if n not in assigned)
    return Blocks(frozenset(hydraulic), tuple(clusters), next(iter(kinetics), None), unassigned)


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
    return {edge: value for edge, (_, value) in known.items()}


def _complete_flows(k: Kirchhoff, known: dict[str, float]) -> dict[str, float]:
    names = list(k.variables)[: k.edges_count]
    kcl = k._kcl.toarray()
    A = kcl[:-1]
    signify = np.abs(kcl).max(axis=0)
    root_w = np.sqrt(np.where(signify > 0, signify, 1.0))
    x = np.zeros(len(names))
    is_known = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        if name in known:
            x[i], is_known[i] = known[name], True
    if not is_known.all():
        rhs = -A[:, is_known] @ x[is_known]
        # Minimises sum(signify * mdot**2), so weighted parallel edges carry equal per-element flow.
        scale = root_w[~is_known]
        x[~is_known] = np.linalg.lstsq(A[:, ~is_known] / scale, rhs, rcond=None)[0] / scale
        x[np.abs(x) < 1e-12 * max(1.0, np.abs(x).max())] = 0.0
    return dict(zip(names, x))


def _check_heated_legs_not_stalled(k: Kirchhoff, blocks: Blocks, flows: dict[str, float]) -> None:
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
    powers_by_fuel = {}
    for fuel in (n for cluster in blocks.thermal for n in cluster if isinstance(n, Fuel)):
        if fuel in powers:
            powers_by_fuel[fuel] = float(powers[fuel])
            continue
        value = _func(agr, fuel, "power", t)
        if value is None:
            raise MissingPowerError(
                f"{fuel} has no 'power' func and no entry in powers; its steady power cannot be "
                f"read from the system (a kinetics node supplies it at run time). Pass "
                f"powers={{{fuel.name!r}: <Watt>}}."
            )
        powers_by_fuel[fuel] = float(np.asarray(value).item())
    return powers_by_fuel


def _side_powers(agr: Aggregator, blocks: Blocks, fuel_powers: dict) -> dict[Channel, SidePowers]:
    sides = {
        n: {"left": np.zeros(n.n), "right": np.zeros(n.n)}
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
    for fuel, heated_sides in wired.items():
        rows = _row_power(fuel, fuel_powers[fuel]) / len(heated_sides)
        for channel, side in heated_sides:
            if len(rows) == channel.n:
                sides[channel][side] += rows
            else:
                warnings.warn(
                    f"{fuel} has {len(rows)} rows but {channel} has {channel.n} cells; its power is "
                    f"spread uniformly over the channel for the seed."
                )
                sides[channel][side] += np.full(channel.n, rows.sum() / channel.n)
    return {c: SidePowers(left=s["left"], right=s["right"]) for c, s in sides.items()}


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


def _mix_at_node(k: Kirchhoff, node, flows: dict, outlets: dict, current: float) -> float:
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
    return N / D if D > 0 else current


def _walk_temperatures(
    agr: Aggregator,
    k: Kirchhoff,
    flows: dict,
    channel_powers: dict,
    temperature,
    t: Second,
    passes: int = 50,
    tolerance: Celsius = 1e-6,
):
    """Walk the flow network in flow direction until the temperatures settle.

    Each pass carries every edge's inlet through its components and mixes the arrivals
    at every node. Returns the outlet temperature per component, the coolant profile
    per channel, and the temperature per junction.
    """
    sources = [comp.T for comp in k.components if isinstance(comp, HeatExchanger)]
    for comp in k.components:
        for name in ("Tin", "Tin_minus"):
            if (pinned := _func(agr, comp, name, t)) is not None:
                sources.append(float(pinned))
    if not sources and temperature is None:
        raise ValueError(
            "The flow network has no temperature source (no HeatExchanger, no pinned Tin func); "
            "pass temperature=<Celsius>."
        )
    if not sources:
        warnings.warn("The flow network has no temperature sink; every temperature is seeded at the reference temperature.")
    T0 = float(temperature) if temperature is not None else float(np.mean(sources))
    nodes = dict.fromkeys(k.g.nodes, T0)
    outlets = dict.fromkeys(k.components, T0)
    profiles = {}
    edges = list(k.g.edges(data=COMPS))
    for _ in range(passes):
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
            mixed = _mix_at_node(k, node, flows, outlets, nodes[node])
            change = max(change, abs(mixed - nodes[node]))
            nodes[node] = mixed
        if change < tolerance:
            break
    return outlets, profiles, nodes


def _abs_pressure(k: Kirchhoff, hydraulic: State, channel: Channel) -> float:
    key = f"(p_abs of {channel.name})"
    if key in hydraulic[k.name]:
        return float(hydraulic[k.name][key])
    return float(k.ref_pressure) if np.ndim(k.ref_pressure) == 0 else 1e5


def _standalone_channels(agr: Aggregator, blocks: Blocks, flows: dict, channel_powers: dict, temperature, t: Second):
    mdots, p_abs, profiles = {}, {}, {}
    for c in (n for cluster in blocks.thermal for n in cluster if isinstance(n, Channel)):
        mdot = flows[c] if c in flows else _func(agr, c, "mdot", t)
        if mdot is None:
            raise MissingFlowError(
                f"{c} has no flow: there is no Kirchhoff, no 'mdot' func and no entry in flows. Pass "
                f"flows={{{c.name!r}: <KgPerS>}} or give the channel an 'mdot' func."
            )
        mdot = float(np.asarray(mdot).item())
        first, second = ("Tin", "Tin_minus") if mdot >= 0 else ("Tin_minus", "Tin")
        inlet = _func(agr, c, first, t)
        inlet = _func(agr, c, second, t) if inlet is None else inlet
        inlet = temperature if inlet is None else inlet
        if inlet is None:
            raise ValueError(f"{c} has no 'Tin' or 'Tin_minus' func; pass temperature=<Celsius>.")
        pressure = _func(agr, c, "p_abs", t)
        mdots[c], p_abs[c] = mdot, 1e5 if pressure is None else float(np.asarray(pressure).item())
        _, profiles[c] = _outlet(c, float(np.asarray(inlet).item()), mdot, channel_powers.get(c))
    return mdots, p_abs, profiles


def _cluster_seed(
    agr: Aggregator,
    cluster,
    mdots: dict,
    p_abs: dict,
    profiles: dict,
    sides: dict,
    T0: float,
    seed_coolant: bool = False,
) -> State:
    """Seed the walls, heat transfer coefficients and fuel interiors of one cluster.

    Each channel side's h comes from ``h_wall`` at the coolant temperature, floored at
    the coolant conductivity over the hydraulic diameter; its wall sits at coolant plus
    heat flux over h, updated over two passes. A fuel interior is the mean of its two
    walls tiled across the plate, and a fuel with one heated side gets that wall on both
    sides. With ``seed_coolant``, each channel also gets its coolant profile and its own
    pressure drop.
    """
    state = {}
    walls = {}
    if seed_coolant:
        for c in (n for n in cluster if isinstance(n, Channel)):
            state[c.name] = {"T_cool": profiles[c]}
    for c in (n for n in cluster if isinstance(n, ChannelAndContacts)):
        mdot, tc, p_abs_c = mdots[c], profiles[c], p_abs[c]
        h_floor = c.fluid.conductivity(tc) / c.pipe.hydraulic_diameter
        h_sides = {}
        for side, width in zip(("left", "right"), c.pipe.heated_parts):
            q = getattr(sides[c], side) / (c.dz * width) if width > 0 else np.zeros(c.n)
            tw = tc.copy()
            for _ in range(2):
                h = np.maximum(c.h_wall(T_wall=tw, T_cool=tc, mdot=mdot, pressure=p_abs_c), h_floor)
                tw = tc + q / h
            walls[(c, side)], h_sides[side] = tw, h
        state.setdefault(c.name, {}).update(h_left=h_sides["left"], h_right=h_sides["right"])
    if seed_coolant:
        for c in (n for n in cluster if isinstance(n, Channel)):
            tc = profiles[c]
            tw = (walls[(c, "left")] + walls[(c, "right")]) / 2.0 if (c, "left") in walls else tc
            state[c.name]["pressure"] = np.sum(c.pressure(T=tc, Tw=tw, mdot=mdots[c]))
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
    pk = blocks.kinetics
    if pk is None:
        return State({})
    fed = [v for u, v, names in agr.graph.out_edges(pk, data=VARS) if isinstance(v, Fuel) and "power" in names]
    power = sum(fuel_powers[f] for f in fed)
    power_input = _func(agr, pk, "power_input", t) if isinstance(pk, PointKineticsWInput) else None
    return point_kinetics_steady_state(pk, power, power_input)


def seed_steady_state(
    agr: Aggregator,
    k: Kirchhoff | None = None,
    *,
    flows: Mapping[Calculation | str, KgPerS] | None = None,
    temperature: Celsius | None = None,
    powers: Mapping[Calculation | str, Watt] | None = None,
    overrides: DictState | None = None,
    t: Second = 0.0,
    strategy: HydraulicStrategyMap | None = None,
    walk_passes: int = 50,
    walk_tolerance: Celsius = 1e-6,
) -> State:
    """A closed-form steady-state guess for a flow network wall-coupled to fuel plates.

    Nothing is solved. Flows come from what is known (``flows``, a pump's fixed flow,
    a closed flapper) and a minimum-norm completion of the current
    law; temperatures are walked around the network in flow direction with each
    heated channel adding its power over flow times heat capacity; pressures come from
    :func:`~stream.composition.subsystems.guess_hydraulic_steady_state` at those
    temperatures; walls sit at coolant plus flux over a floored heat transfer
    coefficient, and fuel interiors are tiled from their walls. Funcs win over walked
    values, as they do in the Aggregator.

    A system without a Kirchhoff (plates that take their coolant from funcs, as
    :func:`~stream.composition.mtr_geometry.symmetric_plate` builds them) is seeded channel by
    channel: the flow from ``flows`` or the channel's ``mdot`` func, the inlet from its
    ``Tin`` func (``Tin_minus`` for reversed flow), the absolute pressure from its
    ``p_abs`` func (1e5 Pa without one), and the channel's own ``pressure`` from its
    pressure drop at the seeded temperatures.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff or None
        The flow solver that sits in ``agr``. If None, the Kirchhoff of ``agr`` is used.
    flows: Mapping[Calculation | str, KgPerS] or None
        Known flows by component or name. Needed only where the network cannot fix a
        flow itself: a head-driven pump, or a pump-free loop (pass a small signed flow).
        Without a Kirchhoff, a channel's entry here wins over its ``mdot`` func.
    temperature: Celsius or None
        Reference temperature. Required when the network has no heat exchanger and no
        pinned inlet; otherwise it only sets the walk's start. Without a Kirchhoff, it
        is the inlet of a channel that has no ``Tin`` or ``Tin_minus`` func.
    powers: Mapping[Calculation | str, Watt] or None
        Fuel powers by fuel or name. Override a fuel's ``power`` func; required for a
        fuel driven by a kinetics node.
    overrides: DictState or None
        Merged last, for calculations the seed does not recognise; a calculation named
        here is not reported as unrecognised.
    t: Second
        Time at which funcs are evaluated.
    strategy: HydraulicStrategyMap or None
        Per-component pressure-drop laws for components the hydraulic guesser does not know.
    walk_passes: int
        The seed walks the network in flow direction, computing each component's outlet
        from its inlet and mixing the arrivals at every junction, and repeats the pass
        until no temperature moves by more than ``walk_tolerance``, at most
        ``walk_passes`` times. A loop settles in a few passes. Default 50.
    walk_tolerance: Celsius
        The largest temperature change that ends the walk. Default 1e-6.

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
        Two known flows on one edge disagree, no temperature source and no
        ``temperature``, or the Kirchhoff or the kinetics node is ambiguous.
    """
    k = _find_kirchhoff(agr, k)
    blocks = decompose(agr, k)
    if k is None:
        fuel_powers = _fuel_powers(agr, blocks, _resolve(agr, powers), t)
        sides = _side_powers(agr, blocks, fuel_powers)
        channel_powers = {c: s.total for c, s in sides.items()}
        mdots, p_abs, profiles = _standalone_channels(agr, blocks, _resolve(agr, flows), channel_powers, temperature, t)
        hydraulic = State({})
        T0 = float(np.mean([np.mean(p) for p in profiles.values()])) if profiles else temperature
    else:
        flows = _complete_flows(k, _known_flows(k, _resolve(agr, flows), t))
        _check_heated_legs_not_stalled(k, blocks, flows)
        fuel_powers = _fuel_powers(agr, blocks, _resolve(agr, powers), t)
        sides = _side_powers(agr, blocks, fuel_powers)
        channel_powers = {c: s.total for c, s in sides.items()}
        outlets, profiles, nodes = _walk_temperatures(
            agr, k, flows, channel_powers, temperature, t, walk_passes, walk_tolerance
        )
        temperatures = {c: profiles.get(c, outlets[c]) for c in k.components}
        temperatures |= {n: nodes[n] for n in k.g.nodes if isinstance(n, Calculation)}
        hydraulic = guess_hydraulic_steady_state(k, {c: flows[k.component_edge(c)] for c in k.components}, temperatures, strategy)
        T0 = float(np.mean(list(nodes.values())))
        for c in k.components:
            if isinstance(c, Channel) and c not in profiles:
                profiles[c] = np.full(c.n, outlets[c])
        mdots = {c: flows[k.component_edge(c)] for c in k.components}
        p_abs = {c: _abs_pressure(k, hydraulic, c) for c in k.components if isinstance(c, ChannelAndContacts)}
    clusters = [
        _cluster_seed(agr, cluster, mdots, p_abs, profiles, sides, T0, seed_coolant=k is None)
        for cluster in blocks.thermal
    ]
    names = sorted(getattr(n, "name", str(n)) for n in blocks.unassigned)
    names = [name for name in names if name not in (overrides or {})]
    if names:
        warnings.warn(
            f"The seed does not recognise {names} and leaves them out; merge their values through overrides."
        )
    return State.merge(hydraulic, *clusters, _kinetics_seed(agr, blocks, fuel_powers, t), overrides or {})


def _attempt(agr: Aggregator, nodes, state: State, label: str, t: Second, **solver_options) -> State:
    try:
        return State.merge(state, solve_block(agr, nodes, state, t=t, **solver_options))
    except AlgRuntimeError as err:
        names = sorted(getattr(n, "name", str(n)) for n in nodes)
        warnings.warn(f"The {label} block {names} did not solve from the seed and keeps its seed values: {err}")
        return state


def guess_steady_state(
    agr: Aggregator,
    k: Kirchhoff | None = None,
    *,
    flows: Mapping[Calculation | str, KgPerS] | None = None,
    temperature: Celsius | None = None,
    powers: Mapping[Calculation | str, Watt] | None = None,
    overrides: DictState | None = None,
    t: Second = 0.0,
    strategy: HydraulicStrategyMap | None = None,
    sweeps: int = 2,
    walk_passes: int = 50,
    walk_tolerance: Celsius = 1e-6,
    **solver_options,
) -> State:
    """A steady-state guess for a flow network wall-coupled to fuel plates.

    Seeds with :func:`seed_steady_state`, then refines the seed block by block with
    :func:`solve_block`. Every thermal cluster is solved first with its hydraulics
    frozen. Each sweep then solves the hydraulic block with its walls frozen, and every
    cluster again, so the walls are solved last, against the final flows. A system
    without a Kirchhoff has no hydraulic block, and a sweep solves its clusters only.
    A block whose solve raises :class:`~stream.solvers.AlgRuntimeError` keeps its
    previous values and is named in a warning.

    Parameters
    ----------
    agr: Aggregator
        The full system.
    k: Kirchhoff or None
        The flow solver that sits in ``agr``. If None, the Kirchhoff of ``agr`` is used.
    flows, temperature, powers, overrides, t, strategy, walk_passes, walk_tolerance:
        As in :func:`seed_steady_state`.
    sweeps: int
        Number of hydraulic-then-clusters sweeps after the first cluster solves. Zero
        or less returns the seed. One sweep is enough on a single forced loop; two are
        worth it where the flows feed back on the walls (parallel channels, natural
        convection). Default 2.
    solver_options:
        Forwarded to every block's :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Returns
    -------
    State
        Loadable into ``agr`` and ready for :meth:`~stream.aggregator.Aggregator.solve_steady`.

    Raises
    ------
    ValueError
        An error raised by :func:`seed_steady_state` (also when the Kirchhoff or the
        kinetics node is ambiguous).
    ~stream.composition.subsystems.MissingFlowError
        A heated channel whose flow nothing fixes.
    MissingPowerError
        A fuel whose steady power is unknown.
    """
    k = _find_kirchhoff(agr, k)
    blocks = decompose(agr, k)
    state = seed_steady_state(
        agr,
        k,
        flows=flows,
        temperature=temperature,
        powers=powers,
        overrides=overrides,
        t=t,
        strategy=strategy,
        walk_passes=walk_passes,
        walk_tolerance=walk_tolerance,
    )
    if sweeps <= 0:
        return state
    for i, cluster in enumerate(blocks.thermal):
        state = _attempt(agr, cluster, state, f"thermal cluster {i}", t, **solver_options)
    for _ in range(sweeps):
        if blocks.hydraulic:
            state = _attempt(agr, blocks.hydraulic, state, "hydraulic", t, **solver_options)
        for i, cluster in enumerate(blocks.thermal):
            state = _attempt(agr, cluster, state, f"thermal cluster {i}", t, **solver_options)
    return state
