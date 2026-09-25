"""Helper functions for creating steady state initial guesses for small and specific subsystems"""

import logging
import warnings
from collections.abc import Mapping
from typing import Callable, Iterable

import numpy as np
from cytoolz import keymap

from stream import Calculation
from stream.calculations import (
    Channel,
    ChannelAndContacts,
    DPCalculation,
    Flapper,
    Fuel,
    Gravity,
    HeatExchanger,
    Junction,
    Kirchhoff,
    PointKinetics,
    PointKineticsWInput,
    Pump,
)
from stream.calculations.kirchhoff import COMPS
from stream.composition.mtr_geometry import symmetric_plate
from stream.errors import StreamError
from stream.physical_models.pressure_drop import local_pressure_by_mdot
from stream.state import State
from stream.units import Celsius, KgPerS, Pascal, Value, Watt
from stream.utilities import just

__all__ = [
    "check_gravity_mismatch",
    "guess_hydraulic_steady_state",
    "point_kinetics_steady_state",
    "symmetric_plate_steady_state",
    "HydraulicStrategy",
    "HydraulicStrategyMap",
    "GravityMismatchError",
    "MissingFlowError",
]

logger = logging.getLogger("stream.subsystems")


def symmetric_plate_steady_state(
    c: ChannelAndContacts,
    f: Fuel,
    mdot: KgPerS,
    p_abs: Pascal,
    power: Watt,
    Tin: Celsius,
    initial_guess_iterations: int = 2,
    **solver_options,
) -> State:
    r"""Steady state for a :func:`.symmetric_plate` system

    Parameters
    ----------
    c: ChannelAndContacts
        A channel (and contacts...) instance
    f: Fuel
        A fuel instance
    mdot: KgPerS
        Desired mass current
    p_abs: Pascal
        Desired absolute pressure at the top of the plate
    power: Watt
        Desired power
    Tin: Celsius
         Desired inlet temperature into the channel. Depending on the sign of ``mdot`` (+, -),
         the inlet temperature is a source term for the (first, last) cell.
    initial_guess_iterations: int
        Before employing a solver, an initial educated guess is assumed. This
        guess should become better educated with each iteration controlled by
        ``precondition``. The iteration floors the wall heat transfer
        coefficient at the conduction scale ``k/Dh``, so correlations with no
        conduction floor (e.g. the Elenbaas natural-convection HTC, whose
        ``h -> 0`` at zero wall superheat) still yield a finite seed.
    solver_options: Dict
        Keyword arguments to control steady state solver behavior

    Returns
    -------
    steady: State
    """
    if initial_guess_iterations < 1:
        raise ValueError(f"Must try at least once to obtain values. Was {initial_guess_iterations}")
    if mdot == 0:
        raise ValueError(
            "mass flow must be nonzero to build a coolant-temperature guess; "
            "for NC starts pass a small signed mdot (e.g. ±0.01)"
        )
    plate = symmetric_plate(c, f, {c: dict(mdot=mdot, p_abs=p_abs, Tin=Tin), f: dict(power=power)}).to_aggregator()

    power_mat = np.zeros(f.shape)
    power_mat[f.meat == 1] = power * f.power_shape
    cp = c.fluid.specific_heat(Tin)
    p_z = np.sum(power_mat, 1)
    q2t_z = p_z / (c.pipe.heated_perimeter * c.dz)
    # Reverse flow enters from the far end: suffix cumsum, not a reversed prefix cumsum (equal only for symmetric power).
    dT = p_z / (np.abs(mdot) * cp)
    tc0 = Tin + (np.cumsum(dT) if mdot >= 0 else np.cumsum(dT[::-1])[::-1])
    tw0 = tc0
    # Conduction-scale floor: a zero-floor HTC (e.g. Elenbaas at the tw0=tc0 seed) returns h=0, and the bare division would send tw0 through inf to NaN.
    h_floor = c.fluid.conductivity(Tin) / c.pipe.hydraulic_diameter
    for _ in range(initial_guess_iterations):
        dp0 = c.pressure(T=tc0, Tw=tw0, mdot=mdot)
        h0 = np.maximum(c.h_wall(T_wall=tw0, T_cool=tc0, mdot=mdot, pressure=p_abs - dp0), h_floor)
        tw0 = (q2t_z / h0) + tc0
    T0 = np.tile(tw0, f.shape[1]).reshape(f.shape[::-1]).T.flatten()
    # Safe because the initial guess iterations are >= 1
    # noinspection PyUnboundLocalVariable
    y0 = plate.load(
        {
            f.name: dict(T=T0, T_wall_left=tw0, T_wall_right=tw0),
            c.name: dict(T_cool=tc0, h_left=h0, h_right=h0, pressure=np.sum(dp0)),
        }
    )

    return plate.save(plate.solve_steady(y0, **solver_options))


def point_kinetics_steady_state(pk: PointKinetics, power: Watt, power_input: Watt = None) -> State:
    r"""Zero reactivity steady :class:`.PointKinetics` steady state

    Parameters
    ----------
    pk: PointKinetics
        a PK instance
    power: Watt
        Desired power at which the reactor operates
    power_input: Watt or None
        If another source of power (besides PK, e.g decay heat) contributes to total power (which should be provided at
        simulation time), the neutronic power generation is ``power - power_input``.
        Note this only applies to :class:`.PointKineticsWInput`.

    Returns
    -------
    steady: State
        Assuming zero reactivity (criticality)
    """

    if isinstance(pk, PointKineticsWInput):
        Pn = power - (power_input or 0.0)
        d = dict(power=power, ck=Pn * (pk.betak / pk.lambdak / pk.Lambda))
        d["pk_power"] = Pn
    else:
        d = dict(power=power, ck=power * (pk.betak / pk.lambdak / pk.Lambda))

    return State({pk.name: d})


class MissingFlowError(StreamError):
    """Error to signify missing :class:`.Kirchhoff` flow data."""

    pass


HydraulicStrategy = Callable[[KgPerS, Celsius], Pascal]
HydraulicStrategyMap = dict[Calculation, HydraulicStrategy]


def _float_values(d: dict[str, Value], keys: Iterable[str], inner_key: str = "pressure") -> Iterable[float]:
    for key in keys:
        y = d[key][inner_key]
        yield y if isinstance(y, (float, int)) else y.item()


def guess_hydraulic_steady_state(
    k: Kirchhoff,
    mdots: dict[Calculation, KgPerS],
    temperature: Celsius,
    strategy: HydraulicStrategyMap | None = None,
) -> State:
    r"""A guess for a :class:`.Kirchhoff` derived system, in which the flows are known

    .. note:: When a component's pressure difference cannot be **physically** determined from the flow,
       the guess is ``0.0``.
       Prominent examples are ideal flow sources such as :class:`.Pump` ``(mdot0=x)`` or a closed :class:`.Flapper`.

    Parameters
    ----------
    k : Kirchhoff
        Kirchhoff Calculation
    mdots : dict[Calculation, KgPerS]
        Known mass flows :math:`\dot{m}` for components in the hydraulic system.
        Supported Calculations are :class:`.DPCalculation` and :class:`.Channel`.
    temperature : Celsius
        Assumed temperature for hydraulic calculations
    strategy : dict[Calculation, Callable[[KgPerS, Celsius], Pascal]] | None
        For unknown calculations, pressure drop functions :math:`\Delta p(\dot{m}, T)`
        may be provided. These are used when the Calculation isn't identified as
        known types or protocols, and failing that, the guess is ``0.0``.

    Returns
    -------
    State
        A guess in which pressures are computed from the known flows, and the flows themselves.
    """

    k_guess = keymap(k.component_edge, mdots)
    s = set(map(k.component_edge, k.components))
    if s != set(k_guess):
        missing_edges = s - set(k_guess)
        missing = sorted(
            getattr(c, "name", str(c)) for c in k.components if k.component_edge(c) in missing_edges
        )
        raise MissingFlowError(
            f"Missing flow data in edges {missing_edges} — provide mdot for their components {missing}."
        )

    strategy = strategy or {}

    def _get_dp(x: Calculation) -> Pascal:
        m = k_guess[k.component_edge(x)]

        match x:
            case Pump():
                # Safe because Pump has x.p.
                # noinspection PyUnresolvedReferences
                return x.p or 0.0
            case Flapper():
                # Closed flapper (t_open = inf): dp is undetermined, guess 0; open: its open-state resistance law.
                if np.isposinf(x.t_open):
                    return 0.0
                return -local_pressure_by_mdot(m, x.fluid.density(temperature), x.f, x._A)
            case DPCalculation():
                # Safe because LumpedComponent has dp_out in its protocol.
                # noinspection PyUnresolvedReferences
                return x.dp_out(Tin=np.array([temperature]), mdot=m, mdot2=0.0)
            case Channel():
                # Safe because Channel has pressure.
                # noinspection PyUnresolvedReferences
                return np.sum(x.pressure(mdot=m, mdot2=0.0, T=(T := np.full(x.n, temperature)), Tw=T))
            case _:
                return strategy.get(x, just(0.0))(m, temperature)

    pressures = {x.name: dict(pressure=_get_dp(x)) for x in k.components}

    def _htc_guess(c: ChannelAndContacts) -> dict[str, Value]:
        # h_left/h_right are algebraic (residual h_calc - h_var), so any finite guess is self-correcting; one is still needed so the State can be loaded.
        T = np.full(c.n, temperature)
        h0 = c.h_wall(T_wall=T, T_cool=T, mdot=k_guess[k.component_edge(c)], pressure=k.ref_pressure or 1e5)
        return dict(h_left=h0, h_right=h0)

    htc = {c.name: _htc_guess(c) for c in k.components if isinstance(c, ChannelAndContacts)}

    junctions = [node for node in k.g.nodes if isinstance(node, Junction)]
    T_vars = ["Tin", "T", "T_wall_left", "T_wall_right", "T_cool"]
    Ts = State.uniform(list(k.components) + junctions, temperature, *T_vars)
    p = np.fromiter(
        _float_values(pressures, map(lambda x: x.name, k.components)),
        dtype=float,
        count=len(k.components),
    )

    a = np.zeros(len(k))
    a[k.variables_by_type["abs_pressure"]] = k.ref_pressure + k._abs_matrix @ p

    return State.merge(Ts, pressures, htc, {k.name: k.save(a) | k_guess})


class GravityMismatchError(StreamError, ValueError):
    pass


def _is_nc_intent(pump: Pump) -> bool:
    """A pump that imposes neither a nonzero head nor a nonzero flow — the cheap proxy for a
    buoyancy-driven (natural-convection) leg, where the flow may run in either direction."""
    forced_head = pump.p is not None and pump.p != 0
    forced_flow = pump.mdot0 is not None and pump.mdot0 != 0
    return not (forced_head or forced_flow)


def _gravity_reverse_flow_sources(k: Kirchhoff) -> Iterable[tuple]:
    """Yield ``(gravity, reversed_flow_source)`` for every Gravity on k's flow graph.

    A component's inlet temperature under reversed flow (``Tin_minus``) is fed by its
    *downstream* neighbour in the series chain ``[tail_junction, *comps, head_junction]``
    (this is exactly the ``Tin_minus`` supplier the Aggregator would route). That neighbour is
    the density temperature buoyancy will use when the flow reverses.
    """
    for u, v, comps in k.g.edges(data=COMPS):
        series = [u, *comps, v]
        for i, comp in enumerate(comps, start=1):
            if isinstance(comp, Gravity):
                yield comp, series[i + 1]


def check_gravity_mismatch(
    k: Kirchhoff,
    temperature: Celsius = 10.0,
    strategy: HydraulicStrategyMap | None = None,
    tol: float = 1e-5,
    head: Pascal = 1.0,
) -> None:
    r"""Report if :math:`\sum_{loop}\Delta p(\dot{m}=0) \neq 0` for any loop

    Usually, if there are no flows at all and thermally equal, total pressure drops in loops should be trivially zero.
    If that's not the case, it is probably due to gravity pressures of differing heights.

    This is a tool to allow users to inspect their models for such glaring issues.
    It relies on :meth:`guess_steady_state`.

    .. note::
        The unclosed-loop test is a **static, zero-flow** check: it evaluates every :math:`\Delta p`
        at :math:`\dot m = 0`, so it is *direction-blind* and cannot see a Gravity that sources the
        wrong temperature under reversed flow. A second, topology-only pass (heuristic, ``warn``-only)
        classifies each Gravity's reversed-flow temperature supplier when a natural-convection pump
        is present; see :data:`_gravity_reverse_flow_sources`.

    Parameters
    ----------
    k : Kirchhoff
        Kirchhoff Calculation
    temperature : Celsius
        Though quite unimportant, some temperature for the hydraulic calculation must be assumed.
    strategy : dict[Calculation, Callable[[KgPerS, Celsius], Pascal]] | None
        For unknown calculations, pressure drop functions :math:`\Delta p(\dot{m}, T)` may be provided.
        These are used when the Calculation isn't identified as known types or protocols, and failing that,
        the guess is ``0.0``. Must be a mapping ``{Calculation: callable}`` or ``None``.
    tol: float
        Tolerance for deciding total pressure drops. Default is 1e-5.
    head: Pascal
        Convert units to meter head for convenience when checking height differences.
        Default output is in Pascal.

    Raises
    ------
    GravityMismatchError
    TypeError
        If ``strategy`` is neither ``None`` nor a mapping.
    """
    if strategy is not None and not isinstance(strategy, Mapping):
        raise TypeError(
            f"strategy must be a HydraulicStrategyMap ({{Calculation: (mdot, T) -> dp}}) or None, "
            f"not {type(strategy).__name__}; the per-component drop functions are looked up by "
            f"strategy.get(component)."
        )
    comps = k.components
    md = dict.fromkeys(comps, 0.0)

    deal_with_pressure_pumps = {p.name: dict(pressure=0.0) for p in comps if isinstance(p, Pump)}
    hs = guess_hydraulic_steady_state(k, md, temperature, strategy)
    s = State.merge(hs, deal_with_pressure_pumps)
    p = np.fromiter(_float_values(s, map(lambda x: x.name, comps)), dtype=float, count=len(comps))
    p_errors = k.kvl_errors(p) / head
    almost_zeros = np.isclose(0.0, p_errors, atol=tol)

    # Warn-only heuristic: only a reversible leg is at risk, so run this only when a natural-convection pump is present.
    if any(isinstance(c, Pump) and _is_nc_intent(c) for c in comps):
        for grav, reverse_src in _gravity_reverse_flow_sources(k):
            if not isinstance(reverse_src, HeatExchanger):
                warnings.warn(
                    f"Gravity {grav.name!r} sources its reversed-flow density temperature from "
                    f"{str(reverse_src)!r} (state-dependent), not a fixed-temperature boundary; under "
                    f"reversed/natural-convection flow its buoyancy will use the hot outlet temperature. "
                    f"Sandwich it between two HeatExchangers, or reset the temperature on the reversed side."
                )

    if np.any(~almost_zeros):
        bad_loops_components = [(k.loop_components(i), p_errors[i].item()) for i in np.flatnonzero(~almost_zeros)]
        raise GravityMismatchError(
            f"There are unclosed loops when flows are 0.\n"
            f"The following is a report of those loop components "
            f"and their aggregate pressure difference (in head = {head} [Pascal]):\n"
            f"{bad_loops_components}"
        )
