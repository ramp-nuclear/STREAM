"""Helper functions for creating steady state initial guesses for small and specific subsystems"""
import logging
from typing import Callable

import numpy as np
from cytoolz import keymap

from stream import Calculation
from stream.calculations import (
    Pump, DPCalculation, Channel, ChannelAndContacts, Fuel, Kirchhoff,
    PointKinetics, PointKineticsWInput, Junction,
)
from stream.composition.mtr_geometry import symmetric_plate
from stream.state import State
from stream.units import Celsius, KgPerS, Pascal, Watt
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
        **solver_options
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
        ``precondition``
    solver_options: Dict
        Keyword arguments to control steady state solver behavior

    Returns
    -------
    steady: State
    """
    if initial_guess_iterations < 1:
        raise ValueError(f"Must try at least once to obtain values. Was {initial_guess_iterations}")
    plate = symmetric_plate(
        c, f, {c: dict(mdot=mdot, p_abs=p_abs, Tin=Tin),
               f: dict(power=power)}
    ).to_aggregator()

    power_mat = np.zeros(f.shape)
    power_mat[f.meat == 1] = power * f.power_shape
    cp = c.fluid.specific_heat(Tin)
    p_z = np.sum(power_mat, 1)
    q2t_z = p_z / (c.pipe.heated_perimeter * c.dz)
    _tc0 = Tin + np.cumsum(p_z / (np.abs(mdot) * cp))
    tc0 = _tc0 if mdot >= 0 else _tc0[::-1]
    tw0 = tc0
    for _ in range(initial_guess_iterations):
        dp0 = c.pressure(T=tc0, Tw=tw0, mdot=mdot)
        h0 = c.h_wall(T_wall=tw0, T_cool=tc0, mdot=mdot, pressure=p_abs - dp0)
        tw0 = (q2t_z / h0) + tc0
    T0 = np.tile(tw0, f.shape[1]).reshape(f.shape[::-1]).T.flatten()
    # Safe because the initial guess iterations are >= 1
    # noinspection PyUnboundLocalVariable
    y0 = plate.load(
        {f.name: dict(T=T0, T_wall_left=tw0, T_wall_right=tw0),
         c.name: dict(T_cool=tc0, h_left=h0, h_right=h0, pressure=np.sum(dp0)),
         }
    )

    return plate.save(plate.solve_steady(y0, **solver_options))


def point_kinetics_steady_state(
        pk: PointKinetics, power: Watt, power_input: Watt = None
) -> State:
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


class MissingFlowError(Exception):
    """Error to signify missing :class:`.Kirchhoff` flow data."""
    pass


HydraulicStrategy = Callable[[KgPerS, Celsius], Pascal]
HydraulicStrategyMap = dict[Calculation, HydraulicStrategy]


def _float_values(d: dict[str, Value],
                  keys: Iterable[str],
                  inner_key: str = "pressure") -> Iterable[float]:
    for key in keys:
        y = d[key][inner_key]
        yield y if isinstance(y, (float, int)) else y.item()


def guess_hydraulic_steady_state(
        k: Kirchhoff, mdots: dict[Calculation, KgPerS], temperature: Celsius,
        strategy: HydraulicStrategyMap | None = None) -> State:
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
        raise MissingFlowError(f"Missing flow data in edges {s - set(k_guess)}")

    strategy = strategy or {}

    def _get_dp(x: Calculation) -> Pascal:
        m = k_guess[k.component_edge(x)]

        match x:
            case Pump():
                # Safe because Pump has x.p.
                # noinspection PyUnresolvedReferences
                return x.p or 0.0
            case DPCalculation():
                # Safe because LumpedComponent has dp_out in its protocol.
                # noinspection PyUnresolvedReferences
                return x.dp_out(Tin=np.array([temperature]), mdot=m, mdot2=0.0)
            case Channel():
                # Safe because Channel has pressure.
                # noinspection PyUnresolvedReferences
                return np.sum(
                    x.pressure(mdot=m, mdot2=0.0, T=(T := np.full(x.n, temperature)), Tw=T))
            case _:
                return strategy.get(x, just(0.0))(m, temperature)

    pressures = {x.name: dict(pressure=_get_dp(x)) for x in k.components}

    junctions = [node for node in k.g.nodes if isinstance(node, Junction)]
    T_vars = ["Tin", "T", "T_wall_left", "T_wall_right", "T_cool"]
    Ts = State.uniform(list(k.components) + junctions, temperature, *T_vars)
    p = np.fromiter(_float_values(pressures, map(lambda x: x.name, k.components)),
                    dtype=float, count=len(k.components))

    a = np.zeros(len(k))
    a[k.variables_by_type["abs_pressure"]] = k.ref_pressure + k._abs_matrix @ p

    return State.merge(Ts, pressures, {k.name: k.save(a) | k_guess})


class GravityMismatchError(ValueError):
    pass


def check_gravity_mismatch(k: Kirchhoff, temperature: Celsius = 10.0,
                           strategy: HydraulicStrategy | None = None,
                           tol: float = 1e-5, head: Pascal = 1.0,
                           ) -> None:
    r"""Report if :math:`\sum_{loop}\Delta p(\dot{m}=0) \neq 0` for any loop

    Usually, if there are no flows at all and thermally equal, total pressure drops in loops should be trivially zero.
    If that's not the case, it is probably due to gravity pressures of differing heights.

    This is a tool to allow users to inspect their models for such glaring issues.
    It relies on :meth:`guess_steady_state`.

    Parameters
    ----------
    k : Kirchhoff
        Kirchhoff Calculation
    temperature : Celsius
        Though quite unimportant, some temperature for the hydraulic calculation must be assumed.
    strategy : dict[Calculation, Callable[[KgPerS, Celsius], Pascal]] | None
        For unknown calculations, pressure drop functions :math:`\Delta p(\dot{m}, T)` may be provided.
        These are used when the Calculation isn't identified as known types or protocols, and failing that,
        the guess is ``0.0``.
    tol: float
        Tolerance for deciding total pressure drops. Default is 1e-5.
    head: Pascal
        Convert units to meter head for convenience when checking height differences.
        Default output is in Pascal.

    Raises
    ------
    GravityMismatchError
    """
    comps = k.components
    md = dict.fromkeys(comps, 0.0)

    deal_with_pressure_pumps = {p.name: dict(pressure=0.0)
                                for p in comps
                                if isinstance(p, Pump)}
    hs = guess_hydraulic_steady_state(k, md, temperature, strategy)
    s = State.merge(hs, deal_with_pressure_pumps)
    p = np.fromiter(_float_values(s, map(lambda x: x.name, comps)),
                    dtype=float, count=len(comps))
    p_errors = k.kvl_errors(p) / head
    almost_zeros = np.isclose(0.0, p_errors, atol=tol)
    if np.any(~almost_zeros):
        bad_loops_components = [
            (k.loop_components(i), p_errors[i].item())
            for i in np.flatnonzero(~almost_zeros)
        ]
        raise GravityMismatchError(
            f"There are unclosed loops when flows are 0.\n"
            f"The following is a report of those loop components "
            f"and their aggregate pressure difference (in head = {head} [Pascal]):\n"
            f"{bad_loops_components}")
