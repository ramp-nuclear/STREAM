"""Tools for post-processing analysis of the thresholds for different phenomena in channels.

These tools are used to analyse the results in post-processing and add these thresholds to the
analysed state.
The pattern to use this is as follows:

>>> from functools import partial
>>> onb_left: ThresholdFunction = partial(Bergles_Rohsenow_T_ONB, direction=Direction.left)
>>> onb_right: ThresholdFunction = partial(Bergles_Rohsenow_T_ONB, direction=Direction.right)
>>> osv: ThresholdFunction = partial(Saha_Zuber_OSV, direction=Direction.left)
>>> post_analysis = threshold_analysis(CHF=Sudo_Kaminaga_CHF, OSV=osv, \
ONB_left=onb_left, ONB_right=onb_right)
>>> # agr = Aggregator(...)
>>> # state = agr.solve_steady(...)
>>> # state = post_analysis(state, agr, "channel")  # Given that "channel" is how the Channel is
>>> #                                               # called in agr

"""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from typing import Callable, Protocol

import numpy as np

from stream.aggregator import Aggregator, Solution
from stream.calculations.break_flow import SATURATION_PRESSURE_FLOOR
from stream.calculations.channel import ChannelAndContacts, ChannelVar, Direction, SaturationReachedError
from stream.calculations.kirchhoff import to_str
from stream.errors import StreamError
from stream.physical_models.heat_transfer_coefficient.temperatures import (
    Bergles_Rohsenow_dT_ONB,
)
from stream.physical_models.thresholds import (
    Fabrega_CHF as _Fabrega_CHF,
)
from stream.physical_models.thresholds import (
    Mirshak_CHF as _Mirshak_CHF,
)
from stream.physical_models.thresholds import (
    Saha_Zuber_OSV_computed_bulk as _Saha_Zuber_OSV,
)
from stream.physical_models.thresholds import (
    Sudo_Kaminaga_CHF as _Sudo_Kaminaga_CHF,
)
from stream.physical_models.thresholds import (
    Whittle_Forgan_OFI as _Whittle_Forgan_OFI,
)
from stream.physical_models.thresholds import (
    boiling_power as _boiling_power,
)
from stream.physical_models.thresholds import (
    saturation_margin,
)
from stream.pipe_geometry import EffectivePipe
from stream.state import CalcState, State, StateTimeseries
from stream.substances import LiquidFuncs, light_water
from stream.units import Celsius, Meter, MPerS2, Value, Watt, WPerM2, g
from stream.utilities import factor


class ThresholdFunction(Protocol):
    """A Protocol for how we expect our input functions to look like for the
    :func:`.threshold_analysis_factory`.

    """

    def __call__(
        self,
        *,
        state: CalcState,
        fluid: LiquidFuncs,
        pipe: EffectivePipe,
        dz: Meter,
        **_,
    ) -> Value: ...


def threshold_analysis(
    **funcs: ThresholdFunction,
) -> Callable[[State, Aggregator, str], State]:
    """A factory to create a function that can analyze an aggregator's State
    to yield a new State with threshold values.

    Parameters
    ----------
    funcs:  ThresholdFunction
        Threshold value functions, named for their threshold.

    Returns
    -------
    Callable[[State, Aggregator, str], State]
        A function that adds threshold values to a state.

    """

    def _analyzer(state: State, agg: Aggregator, calc: str) -> State:
        s = deepcopy(state)
        substate = s[calc]
        kw = {}
        channel: ChannelAndContacts = agg[calc]  # type: ignore
        protocol_params = filter(
            lambda x: x not in {"self", "state", "_"},
            signature(ThresholdFunction.__call__).parameters.keys(),
        )
        for attr in protocol_params:
            try:
                kw[attr] = getattr(channel, attr)
            except AttributeError:
                raise AttributeError(
                    f"The aggregator's {calc} calculation did not have a {attr} "
                    "attribute, and it should have because we analyze channels"
                )
        for key, func in funcs.items():
            substate[key] = func(state=substate, **kw)
        return s

    return _analyzer


STS = StateTimeseries


def transient_threshold_analysis(
    **funcs: ThresholdFunction,
) -> Callable[[STS, Aggregator, str], STS]:
    """A factory to create a function that can analyze an aggregator's StateTimeseries
    to yield a new StateTimeseries with threshold values.

    Parameters
    ----------
    funcs:  Callable[[State, EffectivePipe], Value]
        Threshold value functions, named for their threshold.

    Returns
    -------
    Callable[[StateTimeseries, Aggregator, str], StateTimeseries]
        A function that adds threshold values to a state.

    See Also
    --------
    threshold_analysis
    """
    ta = threshold_analysis(**funcs)

    def _analyzer(state_time_series: STS, agg: Aggregator, calc: str) -> STS:
        return {k: ta(v, agg, calc) for k, v in state_time_series.items()}

    return _analyzer


@dataclass(frozen=True)
class SaturationCrossing:
    """Cells of one channel whose bulk coolant is at or past saturation."""

    channel: str
    cells: list[int]
    T_bulk: list[float]
    Tsat: list[float]


def channel_saturation_crossings(state: State, agg: Aggregator) -> list[SaturationCrossing]:
    """Cells in each :class:`ChannelAndContacts` whose bulk coolant is at or above
    saturation in ``state`` — the validity boundary of the single-phase (with
    subcooled boiling) channel model. Empty when every channel is subcooled.

    Tsat is evaluated per cell at the channel's static pressure, using the exact
    :func:`~stream.physical_models.thresholds.saturation_margin` criterion the
    transient stop guard uses, so the two agree on where validity ends.
    """
    crossings: list[SaturationCrossing] = []
    for node in agg.graph:
        if not isinstance(node, ChannelAndContacts):
            continue
        cs = state[node.name]
        T_bulk = np.atleast_1d(np.asarray(cs[ChannelVar.tbulk], dtype=float))
        Tsat = np.atleast_1d(np.asarray(node.fluid.sat_temperature(cs[ChannelVar.static_pressure]), dtype=float))
        crossed = np.flatnonzero(saturation_margin(T_bulk, Tsat) <= 0.0)
        if crossed.size:
            crossings.append(
                SaturationCrossing(node.name, crossed.tolist(), T_bulk[crossed].tolist(), Tsat[crossed].tolist())
            )
    return crossings


def _as_states(result, agg: Aggregator, times):
    """Normalize a Solution / State / StateTimeseries / raw solve vector / raw trajectory
    to a State or StateTimeseries the checker can scan.

    A :class:`~stream.aggregator.Solution` is saved via ``agg`` so the checker gets its
    TRUE absolute times, not the ``range(len(result))`` axis a raw 2-D trajectory falls
    back to when no ``times`` are supplied. Raw arrays keep that fallback behavior."""
    if isinstance(result, Solution):
        return agg.save(result)
    if isinstance(result, np.ndarray):
        if result.ndim == 1:
            return agg.save(result)
        ts = times if times is not None else range(len(result))
        return {float(t): agg.save(row) for t, row in zip(ts, result)}
    return result


def _is_timeseries(data) -> bool:
    # A State is keyed by calculation name (str); a StateTimeseries by time (number).
    return bool(data) and all(isinstance(k, (int, float, np.number)) for k in data)


def first_saturation_crossing(result, agg: Aggregator, *, times=None):
    """The earliest saturated state and its crossings, or ``None`` if nothing crosses.

    ``result`` may be a single :class:`~stream.state.State` (a steady solution or a
    failed solve's last iterate), a :class:`~stream.state.StateTimeseries`, or a raw
    solve vector / trajectory (e.g. a caught error's ``e.y``, with optional ``times``).
    Returns ``(time_or_None, crossings)``.
    """
    data = _as_states(result, agg, times)
    if _is_timeseries(data):
        for t in sorted(data):
            crossings = channel_saturation_crossings(data[t], agg)
            if crossings:
                return t, crossings
        return None
    crossings = channel_saturation_crossings(data, agg)
    return (None, crossings) if crossings else None


# A raw solve vector/trajectory carries no convergence status, so a checker's verdict may reflect a mid-iteration excursion, not a solution.
_ITERATE_SENTENCE = "input may be a non-converged iterate — verify against a converged solution"


def _is_raw_iterate(result) -> bool:
    """True when the checker input is a raw solve vector/trajectory (a bare
    :class:`numpy.ndarray`), rather than a status-bearing :class:`Solution` or a
    saved :class:`~stream.state.State`."""
    return isinstance(result, np.ndarray)


def _finalize_message(message: str, note: str, raw: bool) -> str:
    """Prepend ``note`` and append the non-converged-iterate caveat (when ``raw``)."""
    if note:
        message = note + message
    if raw:
        message = f"{message} {_ITERATE_SENTENCE}"
    return message


def raise_on_saturation(result, agg: Aggregator, *, times=None, note: str = "") -> None:
    """Raise :class:`SaturationReachedError` if any channel's bulk coolant is at or
    past saturation in ``result`` (the worst-offending channel, earliest time for a
    trajectory). No-op otherwise. Use after a steady solve, or on a caught
    transient/steady failure's state, to attribute it in domain terms; ``note`` is
    prepended to the message (e.g. ``"no converged steady solution — "``).

    When ``result`` is a raw solve vector/trajectory (a bare array, not a
    :class:`~stream.aggregator.Solution` or :class:`~stream.state.State`), the
    message gains one caveat that the input may be a non-converged iterate."""
    found = first_saturation_crossing(result, agg, times=times)
    if found is None:
        return
    _t, crossings = found
    worst = max(crossings, key=lambda c: max(np.subtract(c.T_bulk, c.Tsat)))
    err = SaturationReachedError(worst.channel, worst.cells, worst.T_bulk, worst.Tsat)
    message = _finalize_message(err.message, note, _is_raw_iterate(result))
    if message != err.message:
        err.message = message
        err.args = (message, *err.args[1:])
    raise err


@dataclass(frozen=True)
class DomainViolation:
    """One out-of-domain cell of a Calculation's fluid state.

    Attributes
    ----------
    calc_name: str
        Name of the Calculation whose state is out of domain.
    variable: str
        The offending state variable (e.g. ``'T_cool'`` or ``'static_pressure'``).
    cell: int | None
        Cell index within the variable, or ``None`` for a scalar variable.
    value: float
        The offending value (°C for temperatures, Pa for pressure).
    bound: tuple[float, float] | float
        The violated bound: the fluid validity ``(T_min, T_max)`` range for a
        temperature, or the scalar ``0.0`` floor for a static pressure.
    t: float | None
        Time of the violation for a trajectory; ``None`` for a single State.
    """

    calc_name: str
    variable: str
    cell: int | None
    value: float
    bound: tuple[float, float] | float
    t: float | None


class DomainValidityError(StreamError, RuntimeError):
    """A fluid state lies outside its declared validity domain.

    Raised post-hoc by :func:`raise_on_domain`, never on a solve path.
    ``StreamError``-family, so ``except StreamError`` catches it."""


def _pressure_variable(cs) -> str | None:
    """The state variable to check for a non-negative absolute pressure: the
    canonical static pressure if present, else the first ``*pressure*`` variable
    that is not the signed pressure-*drop* (which is legitimately negative)."""
    if ChannelVar.static_pressure in cs:
        return ChannelVar.static_pressure
    for var in cs:
        if "pressure" in str(var).lower() and str(var) != str(ChannelVar.pressure_drop):
            return var
    return None


def _domain_violations_in_state(state, agg: Aggregator, t) -> list[DomainViolation]:
    """Out-of-domain cells in one State: for every node whose ``fluid`` declares a
    ``validity`` range, its ``T*`` variables against that range and its static
    pressure against ``>= 0``. Non-finite values count as out of domain."""
    violations: list[DomainViolation] = []
    for node in agg.graph:
        validity = getattr(getattr(node, "fluid", None), "validity", None)
        if validity is None:
            continue
        cs = state.get(node.name)
        if cs is None:
            continue
        lo, hi = float(validity[0]), float(validity[1])
        for var, value in cs.items():
            if not str(var).startswith("T"):
                continue
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            scalar = arr.size == 1
            for i, v in enumerate(arr):
                if not np.isfinite(v) or v < lo or v > hi:
                    violations.append(
                        DomainViolation(node.name, str(var), None if scalar else i, float(v), (lo, hi), t)
                    )
        p_var = _pressure_variable(cs)
        if p_var is not None:
            arr = np.atleast_1d(np.asarray(cs[p_var], dtype=float))
            scalar = arr.size == 1
            for i, v in enumerate(arr):
                if not np.isfinite(v) or v < 0.0:
                    violations.append(
                        DomainViolation(node.name, str(p_var), None if scalar else i, float(v), 0.0, t)
                    )
    return violations


def domain_report(result, agg: Aggregator, *, times=None) -> list[DomainViolation]:
    """Fluid-domain violations in ``result``. Accepts the same inputs as the
    saturation checkers (a :class:`~stream.state.State`,
    :class:`~stream.state.StateTimeseries`, raw solve vector/trajectory, or a
    :class:`~stream.aggregator.Solution`).

    Only nodes exposing a ``fluid`` whose ``validity`` range is declared are
    scanned; a node's variables whose names start with ``T`` are checked against
    that range, and its static pressure against ``>= 0``. A converged-but-invalid
    state and a failed iterate are both caught. For a trajectory, violations are
    reported per time, earliest first."""
    data = _as_states(result, agg, times)
    if _is_timeseries(data):
        out: list[DomainViolation] = []
        for t in sorted(data):
            out.extend(_domain_violations_in_state(data[t], agg, float(t)))
        return out
    return _domain_violations_in_state(data, agg, None)


def _domain_severity(v: DomainViolation) -> float:
    """How far outside its bound a violation is (worst wins in ``raise_on_domain``)."""
    if not np.isfinite(v.value):
        return np.inf
    if isinstance(v.bound, tuple):
        lo, hi = v.bound
        return max(lo - v.value, v.value - hi)
    return -v.value  # pressure floor at 0: the more negative, the worse


def _domain_message(v: DomainViolation) -> str:
    where = f"at cell {v.cell} " if v.cell is not None else ""
    if isinstance(v.bound, tuple):
        lo, hi = v.bound
        return (
            f"{v.variable} = {v.value:g} °C {where}of {v.calc_name!r} is outside the "
            f"fluid validity range [{lo:g}, {hi:g}] °C"
        )
    return (
        f"{v.variable} = {v.value:g} Pa {where}of {v.calc_name!r} is negative "
        "(a static/absolute pressure must be >= 0)"
    )


def raise_on_domain(result, agg: Aggregator, *, times=None, note: str = "") -> None:
    """Raise :class:`DomainValidityError` if any fluid state in ``result`` is outside
    its declared validity domain (the worst excursion, earliest time for a
    trajectory). No-op otherwise. Mirrors :func:`raise_on_saturation`: use it after
    a steady solve or on a caught failure's state to attribute it in domain terms;
    ``note`` is prepended, and a raw-array input gains the non-converged-iterate
    caveat."""
    violations = domain_report(result, agg, times=times)
    if not violations:
        return
    times_present = [v.t for v in violations if v.t is not None]
    if times_present:
        earliest = min(times_present)
        candidates = [v for v in violations if v.t == earliest]
    else:
        candidates = violations
    worst = max(candidates, key=_domain_severity)
    message = _finalize_message(_domain_message(worst), note, _is_raw_iterate(result))
    raise DomainValidityError(message)


@dataclass(frozen=True)
class CavitationCrossing:
    """One component whose liquid reached saturation at the absolute pressure routed to it.

    Attributes
    ----------
    component: str
        Name of the component whose ``p_abs`` reached saturation.
    t: float or None
        Time of the first crossing for a trajectory; ``None`` for a single State.
    margin: Celsius
        The smallest subcooling margin over the whole result, ``Tsat(p_abs) - Tin``.
    """

    component: str
    t: float | None
    margin: Celsius


class CavitationError(StreamError, RuntimeError):
    """A component's liquid reached saturation at the absolute pressure it carries.

    Raised post-hoc by :func:`raise_on_cavitation`, never on a solve path.
    ``StreamError``-family, so ``except StreamError`` catches it."""


def _cavitation_margins(state, agg: Aggregator):
    """``(name, margin)`` for every absolute-pressure component of every flow solver in
    ``agg``: the saturation temperature at its routed ``p_abs``, less the temperature it
    carries, at its worst point in ``state``."""
    for node in agg.graph:
        for comp in getattr(node, "abs_pressure_comps", ()):
            fluid = getattr(comp, "fluid", None)
            routed, carried = state.get(node.name), state.get(comp.name)
            if fluid is None or routed is None or carried is None or "Tin" not in carried:
                continue
            p_abs = np.atleast_1d(np.asarray(routed[to_str(("p_abs", comp))], dtype=float))
            Tin = np.atleast_1d(np.asarray(carried["Tin"], dtype=float))
            yield comp.name, float(np.min(fluid.sat_temperature(np.maximum(p_abs, SATURATION_PRESSURE_FLOOR)) - Tin))


def cavitation_crossings(result, agg: Aggregator, *, times=None) -> list[CavitationCrossing]:
    """Components in ``agg`` whose absolute pressure fell to the saturation pressure of the
    liquid they carry — where a single-phase leg starts to flash. Empty when every one of
    them stays subcooled.

    Only components handed to a flow solver's ``abs_pressure_comps`` have an absolute
    pressure to judge, and of those only the ones exposing a ``fluid`` and a ``Tin`` of
    their own are scanned; a heated channel's bulk saturation belongs to
    :func:`channel_saturation_crossings` instead. The saturation temperature is read at
    the pressure floor the break-flow sentinel uses, so a pressure driven to zero yields a
    margin rather than an extrapolation.

    ``result`` may be a single :class:`~stream.state.State`, a
    :class:`~stream.state.StateTimeseries`, a raw solve vector/trajectory (with optional
    ``times``), or a :class:`~stream.aggregator.Solution`. Each crossing reports the
    earliest time it happened and the smallest margin reached over the whole result.
    """
    data = _as_states(result, agg, times)
    if not _is_timeseries(data):
        return [CavitationCrossing(name, None, m) for name, m in _cavitation_margins(data, agg) if m <= 0.0]
    first: dict[str, float] = {}
    worst: dict[str, float] = {}
    for t in sorted(data):
        for name, margin in _cavitation_margins(data[t], agg):
            worst[name] = min(margin, worst.get(name, np.inf))
            if margin <= 0.0 and name not in first:
                first[name] = float(t)
    return [CavitationCrossing(name, t, worst[name]) for name, t in first.items()]


def raise_on_cavitation(result, agg: Aggregator, *, times=None, note: str = "") -> None:
    """Raise :class:`CavitationError` if any component's absolute pressure reached the
    saturation pressure of its liquid in ``result`` (the deepest excursion). No-op
    otherwise. Mirrors :func:`raise_on_saturation`: use it after a steady solve or on a
    caught failure's state to attribute it in domain terms; ``note`` is prepended, and a
    raw-array input gains the non-converged-iterate caveat."""
    crossings = cavitation_crossings(result, agg, times=times)
    if not crossings:
        return
    worst = min(crossings, key=lambda c: c.margin)
    when = "" if worst.t is None else f" from t = {worst.t:g} s"
    message = (
        f"the absolute pressure carried by {worst.component!r} reached the saturation pressure of "
        f"its liquid{when} (subcooling margin {worst.margin:.4g} °C): the single-phase discharge and "
        f"friction laws on that leg hold only while it stays liquid."
    )
    raise CavitationError(_finalize_message(message, note, _is_raw_iterate(result)))


# Below any physical convective coefficient: only a numerically-zero wall coupling (a stagnant / natural-convection cell) trips the q/h blow-up guard.
_H_STAGNANT = 1e-9


def _wall_temp_limit(tbulk: Celsius, q: WPerM2, h, where: str) -> Celsius:
    r"""``tbulk + q/h``, but NaN — with one warning — at cells where ``h`` is
    numerically zero. A stagnant / natural-convection cell has no defined wall-
    temperature limit; returning ``q/h``'s ``inf`` there gives a nonsense number
    with no hint of the cause."""
    h_arr = np.asarray(h, dtype=float)
    stagnant = np.abs(h_arr) <= _H_STAGNANT
    if not stagnant.any():
        return tbulk + q / h
    warnings.warn(
        f"wall-temperature limit undefined at cell(s) {np.flatnonzero(stagnant).tolist()} "
        f"({where}): h≈0 — a stagnant / natural-convection state; returning NaN there "
        "instead of a q/h blow-up.",
        stacklevel=2,
    )
    tw = tbulk + q / np.where(stagnant, 1.0, h_arr)
    return np.where(stagnant, np.nan, tw)


def _tw(state: CalcState, direction: Direction, tbulk: Celsius, inhomogeneity_factor) -> Celsius:
    if ChannelVar.get("heatflux", direction) in state:
        q = state[ChannelVar.get("heatflux", direction)] * inhomogeneity_factor
        h = state[ChannelVar.get("h", direction)]
        return _wall_temp_limit(tbulk, q, h, f"{direction} wall")
    return -np.inf


def twall_limit(*, state: CalcState, inhomogeneity_factor: float = 1.0, **_) -> Celsius:
    """A function that finds the limiting wall temperature.

    We can't just take the physical twall from the calculation because of fuel inhomogeneity, which isn't taken into
    account in the physical solution.

    Parameters
    ----------
    state: CalcState
        The channel state to analyze.
    inhomogeneity_factor: float
        Factor to make flux worse by locally (fuel inhomogeneity, usually).

    Returns
    -------
    Celsius
        The maximal wall temperature for the wall temperature limit check

    """
    tbulk = state[ChannelVar.tbulk]
    twall_right, twall_left = (
        _tw(state, direction, tbulk, inhomogeneity_factor) for direction in (Direction.right, Direction.left)
    )
    return np.maximum(twall_right, twall_left)


def Saha_Zuber_OSV(
    *,
    state: CalcState,
    fluid: LiquidFuncs,
    pipe: EffectivePipe,
    dz: Meter,
    direction: Direction,
    inhomogeneity_factor: float = 1.0,
    **_,
) -> WPerM2:
    """A wrapper for Saha & Zuber based OSV.

    For details, see the underlying :func:`~stream.physical_models.thresholds.Saha_Zuber_OSV`.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.Saha_Zuber_OSV`.

    Parameters
    ----------
    state: State
        The channel state to analyze.
    pipe: EffectivePipe
        The geometry of the flow channel.
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.
    dz: Meter
        Cell length for each cell in the channel.
    direction: Direction
        Which wall direction to take the power shape from.
    inhomogeneity_factor: float
        Factor to make flux worse by locally (fuel inhomogeneity, usually).

    Returns
    -------
    WPerM2
        The flux which for the given local physical state would have caused OSV to occur there.

    """
    tb = state[ChannelVar.tbulk]
    tin = state[ChannelVar.tin]
    pressure = state[ChannelVar.static_pressure]
    q = state[ChannelVar.get("heatflux", direction=direction)]
    coolant = fluid.to_properties(tb, pressure)
    mdot = state[ChannelVar.mass_flow]
    return _Saha_Zuber_OSV(
        T_inlet=tin,
        coolant=coolant,
        mdot=mdot,
        Dh=pipe.hydraulic_diameter,
        area=pipe.area,
        heated_perimeter=pipe.heated_perimeter,
        flux_shape=q,
        dz=dz,
        flux_enworse=inhomogeneity_factor,
    )


def boiling_power(*, state: CalcState, fluid: LiquidFuncs, **__) -> Watt:
    """A wrapper for the :func:`~stream.physical_models.thresholds.boiling_power` function.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.boiling_power`

    Parameters
    ----------
    state: CalcState
        The state of the channel
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.

    Returns
    -------
    Watt
        The power required to reach the saturation temperature.

    """
    mdot = state[ChannelVar.mass_flow]
    tin = state[ChannelVar.tin]
    pressure = state[ChannelVar.static_pressure]
    cp_in = fluid.specific_heat(tin)
    tsat = fluid.sat_temperature(pressure)
    return _boiling_power(mdot=mdot, T_sat=tsat, Tin=tin, cp_in=cp_in)


def Whittle_Forgan_OFI(*, state: CalcState, fluid: LiquidFuncs, pipe: EffectivePipe, **_) -> Watt:
    """A wrapper for the :func:`~stream.physical_models.thresholds.Whittle_Forgan_OFI` function.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.Whittle_Forgan_OFI`

    Parameters
    ----------
    state: State
        The channel state to analyze.
    pipe: EffectivePipe
        The geometry of the flow channel.
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.

    Returns
    -------
    Watt
        The power necessary to achieve OFI conditions according to Whittle & Forgan with Fabrega.

    """
    mdot = state[ChannelVar.mass_flow]
    pressure = state[ChannelVar.static_pressure]
    tin = state[ChannelVar.tin]
    tsat = fluid.sat_temperature(pressure[-1 if mdot >= 0 else 0])
    return _Whittle_Forgan_OFI(
        mdot=mdot,
        sat_temperature=tsat,
        inlet_temperature=tin,
        pipe=pipe,
        cp=fluid.specific_heat,
    )


def Sudo_Kaminaga_CHF(
    *,
    state: CalcState,
    fluid: LiquidFuncs,
    pipe: EffectivePipe,
    gravity: MPerS2 = g,
    **_,
) -> WPerM2:
    """A wrapper for the :func:`~stream.physical_models.thresholds.Sudo_Kaminaga_CHF` function.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.Sudo_Kaminaga_CHF`

    Parameters
    ----------
    state: State
        The channel state to analyze.
    pipe: EffectivePipe
        The geometry of the flow channel.
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.
    gravity: MPerS2
        Gravitational acceleration constant in the channel.

    Returns
    -------
    WPerM2
        The flux necessary at each point to have achieved CHF conditions given
        the rest of the channel stays as is.

    """
    tb = state[ChannelVar.tbulk]
    pressure = state[ChannelVar.static_pressure]
    tsat = fluid.sat_temperature(pressure)
    sat_cool = fluid.to_properties(tsat, pressure)
    mdot = state[ChannelVar.mass_flow]
    return _Sudo_Kaminaga_CHF(T_bulk=tb, sat_coolant=sat_cool, mdot=mdot, pipe=pipe, g=gravity)


def Mirshak_CHF(*, state: CalcState, fluid: LiquidFuncs, pipe: EffectivePipe, **_) -> WPerM2:
    """A wrapper for the :func:`~stream.physical_models.thresholds.Mirshak_CHF` function.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.Mirshak_CHF`

    Parameters
    ----------
    state: State
        The channel state to analyze.
    pipe: EffectivePipe
        The geometry of the flow channel.
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.

    Returns
    -------
    WPerM2
        The flux necessary at each point to have achieved CHF conditions given
        the rest of the channel stays as is.

    """
    tb = state[ChannelVar.tbulk]
    pressure = state[ChannelVar.static_pressure]
    tsat = fluid.sat_temperature(pressure)
    mdot = state[ChannelVar.mass_flow]
    v = mdot / pipe.area / fluid.density(tb)
    return _Mirshak_CHF(T_bulk=tb, T_sat=tsat, pressure=pressure, v=v)


def Fabrega_CHF(*, state: CalcState, fluid: LiquidFuncs, pipe: EffectivePipe, **_) -> WPerM2:
    """A wrapper for the :func:`~stream.physical_models.thresholds.Fabrega_CHF` function.

    See Also
    --------
    :func:`~stream.physical_models.thresholds.Fabrega_CHF`

    Parameters
    ----------
    state: State
        The channel state to analyze.
    pipe: EffectivePipe
        The geometry of the flow channel.
    fluid: LiquidFuncs
        The functional properties of the fluid in the channel.

    Returns
    -------
    WPerM2
        The flux necessary at each point to have achieved CHF conditions given
        the rest of the channel stays as is.

    """
    tin = state[ChannelVar.tin]
    pressure = state[ChannelVar.static_pressure]
    tsat = fluid.sat_temperature(pressure)
    return _Fabrega_CHF(Tin=tin, T_sat=tsat, Dh=pipe.hydraulic_diameter)


def Bergles_Rohsenow_T_ONB(
    *,
    state: CalcState,
    direction: Direction,
    onb_factor: float = 1.0,
    inhomogeneity_factor: float = 1.0,
    **_,
) -> Celsius:
    r"""A wrapper for :func:`~stream.physical_models.heat_transfer_coefficient.temperatures.Bergles_Rohsenow_T_ONB`

    The wall temperature at which ONB would occur according to Bergles and Rohsenow.

    The fluid is set to light water because that's what Bergles & Rohsenow is good for.

    See Also
    --------
    :func:`~stream.physical_models.heat_transfer_coefficient.temperatures.Bergles_Rohsenow_T_ONB`

    Parameters
    ----------
    state: CalcState
        The state of the channel
    direction: Direction
        The direction in the channel we want to analyze.
    onb_factor: float
        Relative uncertainty factor increase for the correlation to account for its uncertainty.
    inhomogeneity_factor: float
        Relative factor by which the local flux must be factored to take fuel inhomogeneity into account.

    Returns
    -------
    ONB_margin: Celsius
        T_wall - T_ONB
    """
    pressure = state[ChannelVar.static_pressure]
    tbulk = state[ChannelVar.tbulk]
    h = state[ChannelVar.get("h", direction)]
    q = state[ChannelVar.get("heatflux", direction)] * inhomogeneity_factor
    twall = _wall_temp_limit(tbulk, q, h, f"{direction} wall")
    tsat = light_water.sat_temperature(pressure)
    br = factor(Bergles_Rohsenow_dT_ONB, by=onb_factor)
    return twall - (tsat + br(pressure, q))
