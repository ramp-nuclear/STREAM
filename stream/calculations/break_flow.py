r"""Break elements — the openings through which a system loses its coolant.

An :class:`Orifice` is the one component behind every break: a hole of a given area
and discharge coefficient, discharging whatever pressure difference stands across it.
What distinguishes a tank-wall hole from a pipe breach from a severed pipe end is where
the orifice sits in the flow graph, not what it computes.

Until it opens it seals the leg it is on (zero flow, pressure left free), so the intact
system can be solved on the very same graph the break will run on.
"""

import logging
import warnings
from typing import Callable, Sequence

import numpy as np

from stream import Calculation, unpacked
from stream.calculation import sealed
from stream.calculations.flapper import continuously_differentiable_relaxation
from stream.calculations.ideal.ideal import LumpedComponent
from stream.physical_models.pressure_drop import mdot_by_local_pressure_smooth
from stream.substances import LiquidFuncs
from stream.units import Array1D, Celsius, KgPerS, Meter, Meter2, Pascal, PerS, Second, Value
from stream.utilities import STREAM_DEBUG, directed_Tin

__all__ = ["Orifice"]

logger = logging.getLogger(__name__)

SATURATION_PRESSURE_FLOOR: Pascal = 700.0
_RE_FLOOR: float = 1.0


@sealed
class Orifice(Calculation):
    r"""A hole discharging a liquid, :math:`\dot{m} = C_d A\sqrt{2\rho\,\Delta p}`
    (written in the inverse, regularized direction), with a life cycle.

    Three things can happen to a break, and all three are smooth in time so a transient
    never has to be restarted:

    - **Opening.** Before ``t_break`` the orifice is sealed: it holds :math:`\dot{m}=0`
      and leaves its pressure free. From ``t_break`` on it ramps to the discharge law
      over ``1/open_rate`` seconds.
    - **Closure.** With ``closes_below=(tank, z)`` the orifice latches shut the first
      time that tank's level reaches ``z`` — the flow frozen at that instant is ramped
      down to zero over ``1/close_rate`` seconds and never reopens. This is how a
      siphon breaker arrests a drain, and how a severed pipe segment is cut out of the
      network without changing the graph.
    - **Flashing.** Given a vena-contracta coefficient ``cc`` and a routed ``p_abs``,
      the orifice watches the pressure at its throat. The single-phase law it uses is
      only valid while that pressure stays above saturation, so crossing raises a
      warning (or stops the run, with ``flashing_terminal``).

    Examples
    --------
    A sealed orifice reports the flow it is holding at zero, whatever the pressure:

    >>> from stream.substances import light_water
    >>> hole = Orifice(light_water, area=1e-4, cd=0.61)
    >>> hole.calculate(np.array([30.0, -5e4]), mdot=0.2, Tin=30.0, t=0.0)
    array([0. , 0.2])

    Once open it follows the discharge law:

    >>> hole.open(0.0)
    >>> float(round(hole.calculate(np.array([30.0, -5e4]), mdot=0.0, Tin=30.0, t=10.0)[1], 6))
    -0.608446

    See Also
    --------
    ~stream.calculations.flapper.Flapper, ~stream.calculations.tank.Tank,
    .mdot_by_local_pressure_smooth
    """

    def __init__(
        self,
        fluid: LiquidFuncs,
        area: Meter2,
        cd: float | Callable[[Value], Value],
        *,
        cc: float | None = None,
        dp_eps: Pascal = 1.0,
        t_break: Second = np.inf,
        open_rate: PerS = 10.0,
        relaxation: Callable[[float], float] = continuously_differentiable_relaxation,
        closes_below: tuple["Tank", Meter] | None = None,
        close_rate: PerS = 10.0,
        flashing_terminal: bool = False,
        mdot_eps: KgPerS | None = None,
        name: str = "Orifice",
    ):
        r"""

        Parameters
        ----------
        fluid: LiquidFuncs
            Coolant properties
        area: Meter2
            Geometric area of the hole.
        cd: float or Callable[[Value], Value]
            Discharge coefficient, either a number or a function of the throat Reynolds
            number (see :mod:`~stream.physical_models.pressure_drop.discharge`). The
            Reynolds number is built from the flow, never from the pressure difference,
            and is floored at 1 so that a stagnant open break still evaluates — every
            discharge correlation returns a near-zero coefficient there, which holds the
            break near zero flow.
        cc: float or None
            Contraction coefficient of the vena contracta. Giving it arms the flashing
            sentinel, which additionally needs ``p_abs`` routed in.
        dp_eps: Pascal
            Half-width of the linear regularization band around ``dp = 0`` in the
            discharge law (see :func:`~.mdot_by_local_pressure_smooth`).
        t_break: Second
            Time the break opens. The default never opens it, leaving the orifice
            sealed until :meth:`open` is called.
        open_rate: PerS
            Reciprocal duration of the opening ramp.
        relaxation: Callable[[float], float]
            Shape of both the opening and the closing ramp, ``r(x<=0) = 0``,
            ``r(x>=1) = 1``.
        closes_below: tuple[Tank, Meter] or None
            A tank and the level at which this orifice latches shut.
        close_rate: PerS
            Reciprocal duration of the closing ramp.
        flashing_terminal: bool
            Stop the transient when the throat pressure reaches saturation, instead of
            only warning.
        mdot_eps: KgPerS or None
            Advection direction-blending width for ``directed_Tin``; ``None`` uses
            ``stream.smoothing.DEFAULT_MDOT_EPS``.
        name: str
            Calculation's name
        """
        self.name = name
        self.fluid = fluid
        self.area = area
        self.cd = cd
        self.cc = cc
        self.dp_eps = dp_eps
        self.t_break = t_break
        self.open_rate = open_rate
        self.relaxation = relaxation
        self.closes_below = closes_below
        self.close_rate = close_rate
        self.flashing_terminal = flashing_terminal
        self.mdot_eps = mdot_eps
        self.t_close = np.inf
        self.m_frozen = 0.0
        self._rho = fluid.density
        self._visc = fluid.viscosity
        self._tsat = fluid.sat_temperature
        self._closed = False
        self._flashed = False

    def open(self, t: Second) -> None:
        """Open the break at time ``t``, ramping in at ``open_rate``."""
        self.t_break = float(t)

    def close(self, t: Second, mdot: KgPerS = 0.0) -> None:
        """Latch the break shut at time ``t``, ramping ``mdot`` down to zero at
        ``close_rate``. A break that is already shut stays shut at its first closure."""
        if not self._closed:
            self._closed = True
            self.t_close = float(t)
            self.m_frozen = float(mdot)

    @unpacked
    def calculate(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        t: Second = None,
        level: Meter = None,
        p_abs: Pascal = None,
        area_factor: float = 1.0,
    ) -> Array1D:
        out = np.empty(2)
        T, dp = variables[0], variables[1]
        time = 0.0 if t is None else float(t)

        if self._closed or time <= self.t_break:
            out[0] = T - Tin
            out[1] = mdot - self._closing_flow(time)
            return out

        relax = self.relaxation((time - self.t_break) * self.open_rate)
        Tin_d = directed_Tin(Tin, Tin_minus, mdot, self.mdot_eps)
        a_eff = self.area * area_factor
        cd_eff = self._cd_eff(mdot, Tin_d, a_eff)
        mdot_calc = -mdot_by_local_pressure_smooth(dp, self._rho(Tin_d), 1.0 / cd_eff**2, a_eff, self.dp_eps)
        out[0] = T - ((1.0 - relax) * Tin + relax * Tin_d)
        out[1] = mdot - relax * mdot_calc
        return out

    # noinspection PyProtocol
    indices = LumpedComponent.indices
    # noinspection PyProtocol
    variables = LumpedComponent.variables
    # noinspection PyProtocol
    mass_vector = LumpedComponent.mass_vector
    __len__ = LumpedComponent.__len__

    def dp_out(self, *, Tin: Celsius, mdot: KgPerS, area_factor: float = 1.0, **_) -> Pascal:
        r"""The pressure gain across the hole at a given flow,
        :math:`-\dot{m}|\dot{m}|/(2\rho (C_d A)^2)` — zero at zero flow, whatever the
        discharge coefficient does there, and always opposing the flow."""
        if not np.any(mdot):
            return 0.0
        a_eff = self.area * area_factor
        throat = self._cd_eff(mdot, Tin, a_eff) * a_eff
        return -mdot * np.abs(mdot) / (2.0 * self._rho(Tin) * throat**2)

    @unpacked
    def event_margin(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS = None,
        Tin: Celsius = None,
        Tin_minus: Celsius | None = None,
        level: Meter = None,
        p_abs: Pascal = None,
        area_factor: float = 1.0,
        **_,
    ) -> Array1D:
        """Margins in a fixed order: latching closure, then flashing onset.

        Either margin is dropped when its sentinel is unarmed, and returns ``+1`` once
        it has fired, so neither can trigger twice.
        """
        margins = []
        if self.closes_below is not None and level is not None:
            margins.append(1.0 if self._closed else float(level) - self.closes_below[1])
        if self.cc is not None and p_abs is not None:
            margins.append(1.0 if self._flashed else self._flashing_margin(mdot, Tin, Tin_minus, p_abs, area_factor))
        return np.array(margins, dtype=float)

    @unpacked
    def change_state(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS = None,
        Tin: Celsius = None,
        Tin_minus: Celsius | None = None,
        t: Second = None,
        level: Meter = None,
        p_abs: Pascal = None,
        area_factor: float = 1.0,
        **_,
    ) -> None:
        if self.closes_below is not None and level is not None and not self._closed:
            if level <= self.closes_below[1]:
                self.close(0.0 if t is None else t, 0.0 if mdot is None else mdot)
                logger.log(STREAM_DEBUG, f"{self} latched shut at t = {self.t_close}")
        if self.cc is not None and p_abs is not None and not self._flashed:
            if self._flashing_margin(mdot, Tin, Tin_minus, p_abs, area_factor) <= 0.0:
                self._flashed = True
                p_vc = self._throat_pressure(mdot, Tin, Tin_minus, p_abs, area_factor)
                warnings.warn(
                    f"{self} reached saturation at its vena contracta ({float(p_vc):.4g} Pa): the "
                    f"single-phase discharge law it applies is valid only while the throat stays "
                    f"liquid, so the flow it reports from here on is an incompressible extrapolation. "
                    f"Reduce the driving pressure difference, or model the break with a critical-flow "
                    f"correlation.",
                    stacklevel=2,
                )

    @unpacked
    def should_continue(self, variables: Sequence[float], **_) -> bool:
        return not (self.flashing_terminal and self._flashed)

    def has_event(self) -> bool:
        """Whether either sentinel is armed — a break that only opens on schedule has
        no event of its own."""
        return self.closes_below is not None or self.cc is not None

    def _closing_flow(self, t: Second) -> KgPerS:
        return self.m_frozen * (1.0 - self.relaxation((t - self.t_close) * self.close_rate))

    def _cd_eff(self, mdot: KgPerS, Tin: Celsius, a_eff: Meter2) -> float:
        if not callable(self.cd):
            return self.cd
        d_h = np.sqrt(4.0 * a_eff / np.pi)
        return self.cd(np.maximum(np.abs(mdot) * d_h / (a_eff * self._visc(Tin)), _RE_FLOOR))

    def _throat_pressure(self, mdot, Tin, Tin_minus, p_abs, area_factor) -> Pascal:
        Tin_d = directed_Tin(Tin, Tin_minus, mdot, self.mdot_eps)
        flux = mdot / (self.area * area_factor)
        return p_abs - flux**2 / (2.0 * self._rho(Tin_d) * self.cc**2)

    def _flashing_margin(self, mdot, Tin, Tin_minus, p_abs, area_factor) -> Celsius:
        Tin_d = directed_Tin(Tin, Tin_minus, mdot, self.mdot_eps)
        p_vc = self._throat_pressure(mdot, Tin, Tin_minus, p_abs, area_factor)
        return float(self._tsat(max(float(p_vc), SATURATION_PRESSURE_FLOOR)) - Tin_d)
