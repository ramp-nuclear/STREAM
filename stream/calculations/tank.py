r"""Free-surface inventories.

A :class:`Tank` is a body of liquid whose surface is exposed to a known pressure: it
owns how much liquid it holds (``level``) and how hot that liquid is (``T``), and it
sits in the flow graph as a node, exactly where a :class:`~.Junction` would.
An :class:`Environment` is the ambient the system may discharge into — a dead end at a
known pressure, with no inventory of its own.

Both are free-surface nodes: the pressure at their surface is imposed rather than
solved for, so mass may enter or leave the network through them.
"""

import logging
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from stream import smoothing
from stream.calculation import Calculation, sealed, unpacked
from stream.calculations.kirchhoff import Junction
from stream.errors import StreamConstructionError
from stream.smoothing import soft_pos
from stream.substances import LiquidFuncs
from stream.units import (
    Array1D,
    Celsius,
    FunctionOfTime,
    KgPerS,
    Meter,
    Meter2,
    Meter3,
    Name,
    Pascal,
    Place,
    Second,
    Watt,
)

__all__ = ["Environment", "Tank"]

logger = logging.getLogger(__name__)

_FLOW_MAPS = ("mdot", "Tin", "Tin_minus")


@sealed
class Environment(Junction):
    """The ambient a system discharges into: a dead end held at a known pressure.

    It behaves as a :class:`~.Junction` for temperature bookkeeping and carries no
    inventory — whatever leaves the system through it is gone.
    """

    def __init__(self, surface_pressure: Pascal | FunctionOfTime = 101325.0, name: str = None):
        """

        Parameters
        ----------
        surface_pressure: Pascal or FunctionOfTime
            Ambient pressure, either a constant or a function of time.
        name: str or None
            Calculation's name
        """
        super().__init__(name=name)
        self.surface_pressure = surface_pressure


@sealed
class Tank(Junction):
    r"""A liquid inventory with a free surface, owning its level and its temperature.

    The tank is a node of the flow graph (a :class:`~.Junction` subclass), so the edges
    leaving and entering it are written in the user's edge list and its adjacent mass
    currents are served to it the way every junction's are. Two states are owned:

    - ``level``, driven by the net inflow,
      :math:`\frac{dL}{dt} = \frac{\sum_k \pm w_k \dot{m}_k}{\rho(T)\,A(L)}`
    - ``T``, driven by the inflow enthalpy,
      :math:`\frac{dT}{dt} = \frac{\sum_k \dot{m}_{\text{in},k} c_p (T_k - T) + Q_{ext}}
      {\rho(T)\,V(L)\,c_p}` — outflow leaves at ``T`` and therefore cancels out.

    While :attr:`pinned` (the default, and what a steady solve needs) the level is held
    at ``level0`` algebraically instead. ``fixed_temperature`` collapses ``T`` to a
    boundary condition for cases where the tank's thermal inertia is not of interest.

    Examples
    --------
    A pinned tank reports how far its level is from where it is held, and a fixed
    temperature is simply anchored:

    >>> from stream.substances import light_water
    >>> tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, fixed_temperature=30.0)
    >>> tank.calculate(np.array([3.0, 30.0]), mdot={})
    array([-1.,  0.])

    Once unpinned, the level follows the net inflow:

    >>> drain = Environment(name="ambient")
    >>> tank.unpin()
    >>> round(float(tank.calculate(np.array([4.0, 30.0]), mdot={drain: 2.0}, Tin_minus={drain: 30.0})[0]), 9)
    -0.001005115

    See Also
    --------
    Environment, ~stream.calculations.kirchhoff.Junction
    """

    def __init__(
        self,
        fluid: LiquidFuncs,
        area: Meter2 | Callable[[Meter], Meter2],
        level0: Meter,
        *,
        z_uncovery: Meter,
        surface_pressure: Pascal | FunctionOfTime = 101325.0,
        fixed_temperature: Celsius | None = None,
        volume: Callable[[Meter], Meter3] | None = None,
        marks: dict[str, Meter] | None = None,
        mdot_starve: KgPerS | None = None,
        mdot_eps: KgPerS | None = None,
        name: str = "Tank",
    ):
        r"""

        Parameters
        ----------
        fluid: LiquidFuncs
            Coolant properties
        area: Meter2 or Callable[[Meter], Meter2]
            Free-surface area, constant or a function of level. A level-dependent area
            must come with a ``volume``.
        level0: Meter
            Initial level, and the level held while :attr:`pinned`.
        z_uncovery: Meter
            Level at which the inventory is considered lost, measured in the same
            datum as ``level``. Reaching it stops the transient.
        surface_pressure: Pascal or FunctionOfTime
            Pressure above the liquid surface (a cover gas, or the atmosphere).
        fixed_temperature: Celsius or None
            When given, ``T`` is held at this value instead of following the energy
            balance.
        volume: Callable[[Meter], Meter3] or None
            Liquid volume as a function of level. Required for a level-dependent
            ``area``; defaults to ``area * level``.
        marks: dict[str, Meter] or None
            Named levels worth knowing about (a nozzle, a weir). Passing one is
            reported when the level falls past it, and does not stop the transient.
        mdot_starve: KgPerS or None
            Stop the transient when the magnitude of a routed reference mass current
            falls to this value — the "circulation stopped while still covered" end,
            as distinct from uncovery. Requires a ``ref_mdot`` supplier.
        mdot_eps: KgPerS or None
            Half-width of the direction-blending band around ``mdot = 0`` used when
            soft-rectifying the incoming mass flows. ``None`` uses
            ``stream.smoothing.DEFAULT_MDOT_EPS`` (read at call time).
        name: str
            Calculation's name
        """
        super().__init__(name=name, mdot_eps=mdot_eps)
        if callable(area) and volume is None:
            raise StreamConstructionError(
                f"{name} was given a level-dependent area but no volume, so the liquid volume "
                f"its energy balance needs cannot be derived. Pass volume=<callable of level> "
                f"next to the area, or give a constant area (whose volume is area * level)."
            )
        self.fluid = fluid
        self.area = area
        self.level0 = level0
        self.z_uncovery = z_uncovery
        self.surface_pressure = surface_pressure
        self.fixed_temperature = fixed_temperature
        self.volume = volume
        self.marks = dict(marks or {})
        self.mdot_starve = mdot_starve
        self.pinned = True
        self._uncovered = False
        self._starved = False
        self._passed = {mark: False for mark in sorted(self.marks)}

    def pin(self) -> None:
        """Hold the level at ``level0``, making it algebraic — what a steady solve needs.

        This changes :attr:`mass_vector`, so an already-built aggregator must be told
        about it (``Aggregator.refresh_mass()``) before the next solve.
        """
        self.pinned = True

    def unpin(self) -> None:
        """Let the level follow the net inflow, making it differential — what a
        transient needs.

        This changes :attr:`mass_vector`, so an already-built aggregator must be told
        about it (``Aggregator.refresh_mass()``) before the next solve.
        """
        self.pinned = False

    @property
    def variables(self) -> dict[Name, Place]:
        """The free-surface level and the bulk liquid temperature."""
        return dict(level=0, T=1)

    def indices(self, variable: Name, asking: Calculation = None) -> Place:
        """The tank serves its own temperature to whatever it is connected to, so both
        ``Tin`` and ``Tin_minus`` resolve to the ``T`` slot."""
        places = dict(Tin=1, Tin_minus=1, level=0, T=1)
        try:
            return places[variable]
        except KeyError:
            raise KeyError(f"{type(self).__name__} does not serve {variable!r} (only {sorted(places)}).") from None

    @property
    def mass_vector(self) -> Sequence[bool]:
        return not self.pinned, (not self.pinned) and self.fixed_temperature is None

    def __len__(self) -> int:
        return 2

    # noinspection PyMethodOverriding
    @unpacked(exclude=_FLOW_MAPS)
    def calculate(
        self,
        variables: Sequence[float],
        *,
        mdot: dict[Calculation, KgPerS],
        Tin: dict[Calculation, Celsius] = None,
        Tin_minus: dict[Calculation, Celsius] = None,
        Q_ext: Watt = 0.0,
        ref_mdot: KgPerS = None,
        t: Second = None,
    ) -> Array1D:
        r"""Inventory and energy residuals.

        Parameters
        ----------
        variables: Sequence[float]
            ``[level, T]``
        mdot: dict[Calculation, KgPerS]
            Mass currents of the components on the adjacent edges.
        Tin, Tin_minus: dict[Calculation, Celsius]
            Temperatures those components carry, depending on
            :math:`\text{sign}(\dot{m})`. Membership decides the sign the current
            enters the inventory with: ``Tin`` components feed the tank, ``Tin_minus``
            components drain it.
        Q_ext: Watt
            External heat rate into the liquid.
        ref_mdot: KgPerS
            Reference current watched by the flow-starvation stop; the residuals ignore it.
        t: Second
            Time.

        Returns
        -------
        out: Array1D
            The level residual (``level - level0`` while pinned, else the level rate)
            and the temperature residual.

        Notes
        -----
        Inflow magnitudes are soft-rectified with :func:`~stream.smoothing.soft_pos`,
        as in :meth:`~.Junction.calculate`, so the energy balance stays smooth through
        a flow reversal on any adjacent edge.
        """
        Tin = Tin or {}
        Tin_minus = Tin_minus or {}
        eps = self.mdot_eps if self.mdot_eps is not None else smoothing.DEFAULT_MDOT_EPS
        level, T = variables[0], variables[1]
        cp = self.fluid.specific_heat(T)

        net = 0.0
        heat = Q_ext
        for comp, T_comp in Tin.items():
            w = self.weights.get(comp, 1.0)
            net += w * mdot[comp]
            heat += w * soft_pos(mdot[comp], eps) * cp * (T_comp - T)
        for comp, T_comp in Tin_minus.items():
            w = self.weights.get(comp, 1.0)
            net -= w * mdot[comp]
            heat += w * soft_pos(-mdot[comp], eps) * cp * (T_comp - T)

        rho = self.fluid.density(T)
        if self.fixed_temperature is not None:
            energy = T - self.fixed_temperature
        elif self.pinned:
            energy = heat
        else:
            energy = heat / (rho * self._volume(level) * cp)
        level_rate = level - self.level0 if self.pinned else net / (rho * self._area(level))
        return np.array([level_rate, energy], dtype=float)

    @unpacked(exclude=_FLOW_MAPS)
    def event_margin(self, variables: Sequence[float], *, ref_mdot: KgPerS = None, **_) -> Array1D:
        """Margins in a fixed order: uncovery, then each mark by name, then starvation.

        A margin that has already fired returns ``+1``, so a refill cannot re-fire it.
        """
        level = variables[0]
        margins = [1.0 if self._uncovered else level - self.z_uncovery]
        margins += [1.0 if passed else level - self.marks[mark] for mark, passed in self._passed.items()]
        if self.mdot_starve is not None:
            starved = self._starved or ref_mdot is None
            margins.append(1.0 if starved else abs(ref_mdot) - self.mdot_starve)
        return np.array(margins, dtype=float)

    @unpacked(exclude=_FLOW_MAPS)
    def change_state(self, variables: Sequence[float], *, ref_mdot: KgPerS = None, **_) -> None:
        level = variables[0]
        if not self._uncovered and level <= self.z_uncovery:
            self._uncovered = True
            logger.warning(f"{self} uncovered: level {float(level):.4g} m is at or below {self.z_uncovery} m")
        for mark, passed in self._passed.items():
            if not passed and level <= self.marks[mark]:
                self._passed[mark] = True
                logger.warning(f"{self} level fell past {mark!r} ({self.marks[mark]} m)")
        if self.mdot_starve is not None and not self._starved and ref_mdot is not None:
            if abs(ref_mdot) <= self.mdot_starve:
                self._starved = True
                logger.warning(
                    f"{self} lost its flow: |mdot| {abs(float(ref_mdot)):.4g} kg/s is at or "
                    f"below {self.mdot_starve} kg/s"
                )

    @unpacked(exclude=_FLOW_MAPS)
    def should_continue(self, variables: Sequence[float], **_) -> bool:
        return not (self._uncovered or self._starved)

    def validate_wiring(self, external: dict[str, dict], funcs: dict[Name, Any]) -> Iterable[str]:
        suppliers = external.get("mdot", {})
        if not suppliers:
            return [
                f"{self} is not wired into a flow graph: nothing supplies its 'mdot', so its "
                f"inventory can never change. Put it on flow edges and hand it to the flow "
                f"solver as a surface node."
            ]

        problems = []
        if self.mdot_starve is not None and "ref_mdot" not in external and "ref_mdot" not in funcs:
            problems.append(
                f"{self} was given mdot_starve={self.mdot_starve} but no reference current is "
                f"routed to it, so the flow-starvation stop can never fire. Mark an edge with "
                f"ref_mdot_for, or drop mdot_starve."
            )

        Tin, Tin_minus = external.get("Tin", {}), external.get("Tin_minus", {})
        for comp in suppliers:
            if (comp in Tin) != (comp in Tin_minus):
                continue
            problems.append(
                f"{comp} carries a mass current to {self} but is wired to it as "
                f"{'both an inflow and an outflow' if comp in Tin else 'neither an inflow nor an outflow'}, "
                f"so that current would be {'counted twice' if comp in Tin else 'left out'} of the "
                f"inventory. Each adjacent component reaches the tank from exactly one side."
            )

        taps = next(
            (
                supplier.surface_taps(self)
                for routed in external.values()
                for supplier in routed
                if hasattr(supplier, "surface_taps")
            ),
            {},
        )
        for comp, sign in taps.items():
            feeds = sign > 0
            if comp in (Tin if feeds else Tin_minus):
                continue
            wired = "an inflow" if comp in Tin else "an outflow" if comp in Tin_minus else "neither"
            problems.append(
                f"{comp} meets {self} with incidence sign {sign:+.0f} (it {'feeds' if feeds else 'drains'} "
                f"the tank) but is wired to it as {wired}, so its current would enter the inventory "
                f"with the wrong sign. Reverse that flow edge, or attach it to the tank's other side."
            )
        return problems

    def _area(self, level: Meter) -> Meter2:
        return self.area(level) if callable(self.area) else self.area

    def _volume(self, level: Meter) -> Meter3:
        return self.volume(level) if self.volume is not None else self.area * level
