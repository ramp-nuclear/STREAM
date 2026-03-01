"""Calculations that only drop the pressure when fluid flows through them.

Their electrical analogs are resistors in an electric circuit, even though most
of the resistors here do not follow an Ohm's law (i.e. they are not linear).

"""

from functools import partial
from itertools import chain
from typing import Callable, Sequence, Protocol

import numpy as np
from numba import njit

from stream.calculation import unpacked, CalcState, Calculation
from stream.calculations.ideal.ideal import LumpedComponent
from stream.physical_models.dimensionless import Re_mdot
from stream.physical_models.pressure_drop import (
    Darcy_Weisbach_pressure_by_mdot,
    gravity_pressure,
    local_pressure_by_mdot,
    local_pressure_factor,
)
from stream.physical_models.pressure_drop.friction import (
    regime_dependent_friction,
    laminar_friction,
    turbulent_friction,
)
from stream.physical_models.pressure_drop.local import (
    sudden_expansion_factor,
    sudden_contraction_factor,
    bend_factor,
)
from stream.pipe_geometry import EffectivePipe
from stream.substances.liquid import LiquidFuncs
from stream.units import (
    Celsius,
    g,
    KgPerS,
    Meter,
    Meter2,
    MPerS2,
    PerMS,
    Pascal,
    KgPerM3,
    KgPerM7,
    KgPerM4S,
    Radians,
)
from stream.utilities import summed


__all__ = [
    "DPCalculation",
    "Resistor",
    "Friction",
    "Gravity",
    "LocalPressureDrop",
    "Bend",
    "ResistorSum",
    "ResistorMul",
    "MultipliableCalculation",
    "RegimeDependentFriction",
    "Screen",
    "VolumetricFlowResistor",
]


class MultipliableCalculation(Calculation, Protocol):
    """A calculation that can be multiplied by a float to create a new calculation with meaning.

    This is used so that we can write 2*resistor to create a resistor that always
    has twice the pressure drop, for example.

    """

    def __mul__(self, other: float) -> "MultipliableCalculation": ...

    def __rmul__(self, other: float) -> "MultipliableCalculation": ...


class DPCalculation(Calculation, Protocol):
    """A calculation that has a `dp_out` method"""

    dp_out = LumpedComponent.dp_out


class ResistorMul:
    """A calculation that is the same as an encapsulated resistor but multiplies its
    pressure loss by a known factor.

    This class uses `__getattr__` magic to act like a calculation so long as its
    resistor is a calculation.

    """

    def __init__(self, factor: float, resistor: DPCalculation):
        if not isinstance(factor, float):
            raise TypeError(f"Cannot multiply object of type {type(resistor)} by non-float type {factor}")
        self.factor = factor
        self.resistor = resistor

    def dp_out(self, **kwargs) -> Pascal:
        return self.factor * self.resistor.dp_out(**kwargs)

    def __deepcopy__(self, memo):
        return type(self)(self.factor, self.resistor)

    def __getattr__(self, item):
        return self.resistor.__getattribute__(item)

    def __mul__(self, other: float) -> "ResistorMul":
        return type(self)(self.factor * other, self.resistor)

    __rmul__ = __mul__

    def __eq__(self, other: "ResistorMul") -> bool:
        if not isinstance(other, ResistorMul):
            return NotImplemented
        else:
            return self.factor == other.factor and self.resistor == other.resistor

    def __hash__(self):
        return hash((self.resistor, self.factor))

    def __len__(self):
        return len(self.resistor)

    def __repr__(self):
        return repr(self.resistor)

    def __str__(self):
        return str(self.resistor)


def _mul(r: DPCalculation, f: float) -> ResistorMul:
    return ResistorMul(f, r)


def _multiplies(cls):
    """Adds multiplications that create a ResistorMul object."""
    cls.__mul__ = _mul
    cls.__rmul__ = _mul
    return cls


@_multiplies
class ResistorSum(LumpedComponent):
    """Adding several LumpedComponent in series, into a single calculation.
    These calculations must leave the temperature unchanged, dealing only with pressure
    drop."""

    def __init__(self, *resistors: DPCalculation, name: str = "R"):
        self.name = name
        unrolled_resistors = summed([r.resistors if isinstance(r, ResistorSum) else [r] for r in resistors], [])
        self.resistors = unrolled_resistors

    def dp_out(self, *, Tin: Celsius, mdot: KgPerS, **kwargs) -> Pascal:
        return sum(r.dp_out(Tin=Tin, mdot=mdot, **kwargs) for r in self.resistors)

    @unpacked
    def should_continue(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        **kwargs,
    ) -> bool:
        return all(
            r.should_continue(variables, Tin=Tin, Tin_minus=Tin_minus, mdot=mdot, **kwargs) for r in self.resistors
        )

    @unpacked
    def change_state(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        **kwargs,
    ):
        [r.change_state(variables, Tin=Tin, Tin_minus=Tin_minus, mdot=mdot, **kwargs) for r in self.resistors]

    def __add__(self, other: "ResistorSum") -> "ResistorSum":
        return ResistorSum(*chain(self.resistors, other.resistors), name=self.name)


@_multiplies
class Resistor(LumpedComponent):
    r"""A simple linear resistor to flow. It ensures
    :math:`\Delta p = \dot{m}R` (Ohm's law)
    """

    def __init__(self, resistance: PerMS, name: str = "R"):
        self.name = name
        self.r = resistance

    def dp_out(self, *, mdot: KgPerS, **_) -> Pascal:
        return -self.r * mdot


@_multiplies
class Friction(LumpedComponent):
    """Resistor quadratic in flow using a given friction coefficient"""

    def __init__(
        self,
        f: float,
        fluid: LiquidFuncs,
        length: Meter,
        hydraulic_diameter: Meter,
        area: Meter2,
        name: str = "Friction",
    ):
        r"""

        Parameters
        ----------
        f: float
            Darcy-Weisbach friction factor
        fluid: LiquidFuncs
            Coolant properties
        length: Meter
        hydraulic_diameter: Meter
        area: Meter2

        See Also
        --------
        .Darcy_Weisbach_pressure_by_mdot, .EffectivePipe
        """
        self.name = name
        self.f = f
        self._dp = partial(Darcy_Weisbach_pressure_by_mdot, L=length, Dh=hydraulic_diameter, A=area)
        self._rho = fluid.density

    def dp_out(self, *, Tin: Celsius, mdot: KgPerS, **_) -> Pascal:
        return -self._dp(mdot=mdot, rho=self._rho(Tin), f=self.f)


@_multiplies
class Gravity(LumpedComponent):
    r"""A Calculation describing in a 0D manner the pressure difference
    incurred by gravity, i.e. :math:`\Delta p = \rho g \Delta h`

    See Also
    --------
    .gravity_pressure
    """

    def __init__(
        self,
        fluid: LiquidFuncs,
        disposition: Meter,
        gravity: MPerS2 = g,
        name: str = "Gravity",
    ):
        r"""

        Parameters
        ----------
        fluid: LiquidFuncs
            Coolant Functional properties
        disposition: Meter
            :math:`\Delta h`, height difference upon which the pressure
            difference is incurred
        gravity: MPerS2
            Gravitational acceleration constant
        """
        self.name = name
        self._rho = fluid.density
        self.g = gravity
        self.h = disposition

    def dp_out(self, *, Tin: Celsius, **_) -> Pascal:
        r"""Returns: :math:`\rho(T_{in})g\Delta h`"""
        return gravity_pressure(rho=self._rho(Tin), dh=self.h, g=self.g)


@_multiplies
class LocalPressureDrop(LumpedComponent):
    """Local pressure drop due to expansion or contraction according to Idelchik chapter 4.

    The appropriate diagrams are 4.2 and 4.10, on pages 246 and 256. [#Idelchik]_

    """

    def __init__(self, fluid: LiquidFuncs, A1: Meter2, A2: Meter2, name: str = "LocalPD"):
        self.name = name
        self._rho = fluid.density
        self._visc = fluid.viscosity
        self.A1 = A1
        self.A2 = A2
        self._area_difference = (1 / (A2**2)) - (1 / (A1**2))
        factors = (sudden_expansion_factor, sudden_contraction_factor)
        pos, neg = factors if A2 >= A1 else factors[::-1]
        self.f_calc = partial(local_pressure_factor, positive_flow=pos, negative_flow=neg)

    def dp_out(self, *, mdot: KgPerS, Tin: Celsius, **_) -> Pascal:
        rho = self._rho(Tin)
        A = min(self.A1, self.A2)
        aratio = min(self.A1 / self.A2, self.A2 / self.A1)
        Dh = np.sqrt(A / np.pi)
        re = Re_mdot(mdot, A, Dh, self._visc(Tin))
        f = self.f_calc(mdot=mdot, aratio=aratio, re=re)
        return -local_pressure_by_mdot(mdot, rho, f, A)

    # noinspection PyMethodOverriding
    @unpacked
    def save(self, vector: Sequence[float], *, mdot: KgPerS, **_) -> CalcState:
        """Tags the information for the calculation input vector.

        Parameters
        ----------
        vector: Sequence[float]
        mdot: KgPerS
            Mass flow rate

        Returns
        -------
        CalcState

        """
        state = Calculation.save(self, vector)
        Tin, dp = vector
        sign = 1 if mdot >= 0 else -1
        density = self._rho(Tin)
        dynamic_difference = (0.5 * mdot**2 / density) * sign * self._area_difference
        state["static_pressure_drop"] = dp - dynamic_difference
        return state


@_multiplies
class Bend(LumpedComponent):
    """Pressure drop due to a low relative curvature bend in a smooth circular/square pipe
    according to Idelchik chapter 6.

    The appropriate diagram is 6.1, on page 424. [#Idelchik]_

    """

    def __init__(
        self,
        fluid: LiquidFuncs,
        hydraulic_diameter: Meter,
        area: Meter2,
        bend_radius: Meter,
        bend_angle: Radians,
        friction_func: Callable[[float], float] = turbulent_friction,
        name: str = "Bend",
    ):
        r"""

        Parameters
        ----------
        fluid: LiquidFuncs
            Coolant properties
        hydraulic_diameter: Meter
            Hydraulic diameter of the pipe.
        area: Meter2
            Cross-sectional area of the pipe.
        bend_radius: Meter
            The pipe's axis radius of curvature, measured from the bend center to the center of the pipe.
        bend_angle: Radians
            The pipe's bend angle.
        friction_func: Callable[[float], float]
            Re-dependent Darcy friction function. Default is :func:`~turbulent_friction`.
        """

        self.name = name
        self._rho = fluid.density
        self._visc = fluid.viscosity
        self.bend_angle = bend_angle
        self.hydraulic_diameter = hydraulic_diameter
        self.area = area
        self.f_func = friction_func

        self._arc_length = bend_radius * bend_angle
        self._relative_curvature = bend_radius / hydraulic_diameter

    def dp_out(self, *, mdot: KgPerS, Tin: Celsius, **_) -> Pascal:
        re = Re_mdot(mdot, self.area, self.hydraulic_diameter, self._visc(Tin))

        f = self.f_func(re)
        friction = f * self._arc_length / self.hydraulic_diameter
        local = bend_factor(self.bend_angle, self._relative_curvature, re)

        k = local + friction
        return -local_pressure_by_mdot(mdot, self._rho(Tin), k, self.area)


@_multiplies
class RegimeDependentFriction(LumpedComponent):
    r"""Friction resistor which depends on the Reynolds number,
    see :func:`.regime_dependent_friction`"""

    def __init__(
        self,
        pipe: EffectivePipe,
        fluid: LiquidFuncs,
        re_bounds: tuple[float, float],
        k_R: float,
        name: str = "Friction",
        laminar: Callable[[float], float] = laminar_friction,
        turbulent: Callable[[float], float] = turbulent_friction,
    ):
        self.name = name
        self._dp = partial(
            Darcy_Weisbach_pressure_by_mdot,
            L=pipe.length,
            Dh=pipe.hydraulic_diameter,
            A=pipe.area,
        )
        self._rho = fluid.density
        self._k_const = (pipe.length / pipe.hydraulic_diameter) / (2 * pipe.area**2)
        self._f = partial(
            regime_dependent_friction,
            fluid=fluid,
            pipe=pipe,
            re_bounds=re_bounds,
            k_R=k_R,
            laminar=laminar,
            turbulent=turbulent,
        )

    def k(self, mdot, t) -> float:
        f = self._f(T_cool=t, mdot=mdot, T_wall=t)
        return self._rho(t) * f * self._k_const

    def dp_out(self, *, Tin: Celsius, mdot: KgPerS, **_) -> Pascal:
        return -self._dp(mdot=mdot, rho=self._rho(Tin), f=self._f(T_cool=Tin, mdot=mdot, T_wall=Tin))


@_multiplies
class VolumetricFlowResistor(LumpedComponent):
    r"""An object that resists flow as:

    .. math::
        \Delta p = kQ^2 +k_{low}Q = k\frac{\dot{m}^2}{\rho^2}
                   + k_{low}\frac{\dot{m}}{\rho}

    """

    def __init__(
        self,
        k: KgPerM7,
        name: str,
        density_func: Callable[[Celsius], KgPerM3],
        klow: KgPerM4S = 0,
    ):
        """

        Parameters
        ----------
        k: KgPerM7
             Resistor constant
        name: str
            The name to give this calculation.
        density_func: Callable[[Celsius], KgPerM3]
            Temperature-dependent coolant density
        klow: KgPerM4S
            The resistor constant at negligible flow.
        """
        self.k = k
        self.klow = klow
        self.name = name
        self._rho = density_func

    def dp_out(self, *, mdot: KgPerS, Tin: Celsius, **_) -> Pascal:
        q = mdot / self._rho(Tin)
        return -self.k * q * np.abs(q) - self.klow * q


@_multiplies
class Screen(LumpedComponent):
    """A resistor to flow due to a circular metal wire mesh"""

    def __init__(
        self,
        clear_area: Meter2,
        total_area: Meter2,
        wire_diameter: Meter,
        fluid: LiquidFuncs,
        name: str = "Screen",
    ):
        r"""A screen-type resistor based on the circular metal wire screen in
        [#IdelchikScreen]_.

        Parameters
        ----------
        clear_area : Meter2
            The total unobstructed mesh area
        total_area : Meter2
            The total area
        wire_diameter : Meter
            Diameter of the circular wire
        fluid : LiquidFuncs
            Coolant property functions
        name : str
            The name to give this calculation.

        References
        ----------
        .. [#IdelchikScreen] Idelchik, "Handbook of Hydraulic Resistance, 4th Edition, p. 598"
        """
        self.clear_area = clear_area
        self.total_area = total_area
        self.d_wire = wire_diameter
        self.fluid = fluid
        self.name = name

    @staticmethod
    @njit
    def factor(clear_area: Meter2, total_area: Meter2, re: float) -> float:
        """A Reynolds dependent Darcy factor.

        Parameters
        ----------
        clear_area: Meter2
            The total unobstructed mesh area
        total_area: Meter2
            The total area
        re: float
            Reynolds No.

        Returns
        -------
        float
            Darcy factor
        """
        f = clear_area / total_area
        factor = 1.3 * (1 - f) + (1 / f - 1) ** 2
        if re > 1000:
            return factor
        if re < 50:
            return factor + 22 / re

        re_list = np.array([50, 100, 150, 200, 300, 400, 500, 1000])
        k_tag = np.array([1.44, 1.24, 1.13, 1.08, 1.03, 1.01, 1.01, 1.00])
        return np.interp(re, re_list, k_tag) * factor

    def dp_out(self, mdot: KgPerS, Tin: Celsius, **kwargs) -> Pascal:
        re = Re_mdot(mdot=mdot, A=self.clear_area, L=self.d_wire, mu=self.fluid.viscosity(Tin))
        return -Darcy_Weisbach_pressure_by_mdot(
            mdot=mdot,
            rho=self.fluid.density(Tin),
            f=self.factor(self.clear_area, self.total_area, re),
            L=1.0,
            Dh=1.0,
            A=self.total_area,
        )
