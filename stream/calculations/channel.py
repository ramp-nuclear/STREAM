"""
Defines both coolant properties and data, and the temporal coolant temperature
derivative calculation.

"""
import logging
from enum import StrEnum
from functools import partial
from typing import Sequence, TypeVar

import numpy as np

from stream.calculation import Calculation, unpacked, CalcState
from stream.physical_models.dimensionless import Re_mdot, Gr, Pe
from stream.physical_models.heat_transfer_coefficient import (
    wall_heat_transfer_coeff, SinglePhaseLiquidHTCExArgs
)
from stream.physical_models.pressure_drop import pressure_diff, static_pressure
from stream.pipe_geometry import EffectivePipe
from stream.substances import LiquidFuncs
from stream.units import (
    Array1D, Celsius, CPerS, KgPerS, KgPerS2, Meter, Name, Pascal,
    Place, WPerM2K, WPerM2, )
from stream.utilities import directed, directed_Tin, STREAM_DEBUG, pair_mean_1d

__all__ = ["Channel", "ChannelAndContacts", "ChannelHeatFlux", "ChannelVar", "Direction"]

logger = logging.getLogger("stream.channel")


class Direction(StrEnum):
    """Directions that are supported by the code.

    We use this enum rather than literal strings because we want other places
    in the code to refer to these values in a way where things won't break if
    we rename directions.

    It is also somewhat more discoverable.

    """
    left = 'left'
    right = 'right'
    inner = 'left'
    outer = 'right'

    def __repr__(self) -> str:
        return str(self)


class ChannelVar(StrEnum):
    """Enum for variable names saved in Channel calculations.

    This class is useful for discoverability, and to increase coherency across
    the setup and analysis of channels, which are often analyzed thoroughly.

    """
    static_pressure = 'static_pressure'
    mass_flow = 'mass_flow'
    absolute_pressure = 'absolute_pressure'
    re = 'Re'
    pe = 'Pe'
    power = 'power'
    twall_left = 'T_wall, left'
    twall_right = 'T_wall, right'
    heatflux_left = 'q, left'
    heatflux_right = 'q, right'
    gr_left = 'Gr, left'
    gr_right = 'Gr, right'
    tbulk = 'T_cool'
    pressure_drop = 'pressure'
    h_left = 'h_left'
    h_right = 'h_right'
    tin = 'T_in'
    tout = 'T_out'
    velocity = 'velocity'

    @classmethod
    def get(cls, key: str, direction: Direction | None) -> "ChannelVar":
        """Get the variable name in a case where direction may be involved.

        Parameters
        ----------
        key: str
            Variable name in general
        direction: left or right
            The directionality, in a case where the channel has multiples of the same thing.

        Returns
        -------
        The appropriate enum member.

        """
        return getattr(cls, f'{key}{f"_{str(direction)}" if direction else ""}')


def coolant_first_order_upwind_dTdt(
        T: Celsius, Tin: Celsius, mdot: KgPerS,
        q_left: WPerM2, q_right: WPerM2,
        fluid: LiquidFuncs, pipe: EffectivePipe, dz: Meter) -> CPerS:
    r"""
    Calculates the first order upwind differencing temperature convection
    equation's temporal derivative. Essentially, this is the equation:

    .. math::
        mc_p\dot{T_i} = \frac{1}{2}\dot{m}(c_{p,i}+c_{p,i-1})(T_{i-1}-T_{i})
        + h\Pi (T_{wall,i} - T_i)

    Parameters
    ----------
    T: Celsius
        Bulk temperature (assumed 1D)
    Tin: Celsius
        Inlet bulk temperature (assumed float)
    q_left, q_right: WPerM2K
        Wall heat flux
    mdot: KgPerS
        Mass flow of coolant, assumed constant along channel
    fluid: LiquidFuncs
        Temperature and pressure dependent coolant properties
    pipe: EffectivePipe
        Geometrical properties of the channel
    dz: Meter
        Cell sizes, which may vary.

    Returns
    -------
    dT/dt: CPerS
        the temporal derivative of the bulk temperature
    """

    rho = fluid.density(T)
    c_bulk = fluid.specific_heat(T)
    cin = fluid.specific_heat(Tin)
    c = directed(pair_mean_1d(directed(c_bulk, mdot), prepend=cin), mdot)

    convection = directed(np.abs(mdot) * c *
                          np.diff(directed(T, mdot), prepend=Tin), mdot)

    heat_transfer = dz * (q_left * pipe.heated_parts[0] + q_right * pipe.heated_parts[1])

    heat_capacity = rho * c_bulk * pipe.area * dz
    return (heat_transfer - convection) / heat_capacity


def _heatflux(T: Celsius, T_left: Celsius, T_right: Celsius, h_left: WPerM2K, h_right: WPerM2K
              ) -> tuple[WPerM2, WPerM2]:
    return h_left * (T_left - T), h_right * (T_right - T)


class Channel(Calculation):
    """
    A channel in a reactor core, this model utilizes the incompressible flow
    assumption. The Channel is considered 1D so that there is no attention to
    transverse flow.
    """

    def __init__(self, z_boundaries: Meter, fluid: LiquidFuncs,
                 pipe: EffectivePipe, pressure_func=pressure_diff,
                 name: str = 'Channel'):
        r"""
        Parameters
        ----------
        z_boundaries: Meter
            boundaries (no. cells + 1). Must be strictly monotonous.
        fluid: LiquidFuncs
            Coolant properties, see :mod:`~.stream.substances.light_water` or
            :mod:`~.stream.substances.heavy_water`.
        pipe: EffectivePipe
            Channel geometry
        pressure_func: Callable
            A function determining the pressure gradient in the channel.
        """
        self.name = name
        self.bounds = z_boundaries
        self.dz = np.abs(np.diff(z_boundaries))
        self.centers = 0.5 * (self.bounds[1:] + self.bounds[:-1])
        self.fluid = fluid
        self.pipe = pipe

        self.pressure = partial(pressure_func, fluid=self.fluid,
                                pipe=self.pipe, dz=self.dz)
        self._dTdt = partial(coolant_first_order_upwind_dTdt, fluid=self.fluid,
                             pipe=self.pipe, dz=self.dz)
        self.n = len(self.dz)
        self._vars = {ChannelVar.tbulk: slice(0, self.n),
                      ChannelVar.pressure_drop: self.n}

    @unpacked
    def calculate(self, variables: Sequence[float], *,
                  T_left: Celsius = None, T_right: Celsius = None,
                  h_left: WPerM2K = 0., h_right: WPerM2K = 0.,
                  Tin: Celsius, Tin_minus: Celsius = None, mdot: KgPerS,
                  mdot2: KgPerS2 = None, **kwargs) -> Array1D:
        r"""Calculate rate of temperature change in each cell by means of
        :func:`First Order Upwind <coolant_first_order_upwind_dTdt>`
        and pressure difference error.

        Parameters
        ----------
        variables: Sequence[float]
            Input variables, see :meth:`.Channel.variables`
        T_left, T_right: Celsius
            Left and right boundary (wall) temperatures
        h_left, h_right: WPerM2K
            Left and right heat transfer coefficients
        Tin: Celsius
            Inlet boundary temperature
        Tin_minus: Celsius
            Outlet boundary temperature
        mdot: KgPerS
            Mass flow rate
        mdot2: KgPerS2
            :math:`\ddot{m}`, time derivative of ``mdot``

        Returns
        -------
        F(y, t): Array1D
            Which comprises the rate of temperature change and a pressure
            difference constraint.
        """
        T_vecs = self._T_vecs(variables, T_left, T_right)
        q_left, q_right = _heatflux(**T_vecs, h_left=h_left, h_right=h_right)
        d = dict(
            T_cool=self._dTdt(Tin=directed_Tin(Tin, Tin_minus, mdot),
                              T=T_vecs['T'],
                              mdot=mdot,
                              q_left=q_left, q_right=q_right
                              ),
            pressure=variables[-1] - np.sum(
                self._dp(mdot=mdot, mdot2=mdot2, **T_vecs)))
        return self.load(d)

    def indices(self, variable: Name, asking=None) -> Place:
        return dict(Tin=self.n - 1, Tin_minus=0,
                    T_coolant=(Ts := slice(0, self.n)),
                    T_cool=Ts, pressure=self.n,
                    T_left=Ts, T_right=Ts, T=Ts, dp=self.n)[variable]

    @property
    def mass_vector(self) -> Sequence[bool]:
        mass = np.ones(len(self), dtype=bool)
        mass[-1] = False
        return mass

    def __len__(self) -> int:
        """Number of cells +1 for pressure difference"""
        return self.n + 1

    @unpacked
    def save(self, vector: Sequence[float], *, T_left=None,
             T_right=None, Tin, Tin_minus=None, mdot,
             p_abs=None, mdot2=None, **kwargs) -> CalcState:
        r"""
        Given input for "calculate" (which is a legal state of the system),
        tag the information, i.e. create a "State" and return it.

        Parameters
        ----------
        vector: Sequence[float]
            Input variables, see :meth:`.Channel.variables`
        T_left, T_right: Celsius
            Left and right boundary (wall) temperatures
        Tin: Celsius
            Inlet boundary temperature
        Tin_minus: Celsius
            Outlet boundary temperature
        mdot: KgPerS
            Mass flow rate
        p_abs: Pascal or None
            Pressure at the channel inlet. If given, the state will include the
            absolute pressure at each cell
        mdot2: KgPerS2
            :math:`\ddot{m}`, time derivative of ``mdot``

        Returns
        -------
        state: CalcState
            The physical state of the channel, which includes additionally
            ``absolute_pressure`` and the ``Re`` number in each cell.
        """
        state: CalcState = super().save(vector)
        T = np.asarray(vector[0:self.n])
        state[ChannelVar.re] = Re_mdot(mdot, self.pipe.area,
                                       self.pipe.hydraulic_diameter,
                                       self.fluid.viscosity(T))
        state[ChannelVar.mass_flow] = float(mdot)
        state[ChannelVar.tin] = float(directed_Tin(Tin, Tin_minus, mdot))
        state[ChannelVar.tout] = T[-1 if mdot >= 0 else 0]
        state[ChannelVar.velocity] = float(mdot) / self.fluid.density(T) / self.pipe.area
        if p_abs is not None:
            dp = self._dp(mdot=mdot, mdot2=mdot2,
                          **self._T_vecs(vector, T_left, T_right))
            absolute_pressure = p_abs + np.cumsum(dp)
            state[ChannelVar.absolute_pressure] = absolute_pressure
            stat_pressure = static_pressure(absolute_pressure, mdot, self.pipe.area,
                                            self.fluid.density(T))
            state[ChannelVar.static_pressure] = stat_pressure
        return state

    def _dp(self, T: Celsius, T_left: Celsius, T_right: Celsius, mdot: KgPerS,
            mdot2: KgPerS2):
        # Wall temperature is average of the 2 walls
        return self.pressure(mdot=mdot, mdot2=mdot2, T=T, Tw=(T_left + T_right) / 2)

    def _T_vecs(self, variables, T_left: Celsius, T_right: Celsius) -> dict:
        return dict(
            T=(T := np.asarray(variables[0:self.n])),
            T_left=T_left if T_left is not None else T,
            T_right=T_right if T_right is not None else T,
            )

    @property
    def variables(self) -> dict[str, Place]:
        """Mapping ``T_cool``, ``pressure`` to vector places"""
        return self._vars


class ChannelHeatFlux(Channel):
    r"""A channel in a reactor core.
    This model utilizes the incompressible flow assumption.
    The Channel is considered 1D so that there is no attention to transverse flow.
    This class differs from the :class:`Channel` class by having a different way to couple with
    an adjunct heat producer. The :class:`Channel` class gives the heat producer a heat transfer coefficient
    and receives its temperature, but this object is coupled by getting the heat flux itself from that neighbor.

    This object is simpler, and thus less useful for actual systems, but more useful for toy problems.
    """

    @classmethod
    def from_channel(cls, channel: Channel):
        """Make a heat flux based channel from a regular one.

        Parameters
        ----------
        channel: Channel

        Returns
        -------
        ChannelHeatFlux

        """
        return cls(z_boundaries=channel.bounds,
                   fluid=channel.fluid,
                   pipe=channel.pipe,
                   pressure_func=channel.pressure,
                   name=channel.name
                   )

    @unpacked
    def calculate(self, variables: Sequence[float], *,
                  Tin: Celsius = None,
                  Tin_minus: Celsius = None,
                  T_left: Celsius = None,
                  T_right: Celsius = None,
                  mdot: KgPerS,
                  mdot2: KgPerS2 = None,
                  q_left: WPerM2 = 0.,
                  q_right: WPerM2 = 0.,) -> Array1D:
        T_vecs = self._T_vecs(variables, T_left, T_right)
        d = dict(
            T_cool=self._dTdt(Tin=directed_Tin(Tin, Tin_minus, mdot),
                              T=T_vecs['T'],
                              mdot=mdot,
                              q_left=q_left, q_right=q_right
                              ),
            pressure=variables[-1] - np.sum(
                self._dp(mdot=mdot, mdot2=mdot2, **T_vecs)))
        return self.load(d)

    @unpacked
    def save(self, vector: Sequence[float], q_left: WPerM2, q_right: WPerM2, **kwargs) -> CalcState:
        s = super().save(vector, **kwargs)
        s[ChannelVar.get('heatflux', Direction.left)] = q_left
        s[ChannelVar.get('heatflux', Direction.right)] = q_right
        s[ChannelVar.power] = sum(self.dz * hp * q for hp, q in zip(self.pipe.heated_parts, (q_left, q_right)))
        return s


class ChannelAndContacts(Channel):
    """
    This class assumes two walls encompass a channel. It calculates the heat
    transfer coefficient to each wall in addition to the Channel properties.
    """

    def __init__(self, z_boundaries: Meter,
                 fluid: LiquidFuncs,
                 pipe: EffectivePipe,
                 h_wall_func: SinglePhaseLiquidHTCExArgs = wall_heat_transfer_coeff,
                 pressure_func=pressure_diff,
                 name: str = 'CC'):
        r"""
        Parameters
        ----------
        z_boundaries: Meter
            boundaries (no. cells + 1). Must be strictly monotonous.
        fluid: LiquidFuncs
            Coolant properties
        pipe: EffectivePipe
            Channel geometry
        h_wall_func: Callable
            A function determining the heat transfer coefficient
        pressure_func: Callable
            A function determining the pressure gradient in the channel.

        See Also
        --------
        .wall_heat_transfer_coeff, .pressure_diff
        """
        super().__init__(z_boundaries=z_boundaries, fluid=fluid, pipe=pipe,
                         pressure_func=pressure_func, name=name)
        self.h_wall_func = h_wall_func
        self._vars |= {ChannelVar.h_left: slice((n := self.n) + 1, 2 * n + 1),
                       ChannelVar.h_right: slice(2 * n + 1, 3 * n + 1)}
        logger.log(STREAM_DEBUG, f"New {self.name}")

    @property
    def mass_vector(self) -> Sequence[bool]:
        mass = np.zeros(len(self), dtype=bool)
        mass[self._vars[ChannelVar.tbulk]] = True
        return mass

    def __len__(self):
        """Number of cells * 3 (for ``T_cool``, ``h_left``, ``h_right``) + 1
        for pressure difference"""
        return 3 * self.n + 1

    def indices(self, variable: Name, asking=None) -> Place:
        try:
            return super().indices(variable, asking=asking)
        except KeyError:
            pass
        return dict(h_left=self.variables['h_right'], h_right=self.variables['h_left'])[variable]

    @property
    def variables(self) -> dict[str, Place]:
        """Mapping ``T_cool``, ``h_left``, ``h_right``, ``pressure``
        to vector places"""
        return self._vars

    def dist_from_edge(self, mdot: KgPerS) -> Meter:
        """Computes the center distances from the channel entrance depending
        on flow direction.

        Parameters
        ----------
        mdot: KgPerS
            Mass flow rate. Sign determines flow direction.

        Returns
        -------

        """
        return self.centers - self.bounds[0] if mdot >= 0 else self.bounds[-1] - self.centers

    def h_wall(self, T_cool: Celsius, T_wall: Celsius, mdot: KgPerS, pressure: Pascal,
               **_) -> WPerM2K | None:
        if T_wall is None:
            return None
        x = self.dist_from_edge(mdot)
        return self.h_wall_func(T_wall=T_wall, T_cool=T_cool, mdot=mdot,
                                pressure=pressure,
                                coolant_funcs=self.fluid,
                                Dh=self.pipe.hydraulic_diameter,
                                depth=self.pipe.depth,
                                A=self.pipe.area, develop_length=x,
                                # Protocol Mismatch!
                                coolant=None  # type: ignore
                                )

    @unpacked
    def save(self, vector: Sequence[float], *,
             T_left=None, T_right=None, Tin, Tin_minus=None,
             mdot, p_abs=None, **kwargs) -> CalcState:
        r"""Given input for "calculate" (which is a legal state of the system),
        tag the information, i.e. create a "State" and return it.

        Parameters
        ----------
        vector: Sequence[float]
            Input variables, see :meth:`.ChannelAndContacts.variables`
        T_left, T_right: Celsius
            Left and right boundary (wall) temperatures
        Tin: Celsius
            Inlet boundary temperature
        Tin_minus: Celsius
            Outlet boundary temperature
        mdot: KgPerS
            Mass flow rate
        p_abs: Pascal or None
            Pressure at the channel inlet. If given, the state will include the
            absolute pressure at each cell


        Returns
        -------
        state: CalcState
            The physical state of the channel, which includes additionally things like
            ``absolute_pressure``, ``Re``, ``ONB, left``, ``ONB, right`` safety
            factor (for each wall, see :func:`.BR_ONB`) number in each cell.
        """
        state: CalcState = super().save(vector, T_left=T_left, T_right=T_right, Tin=Tin,
                                        Tin_minus=Tin_minus, mdot=mdot, p_abs=p_abs,
                                        **kwargs)
        if p_abs is None or (T_left is None and T_right is None):
            return state
        absolute_pressure = state["static_pressure"]
        T = np.asarray(vector[0:self.n])
        h_left = vector[self._vars[ChannelVar.h_left]]
        h_right = vector[self._vars[ChannelVar.h_right]]

        coolant = self.fluid.to_properties(T, absolute_pressure)
        v = mdot / self.pipe.area / coolant.density
        state[ChannelVar.pe] = Pe(coolant.density, v, self.pipe.hydraulic_diameter,
                                  coolant.specific_heat, coolant.conductivity)

        channel_power = 0.
        for direction, wall_temp, h, heated_part in (
                (Direction.left, T_left, h_left, self.pipe.heated_parts[0]),
                (Direction.right, T_right, h_right, self.pipe.heated_parts[1])
                ):
            if wall_temp is not None:
                state[ChannelVar.get('twall', direction)] = wall_temp
                state[ChannelVar.get('gr', direction)] = Gr(
                    coolant.density, coolant.viscosity, coolant.thermal_expansion,
                    T, wall_temp, self.pipe.hydraulic_diameter)
                q = h * (wall_temp - T)
                state[ChannelVar.get('heatflux', direction)] = q
                area = self.dz * heated_part
                p = np.dot(area, q)
                channel_power += p
        state[ChannelVar.power] = channel_power
        return state

    @unpacked
    def calculate(self, variables: Sequence[float], *, T_left: Celsius = None,
                  T_right: Celsius = None, Tin: Celsius,
                  Tin_minus: Celsius = None, mdot: KgPerS, p_abs: Pascal,
                  mdot2: KgPerS2 = None) -> Array1D:
        r"""
        Parameters
        ----------
        variables: Sequence[float]
            Input variables, see :meth:`.ChannelAndContacts.variables`
        T_left, T_right: Celsius
            Left and right boundary (wall) temperatures
        Tin: Celsius
            Inlet boundary temperature
        Tin_minus: Celsius
            Outlet boundary temperature
        mdot: KgPerS
            Mass flow rate
        p_abs: Pascal
            The absolute pressure at the inlet.
        mdot2: KgPerS2
            :math:`\ddot{m}`, time derivative of ``mdot``

        Returns
        -------
        F(y, t): Array1D
            Which consists of the rate of temperature change, and pressure
            difference and wall heat transfer coefficients constraint.
        """
        T_vecs = self._T_vecs(variables, T_left, T_right)
        dp = self._dp(mdot=mdot, mdot2=mdot2, **T_vecs)
        abs_pressure = p_abs + np.cumsum(dp)
        tcool = variables[self._vars[ChannelVar.tbulk]]
        density = self.fluid.density(tcool)
        stat_pressure = static_pressure(abs_pressure, mdot, self.pipe.area, density)
        h_left = self.h_wall(T_wall=T_left, T_cool=tcool, mdot=mdot,
                             pressure=stat_pressure)
        h_right = self.h_wall(T_wall=T_right, T_cool=tcool, mdot=mdot,
                              pressure=stat_pressure)
        h_left, h_right = _other_if_none(h_left, h_right)  # type: WPerM2K
        q_left, q_right = _heatflux(**T_vecs, h_left=h_left, h_right=h_right)

        d = dict(T_cool=self._dTdt(Tin=directed_Tin(Tin, Tin_minus, mdot),
                                   T=T_vecs['T'], mdot=mdot,
                                   q_left=q_left, q_right=q_right),
                 h_left=h_left - variables[self._vars[ChannelVar.h_left]],
                 h_right=h_right - variables[self._vars[ChannelVar.h_right]],
                 pressure=variables[self._vars[ChannelVar.pressure_drop]] - np.sum(dp))

        return self.load(d)


_T = TypeVar("_T")


def _other_if_none(x: _T | None, y: _T | None) -> tuple[_T, _T]:
    """From (x, y) return (x, x) if y is None and vice versa"""
    return y if x is None else x, x if y is None else y
