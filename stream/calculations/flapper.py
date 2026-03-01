r"""Flappers are control-flow elements in hydraulic systems.

A simple implementation is presented simply as :class:`.Flapper`.
"""

import logging
from typing import Callable, Sequence

import numpy as np

from stream import unpacked, Calculation
from stream.calculations.ideal.ideal import LumpedComponent
from stream.physical_models.pressure_drop import mdot_by_local_pressure
from stream.substances import LiquidFuncs
from stream.units import Array1D, Celsius, KgPerS, Meter2, PerS, Second
from stream.utilities import directed_Tin, STREAM_DEBUG


__all__ = ["Flapper", "legacy_relaxation", "continuously_differentiable_relaxation"]

logger = logging.getLogger(__name__)


def continuously_differentiable_relaxation(x):
    """A continuously differentiable relaxation scheme"""
    if x <= 0.0:
        return 0.0
    elif x >= 1.0:
        return 1.0
    return -2 * x**3 + 3 * x**2


def legacy_relaxation(x):
    """Legacy relaxation scheme chosen somewhat arbitrarily"""
    if x <= 0.0:
        return 0.0
    elif x >= 1.0:
        return 1.0
    return x / np.sqrt(4 ** (10 * (1 - x)))


class Flapper(Calculation):
    r"""A Flapper has 2 states, open or close. When closed, there is no flow.
    When open, it is a regular frictional resistor. The condition to change
    state is when :math:`\dot{m} \leq \dot{m}_0` for some user provided
    :math:`\dot{m}_0`"""

    def __init__(
        self,
        open_at_current: KgPerS,
        f: float,
        fluid: LiquidFuncs,
        area: Meter2,
        open_rate: PerS,
        stop_on_open: bool = False,
        relaxation: Callable[[float], float] = legacy_relaxation,
        name: str = "Flapper",
    ):
        r"""

        Parameters
        ----------
        open_at_current: KgPerS
            At this :math:`\dot{m}_0` and lower, the flapper opens and remains
            open
        f: float
            Once open, the Flapper behaves as a resistor (current-squared),
            whose coefficient is ``f``.
            When closed, the returned error on the pressure equation is simply
            :math:`\dot{m}`, which should be zero. In this case the pressure is
            not constrained, as should be.
        open_rate: PerS (float)
            Once the flow condition has been met, the flapper opens gradually,
            transitioning from :math:`\dot{m}=0` to the characteristic
            local pressure drop. This transition rate is determined by
            ``open_rate``.
        stop_on_open: bool
            Control whether a `stop` signal will be given once the flapper
            opens. Default is False.
        relaxation: Callable[[float], float]
            Model for the flow rate when opening the flapper.
            Once the flapper is signalled to open, at time :math:`t_\text{open}`,
            this function :math:`r` controls the gradual transition to the open state.
            The ``open_rate`` parameter = :math:`\lambda` is used such that
            :math:`r(\lambda (t - t_\text{open}))` is the relaxation. Note that this way,
            the function should fulfill :math:`r(x\leq0)=0, r(x\geq1)=1`.
        fluid: LiquidFuncs
            Coolant properties
        area: Meter2

        See Also
        --------
        .local_pressure_by_mdot, .EffectivePipe
        """
        self.name = name
        self.mdot0 = open_at_current
        self.f = f
        self.t_open = np.inf
        self.open_rate = open_rate
        self.fluid = fluid
        self._A = area
        self._rho = fluid.density
        self.stop_on_open = stop_on_open
        self.relaxation = relaxation
        self._flag = False

    @unpacked
    def calculate(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        t: Second,
        **kwargs,
    ) -> Array1D:
        out = np.empty(2)
        T, dp = variables

        if t <= self.t_open:
            out[0] = T - Tin
            out[1] = mdot
        else:
            relax = self.relaxation(float(t - self.t_open) * self.open_rate)
            Tin_d = directed_Tin(Tin, Tin_minus, mdot)
            mdot_calc = -mdot_by_local_pressure(dp, self._rho(Tin_d), self.f, self._A)
            out[0] = T - Tin_d
            out[1] = mdot - relax * mdot_calc
        return out

    # These do follow the protocol and I can't see why they don't.
    # noinspection PyProtocol
    indices = LumpedComponent.indices
    # noinspection PyProtocol
    variables = LumpedComponent.variables
    # noinspection PyProtocol
    mass_vector = LumpedComponent.mass_vector
    __len__ = LumpedComponent.__len__
    dp_out = LumpedComponent.dp_out

    @unpacked
    def should_continue(self, variables: Sequence[float], *, ref_mdot: KgPerS, t: Second, **_) -> bool:
        return not (self.stop_on_open and self.t_open == t and self._flag)

    @unpacked
    def change_state(self, variables: Sequence[float], *, ref_mdot: KgPerS, t: Second, **_) -> None:
        self._flag = False
        if ref_mdot <= self.mdot0 and np.isposinf(self.t_open):
            self.t_open = float(t)
            self._flag = True
            logger.log(STREAM_DEBUG, f"{self} opened at t = {self.t_open}")

    def close(self):
        """Set flapper to be closed (flow is set to zero)"""
        self._flag = False
        self.t_open = np.inf

    def open(self, t: Second):
        """Set flapper to be opened starting at ``t``"""
        self._flag = True
        self.t_open = t
