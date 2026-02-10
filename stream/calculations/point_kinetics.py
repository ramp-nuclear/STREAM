"""
A calculation for the point kinetics neutronics model
"""
import logging
from typing import Callable, Sequence

import numpy as np
from cytoolz.functoolz import curry

from stream import Calculation
from stream.calculation import CalcState
from stream.units import (
    Array, Array1D, Celsius, Name, PerC, PerS, Place, Second, Watt, WPerS)
from stream.utilities import just, STREAM_DEBUG, to_array


logger = logging.getLogger(__name__)


class PointKinetics(Calculation):
    r"""
    The Point Kinetics model is the simplest Neutronics dynamical model. It
    assumes spatial and spectral flux shapes are either unimportant or fixed,
    and deals with the flux by a single number which may be thought of as
    the population size or power.
    The delayed neutron yield processes are paramount to the dynamical
    description, thus they are characterized as well, bundled into k
    characteristic groups. The equations are:

    .. math:: \dot{P} &= \frac{\rho - \beta}{\Lambda} P
              + \sum_k C_k\lambda_k + \frac{S}{\Lambda} \\
              \dot{C}_k &= - C_k\lambda_k + \frac{\beta_k}{ \Lambda} P

    Where:

    - :math:`P`: power

    - :math:`\rho`: reactivity

    - :math:`\beta`: total delayed neutron fraction (1$)

    - :math:`\Lambda`: generation time

    - :math:`C_k`: k-group's contribution to power

    - :math:`\lambda_k` : k-group's decay rate.

    - :math:`S`: Power Source

    In this particular calculation, the reactivity may be influenced by a
    linear thermal feedback
    :math:`\rho = \rho_0 + \alpha_c T_c + \alpha_f T_f` by
    corresponding coolant and fuel elements.
    """

    def __init__(self, generation_time: Second,
                 delayed_neutron_fractions: Array1D,
                 delayed_groups_decay_rates: PerS,
                 temp_worth: dict[Calculation, PerC] | None = None,
                 ref_temp: dict[Calculation, Celsius] | None = None,
                 input_reactivity_func: Callable[
                     [Second, Second], float] = None,
                 SCRAM_condition: Callable[[Watt, WPerS, ...], bool] = None,
                 name: str = 'PK'):
        """
        Parameters
        ----------
        generation_time: Second
            mean time between neutron generations.
        delayed_neutron_fractions: Array1D
            the fractional yield of neutrons in defined delay groups. 1$ worth is the total.
        delayed_groups_decay_rates: PerS
            each group's decay rate.
        temp_worth: dict[Calculation, PerC] or None
            a dictionary whose keys are fuel or channel elements, and values are their temperature worth.
        ref_temp: dict[Calculation, Celsius] or None
            At such temperature/s, temperature feedback is 0.
        input_reactivity_func: Callable[[Second, Second], float] or None
            External reactivity input, a function of (time, SCRAM_time)
        SCRAM_condition: Callable[[Watt, WPerS], bool] or None
            A function returning whether SCRAM should occur, a function of (power [P, Ck], dPdt [P, Ck])


        """
        self.name = name
        self.lambdak = delayed_groups_decay_rates
        self.betak = delayed_neutron_fractions
        self.Lambda = generation_time
        self.temp_worth = temp_worth or {}
        self.T0 = ref_temp or {}
        self.m = len(self.lambdak)
        self.dollar = np.sum(self.betak)
        self.SCRAM_time = None
        self.SCRAM_condition = SCRAM_condition or just(False)
        self.input_reactivity = input_reactivity_func or just(0.0)

        self._A = np.zeros((self.m + 1, self.m + 1))
        self._s = np.zeros(self.m + 1)
        for k in range(1, self.m + 1):
            self._A[0, k] = self.lambdak[k - 1]
            self._A[k, k] = -self.lambdak[k - 1]
            self._A[k, 0] = self.betak[k - 1] / self.Lambda

    def reactivity(self, T: dict[Calculation, Array1D],
                   input_reactivity: float) -> float:
        """Calculate the reactivity, given temperature feedback

        Parameters
        ----------
        T: dict[Calculation, Array1D]
            Mapping of Calculation to temperatures (say in channels, fuels)
        input_reactivity: float

        Returns
        -------
        rho: float
            Calculated reactivity
        """
        return input_reactivity + temperature_reactivity(
            T=T, T0=self.T0, weights=self.temp_worth)

    # noinspection PyMethodOverriding
    def calculate(self, variables: Sequence[float], *,
                  T: dict[Calculation, Celsius] | None = None,
                  source: dict[Calculation, Watt] | None = None,
                  t: dict[Calculation, Second] | None = None,
                  **kwargs) -> Array1D:
        r"""Calculate :math:`\frac{d}{dt}(P, \vec{C}_k)`

        Parameters
        ----------
        variables: Sequence[float]
            :math:`(P, \vec{C}_k)`
        source: dict[Calculation, Watt] or None
            External Power Sources
        T: dict[Calculation, Celsius] or None
            Temperatures which affect reactivity through 'temp_worth'
        t: Second or None
            Time, used to call `input_reactivity`

        Returns
        -------
        dPdt: Array1D
            the change in power and the delayed power fractions
        """
        rho = self.reactivity(
            T or {},
            self.input_reactivity(t[self] if t else t, self.SCRAM_time))
        self._s[0] = (to_array(source).sum() / self.Lambda
                      if source is not None else 0.)
        self._A[0, 0] = (rho - self.dollar) / self.Lambda
        return self._A @ variables + self._s

    def should_continue(self, variables: Sequence[float], **kwargs) -> bool:
        if self.SCRAM_time:
            return not self.SCRAM_time == kwargs['t'][self]
        return True

    def change_state(self, variables: Sequence[float], **kwargs):
        power = variables[self.indices("power")]
        dPdt = self.calculate(variables, **kwargs)[self.indices("power")]
        if not self.SCRAM_time and self.SCRAM_condition(power, dPdt, **kwargs):
            self.SCRAM_time = t = kwargs['t'][self]
            logger.log(STREAM_DEBUG, f"SCRAM Signal given at {t = }")

    @property
    def mass_vector(self) -> Sequence[bool]:
        return np.ones(self.m + 1, dtype=bool)

    def __len__(self) -> int:
        return self.m + 1

    @property
    def variables(self) -> dict[Name, Place]:
        return dict(power=0, ck=slice(1, self.m + 1))

    # noinspection PyMethodOverriding
    def save(self, vector: Sequence[float], *, T=None, source=None, t=None,
             **kwargs) -> CalcState:
        """Given input for "calculate" (which is a legal state of the system), tag the information,
        i.e. create a "State" and return it.

        Parameters
        ----------
        vector: Sequence[float]
        source: dict[Calculation, Watt]
            External Power Sources
        T: dict[Calculation, Celsius]
            Temperatures which affect reactivity through 'temp_worth'
        t: Second or None
            Time, used to call `input_reactivity`

        Returns
        -------
        state: CalcState
            Tagged information regarding system state
        """
        state: CalcState = super().save(vector)
        rho = self.reactivity(T, self.input_reactivity(t[self] if t else t,
                                                       self.SCRAM_time))
        state["reactivity"] = rho
        state["dPdt"] = self.calculate(vector, source=source, T=T, t=t,
                                       **kwargs)[self.indices("power")]
        return state


def temperature_reactivity(T: dict[Calculation, Array],
                           T0: dict[Calculation, Array],
                           weights: dict[Calculation, Array],
                           ) -> float:
    r"""Calculate the reactivity, given temperature feedback

    .. math:: \rho = - \sum_i \vec{w}_i \cdot (\vec{T}-\vec{T}_0)_i

    Parameters
    ----------
    T: dict[Calculation, Array]
        Mapping of Channel calculation to temperatures in channel
    T0: dict[Calculation, Array]
        Reference Temperatures
    weights: dict[Calculation, Array]

    Returns
    -------
    rho: float
        Calculated reactivity
    """
    return -sum(np.dot(w, T[k] - T0[k]) for k, w in weights.items())


@curry
def SCRAM_at_power(power_limit: Watt, power: Watt, **kwargs):
    return power > power_limit


class PointKineticsWInput(PointKinetics):
    r"""The same good-old PK, but with added power_input

    An additional distinction is made between `power` (=total power) and `pk_power`.
    Such a distinction is important when regarding additional sources of power,
    stemming from the decay of fission products, activated structure materials,
    gamma absorption and more. These sources should be passed to this model
    through ``power_input``.

    .. seealso:: :mod:`~stream.physical_models.decay_heat`
    """

    def calculate(self, variables: Sequence[float], *,
                  T: dict[Calculation, Celsius] = None,
                  source: dict[Calculation, Watt] = None,
                  t: dict[Calculation, Second] = None,
                  power_input=None, **kwargs) -> Array1D:
        vals = np.empty(len(self))
        vals[:-1] = super().calculate(variables[:-1], source=source, T=T, t=t, **kwargs)
        vals[-1] = variables[0] + power_input[self] - variables[-1]
        return vals

    @property
    def mass_vector(self) -> Sequence[bool]:
        return [True] * (self.m + 1) + [False]

    def __len__(self): return self.m + 2

    @property
    def variables(self) -> dict[Name, Place]:
        return dict(pk_power=0, ck=slice(1, self.m + 1), power=self.m + 1)
