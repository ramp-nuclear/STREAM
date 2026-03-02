"""
A calculation for the point kinetics neutronics model
"""

import logging
from enum import Enum, StrEnum
from typing import Protocol, Sequence, TypeVar

import numpy as np
from cytoolz.functoolz import curry

from stream import Calculation
from stream.calculation import CalcState, unpacked
from stream.units import (
    Array,
    Array1D,
    Celsius,
    Name,
    PerC,
    PerS,
    Place,
    Second,
    Watt,
    WPerS,
)
from stream.utilities import just

logger = logging.getLogger(__name__)


class InputReactivity(Protocol):
    def __call__(self, state: Enum, t_state: Second, t: Second, **_) -> float:
        """
        Parameters
        ----------
        state: Enum
            Current state
        t_state: Second
            Time when the current state was initiated
        t: Second
            Current time

        Returns
        -------
        rho_c: float
            Inserted control reactivity
        """
        pass


S = TypeVar("S", bound=Enum)


class StateMachine(Protocol):
    """Reactor control state machine. Assued Markovian"""

    def __call__(self, state: S, t: Second, power: Watt, dPdt: WPerS, **kwargs) -> S:
        """
        Parameters
        ----------
        state: Enum
            Current state
        t: Second
            Current time
        power: Watt
            Current power
        dPdt: WPerS
            Current power change rate

        Returns
        -------
        new_state: Enum
            New state, which may be the same as the current state.
        """
        pass


class OneWayToSCRAM(StrEnum):
    NORMAL = "NORMAL"
    SCRAM = "SCRAM"


class ReactivityController:
    r"""Input to :class:`PointKinetics` which depicts the reactivity worth inserted
    due to the reactor control system or postulated events. The control system is
    modelled as a :class:`StateMachine`. This system can include the
    Reactor Protection System (RPS), Reactor Control and Monitoring System (RCMS) and such.

    The state machine is completely user defined such through an 'Enum' class. Then,
    two functionalities are requited:
    1. The reactivity worth response of the controller :math:`w_c(s, t_s, t)` where
       :math:`s` the current state, :math:`t_s` the time in which this state was invoked,
       and :math:`t` the current time.
    2. The state machine transfer function :math:`p: s \rightarrow s'` where `p` receives
       :math:`p(s, t, P, \dot{P}, ...)` provided by :class:`PointKinetics` during simulation.

    After simulation, the `log` attribute of this class contains the history of states, which can be used for analysis and plotting.
    """

    def __init__(
        self,
        input_reactivity: InputReactivity | None = None,
        state_machine: StateMachine = just(OneWayToSCRAM.NORMAL),
        initial_state: S = OneWayToSCRAM.NORMAL,
        initial_time: Second = 0.0,
        abort_states: set[S] | None = None,
    ):
        r"""
        Parameters
        ----------
        input_reactivity: InputReactivity or None
            The reactivity worth response of the controller :math:`w_c(s, t_s, t)` where
            :math:`s` the current state, :math:`t_s` the time in which this state was invoked,
            and :math:`t` the current time. If None, then no reactivity is inserted.
        state_machine: StateMachine
            The state machine transfer function :math:`p: s \rightarrow s'` where `p` receives
            :math:`p(s, t, P, \dot{P}, ...)` provided by :class:`PointKinetics` during simulation.
            If None, then the state machine is static and does not change state.
        initial_state: Enum
            Initial state
        initial_time: Second
            Initial time
        abort_states: set[Enum] or None
            States for which the simulation should stop, through the `should_continue` function.
        """
        self.input_reactivity = input_reactivity or just(0.0)
        self.state = initial_state
        self.t_state = initial_time
        self.log = [(initial_state, initial_time)]
        self.state_machine = state_machine
        self.abort_states = abort_states or set()

    def change_state(self, t: Second, power: Watt, dPdt: WPerS, **kwargs) -> S:
        s = self.state_machine(self.state, t, power, dPdt, **kwargs)
        if self.state != s:
            self.state = s
            self.t_state = t
            self.log.append((s, t))
            logger.info(f"Control State set to {s} at {t = }")
        return S

    def should_continue(self, t: Second) -> bool:
        abort = self.state in self.abort_states and t == self.t_state
        return not abort

    def worth(self, t: Second) -> float:
        """Reactivity worth inserted by the controller as function of time"""
        return self.input_reactivity(self.state, self.t_state, t)

    def worth_history(self, t: Second) -> float:
        sn, tn = self.log[0]
        for i in range(1, len(self.log)):
            sp, tp = self.log[i - 1]
            sn, tn = self.log[i]
            if tp <= t < tn:
                return self.input_reactivity(sp, tp, t)
        return self.input_reactivity(sn, tn, t)

    def reset(self):
        self.state, self.t_state = self.log[0]
        self.log = [self.log[0]]
        return self


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

    def __init__(
        self,
        generation_time: Second,
        delayed_neutron_fractions: Array1D,
        delayed_groups_decay_rates: PerS,
        temp_worth: dict[Calculation, PerC] | None = None,
        ref_temp: dict[Calculation, Celsius] | None = None,
        controls: ReactivityController | None = None,
        name: str = "PK",
    ):
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
        controls: ReactivityController
            Induces reactivity due to a state-machine model of the reactor controls,
            including any postulated accidents.
            By default, does nothing.
        """
        self.name = name
        self.lambdak = delayed_groups_decay_rates
        self.betak = delayed_neutron_fractions
        self.Lambda = generation_time
        self.temp_worth = temp_worth or {}
        self.T0 = ref_temp or {}
        self.m = len(self.lambdak)
        self.dollar = np.sum(self.betak).item()
        self.controls = controls or ReactivityController()

        self._A = np.zeros((self.m + 1, self.m + 1))
        self._s = np.zeros(self.m + 1)
        for k in range(1, self.m + 1):
            self._A[0, k] = self.lambdak[k - 1]
            self._A[k, k] = -self.lambdak[k - 1]
            self._A[k, 0] = self.betak[k - 1] / self.Lambda

    def reactivity(self, T: dict[Calculation, Array1D], input_reactivity: float) -> float:
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
        return input_reactivity + temperature_reactivity(T=T, T0=self.T0, weights=self.temp_worth)

    @unpacked(exclude=("T",))
    def calculate(
        self,
        variables: Sequence[float],
        *,
        T: dict[Calculation, Celsius] | None = None,
        source: Watt | None = None,
        t: Second,
        **kwargs,
    ) -> Array1D:
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
        rhoc = self.controls.worth(t)
        rho = self.reactivity(T if T is not None else {}, rhoc)
        self._s[0] = source / self.Lambda if source is not None else 0.0
        self._A[0, 0] = (rho - self.dollar) / self.Lambda
        return self._A @ variables + self._s

    # noinspection PyProtocol
    def should_continue(self, variables: Sequence[float], *, t: Second, **kwargs) -> bool:
        return self.controls.should_continue(t)

    @unpacked(exclude=("T",))
    def change_state(self, variables: Sequence[float], *, t: Second, **kwargs):
        power = variables[self.indices("power")]
        dPdt = self.calculate(variables, t=t, **kwargs)[self.indices("power")]
        self.controls.change_state(t, power, dPdt, **kwargs)

    @property
    def mass_vector(self) -> Sequence[bool]:
        return np.ones(self.m + 1, dtype=bool)

    def __len__(self) -> int:
        return self.m + 1

    @property
    def variables(self) -> dict[Name, Place]:
        return dict(power=0, ck=slice(1, self.m + 1))

    # noinspection PyMethodOverriding
    @unpacked(exclude=("T",))
    def save(self, vector: Sequence[float], *, T=None, source=None, t, **kwargs) -> CalcState:
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
        rhoc = self.controls.worth_history(t)
        rho = self.reactivity(T or {}, rhoc)
        state["reactivity"] = rho
        state["dPdt"] = self.calculate(vector, source=source, T=T, t=t, **kwargs)[self.indices("power")]
        return state


def temperature_reactivity(
    T: dict[Calculation, Array],
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
    return -sum(np.dot(w, T[k] - T0[k]).item() for k, w in weights.items())


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

    @unpacked(exclude=("T",))
    def calculate(
        self,
        variables: Sequence[float],
        *,
        T: dict[Calculation, Celsius] = None,
        source: Watt | None = None,
        t: Second,
        power_input: Watt | None = None,
        **kwargs,
    ) -> Array1D:
        vals = np.empty(len(self))
        vals[:-1] = super().calculate(variables[:-1], source=source, T=T, t=t, **kwargs)
        vals[-1] = variables[0] + power_input - variables[-1]
        return vals

    @property
    def mass_vector(self) -> Sequence[bool]:
        return [True] * (self.m + 1) + [False]

    def __len__(self):
        return self.m + 2

    @property
    def variables(self) -> dict[Name, Place]:
        return dict(pk_power=0, ck=slice(1, self.m + 1), power=self.m + 1)
