"""Two calculations the loss-of-flow loop needs that the library does not ship: a smooth
one-way check valve and a low-flow trip sensor; plus a horizontal channel pressure law."""
from functools import partial

import numpy as np

from stream.calculation import Calculation, unpacked
from stream.calculations.ideal.ideal import LumpedComponent
from stream.physical_models.pressure_drop import pressure_diff
from stream.smoothing import smooth_pos_weight, smooth_step
from stream.units import Array1D, KgPerS, Pascal, Second


class CheckValve(LumpedComponent):
    """Quadratic resistance forward, ``reverse_ratio`` times stiffer backward beyond ``eps``, zero at zero flow."""

    def __init__(self, k_forward: float, reverse_ratio: float = 1000.0, eps: KgPerS = 0.1, name: str = "CheckValve"):
        self.name = name
        self.k = k_forward
        self.ratio = reverse_ratio
        self.eps = eps

    def dp_out(self, *, mdot: KgPerS, **_) -> Pascal:
        return -self.k * mdot * abs(mdot) * (1 + self.ratio * smooth_pos_weight(-mdot, self.eps))


class LowFlowTrip(Calculation):
    """Latches when the watched ``ref_mdot`` falls to ``threshold``; after ``delay`` the rods
    insert ``worth`` (absolute reactivity) over ``duration`` with a smooth ramp. One state
    variable that stays zero; the sensor exists for its events."""

    def __init__(self, threshold: KgPerS, delay: Second, worth: float, duration: Second, name: str = "Trip"):
        self.name = name
        self.threshold = threshold
        self.delay = delay
        self.worth = worth
        self.duration = duration
        self.arm()

    def arm(self) -> None:
        self.t_trip = np.inf
        self.tripped = False
        self.rods_moving = False

    @property
    def t_rods(self) -> float:
        return self.t_trip + self.delay

    def rho(self, t: Second) -> float:
        if t < self.t_rods:
            return 0.0
        return -self.worth * float(smooth_step(t, self.t_rods, self.t_rods + self.duration))

    def indices(self, variable, asking=None):
        raise KeyError(f"{self.name} serves no variables")

    @property
    def variables(self):
        return {"flag": 0}

    @property
    def mass_vector(self):
        return (False,)

    def __len__(self) -> int:
        return 1

    @unpacked
    def calculate(self, variables, *, ref_mdot: KgPerS, t: Second, **_) -> Array1D:
        return np.array([variables[0]])

    @unpacked
    def event_margin(self, variables, *, ref_mdot: KgPerS, t: Second, **_) -> Array1D:
        armed = 1.0 if self.tripped else ref_mdot - self.threshold
        rods = 1.0 if (not self.tripped or self.rods_moving) else self.t_rods - t
        return np.array([armed, rods])

    @unpacked
    def change_state(self, variables, *, ref_mdot: KgPerS, t: Second, **_) -> None:
        if not self.tripped and ref_mdot <= self.threshold:
            self.t_trip = float(t)
            self.tripped = True
        if self.tripped and t >= self.t_rods:
            self.rods_moving = True

    @unpacked
    def should_continue(self, variables, **_) -> bool:
        return True


def horizontal_pressure(f=None):
    """``pressure_diff`` without gravity, optionally with friction factor ``f``."""
    return partial(pressure_diff, g=0.0) if f is None else partial(pressure_diff, g=0.0, f=f)
