"""The result of a time dependent solution of the initial value problem with an Aggregator"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from stream.units import Array1D, Array2D, Value


class SolveStatus(Enum):
    """How a :class:`Solution` ended.

    A returned Solution is either complete or legitimately stopped — a failure
    raises rather than returning, so there is deliberately no ``FAILED`` member.
    """

    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass(slots=True, frozen=True)
class Event:
    """A transition serviced during a solve.

    Parameters
    ----------
    t: float
        The time at which the event fired.
    stopped: tuple[str, ...]
        Names of the nodes whose ``should_continue`` was False at the event. Empty
        (``()``) for a non-terminal transition/restart record that changed state but
        did not request a stop.
    terminal: bool
        Whether this event ended the run.
    """

    t: float
    stopped: tuple[str, ...]
    terminal: bool


@dataclass(slots=True, frozen=True)
class Solution:
    """The result of asking an :class:`.Aggregator` to solve a system of equations.

    Parameters
    ----------
    time: Array1D
        The times at which the solution was calculated.
    data: Array2D
        The vector of values for each time in the time vector.
        Shaped as (len(time), len(state_vector))
    status: SolveStatus
        Whether the run reached the end of the requested grid (``COMPLETED``) or was
        ended early by a stop event (``STOPPED``). Defaults to ``COMPLETED``.
    events: tuple[Event, ...]
        The transitions serviced during the solve (margin/polling driven), in order.
        Empty by default and on ALG paths (where truncation is signalled by
        ``status``, not by an event record).

    """

    time: Array1D
    data: Array2D
    status: SolveStatus = SolveStatus.COMPLETED
    events: tuple[Event, ...] = ()

    @property
    def completed(self) -> bool:
        """Whether the run reached the end of the requested grid."""
        return self.status is SolveStatus.COMPLETED

    @property
    def t_stop(self) -> float | None:
        """The time the run stopped at (the last solved time) if ``STOPPED``, else ``None``."""
        return float(self.time[-1]) if self.status is SolveStatus.STOPPED else None

    def __getitem__(self, item) -> Value:
        return self.data[item]

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other: "Solution") -> bool:
        # Equality compares time/data only; status/events are ignored, so a STOPPED and a COMPLETED solution with identical arrays are equal.
        if not isinstance(other, Solution):
            return NotImplemented
        try:
            return np.allclose(self.time, other.time) and np.allclose(self.data, other.data)
        except ValueError as e:
            raise ValueError(
                f"Solutions have different shapes: time {np.shape(self.time)} vs {np.shape(other.time)}, "
                f"data {np.shape(self.data)} vs {np.shape(other.data)} — "
                "are they from the same Aggregator (and the same run length)?"
            ) from e
