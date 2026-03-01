"""The result of a time dependent solution of the initial value problem with an Aggregator"""

from dataclasses import dataclass

import numpy as np

from stream.units import Array1D, Array2D, Value


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

    """

    time: Array1D
    data: Array2D

    def __getitem__(self, item) -> Value:
        return self.data[item]

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other: "Solution") -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return np.allclose(self.time, other.time) and np.allclose(self.data, other.data)
