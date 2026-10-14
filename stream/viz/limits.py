"""Design limits a sweep's quantities must respect, declared once on the sweep."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Limit:
    """A limit value and the side of it that is safe.

    ``at_least(1.3)`` means values of 1.3 and above are safe; ``at_most(900.0)``
    means values of 900 and below are. The value is in the frame's own units,
    before any display scale. ``reducer`` names the mesh reducer that finds the
    worst value, ``min`` for ``at_least`` and ``max`` for ``at_most``.
    """

    value: float
    side: str

    @property
    def reducer(self) -> str:
        return "min" if self.side == "above" else "max"

    def unsafe(self, values) -> np.ndarray:
        """Which of ``values`` lie strictly beyond the limit."""
        v = np.asarray(values, dtype=float)
        return v < self.value if self.side == "above" else v > self.value

    def text(self) -> str:
        """``at least 1.3`` or ``at most 900``."""
        return f"{'at least' if self.side == 'above' else 'at most'} {self.value:g}"


def _number(value, helper: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{helper}() takes a number, not {value!r}")
    return float(value)


def at_least(value) -> Limit:
    """A limit the quantity must stay at or above, such as a CHFR of 1.3."""
    return Limit(_number(value, "at_least"), "above")


def at_most(value) -> Limit:
    """A limit the quantity must stay at or below, such as a clad temperature of 900 K."""
    return Limit(_number(value, "at_most"), "below")
