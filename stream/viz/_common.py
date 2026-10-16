"""Helpers shared by the sweep, the questions and the drawing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

from ._envelope import reducer_name
from .labels import Labels

BOUND_COLUMNS = {"nominal": "value", "lower": "lower", "upper": "upper"}


@dataclass(frozen=True)
class _Coords:
    """Cell centres, edges and lengths of a calculation's mesh.

    ``names`` is ``("z",)`` for a channel and ``("z", "x")`` for a plate;
    the other tuples follow the same order. ``mask`` marks the plate's meat.
    """

    names: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    bounds: tuple[np.ndarray | None, ...]
    weights: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None


def _level_text(v: float | str) -> str:
    return v if isinstance(v, str) else format(v, "g")


def _case_text(case: Mapping[str, float | str]) -> str:
    return ", ".join(f"{k}={v if isinstance(v, str) else format(v, 'g')}" for k, v in case.items())


def crossings(x, y, target: float, method: str = "linear") -> list[float]:
    """Every ``x`` at which ``y`` crosses ``target``, ascending; ``x`` sorted."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return []
    if method == "pchip":
        spline = PchipInterpolator(x, y - target)
        return sorted(float(r) for r in spline.roots() if x[0] <= r <= x[-1])
    if method != "linear":
        raise ValueError(f"unknown method {method!r}; use 'linear' or 'pchip'")
    d = y - target
    out = []
    for k in range(len(x) - 1):
        if d[k] == 0:
            out.append(float(x[k]))
        elif d[k] * d[k + 1] < 0:
            out.append(float(x[k] + (x[k + 1] - x[k]) * d[k] / (d[k] - d[k + 1])))
    if d[-1] == 0:
        out.append(float(x[-1]))
    return sorted(set(out))


def y_axis_text(labels: Labels, quantities: list[str], prefix: str = "") -> str:
    """``prefix Display1, Display2 [unit]`` for quantities sharing a unit."""
    displays = ", ".join(labels.display(q) for q in quantities)
    units = [labels.unit(q) for q in quantities if labels.unit(q)]
    head = f"{prefix} " if prefix else ""
    return f"{head}{displays} [{units[0]}]" if units else f"{head}{displays}"


def y_axis(result, labels: Labels) -> tuple[str, str]:
    """The y axis text of ``result`` under ``labels``, and the name whose scale its values carry."""
    if result.kind == "field":
        return labels.axis("z"), "z"
    names = list(result._y_names)
    if result.question.solve_for is not None:
        return labels.axis(names[0]), names[0]
    shown = result._shown_reducer
    name = None if shown is None else reducer_name(shown)
    if name is not None and name.startswith("where_"):
        coordinate = result._coordinate or "z"
        unit = f" [{labels.unit(coordinate)}]" if labels.unit(coordinate) else ""
        return f"{name} {', '.join(labels.display(q) for q in names)}{unit}", coordinate
    prefix = "" if shown is None else reducer_name(shown, labels)
    return y_axis_text(labels, names, prefix), (names[0] if names else "")


def _subset(frame, dims, levels):
    keep = np.ones(len(frame), dtype=bool)
    for d, v in zip(dims, levels):
        column = frame[d].to_numpy()
        keep &= np.isclose(column, v) if np.issubdtype(column.dtype, np.number) else (column == v)
    return frame[keep]


def _with_frame(result, frame, fixed):
    dims = {k: v for k, v in result.dims.items() if k not in fixed}
    return replace(result, frame=frame, dims=dims)


def _panel_text(labels: Labels, fixed: dict) -> str:
    """``Display = value unit`` for a parameter panel level, the display name for a quantity or calculation."""
    parts = []
    for name, level in fixed.items():
        parts.append(labels.display(level) if name in ("quantity", "calculation") else labels.describe({name: level}))
    return ", ".join(parts)


def _limit_text(labels: Labels, q: str, limit) -> str:
    unit = labels[q].unit
    return f"{labels.display(q)} = {labels.value(q, limit.value)}" + (f" {unit}" if unit else "")
