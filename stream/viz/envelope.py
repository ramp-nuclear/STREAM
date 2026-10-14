"""Uncertainty envelopes and the reducers that collapse profiles to numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandas import DataFrame

from .sweep import Coords

REDUCERS = ("min", "max", "mean", "where_min", "where_max")
ENVELOPES = ("lower", "value", "upper")
INNER = ("inner_lower", "inner_upper")


def with_envelopes(table: DataFrame, banded: bool, sigma: float = 1.0, uncertainty: bool = True) -> DataFrame:
    """``table`` with ``lower``/``upper`` and the systematic-only ``inner_lower``/``inner_upper``.

    ``lower = value - sys - sigma * stat`` and the mirror for ``upper``; the
    inner pair uses ``sys`` alone. All four equal ``value`` when the sweep is
    not banded or ``uncertainty`` is ``False``.
    """
    if banded and uncertainty:
        half = table["sys"] + sigma * table["stat"]
        return table.assign(
            lower=table["value"] - half,
            upper=table["value"] + half,
            inner_lower=table["value"] - table["sys"],
            inner_upper=table["value"] + table["sys"],
        )
    v = table["value"]
    return table.assign(lower=v, upper=v, inner_lower=v, inner_upper=v)


@dataclass(frozen=True)
class At:
    """A reducer picking one cell, by coordinate (nearest cell) or by index."""

    z: float | None = None
    x: float | None = None
    cell: int | None = None


def at(*, z: float | None = None, x: float | None = None, cell: int | None = None) -> At:
    """The value at ``z``, at ``x``, or at cell ``cell``."""
    if sum(v is not None for v in (z, x, cell)) != 1:
        raise ValueError("at() takes exactly one of z=, x=, cell=")
    return At(z, x, cell)


def reducer_name(reducer) -> str:
    """A short name for axis labels: ``min``, ``at z=0.35``, a callable's name."""
    if isinstance(reducer, str):
        return reducer
    if isinstance(reducer, At):
        if reducer.cell is not None:
            return f"at cell {reducer.cell}"
        axis, value = ("z", reducer.z) if reducer.z is not None else ("x", reducer.x)
        return f"at {axis}={value:g}"
    return getattr(reducer, "__name__", "custom")


def mesh_shape(rows: DataFrame, rank: int, coords: Coords | None = None) -> tuple[int, ...]:
    """The cell shape of a rank-1 or rank-2 quantity, from ``coords`` or from the rows' own indices."""
    if coords is not None and len(coords.names) == rank:
        return tuple(len(c) for c in coords.centers)
    if rows.empty:
        raise ValueError(f"the shape of a rank-{rank} quantity with no rows can only come from coords, which is None")
    if rank == 1:
        return (int(rows.j.max()) + 1,)
    return int(rows.i.max()) + 1, int(rows.j.max()) + 1


def to_array(rows: DataFrame, rank: int, column: str = "value", coords: Coords | None = None) -> float | np.ndarray:
    """The rows of one quantity in one case as a scalar, ``(n,)`` or ``(m, n)`` array.

    Rows are scattered to the cell their ``i`` and ``j`` name, so a cell with
    no row reads ``NaN`` and its neighbours keep their coordinates. The mesh
    comes from ``coords`` when given and from the largest index otherwise.
    """
    if rank == 0:
        return float(rows[column].iloc[0]) if len(rows) else float("nan")
    shape = mesh_shape(rows, rank, coords)
    places = (rows.j.to_numpy(dtype=int),) if rank == 1 else (rows.i.to_numpy(dtype=int), rows.j.to_numpy(dtype=int))
    kinds = ("row", "column") if rank == 2 else ("cell",)
    for place, size, kind in zip(places, shape, kinds):
        if place.size and place.max() >= size:
            raise ValueError(
                f"the {column!r} rows reach {kind} {place.max()} but the mesh has {size} of them; "
                "the frame and the aggregator disagree"
            )
    out = np.full(shape, np.nan)
    out[places] = rows[column].to_numpy(dtype=float)
    return out


def _axis_index(coords: Coords | None, values: np.ndarray, axis: str | None) -> int | None:
    if axis is None:
        return None
    names = coords.names if coords is not None else ("z", "x")[: values.ndim]
    if axis not in names:
        raise ValueError(f"axis {axis!r} is not one of {', '.join(names)}")
    return names.index(axis)


def _weights(coords: Coords | None, values: np.ndarray) -> np.ndarray:
    if coords is None:
        return np.ones_like(values, dtype=float)
    if values.ndim == 1:
        return np.asarray(coords.weights[0], float)
    return np.outer(coords.weights[0], coords.weights[1])


def _centers(coords: Coords | None, values: np.ndarray, k: int) -> np.ndarray:
    if coords is None:
        return np.arange(values.shape[k], dtype=float)
    return np.asarray(coords.centers[k], float)


def _pick(values: np.ndarray, coords: Coords | None, spec: At):
    if spec.cell is not None:
        if values.ndim != 1:
            raise ValueError("at(cell=...) applies to a profile; use at(z=...) or at(x=...) on a field")
        if not 0 <= spec.cell < values.shape[0]:
            raise ValueError(f"cell {spec.cell} is outside the {values.shape[0]} cells")
        return float(values[spec.cell])
    axis, wanted = ("z", spec.z) if spec.z is not None else ("x", spec.x)
    if coords is None or axis not in coords.names:
        raise ValueError(f"at({axis}=...) needs {axis} coordinates and this calculation has none")
    k = coords.names.index(axis)
    idx = int(np.argmin(np.abs(np.asarray(coords.centers[k]) - wanted)))
    return float(values[idx]) if values.ndim == 1 else np.take(values, idx, axis=k)


def _padded(values: np.ndarray, reducer: str) -> np.ndarray:
    pad = np.inf if reducer in ("min", "where_min") else -np.inf
    return np.where(np.isfinite(values), values, pad)


def reduce_array(values, coords: Coords | None, reducer, axis: str | None = None):
    """Collapse ``values`` with a named reducer, an :class:`At`, or a callable.

    A callable receives ``(values, coords)`` and returns what it likes.
    ``axis`` names the direction collapsed on a field; ``None`` collapses all.
    Cells holding ``NaN`` take no part, and a collapse with no finite cell
    left in it gives ``NaN``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return float(values)
    if callable(reducer) and not isinstance(reducer, At):
        return reducer(values, coords)
    if isinstance(reducer, At):
        return _pick(values, coords, reducer)
    if reducer not in REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; use one of {', '.join(REDUCERS)}, at(...), or a callable")
    k = _axis_index(coords, values, axis)
    finite = np.isfinite(values)
    any_finite = finite.any(axis=k)
    if reducer in ("min", "max"):
        out = np.where(any_finite, getattr(np, reducer)(_padded(values, reducer), axis=k), np.nan)
        return float(out) if k is None else out
    if reducer == "mean":
        w = _weights(coords, values) * finite
        total = np.sum(w, axis=k)
        out = np.sum(np.where(finite, values, 0.0) * w, axis=k) / np.where(total > 0.0, total, np.nan)
        return float(out) if k is None else out
    if values.ndim == 2 and k is None:
        raise ValueError(f"{reducer} on a field needs an axis; pass axis='z' or axis='x'")
    arg = np.argmin if reducer == "where_min" else np.argmax
    picked = arg(_padded(values, reducer), axis=k)
    if values.ndim == 1:
        return float(np.where(any_finite, _centers(coords, values, 0)[picked], np.nan))
    return np.where(any_finite, _centers(coords, values, k)[picked], np.nan)


def reduce_rows(rows: DataFrame, rank: int, coords: Coords | None, reducer, axis: str | None = None) -> dict:
    """``reduce_array`` applied to every envelope column of ``rows``."""
    return {c: reduce_array(to_array(rows, rank, c, coords), coords, reducer, axis) for c in ENVELOPES + INNER}
