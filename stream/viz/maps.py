"""Filled contour maps of a reduced quantity over two parameters."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from .draw import BOUND_COLUMNS, _limit_text, _panel_text, _subset, _with_frame
from .labels import Labels
from .results import y_axis
from .style import Style

MAP_ARGUMENTS = ("color", "style", "marker")


def _check_map_arguments(values: dict, band: str, errorbars: bool) -> None:
    for name, value in values.items():
        if value is not None:
            raise ValueError(f"{name}= does not apply to a map; the colour is the quantity; use col= or row= for a dimension")
    if band != "total":
        raise ValueError("band= does not apply to a map; use bound= to pick an envelope")
    if errorbars:
        raise ValueError("errorbars= does not apply to a map; use bound= to pick an envelope")


def map_levels(result, column: str, scale: float, n: int) -> np.ndarray:
    """``n`` contour bands spanning the finite values of ``column`` over the whole result; one band when they are all equal."""
    values = result.frame[column].to_numpy(float) * scale
    values = values[np.isfinite(values)]
    lo, hi = (values.min(), values.max()) if values.size else (0.0, 1.0)
    if hi <= lo:
        return np.array([lo, lo + max(abs(lo), 1.0) * 1e-6])
    return np.linspace(lo, hi, n + 1)


def _grid_index(levels: np.ndarray, value: float) -> int:
    return int(np.flatnonzero(np.isclose(levels, value, rtol=1e-9, atol=0.0))[0])


def _in_panel(hole: dict, fixed: dict) -> bool:
    return all(hole[p] == v if isinstance(v, str) else np.isclose(hole[p], v) for p, v in fixed.items() if p in hole)


def draw_map(result, ax, style: Style, bound: str, limit: str, labels: Labels, levels: np.ndarray,
             grid: tuple[np.ndarray, np.ndarray]):
    """Filled contours of a reduced quantity over two parameters; returns the contour set for the colour bar.

    The limit contour is where the reduced quantity meets its declared limit: the curve
    ``required(solve_for=x, versus=y)`` would trace, drawn over the whole plane.
    """
    column = BOUND_COLUMNS[bound]
    single = "quantity" in result.frame.columns and len(result.frame)
    q = str(result.frame["quantity"].iloc[0]) if single else result.ylabel_source[0]
    xs, ys, vs = labels.scale(result.x), labels.scale(result.y), labels.scale(y_axis(result, labels)[1])
    gx, gy = grid
    X, Y = np.meshgrid(gx * xs, gy * ys, indexing="ij")
    Z = np.full(X.shape, np.nan)
    filled_at = []
    for row in result.frame.itertuples(index=False):
        i, j = _grid_index(gx, getattr(row, result.x)), _grid_index(gy, getattr(row, result.y))
        Z[i, j] = getattr(row, column) * vs
        if getattr(row, "filled", False):
            filled_at.append((X[i, j], Y[i, j]))
    masked = np.ma.masked_invalid(Z)
    cf = ax.contourf(X, Y, masked, levels=levels, cmap=style.ramp)
    lim = result.limits.get(q)
    if limit == "shade" and lim is None:
        raise ValueError(f"no limit declared for {q!r}; declare it in limits= to shade")
    if limit != "none" and lim is not None:
        level = lim.value * vs
        lo, hi = np.nanmin(Z), np.nanmax(Z)
        if lo < level < hi:
            cs = ax.contour(X, Y, masked, levels=[level], colors=[style.palette[2]], linewidths=style.limit_linewidth)
            ax.clabel(cs, fmt={level: _limit_text(labels, q, lim)}, fontsize=9, inline=True)
        if limit == "shade":
            margin = max(hi - lo, abs(level), 1.0)
            band = [lo - margin, level] if lim.side == "above" else [level, hi + margin]
            if band[0] < band[1]:
                shade = ax.contourf(X, Y, masked, levels=band, colors=["white"], alpha=style.shade_alpha,
                                    hatches=[style.shade_hatch])
                shade.set_edgecolor(style.palette[2])
                shade.set_linewidth(0)
    for hole in result.holes:
        if _in_panel(hole, result.fixed):
            ax.plot(hole[result.x] * xs, hole[result.y] * ys, linestyle="none", marker="x", color=style.palette[2],
                    markersize=7, zorder=5)
    if filled_at:
        fx, fy = zip(*filled_at)
        ax.plot(fx, fy, linestyle="none", marker=style.filled_marker, markerfacecolor="white",
                markeredgecolor=style.palette[-1], zorder=5)
    if style.map_cases:
        ax.plot(result.frame[result.x].to_numpy(float) * xs, result.frame[result.y].to_numpy(float) * ys,
                linestyle="none", marker="o", color=style.palette[-1], markersize=3, alpha=0.6, zorder=4)
    ax.set_xlabel(labels.axis(result.x))
    ax.set_ylabel(labels.axis(result.y))
    ax.grid(False)
    return cf


def plot_map(result, ax, sheet: Style, labels: Labels, color, style, marker, col, row, bound: str, band: str,
             errorbars: bool, limit: str):
    """Draw a map result, one panel per level of ``col`` and ``row``, with one colour bar and one scale."""
    _check_map_arguments(dict(zip(MAP_ARGUMENTS, (color, style, marker))), band, errorbars)
    panel_dims = [d for d in result.dims if d not in (col, row)]
    if panel_dims:
        raise ValueError(f"{panel_dims[0]!r} varies; a map takes it on col= or row=")
    for name, dim in (("col", col), ("row", row)):
        if dim is not None and dim not in result.dims:
            raise ValueError(
                f"{name}={dim!r} does not vary in this result; the varying dimensions are {', '.join(result.dims) or 'none'}"
            )
    bar_text, bar_name = y_axis(result, labels)
    levels = map_levels(result, BOUND_COLUMNS[bound], labels.scale(bar_name), sheet.map_levels)
    grid = (np.asarray(result.level_space[result.x], float), np.asarray(result.level_space[result.y], float))
    if col is None and row is None:
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        cf = draw_map(result, ax, sheet, bound, limit, labels, levels, grid)
        ax.set_title(result.subtitle_with(labels), fontsize=sheet.subtitle_size, color=sheet.palette[-1])
        fig.colorbar(cf, ax=ax).set_label(bar_text)
        return fig, ax
    cols = result.dims[col] if col else [None]
    rows = result.dims[row] if row else [None]
    w, h = sheet.panel_size
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(w * len(cols), h * len(rows)), sharex=True, sharey=True,
                             squeeze=False)
    cf = None
    for (ri, rv), (ci, cv) in product(enumerate(rows), enumerate(cols)):
        panel_ax = axes[ri, ci]
        fixed = {k: v for k, v in ((row, rv), (col, cv)) if k is not None}
        panel = _with_frame(result, _subset(result.frame, list(fixed), list(fixed.values())), fixed)
        panel = replace(panel, fixed={**result.fixed, **fixed})
        cf = draw_map(panel, panel_ax, sheet, bound, limit, labels, levels, grid)
        title = ", ".join(p for p in (_panel_text(labels, fixed), result.subtitle_with(labels)) if p)
        panel_ax.set_title(title, fontsize=sheet.subtitle_size, color=sheet.palette[-1])
        if ci:
            panel_ax.set_ylabel("")
        if ri != len(rows) - 1:
            panel_ax.set_xlabel("")
    fig.colorbar(cf, ax=axes.ravel().tolist()).set_label(bar_text)
    return fig, axes
