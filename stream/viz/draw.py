"""Matplotlib drawing of results under the style context."""

from __future__ import annotations

from itertools import product
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .encoding import Encoding, encode
from .labels import Labels
from .results import y_axis
from .style import DEFAULT_STYLE, Style

BOUNDS = ("nominal", "lower", "upper")
BANDS = ("total", "split", "none")
FIELD_ARGUMENTS = ("color", "style", "marker", "col", "row")


def _check(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; use one of {', '.join(allowed)}")


def _levels_of(result, enc: Encoding):
    dims = [d for d in result.dims if d not in (enc.col, enc.row)]
    return dims, list(product(*(result.dims[d] for d in dims)))


def _subset(frame, dims, levels):
    keep = np.ones(len(frame), dtype=bool)
    for d, v in zip(dims, levels):
        column = frame[d].to_numpy()
        keep &= np.isclose(column, v) if np.issubdtype(column.dtype, np.number) else (column == v)
    return frame[keep]


def draw_lines(result, enc: Encoding, ax, style: Style, bound: str, band: str, errorbars: bool, labels: Labels) -> tuple[bool, bool]:
    """Draw one line per level combination; return whether the total and the inner band were drawn."""
    dims, combos = _levels_of(result, enc)
    xs = labels.scale(result.x)
    ys = labels.scale(y_axis(result, labels)[1])
    line_column = {"nominal": "value", "lower": "lower", "upper": "upper"}[bound]
    show_band = result.banded and bound == "nominal" and band != "none"
    drawn = inner_drawn = False
    for levels in combos:
        sub = _subset(result.frame, dims, levels).sort_values(result.x)
        if sub.empty:
            continue
        props = enc.props(dict(zip(dims, levels)))
        props.setdefault("color", style.palette[0])
        if result.kind == "profile" and not combos[1:]:
            props.setdefault("marker", "o")
        x, y = sub[result.x].to_numpy() * xs, sub[line_column].to_numpy() * ys
        if show_band and errorbars:
            lo, hi = sub["lower"].to_numpy() * ys, sub["upper"].to_numpy() * ys
            if np.any(hi > lo):
                ax.errorbar(x, y, yerr=[y - lo, hi - y], capsize=3, **props)
                drawn = True
                continue
        ax.plot(x, y, **props)
        if show_band:
            lo, hi = sub["lower"].to_numpy() * ys, sub["upper"].to_numpy() * ys
            if np.any(hi > lo):
                ax.fill_between(x, lo, hi, color=props["color"], alpha=style.band_alpha, linewidth=0)
                drawn = True
                if band == "split":
                    ilo, ihi = sub["inner_lower"].to_numpy() * ys, sub["inner_upper"].to_numpy() * ys
                    if np.any(ihi > ilo):
                        ax.fill_between(x, ilo, ihi, color=props["color"], alpha=style.band_alpha_inner, linewidth=0)
                        inner_drawn = True
    return drawn, inner_drawn


def draw_field(field, ax, style: Style, bound: str, labels: Labels) -> None:
    """Draw a rank-2 quantity as a colour mesh with a colour bar."""
    column = {"nominal": "value", "lower": "lower", "upper": "upper"}[bound]
    zs, xs, vs = labels.scale("z"), labels.scale("x"), labels.scale(field.quantity)
    zb = field.z_bounds if field.z_bounds is not None else np.arange(len(field.z) + 1) - 0.5
    xb = field.x_bounds if field.x_bounds is not None else np.arange(len(field.xs) + 1) - 0.5
    mesh = ax.pcolormesh(xb * xs, zb * zs, field.arrays[column] * vs, cmap=style.ramp, shading="flat")
    bar = ax.figure.colorbar(mesh, ax=ax)
    bar.set_label(labels.axis(field.quantity))
    if field.mask is not None and field.mask.any():
        rows = np.flatnonzero(field.mask.any(axis=1))
        cols = np.flatnonzero(field.mask.any(axis=0))
        ax.add_patch(
            Rectangle(
                (xb[cols[0]] * xs, zb[rows[0]] * zs),
                (xb[cols[-1] + 1] - xb[cols[0]]) * xs,
                (zb[rows[-1] + 1] - zb[rows[0]]) * zs,
                fill=False, edgecolor=style.palette[-1], linewidth=1.8, label="meat",
            )
        )
    ax.set_xlabel(labels.axis("x"))
    ax.set_ylabel(labels.axis("z"))
    ax.grid(False)


def _band_texts(band: str, inner_drawn: bool, sigma_text: str, labels: Labels) -> list[tuple[str | None, bool]]:
    """Each band entry's legend text and whether it is the inner systematic swatch."""
    if band == "split":
        total = labels["band_total"].display
        inner = [(labels["band_sys"].display, True)] if inner_drawn else []
        return inner + [(None if total is None else f"{total}{sigma_text}", False)]
    text = labels["band"].display
    return [(None if text is None else f"{text}{sigma_text}", False)]


def build_legend(ax, enc: Encoding, band_drawn: bool, inner_drawn: bool, band: str, sigma_text: str, labels: Labels, style: Style) -> None:
    """Put a legend on ``ax`` when there is a block or a band entry to show."""
    handles, texts = [], []
    for block in enc.legend:
        handles.append(Line2D([], [], color="none", label=block.title))
        texts.append(block.title)
        for text, props in block.entries:
            props = {"color": style.palette[-1], **props}
            handles.append(Line2D([], [], **props))
            texts.append(f"  {text}")
    if band_drawn:
        for text, is_inner in _band_texts(band, inner_drawn, sigma_text, labels):
            if text is None:
                continue
            alpha = style.band_alpha_inner if is_inner else style.band_alpha
            handles.append(Patch(facecolor=style.palette[-1], alpha=alpha))
            texts.append(text)
    if not texts:
        return
    if len(enc.legend) == 1 and not band_drawn:
        block = enc.legend[0]
        ax.legend(handles[1:], [t.strip() for t in texts[1:]], title=block.title)
        return
    ax.legend(handles, texts)


def finish_axes(ax, result, labels: Labels, style: Style, title: str) -> None:
    """Label the axes and put ``title`` above them in the subtitle size."""
    ax.set_xlabel(labels.axis(result.x))
    ax.set_ylabel(y_axis(result, labels)[0])
    ax.set_title(title, fontsize=style.subtitle_size, color=style.palette[-1])


def _panel_text(labels: Labels, fixed: dict) -> str:
    """``Display = value unit`` for a numeric panel level, the level's display name otherwise."""
    parts = []
    for name, level in fixed.items():
        parts.append(labels.describe({name: level}) if isinstance(level, Real) else labels.display(level))
    return ", ".join(parts)


def _check_field_arguments(values: dict, band: str, errorbars: bool) -> None:
    for name, value in values.items():
        if value is not None:
            raise ValueError(f"{name}={value!r} draws lines and a field draws a map; drop {name}=")
    if band != "total":
        raise ValueError(f"band={band!r} shades lines and a field draws a map; use bound= to pick an envelope")
    if errorbars:
        raise ValueError("errorbars=True draws bars on lines and a field draws a map; use bound= to pick an envelope")


def plot(result, ax=None, color=None, style=None, marker=None, col=None, row=None, bound="nominal", band="total",
         errorbars=False, style_sheet=None, labels=None):
    """Draw ``result`` and return ``(figure, axes)``; ``axes`` is an array for a panel grid."""
    _check("bound", bound, BOUNDS)
    _check("band", band, BANDS)
    sheet = style_sheet or DEFAULT_STYLE
    merged = result.labels.merged(labels)
    if ax is not None and (col or row):
        raise ValueError("ax= draws into one axes and cannot be combined with col= or row=")
    sigma_text = f" ({result.sigma:g}σ)" if result.sigma != 1.0 else ""
    with sheet.context():
        if result.kind == "field":
            _check_field_arguments(dict(zip(FIELD_ARGUMENTS, (color, style, marker, col, row))), band, errorbars)
            fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
            draw_field(result, ax, sheet, bound, merged)
            ax.set_title(result.subtitle_with(merged), fontsize=sheet.subtitle_size, color=sheet.palette[-1])
            return fig, ax
        enc = encode(result.dims, result.numeric, merged, sheet, color, style, marker, col, row, result.level_space)
        if enc.col is None and enc.row is None:
            fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
            drawn, inner_drawn = draw_lines(result, enc, ax, sheet, bound, band, errorbars, merged)
            finish_axes(ax, result, merged, sheet, result.subtitle_with(merged))
            build_legend(ax, enc, drawn, inner_drawn, band, sigma_text, merged, sheet)
            return fig, ax
        cols = result.dims.get(enc.col, [None]) if enc.col else [None]
        rows = result.dims.get(enc.row, [None]) if enc.row else [None]
        w, h = sheet.panel_size
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(w * len(cols), h * len(rows)), sharex=True, sharey=True, squeeze=False)
        drawn_any = inner_any = False
        for (ri, rv), (ci, cv) in product(enumerate(rows), enumerate(cols)):
            panel_ax = axes[ri, ci]
            fixed = {k: v for k, v in ((enc.row, rv), (enc.col, cv)) if k is not None}
            sub = _subset(result.frame, list(fixed), list(fixed.values()))
            panel = _with_frame(result, sub, fixed)
            drawn, inner_drawn = draw_lines(panel, enc, panel_ax, sheet, bound, band, errorbars, merged)
            drawn_any, inner_any = drawn_any or drawn, inner_any or inner_drawn
            title = ", ".join(p for p in (_panel_text(merged, fixed), result.subtitle_with(merged)) if p)
            finish_axes(panel_ax, panel, merged, sheet, title)
            if ci:
                panel_ax.set_ylabel("")
            if ri != len(rows) - 1:
                panel_ax.set_xlabel("")
        build_legend(axes[0, -1], enc, drawn_any, inner_any, band, sigma_text, merged, sheet)
        fig.tight_layout()
        return fig, axes


def _with_frame(result, frame, fixed):
    from dataclasses import replace

    dims = {k: v for k, v in result.dims.items() if k not in fixed}
    return replace(result, frame=frame, dims=dims)
