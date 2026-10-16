"""Matplotlib drawing of results under the style context."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ._encoding import Encoding, LegendBlock, encode
from ._inverse import crossings
from ._results import y_axis
from .labels import Labels
from .style import DEFAULT_STYLE, Style

BOUNDS = ("nominal", "lower", "upper")
BANDS = ("total", "split", "none")
LIMITS = ("line", "shade", "none")
CROSSINGS = ("none", "mark", "legend")
BOUND_COLUMNS = {"nominal": "value", "lower": "lower", "upper": "upper"}
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


@dataclass(frozen=True)
class Crossing:
    """Where one drawn curve first meets the limit of its quantity, in frame units."""

    levels: dict
    props: dict
    quantity: str
    x: float


def binder_colors(frame, style: Style) -> dict[str, str]:
    """A palette colour per binder, in order of first appearance."""
    names = list(dict.fromkeys(frame.loc[np.isfinite(frame["value"].to_numpy(float)), "binder"].astype(str)))
    return {name: style.palette[k % len(style.palette)] for k, name in enumerate(names)}


def draw_lines(result, enc: Encoding, ax, style: Style, bound: str, band: str, errorbars: bool, labels: Labels,
               binding: dict[str, str] | None = None, crossing: str = "none") -> tuple[bool, bool, bool, list[Crossing]]:
    """Draw one line per level combination.

    Return whether the total band, the inner band and an interpolated point
    were drawn, and the first crossing of the limit by each curve when
    ``crossing`` is not ``"none"``. ``binding`` maps each binder to its
    colour; the points of each binder get a marker in it. Interpolated
    points get an open marker.
    """
    dims, combos = _levels_of(result, enc)
    xs = labels.scale(result.x)
    ys = labels.scale(y_axis(result, labels)[1])
    line_column = BOUND_COLUMNS[bound]
    show_band = result._spec.banded and bound == "nominal" and band != "none"
    drawn = inner_drawn = filled_drawn = False
    found = []
    for levels in combos:
        sub = _subset(result.frame, dims, levels).sort_values(result.x)
        if sub.empty:
            continue
        props = enc.props(dict(zip(dims, levels)))
        props.setdefault("color", style.palette[0])
        if result.kind == "profile" and not combos[1:]:
            props.setdefault("marker", "o")
        if crossing != "none":
            hit = _first_crossing(result, sub, line_column)
            if hit is not None:
                found.append(Crossing(dict(zip(dims, levels)), dict(props), *hit))
        x, y = sub[result.x].to_numpy() * xs, sub[line_column].to_numpy() * ys
        if show_band and errorbars:
            lo, hi = sub["lower"].to_numpy() * ys, sub["upper"].to_numpy() * ys
            if np.any(hi > lo):
                ax.errorbar(x, y, yerr=[y - lo, hi - y], capsize=3, **props)
                _draw_binders(ax, sub, x, y, binding)
                filled_drawn = _draw_filled(ax, sub, x, y, props["color"], style) or filled_drawn
                drawn = True
                continue
        ax.plot(x, y, **props)
        _draw_binders(ax, sub, x, y, binding)
        filled_drawn = _draw_filled(ax, sub, x, y, props["color"], style) or filled_drawn
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
    return drawn, inner_drawn, filled_drawn, found


def _first_crossing(result, sub, column: str) -> tuple[str, float] | None:
    q = _quantities_on(result)[0] if result.binding or "quantity" not in sub.columns else str(sub["quantity"].iloc[0])
    found = crossings(sub[result.x].to_numpy(float), sub[column].to_numpy(float), result.limits[q].value, "linear")
    return (q, found[0]) if found else None


def _draw_filled(ax, sub, x, y, color: str, style: Style) -> bool:
    if "filled" not in sub.columns:
        return False
    f = sub["filled"].to_numpy(bool)
    if not f.any():
        return False
    ax.plot(x[f], y[f], linestyle="none", marker=style.filled_marker, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.4, zorder=4)
    return True


def _draw_binders(ax, sub, x, y, binding: dict[str, str] | None) -> None:
    for name, color in (binding or {}).items():
        hit = (sub["binder"].astype(str) == name).to_numpy()
        if hit.any():
            ax.plot(x[hit], y[hit], linestyle="none", marker="o", color=color, zorder=3)


def _cell_bounds(field) -> tuple[np.ndarray, np.ndarray]:
    zb = field.z_bounds if field.z_bounds is not None else np.arange(len(field.z) + 1) - 0.5
    xb = field.x_bounds if field.x_bounds is not None else np.arange(len(field.xs) + 1) - 0.5
    return zb, xb


def draw_field(field, ax, style: Style, bound: str, labels: Labels) -> None:
    """Draw a rank-2 quantity as a colour mesh with a colour bar."""
    column = BOUND_COLUMNS[bound]
    zs, xs, vs = labels.scale("z"), labels.scale("x"), labels.scale(field.quantity)
    zb, xb = _cell_bounds(field)
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


def _limit_text(labels: Labels, q: str, limit) -> str:
    unit = labels[q].unit
    return f"{labels.display(q)} = {labels.value(q, limit.value)}" + (f" {unit}" if unit else "")


def _quantities_on(result) -> list[str]:
    if "quantity" in result.frame.columns:
        return list(dict.fromkeys(result.frame["quantity"].astype(str)))
    return list(result._spec.ylabel_source)


def _shade(ax, style: Style, color: str, span: tuple[float, float], x=None, y=None) -> None:
    kwargs = dict(facecolor="white", alpha=style.shade_alpha, hatch=style.shade_hatch, edgecolor=color, linewidth=0)
    if x is None:
        ax.axhspan(span[0], span[1], **kwargs)
    else:
        ax.fill_between(x, y, span[1], **kwargs)


def _is_required(result) -> bool:
    return result.kind == "curve" and result._spec.reducer is None and bool(result._spec.targets)


def _limit_color(enc: Encoding, style: Style, q: str) -> str:
    return enc.maps["color"][q] if enc.channels.get("color") == "quantity" else style.palette[2]


def _one_curve_error(result) -> ValueError:
    if "quantity" in result.dims:
        return ValueError("shade needs one quantity per axes; put quantity on col= or row=")
    if "calculation" in result.dims:
        return ValueError("shade needs one curve per axes; put calculation on col= or row=")
    if result.dims:
        return ValueError(f"shade needs one curve per axes; put {next(iter(result.dims))} on col= or row=")
    return ValueError("shade needs one curve per axes")


def required_side(result) -> str | None:
    """The side of the crossing where the quantities drawn on these axes are safe, read from the frame.

    ``None`` when no drawn crossing has a side; raises when the drawn crossings disagree.
    """
    frame = result.frame
    finite = frame[np.isfinite(frame["value"].to_numpy(float)) & frame["safe"].isin(["below", "above"]).to_numpy()]
    sides = dict.fromkeys(finite["safe"])
    if len(sides) > 1:
        below = finite.loc[finite["safe"] == "below", "quantity"].iloc[0]
        above = finite.loc[finite["safe"] == "above", "quantity"].iloc[0]
        raise ValueError(
            f"{below} is safe below the required {result._spec.ylabel_source[0]} and {above} above it; "
            "shade one quantity per axes"
        )
    return next(iter(sides), None)


def _check_shade(result, enc: Encoding) -> None:
    if _is_required(result):
        unknown = list(dict.fromkeys(result.frame.loc[result.frame["safe"].isna(), "quantity"].astype(str)))
        if unknown:
            raise ValueError(f"shading needs the side of the limit; declare {', '.join(unknown)} in limits=")
        required_side(result)
        if len(result.frame) != len(result.frame.drop_duplicates(result.x)):
            raise _one_curve_error(result)
        return
    if result.limit_note is not None:
        raise ValueError(result.limit_note)
    drawn = _quantities_on(result)
    if len(drawn) > 1 and not result.binding:
        raise ValueError("shade needs one quantity per axes; put quantity on col= or row=")
    if "calculation" in enc.channels.values():
        raise _one_curve_error(result)
    if drawn[0] not in result.limits:
        raise ValueError(f"no limit declared for {drawn[0]!r}; declare it in limits= to shade")


def draw_limit_lines(result, ax, enc: Encoding, style: Style, labels: Labels, limit: str) -> None:
    """Dashed lines at the declared limits of the drawn quantities; checks a shade request first."""
    if limit == "shade":
        _check_shade(result, enc)
    if limit == "none" or _is_required(result):
        return
    ys = labels.scale(y_axis(result, labels)[1])
    drawn = _quantities_on(result)
    for q in drawn[:1] if result.binding else drawn:
        lim = result.limits.get(q)
        if lim is None:
            continue
        color = _limit_color(enc, style, q)
        y = lim.value * ys
        ax.axhline(y, linestyle="--", linewidth=style.limit_linewidth, color=color, zorder=2)
        ax.annotate(_limit_text(labels, q, lim), xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=9, color=color)


def shade_limits(result, ax, enc: Encoding, style: Style, labels: Labels, limit: str) -> None:
    """Hatch the unsafe side up to the current axes edge; the axes limits are kept as they are."""
    if limit != "shade":
        return
    lo, hi = ax.get_ylim()
    ys = labels.scale(y_axis(result, labels)[1])
    if _is_required(result):
        xs = labels.scale(result.x)
        side = required_side(result)
        if side is None:
            return
        sub = result.frame.sort_values(result.x)
        x, y = sub[result.x].to_numpy(float) * xs, sub["value"].to_numpy(float) * ys
        edge = hi if side == "below" else lo
        _shade(ax, style, style.palette[2], (lo, edge), x=x, y=y)
    else:
        q = _quantities_on(result)[0]
        lim = result.limits[q]
        y = lim.value * ys
        _shade(ax, style, _limit_color(enc, style, q), (lo, y) if lim.side == "above" else (y, hi))
    ax.set_ylim(lo, hi)


def draw_limits(result, ax, enc: Encoding, style: Style, labels: Labels, limit: str) -> None:
    """Dashed lines at the declared limits of the drawn quantities, and the unsafe side shaded on request."""
    draw_limit_lines(result, ax, enc, style, labels, limit)
    if limit == "shade":
        ax.autoscale_view()
        shade_limits(result, ax, enc, style, labels, limit)


def _check_crossing(result, crossing: str) -> None:
    if crossing == "none":
        return
    if _is_required(result):
        raise ValueError("crossing= needs a horizontal limit; a required result has none, its y axis is the crossing")
    if result.kind in ("map", "field"):
        raise ValueError(f"crossing= does not apply to a {result.kind}")
    if result.limit_note is not None:
        raise ValueError(result.limit_note)
    drawn = _quantities_on(result)
    for q in drawn[:1] if result.binding else drawn:
        if q not in result.limits:
            raise ValueError(f"no limit declared for {q!r}; declare it in limits= to mark its crossing")


def _crossing_text(labels: Labels, result, xc: float) -> str:
    unit = labels[result.x].unit
    return labels.value(result.x, xc) + (f" {unit}" if unit else "")


def draw_crossings(result, ax, found: list[Crossing], style: Style, labels: Labels, foot: bool) -> None:
    """A dashed drop line and a marker where each curve meets its limit, the value at the foot on request.

    The axes limits are kept as they are.
    """
    if not found:
        return
    lo, hi = ax.get_ylim()
    xs = labels.scale(result.x)
    ys = labels.scale(y_axis(result, labels)[1])
    for c in found:
        color = c.props["color"]
        x, y = c.x * xs, result.limits[c.quantity].value * ys
        ax.vlines(x, lo, y, linestyles="--", linewidth=style.crossing_linewidth, colors=color, zorder=2)
        ax.plot(x, y, linestyle="none", marker="o", markersize=style.markersize, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        if foot:
            ax.annotate(_crossing_text(labels, result, c.x), xy=(x, lo), xycoords="data", xytext=(3, 3),
                        textcoords="offset points", rotation=90, ha="left", va="bottom", fontsize=9, color=color)
    ax.set_ylim(lo, hi)


def _level_text(result, labels: Labels, dim: str, level) -> str:
    return labels.value(dim, level) if dim in result._spec.numeric else labels.display(level)


def crossing_legend(result, enc: Encoding, found: list[Crossing], labels: Labels) -> Encoding:
    """The encoding with each crossing value added to the legend."""
    if not found:
        return enc
    shared = len({c.quantity for c in found}) == 1
    q = found[0].quantity
    title = f"{labels.display(result.x)} at {_limit_text(labels, q, result.limits[q]) if shared else 'the limit'}"
    encoded = list(dict.fromkeys(enc.channels.values()))
    if len(encoded) == 1:
        (dim,) = encoded
        order = sorted(result.dims[dim]) if dim in result._spec.numeric else list(result.dims[dim])
        extra = {order.index(c.levels[dim]): f" ({_crossing_text(labels, result, c.x)})" for c in found}
        (block,) = enc.legend
        entries = [(text + extra.get(k, ""), props) for k, (text, props) in enumerate(block.entries)]
        return replace(enc, legend=[LegendBlock(f"{block.title}; {title}", entries)])
    entries = []
    for c in found:
        text = _crossing_text(labels, result, c.x)
        head = [_level_text(result, labels, d, v) for d, v in c.levels.items()]
        if not shared:
            head.append(_limit_text(labels, c.quantity, result.limits[c.quantity]))
        entries.append((f"{', '.join(head)}: {text}" if head else text, c.props))
    return replace(enc, legend=[*enc.legend, LegendBlock(title, entries)])


def draw_field_limit(field, ax, style: Style, labels: Labels, limit: str, bound: str = "nominal") -> None:
    """The limit contour over a field image, and the cells beyond it hatched on request."""
    lim = field.limits.get(field.quantity)
    if limit == "none" or lim is None:
        if limit == "shade" and lim is None:
            raise ValueError(f"no limit declared for {field.quantity!r}; declare it in limits= to shade")
        return
    zs, xs, vs = labels.scale("z"), labels.scale("x"), labels.scale(field.quantity)
    xx, zz = np.meshgrid(field.xs * xs, field.z * zs)
    drawn = field.arrays[BOUND_COLUMNS[bound]]
    values = drawn * vs
    level = lim.value * vs
    if np.nanmin(values) < level < np.nanmax(values):
        cs = ax.contour(xx, zz, values, levels=[level], colors=[style.palette[2]], linewidths=style.limit_linewidth)
        ax.clabel(cs, fmt={level: _limit_text(labels, field.quantity, lim)}, fontsize=9, inline=True)
    unsafe = lim.unsafe(drawn)
    if limit == "shade" and unsafe.any():
        zb, xb = _cell_bounds(field)
        cells = np.ma.masked_array(np.zeros(unsafe.shape), mask=~unsafe)
        ax.pcolor(xb * xs, zb * zs, cells, cmap=ListedColormap(["white"]), alpha=style.shade_alpha,
                  edgecolor=style.palette[2], linewidth=0, hatch=style.shade_hatch)


def _band_texts(band: str, inner_drawn: bool, sigma_text: str, labels: Labels) -> list[tuple[str | None, bool]]:
    """Each band entry's legend text and whether it is the inner systematic swatch."""
    if band == "split":
        total = labels["band_total"].display
        inner = [(labels["band_sys"].display, True)] if inner_drawn else []
        return inner + [(None if total is None else f"{total}{sigma_text}", False)]
    text = labels["band"].display
    return [(None if text is None else f"{text}{sigma_text}", False)]


def build_legend(ax, enc: Encoding, band_drawn: bool, inner_drawn: bool, band: str, sigma_text: str, labels: Labels, style: Style,
                 binding: dict[str, str] | None = None, filled_drawn: bool = False) -> None:
    """Put a legend on ``ax`` when there is a block, a band or an interpolated entry to show."""
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
    if filled_drawn:
        handles.append(Line2D([], [], linestyle="none", marker=style.filled_marker, markerfacecolor="white",
                              markeredgecolor=style.palette[-1]))
        texts.append(labels["filled"].display)
    if binding:
        handles.append(Line2D([], [], color="none", label=labels["binding"].display))
        texts.append(labels["binding"].display)
        for name, color in binding.items():
            handles.append(Line2D([], [], linestyle="none", marker="o", color=color))
            texts.append(f"  {name}")
    if not texts:
        return
    if len(enc.legend) == 1 and not band_drawn and not binding and not filled_drawn:
        block = enc.legend[0]
        ax.legend(handles[1:], [t.strip() for t in texts[1:]], title=block.title)
        return
    if not enc.legend and not band_drawn and binding and not filled_drawn:
        ax.legend(handles[1:], [t.strip() for t in texts[1:]], title=labels["binding"].display)
        return
    ax.legend(handles, texts)


def finish_axes(ax, result, labels: Labels, style: Style, title: str) -> None:
    """Label the axes and put ``title`` above them in the subtitle size."""
    ax.set_xlabel(labels.axis(result.x))
    ax.set_ylabel(y_axis(result, labels)[0])
    ax.set_title(title, fontsize=style.subtitle_size, color=style.palette[-1])


def _panel_text(labels: Labels, fixed: dict) -> str:
    """``Display = value unit`` for a parameter panel level, the display name for a quantity or calculation."""
    parts = []
    for name, level in fixed.items():
        parts.append(labels.display(level) if name in ("quantity", "calculation") else labels.describe({name: level}))
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
         errorbars=False, limit="line", crossing="none", style_sheet=None, labels=None):
    """Draw ``result`` and return ``(figure, axes)``; ``axes`` is an array for a panel grid.

    Parameters
    ----------
    result:
        The answer to draw, from profile, field, reduce or required.
    ax:
        Draw into this axes instead of a new figure. Not with ``col`` or ``row``.
    color, style, marker:
        Dimensions on colour, line style and marker. Unassigned dimensions
        take them in that order, ``quantity`` first. Not for a map, whose
        colour is the quantity, nor for a field.
    col, row:
        Dimensions on a grid of panels. Not for a field.
    bound:
        Which envelope column is drawn:

        * ``"nominal"``: ``value`` (the default);
        * ``"lower"``: the lower envelope;
        * ``"upper"``: the upper envelope.
    band:
        The uncertainty band around a nominal line. Not for a map or a field.

        * ``"total"``: one band from ``lower`` to ``upper`` (the default);
        * ``"split"``: the total band with the systematic part shaded darker inside it;
        * ``"none"``: no band.
    errorbars:
        Draw the band as error bars instead of a fill. Not for a map or a field.
    limit:
        The declared limit of each drawn quantity:

        * ``"line"``: a dashed line at the limit, a contour on a map or a field,
          nothing extra on a ``required`` result (the default);
        * ``"shade"``: the line and the unsafe side hatched; on a ``required``
          result, the side of the curve where the limit is not met. Needs one
          quantity (or one curve) per axes; each panel counts on its own.
          Raises when the result carries no limit for the drawn quantity (none
          declared, or a ``where_*`` or custom reducer), when several
          quantities share the axes, or on a ``required`` result whose side is
          unknown;
        * ``"none"``: neither.
    crossing:
        Where each curve meets the declared limit of its quantity, at its
        first linear crossing in ``x`` on the drawn ``bound``:

        * ``"none"``: nothing (the default);
        * ``"mark"``: for every drawn curve, a dashed drop line in the
          curve's colour from the x axis up to the limit, a marker where the
          curve meets the limit, and the x value printed along the foot of
          the line;
        * ``"legend"``: the drop line and marker, with the x value in the
          legend instead of at the foot; on a panel grid the value is printed
          at the foot inside each panel, as ``"mark"`` does, because the
          legend is shared.

        A curve that never meets its limit gets no mark. For a ``profile`` or
        a ``reduce`` curve only. Raises on a ``required`` result, a map or a
        field, and when a drawn quantity carries no limit (none declared, or
        a ``where_*`` or custom reducer).
    style_sheet:
        A :class:`Style` for the look (sizes, colours, the map's levels, the
        markers of filled cases).
    labels:
        Display names merged over the result's own for this figure. An
        entry is a ``Label`` or a shorthand tuple; the 4-tuple
        ``(display, unit, scale, fmt)`` also sets how that name's numbers
        print, with ``fmt`` a format spec like ``".1f"`` or a callable on the
        scaled number. A name without a format prints with the style
        sheet's ``number_format``.
    """
    _check("bound", bound, BOUNDS)
    _check("band", band, BANDS)
    _check("limit", limit, LIMITS)
    _check("crossing", crossing, CROSSINGS)
    _check_crossing(result, crossing)
    sheet = style_sheet or DEFAULT_STYLE
    merged = result._spec.labels.merged(labels).with_default_format(sheet.number_format)
    if ax is not None and (col or row):
        raise ValueError("ax= draws into one axes and cannot be combined with col= or row=")
    sigma_text = f" ({format(result._spec.sigma, merged.default_format)}σ)" if result._spec.sigma != 1.0 else ""
    with sheet.context():
        if result.kind == "field":
            _check_field_arguments(dict(zip(FIELD_ARGUMENTS, (color, style, marker, col, row))), band, errorbars)
            fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
            draw_field(result, ax, sheet, bound, merged)
            draw_field_limit(result, ax, sheet, merged, limit, bound)
            ax.set_title(result.subtitle_with(merged), fontsize=sheet.subtitle_size, color=sheet.palette[-1])
            return fig, ax
        if result.kind == "map":
            from ._maps import plot_map

            return plot_map(result, ax, sheet, merged, color, style, marker, col, row, bound, band, errorbars, limit)
        enc = encode(result.dims, result._spec.numeric, merged, sheet, color, style, marker, col, row, result._spec.level_space)
        binding = binder_colors(result.frame, sheet) if result.binding else None
        if enc.col is None and enc.row is None:
            fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
            drawn, inner_drawn, filled_drawn, found = draw_lines(result, enc, ax, sheet, bound, band, errorbars, merged,
                                                                 binding, crossing)
            draw_limits(result, ax, enc, sheet, merged, limit)
            if found:
                ax.autoscale_view()
                draw_crossings(result, ax, found, sheet, merged, foot=crossing == "mark")
            if crossing == "legend":
                enc = crossing_legend(result, enc, found, merged)
            finish_axes(ax, result, merged, sheet, result.subtitle_with(merged))
            build_legend(ax, enc, drawn, inner_drawn, band, sigma_text, merged, sheet, binding=binding,
                         filled_drawn=filled_drawn)
            return fig, ax
        cols = result.dims.get(enc.col, [None]) if enc.col else [None]
        rows = result.dims.get(enc.row, [None]) if enc.row else [None]
        w, h = sheet.panel_size
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(w * len(cols), h * len(rows)), sharex=True, sharey=True, squeeze=False)
        drawn_any = inner_any = filled_any = False
        panels = []
        for (ri, rv), (ci, cv) in product(enumerate(rows), enumerate(cols)):
            panel_ax = axes[ri, ci]
            fixed = {k: v for k, v in ((enc.row, rv), (enc.col, cv)) if k is not None}
            sub = _subset(result.frame, list(fixed), list(fixed.values()))
            panel = _with_frame(result, sub, fixed)
            drawn, inner_drawn, filled_drawn, found = draw_lines(panel, enc, panel_ax, sheet, bound, band, errorbars, merged,
                                                                 binding, crossing)
            panels.append((panel, panel_ax, found))
            drawn_any, inner_any, filled_any = drawn_any or drawn, inner_any or inner_drawn, filled_any or filled_drawn
            finish_axes(panel_ax, panel, merged, sheet, _panel_text(merged, fixed))
            if ci:
                panel_ax.set_ylabel("")
            if ri != len(rows) - 1:
                panel_ax.set_xlabel("")
        for panel, panel_ax, _ in panels:
            draw_limit_lines(panel, panel_ax, enc, sheet, merged, limit)
        if limit == "shade" or any(hits for _, _, hits in panels):
            axes[0, 0].autoscale_view()
        for panel, panel_ax, found in panels:
            shade_limits(panel, panel_ax, enc, sheet, merged, limit)
            draw_crossings(panel, panel_ax, found, sheet, merged, foot=True)
        build_legend(axes[0, -1], enc, drawn_any, inner_any, band, sigma_text, merged, sheet, binding=binding,
                     filled_drawn=filled_any)
        subtitle = result.subtitle_with(merged)
        if subtitle:
            fig.suptitle(subtitle, fontsize=sheet.subtitle_size, color=sheet.palette[-1])
        fig.tight_layout()
        return fig, axes


def _with_frame(result, frame, fixed):
    dims = {k: v for k, v in result.dims.items() if k not in fixed}
    return replace(result, frame=frame, dims=dims)
