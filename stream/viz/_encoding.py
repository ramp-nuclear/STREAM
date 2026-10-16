"""Which dimension goes to which visual channel, and what the legend says."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path

from .labels import STREAM_LABELS, Labels
from .style import Style

CAPACITY = {"color": 6, "style": 4, "marker": 6}
LINESTYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D", "v", "P")
PROPERTY = {"color": "color", "style": "linestyle", "marker": "marker"}
PACKAGE = str(Path(__file__).parent) + os.sep


@dataclass
class LegendBlock:
    """One titled group of legend entries, ``(text, line properties)`` each."""

    title: str
    entries: list[tuple[str, dict]]


@dataclass
class Encoding:
    """The assignment of dimensions to channels for one figure."""

    channels: dict[str, str]
    maps: dict[str, dict]
    legend: list[LegendBlock]
    col: str | None = None
    row: str | None = None
    numeric: frozenset[str] = field(default_factory=frozenset)

    def props(self, levels: dict) -> dict:
        """Line properties for one combination of dimension levels."""
        out = {}
        for channel, dim in self.channels.items():
            out[PROPERTY[channel]] = self.maps[channel][levels[dim]]
        return out


def _ramp_position(space: list, level) -> int | None:
    for k, v in enumerate(space):
        if isclose(float(v), float(level), rel_tol=1e-9, abs_tol=0.0):
            return k
    return None


def _color_map(levels: list, numeric: bool, labels: Labels, style: Style, space: list | None = None) -> dict:
    if numeric:
        ordered = sorted(levels)
        positions = [_ramp_position(space or [], v) for v in ordered]
        if None in positions:
            space, positions = ordered, list(range(len(ordered)))
        colors = style.ramp_colors(len(space))
        return {level: colors[k] for level, k in zip(ordered, positions)}
    registry = [k for k in labels if k in levels]
    ordered = registry + [v for v in dict.fromkeys(levels) if v not in registry]
    out, k = {}, 0
    for level in ordered:
        fixed = labels[level].color
        if fixed is not None:
            out[level] = fixed
        else:
            out[level] = style.palette[k % len(style.palette)]
            k += 1
    return out


def _caller_level() -> int:
    frame, level = sys._getframe(2), 2
    while frame is not None and frame.f_code.co_filename.startswith(PACKAGE):
        frame, level = frame.f_back, level + 1
    return level


def _entries(dim: str, levels: list, numeric: bool, labels: Labels) -> list[str]:
    if not numeric:
        return [labels.display(v) for v in levels]
    texts = [labels.value(dim, v) for v in levels]
    first: dict[str, float] = {}
    for v, text in zip(levels, texts):
        if text in first and first[text] != v:
            warnings.warn(
                f"{dim} levels {first[text]:g} and {v:g} both print as {text!r}; "
                f"declare a finer format for {dim} in labels=",
                stacklevel=_caller_level(),
            )
            break
        first.setdefault(text, v)
    return texts


def encode(dims: dict, numeric: frozenset, labels: Labels, style: Style, color=None, style_=None, marker=None,
           col=None, row=None, level_space: dict | None = None) -> Encoding:
    """Assign the varying dimensions to colour, line style, marker and panels.

    Explicit arguments name dimensions; the rest are assigned in order:
    ``quantity`` to colour when it varies, then the remaining dimensions
    to colour, style and marker in turn. A fourth line dimension raises.
    ``level_space`` gives each parameter's full list of levels in the sweep;
    a numeric level takes its ramp position there, so it keeps its colour
    when another figure shows fewer levels.
    """
    labels = STREAM_LABELS.merged(labels).with_default_format(getattr(labels, "default_format", STREAM_LABELS.default_format))
    requested = {"color": color, "style": style_, "marker": marker, "col": col, "row": row}
    seen: dict[str, str] = {}
    for channel, dim in requested.items():
        if dim is None:
            continue
        if dim not in dims:
            raise ValueError(f"{channel}={dim!r} does not vary in this result; the varying dimensions are {', '.join(dims) or 'none'}")
        if dim in seen:
            raise ValueError(f"{dim!r} is assigned to both {seen[dim]} and {channel}")
        seen[dim] = channel
    free_channels = [c for c in ("color", "style", "marker") if requested[c] is None]
    remaining = [d for d in dims if d not in seen]
    if "quantity" in remaining and "color" in free_channels:
        requested["color"] = "quantity"
        free_channels.remove("color")
        remaining.remove("quantity")
    for dim in remaining:
        if not free_channels:
            raise ValueError(
                f"{dim!r} varies but colour, line style and marker are all taken; put it on a panel grid "
                f"with col= or row=, or fix it with a where= key"
            )
        requested[free_channels.pop(0)] = dim
    channels = {c: requested[c] for c in ("color", "style", "marker") if requested[c] is not None}
    maps, legend = {}, []
    for channel, dim in channels.items():
        levels = sorted(dims[dim]) if dim in numeric else list(dims[dim])
        if len(levels) > CAPACITY[channel]:
            raise ValueError(f"{dim!r} has {len(levels)} levels, more than {channel} can show ({CAPACITY[channel]})")
        if channel == "color":
            maps[channel] = _color_map(levels, dim in numeric, labels, style, (level_space or {}).get(dim))
        else:
            table = LINESTYLES if channel == "style" else MARKERS
            maps[channel] = dict(zip(levels, table))
        texts = _entries(dim, levels, dim in numeric, labels)
        entries = [(t, {PROPERTY[channel]: maps[channel][v]}) for t, v in zip(texts, levels)]
        legend.append(LegendBlock(labels.axis(dim), entries))
    return Encoding(channels, maps, legend, requested["col"], requested["row"], numeric)
