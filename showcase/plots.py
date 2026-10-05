"""Figures shared by the showcase notebooks.

Every function draws into ``ax`` when one is given, otherwise into a new figure, and
returns the Figure. Colours follow the entity (channel, leg or run status), never the
order in which series are drawn.
"""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = dict(hot="#eb6834", warm="#1baf7a", wide="#eda100", bypass="#2a78d6", flapper="#e87ba4", pump="#008300")
STATUS_COLORS = dict(completed="#0ca30c", saturation="#fab219", failed="#d03b3b", timeout="#d03b3b")
STATUS_MARKERS = dict(completed="o", saturation="^", failed="X", timeout="s")
NEUTRAL = "#6b6b6b"
INK = "#222222"
LEGS = ("hot", "warm", "wide", "bypass", "flapper", "pump")

_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 8.0,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.frameon": False,
}


@contextmanager
def style():
    """Apply the showcase rcParams for the duration of the block."""
    with mpl.rc_context(_RC):
        yield


def _axes(ax, figsize=(7.5, 3.6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax
    return ax.figure, ax


def _color(name: str) -> str | None:
    for key, value in COLORS.items():
        if key in name:
            return value
    return None


def _legend(ax) -> None:
    if len(ax.get_legend_handles_labels()[1]) >= 2:
        ax.legend(loc="best")


def hline(ax, y: float, label: str) -> None:
    """A dashed neutral reference line at ``y`` with ``label`` at its right end."""
    ax.axhline(y, color=NEUTRAL, linestyle="--", linewidth=1.5)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", color=NEUTRAL)


def flows(df: pd.DataFrame, ax=None, zoom: tuple[float, float] | None = None, legs=None):
    """Mass flow of each leg in ``df`` (columns ``t`` and leg names) against time.

    ``zoom`` restricts the time axis and fits the vertical axis to the drawn legs
    inside that window; a zero line marks flow reversal.
    """
    fig, ax = _axes(ax)
    legs = [k for k in (legs or LEGS) if k in df.columns]
    for k in legs:
        ax.plot(df["t"], df[k], color=COLORS[k], label=k.capitalize())
    if zoom is not None:
        window = df[(df["t"] >= zoom[0]) & (df["t"] <= zoom[1])]
        values = window[legs].to_numpy()
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        pad = 0.08 * (hi - lo or 1.0)
        ax.set_xlim(*zoom)
        ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0.0, color=NEUTRAL, linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass flow [kg/s]")
    _legend(ax)
    return fig


def peaks(df: pd.DataFrame, tsat: float, ax=None):
    """Hottest coolant cell of each channel column in ``df`` against time, with the
    saturation temperature ``tsat`` (°C) as a dashed line."""
    fig, ax = _axes(ax)
    for k in [c for c in df.columns if c != "t"]:
        ax.plot(df["t"], df[k], color=_color(k) or INK, label=k.capitalize())
    hline(ax, tsat, f"Saturation {tsat:.1f} °C")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Peak coolant temperature [°C]")
    _legend(ax)
    return fig


def profile(z, T_cool, T_wall=None, ax=None, color: str = COLORS["hot"], title: str | None = None,
            legend: bool = True):
    """Axial coolant (solid) and wall (dashed) temperature against height ``z`` (m)."""
    fig, ax = _axes(ax, figsize=(3.2, 3.6))
    ax.plot(T_cool, z, color=color, label="Coolant")
    if T_wall is not None:
        ax.plot(T_wall, z, color=color, linestyle="--", label="Wall")
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Height [m]")
    if title:
        ax.set_title(title)
    if legend:
        _legend(ax)
    return fig


def schedule(events: dict[str, float], ax=None):
    """Named event times (s) on a logarithmic time line; labels sit on four staggered
    levels joined to their markers by leader lines."""
    fig, ax = _axes(ax, figsize=(7.5, 2.8))
    items = sorted(events.items(), key=lambda kv: kv[1])
    times = np.array([t for _, t in items], dtype=float)
    lo, hi = times.min() / 2, times.max() * 2
    ax.hlines(0.0, lo, hi, color=NEUTRAL, linewidth=1.5)
    ax.plot(times, np.zeros_like(times), "o", color=INK, markersize=8)
    for i, (name, t) in enumerate(items):
        dy = (14, -14, 46, -46)[i % 4]
        ax.annotate(f"{name}\n{t:.0f} s", xy=(t, 0.0), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top",
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=1.0, shrinkA=0, shrinkB=5))
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False, axis="y")
    ax.set_xlabel("Time [s]")
    return fig


def sweep_line(df: pd.DataFrame, x: str, ys, ax=None, xlabel: str | None = None, ylabel: str | None = None):
    """Columns ``ys`` of a sweep table against column ``x``, one marked line each."""
    fig, ax = _axes(ax)
    ys = [ys] if isinstance(ys, str) else list(ys)
    data = df.sort_values(x)
    for y in ys:
        ax.plot(data[x], data[y], marker="o", color=_color(y) or INK, label=y)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or (ys[0] if len(ys) == 1 else "Value"))
    _legend(ax)
    return fig


def status_strip(df: pd.DataFrame, x: str, ax=None, xlabel: str | None = None):
    """Each run's ``status`` as a coloured marker at its value of ``x``."""
    fig, ax = _axes(ax, figsize=(7.5, 1.6))
    for status in STATUS_COLORS:
        rows = df[df["status"] == status]
        if len(rows):
            ax.plot(rows[x], np.zeros(len(rows)), linestyle="none", marker=STATUS_MARKERS[status],
                    color=STATUS_COLORS[status], markersize=10, label=status.capitalize())
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False, axis="y")
    ax.set_xlabel(xlabel or x)
    if len(ax.get_legend_handles_labels()[1]):
        ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.35))
    return fig
