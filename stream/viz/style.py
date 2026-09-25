"""The look of every figure the module draws."""

from __future__ import annotations

from dataclasses import dataclass, replace

import matplotlib
from matplotlib import rc_context
from matplotlib.colors import to_hex


@dataclass(frozen=True)
class Style:
    """Colours, sizes and conventions applied while a figure is drawn.

    ``palette`` colours categorical levels, ``ramp`` names the colormap used
    for numeric levels, and the two alphas shade the total and the inner
    systematic band. Every field can be changed with :meth:`replace`.
    """

    palette: tuple[str, ...] = ("#2980b9", "#1abc9c", "#e67e22", "#8e44ad", "#7f8c8d", "#2c3e50")
    ramp: str = "viridis_r"
    band_alpha: float = 0.18
    band_alpha_inner: float = 0.35
    figsize: tuple[float, float] = (7.2, 4.2)
    panel_size: tuple[float, float] = (4.4, 3.6)
    linewidth: float = 1.8
    markersize: float = 5.0
    legend_frame: bool = False
    grid_alpha: float = 0.25
    subtitle_size: float = 9.5

    def replace(self, **changes) -> Style:
        """A copy with the given fields changed."""
        return replace(self, **changes)

    def context(self):
        """A matplotlib ``rc_context`` carrying this style, for use in a ``with``."""
        return rc_context(
            {
                "axes.prop_cycle": matplotlib.cycler(color=list(self.palette)),
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": self.grid_alpha,
                "legend.frameon": self.legend_frame,
                "lines.linewidth": self.linewidth,
                "lines.markersize": self.markersize,
                "figure.figsize": self.figsize,
                "axes.titlesize": 11,
                "axes.labelcolor": self.palette[-1],
            }
        )

    def ramp_colors(self, n: int) -> list[str]:
        """``n`` hex colours from the ramp, light to dark, avoiding the pale end."""
        cmap = matplotlib.colormaps[self.ramp]
        if n == 1:
            return [to_hex(cmap(0.95))]
        return [to_hex(cmap(0.25 + 0.7 * k / (n - 1))) for k in range(n)]


DEFAULT_STYLE = Style()
