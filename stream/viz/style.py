"""The look of every figure the module draws."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

import matplotlib
from matplotlib import rc_context
from matplotlib.colors import to_hex

__all__ = ["Style"]


@dataclass(frozen=True)
class Style:
    """Colours, sizes and conventions applied while a figure is drawn.

    ``palette`` colours categorical levels, ``ramp`` names the colormap used
    for numeric levels, and the two alphas shade the total and the inner
    systematic band. Every field can be changed with :meth:`replace`.

    ``limit_color`` is the colour of a declared limit wherever one is drawn
    (the dashed line, the contour on a map or a field, the hatch edge);
    ``limit_linewidth``, ``shade_alpha`` and ``shade_hatch`` draw it and
    ``crossing_linewidth`` the drop line where a curve meets it;
    ``map_levels`` is the number of filled contour bands of a map and
    ``map_cases`` adds a dot at every solved case; ``filled_marker`` marks a
    case filled by interpolation.

    ``number_format`` is the Python format spec of every printed number whose
    label declares no format of its own.

    ``with style:`` applies the style to everything drawn inside the block, as
    ``with style.context():`` does.
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
    limit_color: str = "#e67e22"
    limit_linewidth: float = 1.6
    crossing_linewidth: float = 1.2
    shade_alpha: float = 0.55
    shade_hatch: str = "////"
    map_levels: int = 12
    map_cases: bool = False
    filled_marker: str = "o"
    number_format: str = ".4g"

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

    def __enter__(self) -> Self:
        context = self.context()
        context.__enter__()
        object.__setattr__(self, "_entered", (*getattr(self, "_entered", ()), context))
        return self

    def __exit__(self, *exc) -> None:
        *rest, context = self._entered
        object.__setattr__(self, "_entered", tuple(rest))
        context.__exit__(*exc)

    def ramp_colors(self, n: int) -> list[str]:
        """``n`` hex colours from the ramp, light to dark, avoiding the pale end."""
        cmap = matplotlib.colormaps[self.ramp]
        if n == 1:
            return [to_hex(cmap(0.95))]
        return [to_hex(cmap(0.25 + 0.7 * k / (n - 1))) for k in range(n)]


DEFAULT_STYLE = Style()
