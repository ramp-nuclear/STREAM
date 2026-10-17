"""Display names, units and scales for everything the figures print."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

__all__ = ["Label", "Labels"]


@dataclass(frozen=True)
class Label:
    """How one name is shown.

    Parameters
    ----------
    display:
        Text used on axes, legend titles and subtitles. ``None`` hides the
        entry from legends while the thing it labels is still drawn.
    unit:
        Unit of the number after ``scale`` is applied, in matplotlib math
        text. An empty string means dimensionless and prints no bracket.
    scale:
        Factor applied to the stored numbers before drawing or printing.
    fmt:
        How a number of this name prints: a Python format spec applied after
        ``scale`` (``".1f"``, ``".3g"``, ``",.0f"``, ``".2e"``) or a callable
        taking the scaled number and returning text. ``None`` uses the
        registry's default.
    color:
        Optional fixed colour for this name when it is a legend level.
    """

    display: str | None
    unit: str = ""
    scale: float = 1.0
    fmt: str | Callable[[float], str] | None = None
    color: str | None = None


def _as_label(name: str, spec) -> Label:
    """Turn a registry entry into a :class:`Label`.

    Accepted forms: a ``Label``; ``None`` (hidden); a string (display only);
    a 2-tuple ``(display, unit)``; a 3-tuple ``(display, unit, scale)``;
    a 4-tuple ``(display, unit, scale, fmt)`` with ``fmt`` a format spec or a callable.
    """
    if spec is None:
        return Label(None)
    if isinstance(spec, Label):
        return spec
    if isinstance(spec, str):
        return Label(spec)
    if (isinstance(spec, tuple) 
        and 2 <= len(spec) <= 4 
        and all(isinstance(s, str) for s in spec[:2]) 
        and (len(spec) < 4 or isinstance(spec[3], str) or callable(spec[3]))
        ):
            return Label(*spec)
    raise TypeError(
        f"label for {name!r} must be a string, a (display, unit) tuple, a (display, unit, scale) "
        f"tuple, a (display, unit, scale, fmt) tuple with fmt a format spec or a callable, a Label "
        f"or None, not {spec!r}"
    )


class Labels(Mapping[str, Label]):
    """A registry from internal names to :class:`Label` records.

    Lookups never fail: a name with no entry displays as itself with no unit.
    Later registries win in :meth:`merged`, their ``default_format`` too.

    A number prints with its label's ``fmt`` when it declares one, else with
    ``default_format``, a Python format spec (``".4g"`` unless given). A
    registry's own ``default_format`` reaches ``Result.subtitle``,
    ``Result.ylabel`` and direct :meth:`value` and :meth:`describe` calls;
    in figures :meth:`~stream.viz.Result.plot` replaces it with the style
    sheet's ``number_format``.
    """

    def __init__(self, entries: Mapping[str, object] | None = None, *, default_format: str = ".4g"):
        self._entries: dict[str, Label] = {k: _as_label(k, v) for k, v in (entries or {}).items()}
        self.default_format = default_format

    def __getitem__(self, name: str) -> Label:
        return self._entries.get(name, Label(name))

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def known(self, name: str) -> bool:
        """Whether ``name`` has an entry."""
        return name in self._entries

    def merged(self, other: Labels | Mapping[str, object] | None) -> Labels:
        """A new registry with ``other``'s entries overriding this one's.

        When ``other`` is a :class:`Labels`, its ``default_format`` wins too.
        """
        merged = Labels(default_format=other.default_format if isinstance(other, Labels) else self.default_format)
        merged._entries = dict(self._entries)
        if other is not None:
            merged._entries.update(Labels(other)._entries if not isinstance(other, Labels) else other._entries)
        return merged

    def with_default_format(self, fmt: str) -> Labels:
        """A copy whose numbers without a format of their own print with ``fmt``."""
        copy = self.merged(None)
        copy.default_format = fmt
        return copy

    def unit(self, name: str) -> str | None:
        """The unit string, or ``None`` when the name has no entry."""
        return self._entries[name].unit if name in self._entries else None

    def scale(self, name: str) -> float:
        return self[name].scale

    def display(self, name) -> str:
        """The display text of ``name``, or ``name`` itself when the entry hides it."""
        label = self[name]
        return str(name) if label.display is None else str(label.display)

    def axis(self, name: str) -> str:
        """``Display [unit]``, or ``Display`` when the unit is empty."""
        label = self[name]
        return f"{self.display(name)} [{label.unit}]" if label.unit else self.display(name)

    def value(self, name: str, x: float | str) -> str:
        """``x`` scaled and formatted for printing; a string is returned as it is.

        The format is the label's ``fmt`` when it has one, else ``default_format``.
        """
        if isinstance(x, str):
            return x
        label = self[name]
        y = x * label.scale
        fmt = label.fmt or self.default_format
        return fmt(y) if callable(fmt) else format(y, fmt)

    def describe(self, fixed: Mapping[str, float | str]) -> str:
        """``Display = value unit`` for each fixed parameter, comma separated."""
        parts = []
        for name, x in fixed.items():
            label = self[name]
            unit = f" {label.unit}" if label.unit else ""
            parts.append(f"{self.display(name)} = {self.value(name, x)}{unit}")
        return ", ".join(parts)


STREAM_LABELS = Labels(
    {
        "T_cool": ("Coolant temperature", "°C"),
        "T": ("Temperature", "°C"),
        "T_in": ("Inlet temperature", "°C"),
        "T_out": ("Outlet temperature", "°C"),
        "Tin": ("Inlet temperature", "°C"),
        "T_wall, left": ("Wall temperature, left", "°C"),
        "T_wall, right": ("Wall temperature, right", "°C"),
        "T_wall_left": ("Wall temperature, left", "°C"),
        "T_wall_right": ("Wall temperature, right", "°C"),
        "q, left": ("Heat flux, left", "W/m$^2$"),
        "q, right": ("Heat flux, right", "W/m$^2$"),
        "h_left": ("Heat transfer coefficient, left", "W/m$^2$K"),
        "h_right": ("Heat transfer coefficient, right", "W/m$^2$K"),
        "static_pressure": ("Static pressure", "Pa"),
        "absolute_pressure": ("Absolute pressure", "Pa"),
        "pressure": ("Pressure drop", "Pa"),
        "static_pressure_drop": ("Static pressure drop", "Pa"),
        "mass_flow": ("Mass flow", "kg/s"),
        "velocity": ("Velocity", "m/s"),
        "Re": ("Reynolds number", "", 1.0, ".0f"),
        "Pe": ("Péclet number", "", 1.0, ".0f"),
        "Gr, left": ("Grashof number, left", ""),
        "Gr, right": ("Grashof number, right", ""),
        "power": ("Power", "W"),
        "reactivity": ("Reactivity", ""),
        "dPdt": ("Power rate", "W/s"),
        "ck": ("Precursor concentration", ""),
        "level": ("Level", "m"),
        "z": ("z", "m"),
        "x": ("x", "m"),
        "cell": ("Cell", ""),
        "quantity": "Quantity",
        "calculation": "Calculation",
        "band": "uncertainty",
        "band_sys": "systematic",
        "band_total": "systematic + statistical",
        "binding": "binding",
        "filled": "interpolated",
    }
)
