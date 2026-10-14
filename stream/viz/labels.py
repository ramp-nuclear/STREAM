"""Display names, units and scales for everything the figures print."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass


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
    color:
        Optional fixed colour for this name when it is a legend level.
    """

    display: str | None
    unit: str = ""
    scale: float = 1.0
    color: str | None = None


def as_label(name: str, spec) -> Label:
    """Turn a registry entry into a :class:`Label`.

    Accepted forms: a ``Label``; ``None`` (hidden); a string (display only);
    a 2-tuple ``(display, unit)``; a 3-tuple ``(display, unit, scale)``.
    """
    if spec is None:
        return Label(None)
    if isinstance(spec, Label):
        return spec
    if isinstance(spec, str):
        return Label(spec)
    if isinstance(spec, tuple) and 2 <= len(spec) <= 3 and all(isinstance(s, str) for s in spec[:2]):
        return Label(*spec)
    raise TypeError(
        f"label for {name!r} must be a string, a (display, unit) tuple, a (display, unit, scale) "
        f"tuple, a Label or None, not {spec!r}"
    )


class Labels(Mapping[str, Label]):
    """A registry from internal names to :class:`Label` records.

    Lookups never fail: a name with no entry displays as itself with no unit.
    Later registries win in :meth:`merged`.
    """

    def __init__(self, entries: Mapping[str, object] | None = None):
        self._entries: dict[str, Label] = {k: as_label(k, v) for k, v in (entries or {}).items()}

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
        """A new registry with ``other``'s entries overriding this one's."""
        merged = Labels()
        merged._entries = dict(self._entries)
        if other is not None:
            merged._entries.update(Labels(other)._entries if not isinstance(other, Labels) else other._entries)
        return merged

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

    def value(self, name: str, x: float) -> str:
        """``x`` scaled and formatted for a legend entry."""
        return format(x * self[name].scale, ".4g")

    def describe(self, fixed: Mapping[str, float]) -> str:
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
        "Re": ("Reynolds number", ""),
        "Pe": ("Péclet number", ""),
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
    }
)
