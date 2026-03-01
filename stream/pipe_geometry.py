r"""
Pipe Geometry
=============
A pipe geometry interface used by different calculations, defined through
:class:`EffectivePipe`.
"""

from dataclasses import dataclass, field
from typing import TypeVar, Literal

import numpy as np

from stream.units import Meter, Meter2


__all__ = ["EffectivePipe"]

_T = TypeVar("_T", bound="EffectivePipe")


@dataclass(frozen=True, slots=True)
class EffectivePipe:
    # noinspection PyUnresolvedReferences
    r"""
    A simple constant-cross-sectional pipe's parameters, where the parameters
    are "effectively" fitting to a circular pipe, and there is a distinction
    between "wet" (for fluid flow) and "heated" (for heat transfer) geometries.

    Essentially, for both perimeters, :math:`P_\text{heated}, P_\text{wet}`,
    the appropriate diameter is given using the flow area :math:`A`:

    .. math:: D = 4A / P

    Parameters
    ----------
    length: Meter
        Pipe Length
    heated_perimeter: Meter
        The cross-sectional heated perimeter
    wet_perimeter: Meter
        The cross-sectional wet (hydraulic) perimeter
    area: Meter2
        Flow cross-section
    heated_parts: tuple[Meter, ...] or None
        For a cross-section of the pipe, there may be several distinct regions
        which contribute flux (e.g. in MTR there are 2 in most cases).
        This is also the default case.
    width: Meter | None
        Channel width. When this is None, that means one should not use this attribute.
        This property is currently only used in the Sudo & Kaminaga CHF for one
        of its terms.
        See that usage in :func:`~stream.physical_models.thresholds.Sudo_Kaminaga_CHF`.
        For rectangular channels this is the longer side of the channel, which is commonly also the heated one.
        In annular channels, this is the hydraulic diameter of the channel.
        The reference for why this is so is by [#Mishima]_.
    depth: Meter | None
        Channel depth. This property is in use in the Elenbaas correlation
        for natural convection between parallel plates
        `stream.physical_models.heat_transfer_coeffcient.natural_convection.Elenbaas_h_spl`
        For rectangular channels this is the shorter side of the channel cross-section.

    References
    ----------
    .. [#Mishima] K. Mishima, H. Nishihara, T. Shibata, "CHF Correlations Related
       to the Core Cooling of a Research Reactor", JAERI-M 84-073, International
       Meeting on Reduced Enrichment for Research and Test Reactors, Tokai, Japan, 1983.
    """

    length: Meter
    heated_perimeter: Meter
    wet_perimeter: Meter
    area: Meter2
    heated_parts: tuple[Meter, ...] = ()
    width: Meter | None = None
    depth: Meter | None = None
    heated_diameter: Meter = field(init=False)
    hydraulic_diameter: Meter = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "heated_diameter", 4 * self.area / self.heated_perimeter)
        object.__setattr__(self, "hydraulic_diameter", 4 * self.area / self.wet_perimeter)
        object.__setattr__(
            self,
            "heated_parts",
            (self.heated_parts if self.heated_parts else (self.heated_perimeter / 2, self.heated_perimeter / 2)),
        )
        assert np.isclose(np.sum(self.heated_parts), self.heated_perimeter, rtol=1e-15), (
            f"The partitions of P_heated: {self.heated_parts} do not sum up to the total within relative tolerance 1e-15"
        )

    @classmethod
    def rectangular(
        cls,
        length: Meter,
        edge1: Meter,
        edge2: Meter,
        heated_edge: Meter,
        one_sided: Literal["left", "right", None] = None,
    ) -> "EffectivePipe":
        """Create an EffectivePipe geometry for a channel with a rectangular cross-section,
        which is heated on two opposite sides.

        Parameters
        ----------
        length: Meter
            Length along the flow.
        edge1, edge2: Meter
            Edges of the rectangular cross-section.
        heated_edge: Meter
            The heated section length which may differ from the edge itself.
        one_sided: bool
            A flag for whether heating happens on two opposite edges (False) or on just one edge (True).

        Returns
        -------
        EffectivePipe
        """
        parts = {
            "left": (heated_edge, 0.0),
            "right": (0.0, heated_edge),
            None: (heated_edge, heated_edge),
        }[one_sided]
        width = max(edge1, edge2)
        depth = min((edge1, edge2))
        return cls(
            length=length,
            heated_perimeter=heated_edge if one_sided else 2 * heated_edge,
            wet_perimeter=2 * (edge1 + edge2),
            area=edge1 * edge2,
            heated_parts=parts,
            width=width,
            depth=depth,
        )

    @classmethod
    def circular(cls, length: Meter, diameter: Meter):
        """Create an EffectivePipe geometry for a channel with a circular cross-section, which is fully heated.

        Parameters
        ----------
        length: Meter
            Length along the flow.
        diameter: Meter
            Diameter of the pipe.

        Returns
        -------
        EffectivePipe
        """
        perimeter = diameter * np.pi
        return cls(
            length=length,
            heated_perimeter=perimeter,
            wet_perimeter=perimeter,
            area=np.pi * diameter**2 / 4,
            heated_parts=(perimeter, 0.0),
            width=diameter,
        )
