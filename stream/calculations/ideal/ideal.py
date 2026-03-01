r"""
The :class:`~stream.calculations.ideal.ideal.LumpedComponent` is used as
a base class for methods and properties which many other ``Calculation``\ s use.
The overarching concept is that of a Lumped object in an idealized Kirchoff model
for flow, with an analogy to electronic circuits.
Most Primary Circuit ``Calculation``\ s just deal with changes in pressure and outlet
temperature, and these often share much of their implementation.
Some calculations may benefit from methods or properties of the LumpedComponent,
and if they are not just LumpedComponents, they should use composition rather
than inheritance, as done in :class:`~stream.calculations.flapper.Flapper`.

If you want to represent a 0-dimensional, no inner-working ``Calculation``, which
only serves to have a pressure drop or a temperature change, you may find this
namespace to be useful.
"""

import logging
from abc import ABCMeta
from typing import Sequence

import numpy as np

from stream import unpacked, Calculation
from stream.units import Array1D, Celsius, KgPerS, Name, Place, Pascal
from stream.utilities import directed_Tin


__all__ = ["LumpedComponent"]

logger = logging.getLogger(__name__)


class LumpedComponent(Calculation, metaclass=ABCMeta):
    r"""0-dimensional Kirchhoff-oriented flow components

    An ideal simple component would only need two variables:

    1. Incoming temperature (Tin)
    2. Incoming mass current (mdot)

    These calculations compute only the outlet temperature and the pressure difference
    on them. The equations are assumed to be algebraic.

    """

    # noinspection PyPep8Naming
    @unpacked
    def calculate(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        **kwargs,
    ) -> Array1D:
        r"""Compute the algebraic residual values for this Calculation.

        Parameters
        ----------
        variables: Sequence[float]
            Input variables, specifically [Tin, dp]
        Tin: Celsius
            Inlet Temperature.
        Tin_minus: Celsius or None
            Inlet Temperature in case the flow is reversed.
        mdot: KgPerS
            Mass flow rate.
        kwargs: dict
            Other input. It takes precedence over the following
            ``dict(dp=dp, Tin=Tin, mdot=mdot)`` when merged, which is then
            input for ``self.T_out`` and ``self.dp_out``.

        Returns
        -------
        errors: Array1D
            The error in ``variables``
        """
        inputs = dict(dp=variables[1], Tin=directed_Tin(Tin, Tin_minus, mdot), mdot=mdot) | kwargs
        out = np.empty(2)
        out[0] = variables[0] - self.T_out(**inputs)
        out[1] = variables[1] - self.dp_out(**inputs)
        return out

    def indices(self, variable: Name, asking=None) -> int:
        return dict(Tin=0, Tin_minus=0, pressure=1)[variable]

    indices.__doc__ = Calculation.indices.__doc__

    @property
    def mass_vector(self) -> tuple[bool, bool]:
        """A mass vector, which is just two False values."""
        return False, False

    @property
    def variables(self) -> dict[Name, Place]:
        return dict(Tin=0, pressure=1)

    variables.__doc__ = Calculation.variables.__doc__

    def __len__(self) -> int:
        return 2

    # noinspection PyPep8Naming
    def T_out(self, *, Tin: Celsius, **_) -> Celsius:
        """Outlet temperature calculation.

        Parameters
        ----------
        Tin: Celsius
            Inlet Temperature

        """
        return Tin

    def dp_out(self, *, mdot: KgPerS, Tin: Celsius, **_) -> Pascal:
        """Pressure drop calculation.

        Parameters
        ----------
        mdot: KgPerS
            Mass flow rate
        Tin: Celsius
            Inlet temperature.

        Returns
        -------
        Pascal

        """
        return 0.0
