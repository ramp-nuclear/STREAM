r"""
Consider the case of incompressible (thus barotropic), inviscid, flow. Along a
streamline or any two points in irrotational flow, the Bernoulli integral
equation is for some body force potential :math:`\psi`:

.. math::
    \int_1^2 \frac{\partial \vec{v}}{\partial t} \cdot d\vec{r}
    + \int_1^2 \frac{dp}{\rho} + \Delta\left(\psi + \frac{v^2}{2}\right) = 0

One may write the first term as follows:

.. math::
    \int_1^2 \frac{\partial \vec{v}}{\partial t} \cdot d\vec{r}
    = \frac{1}{\rho}\frac{d\dot{m}}{dt}\sum_{n=1}^2\frac{l_n}{A_n}

Which is the inertia term, and may be written with equivalent inertia 1/length
:math:`(l/A)_{Total}`.

References
----------
.. [#Todreas] Nuclear Systems Volume I, Thermal Hydraulic Fundamentals, Todreas & Kazimi, 2nd edition.
"""

from typing import Callable

from stream.calculations.ideal.ideal import LumpedComponent
from stream.units import PerM, KgPerS, KgPerS2, Pascal


__all__ = ["bilinear", "Inertia"]


Inertia_Coefficient = PerM | Callable[[...], PerM]


class Inertia(LumpedComponent):
    r"""Flow inertia. Mathematically speaking, it is equivalent to an electrical inductor.

    The equation represented here is really:

    .. math:: \Delta p = L \frac{d\dot{m}}{dt}

    Where :math:`L` is the inertia.
    For more information, please see [#Todreas]_"""

    def __init__(self, inertia: PerM, name="Inertia"):
        r"""
        Parameters
        ----------
        inertia: PerM | Callable[[...], PerM]
            Moment of inertia. In terms of "geometrical" inertia, this may be viewed
            as :math:`(l/A)_T`.

        Notes
        -----
        - :math:`A` is the equivalent cross-sectional flow area of the system.
        - :math:`l` is the equivalent flow length of the system.
        """
        self.name = name
        self._L = inertia

    def _coefficient(self, **kwargs):
        return self._L(**kwargs) if callable(self._L) else self._L

    def dp_out(self, *, mdot2: KgPerS2, **kwargs) -> Pascal:
        r"""The following relation is implemented:

        .. math:: \Delta p = L \frac{d\dot{m}}{dt}
        """
        return -self._coefficient(**kwargs) * mdot2


def bilinear(L0: PerM, mdot0: KgPerS):
    r"""Creates a bi-linear inertia function, to be used in :class:`Inertia`.

    .. math::
        L=
        \begin{cases}
            L_0 (\dot{m}/\dot{m}_0) & \text{if $\dot{m} < \dot{m}_0$} \\
            L_0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    L0 : PerM
        Inertia constant
    mdot0 : KgPerS
        The bi-linear `knee` current, under which the inertia is linearly decreasing.

    Returns
    -------
    bi : Callable[[KgPerS, ...], PerM]
        The inertia function :math:`L(\dot{m})`
    """

    def bi_(mdot, **kwargs):
        return (mdot / mdot0) * L0 if mdot < mdot0 else L0

    return bi_
