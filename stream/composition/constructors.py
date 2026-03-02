r"""Various alternative constructors for some calculations"""

from typing import Literal, Protocol, Sequence

import numpy as np

from stream import Calculation, unpacked
from stream.calculations import Friction, Pump, Resistor
from stream.calculations.ideal.resistors import DPCalculation
from stream.units import Array1D, Functional, KgPerS, Pascal, Place

__all__ = ["Calculation_factory", "ResistorFromKnownPoint"]


def ResistorFromKnownPoint(
    dp: Pascal = None,
    mdot: KgPerS = None,
    behavior: Literal["constant", "linear", "parabolic"] = "parabolic",
    name: str = None,
    **kwargs,
) -> DPCalculation:
    r"""Construct a resistor to flow from a known point in the :math:`(\Delta p, \dot{m})` phase space.

    Parameters
    ----------
    dp: Pascal or None
        Known pressure difference
    mdot: KgPerS or None
        Known mass current
    behavior: Literal["constant", "linear", "parabolic"]
        Which :math:`\Delta p (\dot{m})` dependence should the resistor have.
    name: str or None
        Optional name, see :ref:`Calculation`

    Returns
    -------
    DPCalculation
        A resistor guaranteed to be working at the desired phase point

    Examples
    --------
    >>> r = ResistorFromKnownPoint(-1.0, 1.0, "linear")
    >>> r.dp_out(Tin=25., mdot=3.0)
    -3.0
    >>> r = ResistorFromKnownPoint(-1.0, behavior="constant")
    >>> r.dp_out(Tin=25., mdot=3.0)
    0.0
    """

    assert dp is not None or mdot is not None, "At least one of (dp, mdot) must be specified"

    if behavior == "constant":
        return Pump(pressure=dp, mdot0=mdot, name=name)

    assert dp is not None and mdot is not None, "For non-ideal current or head sources, dp must be specified"
    if behavior == "linear":
        return Resistor(resistance=-dp / mdot, name=name)
    if behavior == "parabolic":
        assert dp <= 0, "A resistor with a parabolic dp(mdot) dependence can not be positive"
        rho0 = kwargs["fluid"].density(kwargs.pop("Tin"))
        return Friction(
            f=2 * abs(dp) * rho0 / (mdot**2),
            length=1.0,
            hydraulic_diameter=1.0,
            area=1.0,
            name=name,
            **kwargs,
        )
    raise ValueError(f"Behavior type for point based resistor unknown: {behavior}")


class _CalcFactory(Protocol):
    def __call__(self, name: str | None = None) -> Calculation: ...


def Calculation_factory(
    calculate: Functional,
    mass_vector: Sequence[bool],
    variables: dict[str, Place],
    unpack: bool = True,
) -> _CalcFactory:
    r"""A simple :class:`~stream.calculation.Calculation` factory.

    This factory does not support non-default methods `load`, `save`, `should_continue`.

    Parameters
    ----------
    calculate: Functional
        See :meth:`~stream.calculation.Calculation.calculate`
    mass_vector: Sequence[bool]
        See :meth:`~stream.calculation.Calculation.mass_vector`
    variables: dict[str, Place]
        See :meth:`~stream.calculation.Calculation.variables`
    unpack: bool or None
        Whether :func:`~stream.calculation.unpacked` should be applied to `calculate`.
        Default is ``True``.

    Returns
    -------
    class: Callable
        A Calculation class
    """
    length = len(mass_vector)

    class Constructed(Calculation):
        def __init__(self, name=None):
            self.name = name
            self.calculate = unpacked(calculate) if unpack else calculate

        def calculate(self, y: Sequence[float], **kwargs) -> Array1D:
            pass

        @property
        def mass_vector(self) -> Sequence[bool]:
            return np.asarray(mass_vector, dtype=bool)

        def __len__(self) -> int:
            return length

        @property
        def variables(self) -> dict[str, Place]:
            return variables

    return Constructed
