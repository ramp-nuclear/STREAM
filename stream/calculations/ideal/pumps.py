from typing import Sequence

import numpy as np

from stream.calculation import unpacked
from stream.calculations.ideal.ideal import LumpedComponent
from stream.units import KgPerS, Pascal, Celsius, Array1D


__all__ = ["Pump"]

from stream.utilities import directed_Tin


class Pump(LumpedComponent):
    r"""Hydraulic component constraining :math:`\Delta p` or :math:`\dot{m}`.
    It does not affect temperature values, merely propagates it.

    If the chosen constraint is mass flow rate (:math:`\dot{m}=\dot{m}_0`),
    then the pressure difference cannot be constrained, and vice versa.
    One can provide externally a time-dependent value of pressure difference
    or mass flow rate.
    """

    def __init__(
        self,
        pressure: Pascal | None = None,
        mdot0: KgPerS | None = None,
        name: str = "Pump",
    ):
        r"""
        Parameters
        ----------
        pressure: Pascal or None
            Optional Static desired :math:`\Delta p`
        mdot0: KgPerS or None
             Optional Static desired :math:`\dot{m}`
        name: str or None
            Optional name, see :ref:`Calculation`

        Raises
        ------
        ValueError
            If both ``pressure`` and ``mdot0`` are not None.
        """
        if pressure is not None and mdot0 is not None:
            raise ValueError(
                f"One cannot impose both pressure and current, in this case you tried {pressure=}, {mdot0=}"
            )
        self.name = name
        self.p = pressure
        self.mdot0 = mdot0

    @unpacked
    def calculate(
        self,
        variables: Sequence[float],
        *,
        mdot: KgPerS,
        Tin: Celsius,
        Tin_minus: Celsius | None = None,
        pressure: Pascal | None = None,
        mdot0: KgPerS | None = None,
        **_,
    ) -> Array1D:
        if pressure is mdot0 is self.p is self.mdot0 is None:
            raise ValueError("One has to impose either the pressure or the flow in a pump")
        out = np.empty(2)
        out[0] = variables[0] - self.T_out(Tin=directed_Tin(Tin, Tin_minus, mdot), mdot=mdot)
        if pressure is not None or self.p is not None:
            out[1] = variables[1] - (self.p if pressure is None else pressure)
        else:
            out[1] = mdot - (self.mdot0 if mdot0 is None else mdot0)
        return out
