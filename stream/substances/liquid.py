r"""
Liquid Properties and Interface
===============================

Two dataclasses are defined, which serve as an interface to the rest of the code whenever liquid properties such as
coolant density, latent heat etc.

These classes are :class:`LiquidFuncs`, which is a template for adding such properties as functions of the
:math:`(T, p)`, phase space. Applying these functions for such a phase space point yields a :class:`Liquid` object,
containing the same properties, actualized.
"""

from dataclasses import dataclass
from typing import Callable

from scipy.constants import atm

from stream.units import (
    Celsius,
    JPerKg,
    JPerKgK,
    KgPerM3,
    NPerM,
    PaS,
    Pascal,
    PerC,
    WPerMK,
)

__all__ = ["Liquid", "LiquidFuncs"]


@dataclass(frozen=True, slots=True)
class Liquid:
    """
    Bulk properties of a liquid.

    Parameters
    ----------
    density: KgPerM3
        Liquid density
    specific_heat: JPerKgK
        Liquid specific heat
    viscosity: PaS
        Liquid viscosity
    conductivity: WPerMK
        Liquid thermal conductivity
    vapor_density: KgPerM3
        Vapor density
    sat_temperature: Celsius
        Saturation (Boiling) temperature
    surface_tension: NPerM
        Liquid surface tension
    latent_heat: JPerKg
        Liquid-Vapor transition specific latent heat
    thermal_expansion: PerC
        Thermal volume expansion coefficient
    """

    density: KgPerM3
    vapor_density: KgPerM3
    specific_heat: JPerKgK
    viscosity: PaS
    sat_temperature: Celsius
    surface_tension: NPerM
    latent_heat: JPerKg
    conductivity: WPerMK
    thermal_expansion: PerC

    def __getitem__(self, item) -> "Liquid":
        return Liquid(*(getattr(self, attr)[item] for attr in self.__slots__))


@dataclass(frozen=True, repr=False)
class LiquidFuncs:
    """
    A dataclass containing functions of temperature and pressure depicting several properties of a liquid


    Parameters
    ----------
    density: Callable[[Celsius], KgPerM3]
        Liquid density
    specific_heat: Callable[[Celsius], JPerKgK]
        Liquid specific heat
    viscosity: Callable[[Celsius], PaS]
        Liquid viscosity
    conductivity: Callable[[Celsius], WPerMK]
        Liquid thermal conductivity
    vapor_density: Callable[[Celsius], KgPerM3]
        Vapor density
    sat_temperature: Callable[[Pascal], Celsius]
        Saturation (Boiling) temperature. Note the pressure dependence
    surface_tension: Callable[[Celsius], NPerM]
        Liquid surface tension
    latent_heat: Callable[[Celsius], JPerKg]
        Liquid-Vapor transition specific latent heat
    thermal_expansion: Callable[[Celsius], PerC]
        Thermal volume expansion coefficient
    """

    density: Callable[[Celsius], KgPerM3]
    vapor_density: Callable[[Celsius], KgPerM3]
    specific_heat: Callable[[Celsius], JPerKgK]
    viscosity: Callable[[Celsius], PaS]
    sat_temperature: Callable[[Pascal], Celsius]
    surface_tension: Callable[[Celsius], NPerM]
    latent_heat: Callable[[Celsius], JPerKg]
    conductivity: Callable[[Celsius], WPerMK]
    thermal_expansion: Callable[[Celsius], PerC]

    def to_properties(self, T: Celsius, p: Pascal = atm) -> Liquid:
        """
        Apply T, p once to receive every property of the liquid at the
        given phase space point.

        Parameters
        ----------
        T: Celsius
            Temperature
        p: Pascal
            Pressure. Specifically used only for the saturation temperature,
            so it is optional and 1 Atmosphere (101,325 Pa) by default.

        Returns
        -------
        liquid: Liquid
            A Liquid Dataclass containing the aforementioned properties.
        """
        return Liquid(
            density=self.density(T),
            vapor_density=self.vapor_density(T),
            specific_heat=self.specific_heat(T),
            viscosity=self.viscosity(T),
            sat_temperature=self.sat_temperature(p),
            surface_tension=self.surface_tension(T),
            latent_heat=self.latent_heat(T),
            conductivity=self.conductivity(T),
            thermal_expansion=self.thermal_expansion(T),
        )
