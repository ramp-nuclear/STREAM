r"""
Defining contact terms for Coolant-Clad interactions:
=====================================================
In general this module deals with the fact that at the transition from single
phase flow to a more complex subcooled (or compressed) boiling, heat flux which
in the single phase case contributes to a temperature rise, has other contributions
in the subcooled boiling phase: Latent heat, bubble forming (surface tension) among other
phenomena.

Some explanation regarding the jargon of this module:
-----------------------------------------------------
- SPL - Single Phase Liquid region.
- INC - Incipience Point, beyond which boiling occurs.
- SCB - Subcooled Boiling region.


In this module several functions relating to the calculation of the
heat transfer coefficient between coolant and walls are defined.

References
----------
.. [#BR] A. E. Rohsenow, W. M. Bergles, "The Determination of Forced Convection
        Surface Boiling Heat Transfer", Journal of Heat Transfer, 1964, p. 385
.. [#Collier] J. G. Thome, J. R. Collier, "Convective Boiling and Condensation",
    Oxford Science, 1994
"""
from typing import Protocol

import numpy as np

from stream.physical_models.dimensionless import Re_mdot
from stream.physical_models.heat_transfer_coefficient.laminar import Marco_Han_Nusselt, two_sided_heating_nusselt
from stream.physical_models.heat_transfer_coefficient.single_phase import (
    regime_dependent_h_spl,
    spl_htc,
    maximal_h_spl, SinglePhaseLiquidHTCExArgs,
)
from stream.physical_models.heat_transfer_coefficient.subcooled_boiling import (
    Bergles_Rohsenhow_SCB_heat_flux, McAdams_SCB_heat_flux, regime_dependent_q_scb,
    Bergles_Rohsenhow_partial_SCB, )
from stream.physical_models.heat_transfer_coefficient.temperatures import (
    film_temperature, Bergles_Rohsenow_T_ONB, wall_temperature
)
from stream.physical_models.heat_transfer_coefficient.turbulent import (
    Dittus_Boelter_h_spl,
    Dittus_Boelter)
from stream.substances import LiquidFuncs, Liquid
from stream.units import (Celsius, KgPerS, Meter, Meter2, Pascal, WPerM2, WPerM2K, Value)

__all__ = [
    "Bergles_Rohsenhow_partial_SCB",
    "Bergles_Rohsenhow_SCB_heat_flux",
    "Bergles_Rohsenow_T_ONB",
    "Dittus_Boelter",
    "Dittus_Boelter_h_spl",
    "film_temperature",
    "Marco_Han_Nusselt",
    "two_sided_heating_nusselt",
    "maximal_h_spl",
    "McAdams_SCB_heat_flux",
    "regime_dependent_h_spl",
    "regime_dependent_q_scb",
    "spl_htc",
    "wall_heat_transfer_coeff",
    "wall_temperature",
    "SinglePhaseLiquidHTCExArgs"
    ]


class SinglePhaseLiquidHTC(Protocol):
    def __call__(self, coolant: Liquid, mdot: KgPerS, Dh: Meter, A: Meter2,
                 T_cool: Celsius, T_wall: Celsius, coolant_funcs: LiquidFuncs,
                 pressure: Pascal
                 ) -> WPerM2K:
        """Single Phase Liquid Heat Transfer Coefficient function.

        Parameters
        ----------
        coolant: Liquid
            Coolant `film` properties. See in :func:`~.wall_heat_transfer_coeff`
        mdot: KgPerS
            Coolant mass flow
        Dh: Meter
            Hydraulic diameter
        A: Meter2
            Flow area
        T_cool: Celsius
            Coolant bulk temperature
        T_wall: Celsius
            Wall temperature
        coolant_funcs: LiquidFuncs
            Coolant properties functions.
        pressure: Pascal
            Static pressure of the liquid.

        Returns
        -------
        h: WPerM2K
            Heat transfer coefficient
        """


class SubCooledBoilingFluxFunction(Protocol):
    def __call__(self, T_wall: Celsius, coolant: Liquid, re: Value) -> WPerM2:
        """
        Parameters
        ----------
        T_wall: Celsius
            Wall temperatures
        coolant: Liquid
            Saturated Coolant properties
        re: Value
            Reynolds Number

        Returns
        -------
        q: WPerM2
            heat flux at the Subcooled Boiling (SCB) regime
        """
        ...


class FilmFunction(Protocol):
    def __call__(self, *, T_cool: Celsius, T_wall: Celsius) -> Celsius:
        return ...


class IncipienceTemperatureFunction(Protocol):
    def __call__(self, pressure: Pascal, q_spl: WPerM2, Tsat: Celsius) -> Celsius:
        return ...


class PartialSCBFactorFunction(Protocol):
    def __call__(self, q_spl: WPerM2, q_scb: WPerM2, q_scb_inc: WPerM2) -> Value:
        ...


def wall_heat_transfer_coeff(
        *,
        T_wall: Celsius,
        T_cool: Celsius,
        mdot: KgPerS,
        pressure: Pascal,
        coolant_funcs: LiquidFuncs,
        Dh: Meter,
        A: Meter2,
        h_spl: SinglePhaseLiquidHTCExArgs = Dittus_Boelter_h_spl,
        q_scb: SubCooledBoilingFluxFunction = Bergles_Rohsenhow_SCB_heat_flux,
        film: FilmFunction = film_temperature,
        incipience: IncipienceTemperatureFunction = Bergles_Rohsenow_T_ONB,
        partial_scb: PartialSCBFactorFunction = Bergles_Rohsenhow_partial_SCB,
        **kwargs,
        ) -> WPerM2K:
    r"""
    Computes the heat transfer coefficient according to [#BR]_.

    Parameters
    ----------
    T_wall: Celsius
        Wall temperature.
    T_cool: Celsius
        Coolant bulk temperature.
    mdot: KgPerS
        Mass current.
    pressure: Pascal
        Coolant absolute static pressure at each cell.
    coolant_funcs: LiquidFuncs
        Coolant properties functions.
    Dh: Meter
        The effective hydraulic diameter used for Re, Nu values.
    A: Meter2
        Cross sectional flow area.
    h_spl: SinglePhaseLiquidHTC
        The single phase liquid (SPL) heat transfer coefficient function, which is returned when there is no boiling.
    q_scb: SubCooledBoilingFluxFunction
        The subcooled boiling heat flux (SCB) function (which returns W/M2).
    partial_scb: PartialSCBFactorFunction
        An interpolation function between the SPL and SCB regimes, returning the factor by which q''_SPL is multiplied.
    film: FilmFunction
        Film temperature function, for which the coolant properties for heat transfer are calculated.
    incipience: IncipienceTemperatureFunction
        Boiling INC (incipience) temperature function.
    kwargs:
        Extra arguments for h_spl.

    Returns
    -------
    h: WPerM2K
        heat transfer coefficient of clad-coolant
    """
    T_film = film(T_cool=T_cool, T_wall=T_wall)
    cool = coolant_funcs.to_properties(T_film, pressure)
    T_sat = np.atleast_1d(cool.sat_temperature)
    kwargs.pop('coolant', None)
    h0 = h_spl(coolant=cool, mdot=mdot, Dh=Dh, A=A, T_cool=T_cool, T_wall=T_wall,
               coolant_funcs=coolant_funcs, **kwargs)
    q_spl: WPerM2 = h0 * np.abs(T_wall - T_cool)

    T_wall_inc = incipience(pressure, q_spl, T_sat)

    boiling = (T_wall > T_wall_inc) & (q_spl != 0)
    if np.any(boiling):
        T_cool, T_sat, pressure, T_wall, T_wall_inc, h0, q_spl = map(
            np.atleast_1d, [T_cool, T_sat, pressure, T_wall, T_wall_inc, h0, q_spl])
        re = Re_mdot(mdot, A, Dh, coolant_funcs.viscosity(T_cool[boiling]))
        sat_cool = coolant_funcs.to_properties(T_sat[boiling], pressure[boiling])
        q_scb_wall = q_scb(T_wall[boiling], sat_cool, re=re)
        q_scb_inc = q_scb(T_wall_inc[boiling], sat_cool, re=re)

        h0[boiling] *= partial_scb(q_spl[boiling], q_scb_wall, q_scb_inc)
    return h0
