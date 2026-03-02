"""
Reactor Safety Thermohydraulic Thresholds
=========================================
Ensuring the reactor's safety is top priority.
Thus, certain situations are to be avoided during normal operation and accidents.
In the case of thermohydraulics, several conditions are telltale signs of incoming doom.
Within this module different TH thresholds are contained, allowing for post-run analysis of their appearance:

.. list-table::
    :name: TH limits
    :header-rows: 1
    :widths: 30, 50

    * - Condition / Threshold
      - Implemented Correlations
    * - **ONB** - Onset of Nucleate Boiling
      - :func:`Bergles Rosenhow <~stream.physical_models.Bergles_Rohsenow_T_ONB>` [#BR]_
    * - **OFI** - Onset of Flow Instability
      - :func:`Whittle Forgan <Whittle_Forgan_OFI>` [#WF]_
    * - **OSV** - Onset of Significant Void
      - :func:`Saha Zuber <Saha_Zuber_OSV_computed_bulk>` [#SahaZuber]_,
        :func:`Saha Zuber (Deprecated) <Saha_Zuber_OSV>` [#SahaZuber]_,
    * - **CHF** - Critical Heat Flux
      - :func:`Sudo Kaminaga <Sudo_Kaminaga_CHF>` [#Sudo]_,
        :func:`Mirshak <Mirshak_CHF>` [#Mirshak]_,
        :func:`Fabrega <Fabrega_CHF>` [#Fabrega]_,
    * - **BP** - Boiling Power
      - :func:`Boiling Power <boiling_power>` [#BPR]_

References
----------
.. [#SahaZuber] Convective Boiling and Condensation (3rd edition),
    Collier and Thome, p. 226-227.
.. [#BR] A. E. Rohsenow, W. M. Bergles, "The Determination of Forced Convection
    Surface Boiling Heat Transfer", Journal of Heat Transfer, vol. C86, p. 385,
    1964.
.. [#WF] R. H. Whittle and R. Forgan, "A correlation for the minima in the
    pressure drop versus flow-rate curves for sub-cooled water flowing in the
    narrow heated channels," Nuclear Engineering and Design, vol. 6, pp. 89-99,
    1967.
.. [#Mirshak] S. Mirshak, W. S. Durant and R. H. Towell,
    "Heat Flux at Burnout". E. I. du Pont de Nemours & Co., Aiken,
    South Carolina, 1959.
.. [#Fabrega] S. Fabréga, "Thermal Calculations for Water Cooled Research
    Reactors", (translated from French by J. Rodd), AAEC, Orig, Grenoble,
    France, 1971.
.. [#FabregaFrench] S. Fabréga, "Le Calcul Thermique Des Reacteurs De Recherche
    Refroidis Par Eau", Centre d'Estudes Nucléaires de Grenoble, CEA-R-4114, 1971.
.. [#Sudo] M. Kaminaga, K. Yamamoto, Y. Sudo, "Improvement of Critical Heat
    Flux Correlation for Research Reactors using Plate-Type Fuel", Journal of
    Nuclear Science and technology, vol. 35, no. 12, pp. 943-951, 1998.
.. [#BPR] CONVEC v 3.40, section 7.2.4, page 20, Revision II
"""

from typing import Callable

import numpy as np
from scipy.integrate import quad

from stream.physical_models.dimensionless import Pe
from stream.pipe_geometry import EffectivePipe
from stream.substances import Liquid
from stream.units import (
    Celsius,
    JPerKgK,
    KgPerS,
    Meter,
    Meter2,
    MPerS,
    MPerS2,
    Pascal,
    Value,
    Watt,
    WPerM2,
)
from stream.units import (
    g as gravity,
)
from stream.utilities import directed

__all__ = [
    "Saha_Zuber_OSV",
    "Saha_Zuber_OSV_computed_bulk",
    "Fabrega_CHF",
    "Mirshak_CHF",
    "Sudo_Kaminaga_CHF",
    "Whittle_Forgan_OFI",
    "boiling_power",
]


def Saha_Zuber_OSV(T_bulk: Celsius, coolant: Liquid, u: MPerS, Dh: Meter) -> WPerM2:
    r"""
    Calculates the minimal q'' heat flux at which OSV occurs, according to Saha & Zuber (1974) [#SahaZuber]_:

    for :math:`\text{Pe} < 70,000`:

    .. math::
        T_\text{sat} - T_\text{bulk}
        = 0.0022\frac{q''_\text{OSV}D_h}{k_\text{bulk}}

    for :math:`\text{Pe} > 70,000`:

    .. math::
        T_\text{sat} - T_\text{bulk}
        = 153.8 \frac{q''_\text{OSV}}{c_{p, \text{bulk}}G}

    .. danger::
        This method assumes that the coolant temperature does not change when the flux changes.
        This is not what people usually mean when they use OSV.
        See :func:`Saha_Zuber_OSV_computed_bulk` for what you should probably use.
        This function is deprecated as the main usage and may change its name to reflect that in the future.

    Parameters
    ----------
    T_bulk: Celsius
        Coolant bulk temperature.
    coolant: Liquid
        Coolant properties at T_bulk and appropriate pressure.
    u: MPerS
        Coolant flow velocity.
    Dh: Meter
        Hydraulic diameter

    Returns
    -------
    q'': WPerM2
        OSV (Onset of Significant Void) heat flux.

    """
    dT = coolant.sat_temperature - T_bulk
    pe = Pe(
        rho=(rho := coolant.density),
        v=u,
        L=Dh,
        cp=(cp := coolant.specific_heat),
        k=(k := coolant.conductivity),
    )
    G = rho * np.abs(u)
    Nu_c = 455
    St_c = 0.0065
    return dT * (k / Dh * Nu_c * (pe < 7e4) + cp * G * St_c * (pe > 7e4))


def Saha_Zuber_OSV_computed_bulk(
    T_inlet: Celsius,
    coolant: Liquid,
    mdot: KgPerS,
    Dh: Meter,
    area: Meter2,
    heated_perimeter: Meter,
    flux_shape: Value,
    dz: Meter,
    flux_enworse: float = 1.0,
) -> WPerM2:
    r"""
    Calculates the minimal q'' heat flux at which OSV occurs, according to
    Saha & Zuber (1974) [#SahaZuber]_ (see :func:`Saha_Zuber_OSV`), but the bulk temperature is
    computed as though the flux through the channel is indeed :math:`q''_{OSV}`.

    Saha & Zuber's correlation may be written as:

    .. math:: T_\text{sat} - T_\text{bulk} =  \frac{q''_{OSV}}{X}

    Where :math:`X` is, for :math:`\text{Pe} < 70,000`:

    .. math::
        X = \frac{\text{Nu}_c}{\text{Pe}}Gc_p = \frac{k}{D_h}\text{Nu}_c

    for :math:`\text{Pe} > 70,000`:

    .. math::
        X = \text{St}_c G c_p

    and :math:`\text{St}_c = 0.0065, \text{Nu}_c = 455`.

    Assuming :math:`q''(z)_{OSV} = \alpha q''(z)`, and the flux is given for the
    entire ``heated_perimeter`` (:math:`H_p`):

    .. math:: T_\text{bulk} = T_{in} + \frac{H_p}{\dot{m}c_p}\alpha \int^z q'' dz'

    meaning:

    .. math::
     \alpha = \frac{q''_{OSV}}{q''}
     = \frac{T_\text{sat} - T_{in}}{\frac{q''}{X} + \frac{H_p}{\dot{m}c_p}\int^z q'' dz'}

    and finally, :math:`q''_{OSV}` is independent of the normalization of :math:`q''`:

    .. math:: q''_{OSV}
     = \frac{X(T_\text{sat} - T_{in})}{1 +
     \frac{XH_p}{\dot{m}c_p}\frac{\int^z q'' dz'}{q''(z)}}

    Parameters
    ----------
    T_inlet: Celsius
        Coolant bulk temperature.
    coolant: Liquid
        Coolant properties at T_bulk and appropriate pressure.
    mdot: KgPerS
        Mass flow rate
    Dh: Meter
        Hydraulic diameter
    area: Meter2
        Flow area
    heated_perimeter: Meter
        Heated perimeter
    flux_shape: Value
        A one dimensional heat flux distribution, which can be freely normalized.
    dz: Meter
        Axial cell lengths
    flux_enworse: float
        Factor to make the flux worse by. This is for local flux disturbance effects like fuel homogeneity.


    Returns
    -------
    q'': WPerM2
        OSV (Onset of Significant Void) heat flux.

    """
    dT = directed(coolant.sat_temperature - T_inlet, mdot)
    rho = coolant.density
    G = np.abs(mdot / area)
    u = G / rho
    pe = Pe(
        rho=rho,
        v=u,
        L=Dh,
        cp=(cp := coolant.specific_heat),
        k=(k := coolant.conductivity),
    )
    Nu_c = 455
    St_c = 0.0065

    coefficient = directed(k / Dh * Nu_c * (pe <= 7e4) + cp * G * St_c * (pe > 7e4), mdot)
    cumulative = np.cumsum(directed(flux_shape * dz, mdot))
    power_factor = directed(heated_perimeter / (np.abs(mdot) * cp), mdot)
    shape_factor = cumulative / directed(flux_shape * flux_enworse, mdot)
    denominator = 1.0 + coefficient * power_factor * shape_factor
    return directed(coefficient * dT / denominator, mdot)


def boiling_power(
    mdot: KgPerS,
    T_sat: Celsius,
    Tin: Celsius,
    cp_in: JPerKgK,
) -> Watt:
    r"""The limit for "Boiling power ratio" as per TERMIC(CONVEC) [#BPR]_ - ratio of the channel power leading to
    saturated water temperature at the channel outlet to the current channel power:

    .. math::
        BP = \dot{m} c_\text{p}(T_\text{sat} - T_\text{inlet})

    Parameters
    ----------
    mdot: KgPerS
        Mass flow rate
    Tin: Celsius
        Bulk coolant temperature @ inlet
    cp_in: JPerKgK
        Specific heat of the coolant at Tin
    T_sat: Celsius
        Saturation temperature

    Returns
    -------
    Watt
        Power required to reach the saturation temperature in the channel.

    """
    return np.abs(mdot) * cp_in * (T_sat - Tin)


def Whittle_Forgan_OFI(
    mdot: KgPerS,
    sat_temperature: Pascal,
    inlet_temperature: Celsius,
    pipe: EffectivePipe,
    cp: Callable[[Celsius], JPerKgK],
) -> Watt:
    r"""The limit for "Onset of Flow Instability" as per Whittle and Forgan [#WF]_,
    but with a correction by Fabréga [#FabregaFrench]_.
    In essence, power for OFI is given by:

    .. math::
        P_\text{RD} = \frac{\dot{m} c_\text{p}(T_\text{sat} - T_\text{inlet})}
        {1 + 3.15 (D_\text{h} / L_\text{h}) (1.08G)^{0.29}}

    Where :math:`G= \rho u = \dot{m}/A` is the mass flow flux, given in CGS.
    The reference says so: "G: vitesse massique à l'entrée et évaluée en CGS"

    Parameters
    ----------
    mdot: KgPerS
        Mass flow rate
    sat_temperature: Celsius
        Saturation temperature at the outlet.
    inlet_temperature: Celsius
        Bulk temperature at the fuel inlet.
    pipe: EffectivePipe
        The geometry of the flow region.
    cp: Callable[[Celsius], JPerKgK]
        Strategy for computing the specific heat of the coolant.

    Returns
    -------
    PRD_OFI: Watt
        OFI limit power

    """
    G = np.abs(mdot) / pipe.area
    G /= 10  # G must be in CGS, sadly.
    cp_int = quad(cp, inlet_temperature, sat_temperature)[0]  # type: ignore
    return np.abs(mdot) * cp_int / (1.0 + 3.15 * (pipe.hydraulic_diameter / pipe.length) * (1.08 * G) ** 0.29)


def Sudo_Kaminaga_CHF(
    T_bulk: Celsius,
    sat_coolant: Liquid,
    mdot: KgPerS,
    pipe: EffectivePipe,
    g: MPerS2 = gravity,
) -> WPerM2:
    r"""The limit heat flux for CHF according to Kaminaga et al. [#Sudo]_.

    Parameters
    ----------
    T_bulk: Celsius
        Bulk coolant temperature
    sat_coolant: Liquid
        Coolant properties at saturation temperature
    mdot: KgPerS
        Coolant mass flow rate
    pipe: EffectivePipe
        Channel Geometry
    g: MPerS2
        Gravitational acceleration constant

    Notes
    -----
    The width of the channel is taken from the pipe from an attribute called `width`.
    This attribute is only ever really used here, and it is geometrically set for most channels,
    and can be manually set for custom pipes, or left as None.

    This width should not be taken as the heated width, according to Mishima's experiments.
    In the original works by Mishima [#Mishima]_, which Sudo & Kaminage cite [#Sudo]_,
    they use the hydraulic diameter for annular channels, and the channel
    width for their rectangular case, where the heated region was 30mm out of 40mm.
    Thus, heated width would be the wrong quantity to use here.

    References
    ----------
    .. [#Mishima] K. Mishima, H. Nishihara, T. Shibata, "CHF Correlations Related
       to the Core Cooling of a Research Reactor", JAERI-M 84-073, International
       Meeting on Reduced Enrichment for Research and Test Reactors, Tokai, Japan, 1983.

    Returns
    -------
    q'': WPerM2
        CHF (Critical Heat Flux)
    """
    drho = (rho_l := sat_coolant.density) - (rho_v := sat_coolant.vapor_density)
    hfg = sat_coolant.latent_heat
    cp = sat_coolant.specific_heat
    Tsat = sat_coolant.sat_temperature
    Aht = sum(pipe.heated_parts) * pipe.length
    lamda = np.sqrt(sat_coolant.surface_tension / drho / g)

    q1 = _SKq1(G_star=(G_star := mdot / pipe.area / np.sqrt(lamda * drho * rho_v * g)))

    q2 = _SKq2(
        A_ratio=(A_ratio := pipe.area / Aht),
        dT_inlet=(dT_inlet := (cp / hfg) * (Tsat[0] - T_bulk[0])),
        G_star=G_star,
    )

    q3 = _SKq3(
        A_ratio=A_ratio,
        w=pipe.width,
        lamda=lamda,
        dT_inlet=dT_inlet,
        rho_v=rho_v,
        rho_l=rho_l,
    )

    q4 = _SKq4(G_star=G_star, dT_outlet=(cp / hfg) * (Tsat[-1] - T_bulk[-1]))

    q_star = np.zeros_like(G_star)
    # Downward
    q_star[G_star >= 0] = np.maximum(np.minimum(q2, q4), q3)[G_star >= 0]
    # Upward
    q_star[G_star < 0] = np.maximum(np.maximum(np.minimum(q2, q4), q1), q3)[G_star < 0]

    return q_star * hfg * np.sqrt(lamda * drho * rho_v * g)


def _SKq1(G_star):
    return 0.005 * np.abs(G_star) ** 0.611


def _SKq2(A_ratio, G_star, dT_inlet):
    return A_ratio * np.abs(G_star) * dT_inlet


def _SKq3(A_ratio, w, lamda, dT_inlet, rho_v, rho_l):
    r"""One of the form functions for the Sudo Kaminaga CHF correlation.

    The term is of the form:
    .. math::
       0.7 \frac{A_{flow}}{A_{heated}}
       \frac{\sqrt{\frac{W}{\lambda}}}{\left(1+\left(\frac{\rho_v}{\rho_l}\right^{0.25}\right)^2}
       (1+CdT)

    The :math:`1+CdT` term appears as both :math:`1+dT` and :math:`1+3dT`,
    but :math:`1+3dT` is more lenient so without clear evidence we would have to take the other.
    Thus, it could be unclear in which cases one should use :math:`1+dT` vs :math:`1+3dT`.

    The :math:`1+dT` term is found in Sudo and Kaminaga's IGORR presentation [SKIGORR]_.
    The :math:`1+3dT` term is found in Sudo and Kaminaga's 1998 paper in JNST [#SudoJNST]_.
    Because the :math:`1+3dT` term is newer and used for counter current in vertical rectangular
    channels at subcooled conditions, it seems that it should fit, and it specifically mentions
    that the authors propose it for the use in flow reversal of downward flow reactors in LOFA,
    for channels of our width and our conditions.

    Even so, someone claimed that they know of newer materials where the authors said that they
    were not allowed to use the :math:`1+3dT` in practice with their regulator.
    Why :math:`1+dT` is used rather than no subcooling at all is still a mystery to this author.
    The reference for the counter-example is still pending. TODO: Get the reference and clear this.

    References
    ----------
    [#SKIGORR] M. Kaminaga, Y. Sudo, T. Kodaira, N. Ohnishi, "CHF Correlation
    Scheme Proposed for Research Reactors Using Plate-Type Fuel - New CHF
    Correlation Under CCFL Condition", IGORR-IV, Garlinburg, Tennessee USA, 25.5.1995.
    [#SudoJNST] M. Kaminaga, K. Yamamoto, Y. Sudo, "Improvement of Critical Heat
    Flux Correlation for Research Reactors using Plate-Type Fuel", Journal of
    Nuclear Science and Technology, Vol. 35, No. 12, pages 943-951, 1998.

    """
    return 0.7 * A_ratio * np.sqrt(w / lamda) * (1 + dT_inlet) / (1 + (rho_v / rho_l) ** 0.25) ** 2


def _SKq4(G_star, dT_outlet):
    q4 = np.full_like(G_star, np.inf)
    _Gs = G_star[G_star != 0]
    _dT = dT_outlet[G_star != 0]
    q4[G_star != 0] = _SKq1(_Gs) * (1 + 5000 * _dT / np.abs(_Gs))
    return q4


def Mirshak_CHF(T_bulk: Celsius, T_sat: Celsius, pressure: Pascal, v: MPerS) -> WPerM2:
    r"""The limit for CHF as measured and calculated by Mirshak [#Mirshak]_ for rapid flows (v > 1.5 m/s)

    .. math:: 1.51\cdot 10^6 (1 + 0.1198v)(1 + 0.00914
        (T_\text{sat} - T_\text{bulk}))(1+1.9\cdot 10^{-6} p)

    Parameters
    ----------
    T_bulk: Celsius
        Bulk coolant temperature
    T_sat: Celsius
        Saturation temperature
    pressure: Pa
        pressure in channel
    v: MPerS
        Coolant flow velocity.

    Returns
    -------
    q'': WPerM2
        Heat flux where Critical Heat Flux (CHF) begins
    """
    return 1.51e6 * (1 + 0.1198 * v) * (1 + 0.00914 * (T_sat - T_bulk)) * (1 + 0.19e-5 * pressure)


def Fabrega_CHF(Tin: Celsius, T_sat: Celsius, Dh: Meter) -> WPerM2:
    r"""The limit for CHF as measured and calculated by Fabrega [#Fabrega]_ for slow flows (v < 0.5 m/s)

    .. math:: 10^7 D_\text{h}(0.023(T_\text{sat} - T_\text{bulk in}) + 4.56)

    Parameters
    ----------
    Tin: Celsius
        Bulk coolant temperature at inlet
    T_sat: Celsius
        Saturation temperature
    Dh: Meter
        Channel hydraulic diameter

    Returns
    -------
    q'': WPerM2
        Heat flux where Critical Heat Flux (CHF) begins
    """
    return 1e7 * Dh * (0.023 * (T_sat - Tin) + 4.56)
