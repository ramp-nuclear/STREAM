r""":math:`\text{H}_2\text{O}` constants and correlations, as given in [#simantov]_.
The user facing function is of type :class:`~.stream.substances.liquid.LiquidFuncs`

>>> type(light_water)
<class 'stream.substances.liquid.LiquidFuncs'>
"""

from numba import njit
import numpy as np

from stream.substances import LiquidFuncs
from stream.units import (
    Celsius,
    JPerKg,
    JPerKgK,
    KgPerM3,
    kilo,
    NPerM,
    PaS,
    Pascal,
    WPerMK, PerC,
)
from stream.utilities import to_Fahrenheit

__all__ = ["light_water"]


@njit
def _viscosity(T: Celsius) -> PaS:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    mu: PaS
        Dynamic viscosity of saturated H2O

    Examples
    --------
    >>> _viscosity(90.)
    0.00031444961652895464
    """
    A = -6.325203964
    B = 8.705317e-3
    C = -0.088832314
    D = -9.657e-7
    return np.exp((A + C * T) / (1 + B * T + D * T ** 2))


@njit
def _specific_heat(T: Celsius) -> JPerKgK:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    cp: JPerKgK
        Specific heat of saturated H2O

    Examples
    --------
    >>> _specific_heat(8.)
    4179.863745234987
    >>> _specific_heat(8.) == _specific_heat(-8.)
    True
    >>> _specific_heat(50.)
    4181.4264285644285
    """
    T = np.abs(T)
    A = 17.48908904
    B = -1.67507e-3
    C = -0.03189591
    D = -2.8748e-6
    return np.sqrt((A + C * T) / (1 + B * T + D * T ** 2)) * kilo


@njit
def _conductivity(T: Celsius) -> WPerMK:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    k: WPerMK
        Saturated thermal conductivity

    Examples
    --------
    >>> _conductivity(50.)
    0.6419141378687501
    """
    A = 0.5677829144
    B = 1.8774171e-3
    C = -8.1790e-6
    D = 5.66294775e-9
    return np.abs(A + B * T + C * T ** 2 + D * T ** 3)


@njit
def _density(T: Celsius) -> KgPerM3:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    rho: KgPerM3
        Saturated liquid density

    Examples
    --------
    >>> _density(50.)
    987.27431208
    >>> _density(100.)
    959.13959928
    """
    A = 1004.789042
    B = -0.046283
    C = -7.9738e-4
    TF = to_Fahrenheit(T)
    return np.abs(A + B * TF + C * TF ** 2)


@njit
def _thermal_expansion(T: Celsius) -> PerC:
    r"""Thermal volume expansion coefficient, :math:`\beta`, which is defined as:

    .. math:: \beta = -\frac{1}{\rho} \left(\frac{\partial \rho}{\partial T}\right)_p

    It is derived analytically from ``_density(T)``.

    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    beta: PerC
        Thermal volume expansion coefficient

    Examples
    --------
    >>> _thermal_expansion(20.) * 1e6
    279.0788203166585
    >>> _thermal_expansion(100.) * 1e6
    721.3442303074213
    """
    B = -0.046283
    C = -7.9738e-4
    TF = to_Fahrenheit(T)
    return - 1.8 * (B + 2 * C * TF) / _density(T)


@njit
def _sat_temperature(P: Pascal) -> Celsius:
    """
    Parameters
    ----------
    P: Pascal
        Fluid absolute pressure

    Returns
    -------
    Tsat: Celsius
        Saturation temperature

    Examples
    --------
    >>> _sat_temperature(1e5)
    99.63072810857243
    >>> _sat_temperature(0.5e5)
    81.28047959788387
    >>> _sat_temperature(2e5)
    120.29401952865119
    >>> from scipy.constants import atm
    >>> _sat_temperature(atm)
    100.00030833371952
    """
    X = np.log(np.abs(P) * 1e-6)
    A = 179.9600321
    B = -0.1063030
    C = 24.2278298
    D = 2.951e-4
    return (A + C * X) / (1 + B * X + D * X ** 2)


@njit
def _latent_heat(T: Celsius) -> JPerKg:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    hfg: JPerKg
        Latent heat for the liquid-gas phase transition

    Examples
    --------
    >>> _latent_heat(50.)
    2382729.243923866
    >>> _latent_heat(100.)
    2257149.1343506747
    """
    A = 6254828.560
    B = -11742.337953
    C = 6.336845
    D = -0.049241
    return 1e3 * np.sqrt(np.abs(A + B * T + C * T ** 2 + D * T ** 3))


@njit
def _surface_tension(T: Celsius) -> NPerM:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    sigma: NPerM
        Saturated liquid surface tension

    Examples
    --------
    >>> _surface_tension(50.)
    0.06794675477982745
    >>> _surface_tension(100.)
    0.05891594230703328
    """
    X = np.abs(373.99 - T) / 647.15
    A = 235.8e-3
    B = 1.256
    C = -0.625
    return A * (X ** B) * np.abs(1 + C * X)


@njit
def _vapor_density(T: Celsius) -> KgPerM3:
    """
    Parameters
    ----------
    T: Celsius
        Temperature of fluid

    Returns
    -------
    rho_g: KgPerM3
        Saturated vapor density

    Examples
    --------
    >>> _vapor_density(100.)
    0.5978051373615001
    >>> _vapor_density(50.)
    0.08307666133931553
    """
    A = -4.375094e-4
    B = -6.947700e-3
    C = 7.662589e-4
    D = 2.418897e-5
    E = -5.963920e-6
    F = -4.227966e-8
    G = 2.867976e-7
    H = 2.594175e-11
    return (A + C * T + E * T ** 2 + G * T ** 3) / (
        1 + B * T + D * T ** 2 + F * T ** 3 + H * T ** 4
    )


light_water = LiquidFuncs(
    density=_density,
    vapor_density=_vapor_density,
    specific_heat=_specific_heat,
    viscosity=_viscosity,
    sat_temperature=_sat_temperature,
    surface_tension=_surface_tension,
    latent_heat=_latent_heat,
    conductivity=_conductivity,
    thermal_expansion=_thermal_expansion,
)
