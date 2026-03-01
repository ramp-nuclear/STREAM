r""":math:`\text{D}_2\text{O}` constants and correlations,
as given in [#simantov]_. The user facing function is of type
:class:`~.stream.data_classes.LiquidFuncs`

>>> type(heavy_water)
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
    NPerM,
    PaS,
    Pascal,
    WPerMK,
    PerC,
)
from stream.utilities import to_Fahrenheit

__all__ = ["heavy_water"]


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
    1095.7419670000002
    >>> _density(100.)
    1063.4244970000002
    """
    TF = to_Fahrenheit(T)
    A = 1117.772605
    B = -0.077855
    C = -8.42e-4
    return A + B * TF + C * TF**2


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
    312.34463951465654
    >>> _thermal_expansion(100.) * 1e6
    736.0686181371651
    """
    B = -0.077855
    C = -8.42e-4
    TF = to_Fahrenheit(T)
    return -1.8 * (B + 2 * C * TF) / _density(T)


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
    >>> _specific_heat(50.)
    4220.658975628751
    >>> _specific_heat(100.)
    4162.210117465748

    """
    Tl = (1.8 * T + 491.67) * 1e-4
    A = 2.237124
    B = 122.217151
    C = -2303.384060
    D = 13555.737878
    return 1000 * (A + B * Tl + C * Tl**2 + D * Tl**3)


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
    >>> _viscosity(50.)
    0.0006441125212510078
    >>> _viscosity(100.)
    0.0003301433604774831
    """
    TF = to_Fahrenheit(T)
    A = -1.111606e-4
    B = 9.46e-8
    C = 0.0873655375
    D = 0.4111103409
    return A + B * TF + C / TF + D / (TF**2)


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
    0.6167873183429435
    >>> _conductivity(100.)
    0.6357784886396809
    """
    Tl = (1.8 * T + 491.67) * 1e-4
    A = -0.4521496
    B = 36.0743280
    C = -357.9973221
    D = 924.0219962
    return A + B * Tl + C * Tl**2 + D * Tl**3


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
    100.98975482398993
    >>> _sat_temperature(0.5e5)
    82.7830309880722
    >>> _sat_temperature(2e5)
    121.5058319422803
    >>> from scipy.constants import atm
    >>> _sat_temperature(atm)
    101.35704815210188
    """
    X = np.log(np.abs(P) * 1e-6)
    A = 5.194927982
    B = 0.236771673
    C = -2.615268e-3
    D = 1.708386e-3
    return np.exp(A + B * X + C * X**2 + D * X**3)


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
    0.06809951822968323
    >>> _surface_tension(100.)
    0.059250184550697166

    """
    X = np.abs(373.99 - T) / 647.15
    A = 2.44835759e-1
    B = 1.269
    C = -6.60709649e-1
    return A * (X**B) * (1 + C * X)


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
    >>> _vapor_density(50.)
    0.08342446145018677
    >>> _vapor_density(100.)
    0.6309356177290303

    """
    A = -5.456208705
    B = 2.386228e-3
    C = 0.060526809
    D = -1.15778e-5
    E = -1.1136e-4
    return np.exp((A + C * T + E * (T**2)) / (1 + B * T + D * (T**2)))


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
    2199499.183881408
    >>> _latent_heat(100.)
    2076983.0825663893
    """
    X = np.abs(371.49 - T)
    A = 508093.6669
    B = 17006.921765
    C = -11.009078
    return np.sqrt(A + B * X + C * X**2) * 1000


heavy_water = LiquidFuncs(
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
