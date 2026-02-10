"""Temperatures used in the calculation of the heat transfer coefficient"""

import numpy as np
from numba import njit

from stream.units import Celsius, WPerM2K, Pascal, WPerM2


def wall_temperature(
        T_cool: Celsius, T_clad: Celsius, h_cool: WPerM2K, h_clad: WPerM2K
        ) -> Celsius:
    r"""
    Computes the interface temperature, under the assumption that the interface's thermal inertia is 0, thus
    incoming and outgoing fluxes should be equal. Then temperature differences should behave as follows:

    .. math::
        (T_\text{clad} - T_\text{wall}) h_\text{clad}
        = (T_\text{wall} - T_\text{cool}) h_\text{cool}


    Parameters
    ----------
    T_cool: Celsius
        Coolant bulk temperature
    T_clad: Celsius
        Clad temperature, closest to the interface.
    h_cool: WPerM2K
        Heat transfer coefficient between wall and coolant bulk.
    h_clad: WPerM2K
        Clad heat transfer coefficient (conductivity / distance) between T_clad placement and T_wall placement.

    Returns
    -------
    T_wall: Celsius
        Wall temperature
    """
    ratio = np.where(h_clad == np.inf, 1, h_clad / (h_clad + h_cool))
    return ratio * T_clad + (1 - ratio) * T_cool


@njit
def Bergles_Rohsenow_T_ONB(pressure: Pascal, q_spl: WPerM2, Tsat: Celsius) -> Celsius:
    r"""The wall temperature at which nucleate boiling begins at certain conditions.
    This is simply the saturation temperature combined with the Bergles & Rohsenow correlation [#BR]_ [#Collier]_.

    Like all Bergles & Rohsenow, this is for water only.

    See Also
    --------
    .Bergles_Rohsenow_dT_ONB

    Parameters
    ----------
    pressure: Pascal
        Absolute pressure of coolant
    q_spl: WPerM2
        Single phase heat flux
    Tsat: Celsius
        Saturation (Boiling) temperature of coolant

    Returns
    -------
    T_inc: Celsius
        Incipience temperature

    Examples
    --------
    >>> Bergles_Rohsenow_T_ONB(Tsat=100., q_spl=0., pressure=1e10)
    100.0
    """
    return Tsat + Bergles_Rohsenow_dT_ONB(pressure, q_spl)


@njit
def Bergles_Rohsenow_dT_ONB(pressure: Pascal, q_spl: WPerM2) -> Celsius:
    r"""A correlation by Bergles-Rohsenow which determines the difference between
    the wall temperature at nucleate boiling and the saturation temperature.
    [#BR]_ [#Collier]_

    Like all Bergles & Rohsenow, this is for water only.

    .. math:: \Delta T_\text{ONB} = 0.556 \left( \frac{q''_\text{ONB}}{1082 p^{1.156}} \right)
        ^{0.463 p^{0.0234}}

    Where in the above expression the absolute pressure is given in **Bar**.

    Parameters
    ----------
    pressure: Pascal
        Absolute pressure of coolant
    q_spl: WPerM2
        Single phase heat flux.

    Examples
    --------
    >>> Bergles_Rohsenow_dT_ONB(q_spl=0., pressure=1e10)
    0.0

    Returns
    -------
    Celsius
        The difference between the wall temperature and saturation temperature
        at which nucleate boiling begins.

    """
    p = pressure / 1e5
    return 0.556 * (q_spl / 1082 / p ** 1.156) ** (0.463 * p ** 0.0234)


@njit
def film_temperature(*, T_cool: Celsius, T_wall: Celsius) -> Celsius:
    r"""Determine the coolant temperature at the wall interface.
    This temperature may be used to assess the thermodynamical properties of the coolant there.

    Parameters
    ----------
    T_cool: Celsius
        Coolant bulk temperature.
    T_wall: Celsius
        Wall interface temperature.

    Returns
    -------
    film: Celsius
        "Film" temperature.

    Examples
    --------
    >>> film_temperature(20., 20.)
    20.0
    """
    return (T_cool + T_wall) / 2
