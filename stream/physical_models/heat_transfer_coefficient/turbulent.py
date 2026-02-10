"""Turbulent Single Phase Heat Transfer Coefficient"""
from numba import njit

from stream.physical_models.dimensionless import Re_mdot, Pr
from stream.substances import Liquid
from stream.units import KgPerS, Meter, Meter2, WPerM2K, Value


@njit
def Dittus_Boelter(re: Value, pr: Value) -> Value:
    r"""The Dittus-Boelter equation calculates the Nusselt number.
    Where Reynolds is given for the inside diameter of an equivalent circular smooth duct.

    .. math:: \text{Nu} = 0.023\text{Re}^{0.8}\text{Pr}^{0.4}

    Its application is in the following phase subspace:

    Parameters
    ----------
    re: Value
        Reynolds number -> 10,000
    pr: Value
        Prandtl number, in [0.6, 160]

    Returns
    -------
    Nu: Value
        The Nusselt number

    Examples
    --------
    >>> Dittus_Boelter(re=1., pr=1.)
    0.023
    """
    return 0.023 * (re ** 0.8) * (pr ** 0.4)


def Dittus_Boelter_h_spl(
        *, coolant: Liquid, mdot: KgPerS, Dh: Meter, A: Meter2, **_
) -> WPerM2K:
    r"""
    Single phase liquid heat transfer coefficient.
    This is done through the Dittus-Boelter correlation, which holds for certain very high Reynolds.

    .. math:: h_\text{SPL} = (k / D_h)\text{Nu}
        = 0.023(k / D_h)\text{Re}^{0.8}\text{Pr}^{0.4}

    Parameters
    ----------
    coolant: Liquid
        Coolant properties
    mdot: KgPerS
        Characteristic flow mass current
    Dh: Meter
        The effective hydraulic diameter used for Re, Nu values.
    A: Meter2
        Cross sectional flow area.

    Returns
    -------
    h_SPL:  WPerM2K
        h_SPL heat transfer coefficient
    """
    cp = coolant.specific_heat
    mu = coolant.viscosity
    k = coolant.conductivity
    Nu = Dittus_Boelter(Re_mdot(mdot, A, Dh, mu), Pr(cp, mu, k))
    return (k / Dh) * Nu
