"""Natural Convection Single Phase Heat Transfer Coefficient"""

import numpy as np
from numba import njit

from stream.physical_models.dimensionless import Ra
from stream.substances import Liquid
from stream.units import (
    Celsius,
    JPerKgK,
    KgPerM3,
    Meter,
    PaS,
    PerC,
    Value,
    WPerM2K,
    WPerMK,
)


@njit
def _Elenbaas(
    rho: KgPerM3,
    mu: PaS,
    cp: JPerKgK,
    k: WPerMK,
    beta: PerC,
    T: Celsius,
    Twall: Celsius,
    Lh: Meter,
    S: Meter,
) -> Value:
    ra = Ra(rho=rho, mu=mu, cp=cp, k=k, beta=beta, T=T, Twall=Twall, Dh=S)
    return (1 / 24) * ra * (S / Lh) * (1 - np.exp(-35 * Lh / (ra * S))) ** 0.75


def Elenbaas_h_spl(*, coolant: Liquid, depth: Meter, T_cool: Celsius, T_wall: Celsius, Lh: Meter, **_) -> WPerM2K:
    r"""A semi-empirical correlation for :class:`~.SinglePhaseLiquidHTC` made by
    Elenbaas [1]_ [2]_ for the case of `symmetrically heated, isothermal plates`:

    .. math:: \text{Nu}_S=\frac{1}{24}\frac{S}{L}\text{Ra}_S
              \left\{ 1-\exp \left[ -\frac{35}{Ra_S(S/L)} \right] \right\}^{3/4}

    Parameters
    ----------
    coolant: Liquid
        Coolant properties
    depth: Meter
        The depth, length between the two parallel plates
    T_cool: Celsius
        Coolant bulk temperature
    T_wall: Celsius
        Surface temperature
    Lh: Meter
        Heated length

    Returns
    -------
    WPerM2K
        Heat transfer coefficient

    References
    ----------
    .. [1] Elenbaas, W., Physica, 9, 1, 1942.
    .. [2] Incropera, Frank P. et al., Fundamentals of heat and mass transfer, 6th edition, section 9.7, p. 584.
    """
    S = depth
    k = coolant.conductivity
    Nu = _Elenbaas(
        rho=coolant.density,
        mu=coolant.viscosity,
        cp=coolant.specific_heat,
        k=k,
        beta=coolant.thermal_expansion,
        T=T_cool,
        Twall=T_wall,
        Lh=Lh,
        S=S,
    )
    return (k / S) * Nu
