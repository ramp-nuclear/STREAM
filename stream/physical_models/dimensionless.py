"""Several flow properties and equations are given here, including
dimensionless parameters and corresponding correlations"""

import numpy as np
from numba import njit

from stream.units import (
    Celsius,
    g as local_gravity,
    JPerKgK,
    KgPerM3,
    KgPerS,
    Meter,
    Meter2,
    MPerS,
    MPerS2,
    PaS,
    PerC,
    Value,
    WPerM2K,
    WPerMK,
)


__all__ = ["flow_regimes", "Gr", "Nu", "Pe", "Pr", "Ra", "Re", "Re_mdot"]


@njit
def Re(rho: KgPerM3, u: MPerS, L: Meter, mu: PaS) -> Value:
    r"""
    The Reynolds number (Re) is defined as the ratio between inertial forces
    and viscous forces. :math:`\text{Re} = \rho u L / \mu`

    Parameters
    ----------
    rho: KgPerM3
        fluid density
    u: MPerS
        characteristic or equivalent velocity
    L: Meter
        characteristic or equivalent length
    mu: PaS
        dynamic viscosity of the fluid

    Returns
    -------
    Re: Value
        The Reynolds number

    Examples
    --------
    >>> Re(rho=0., u=1., L=1., mu=1.)
    0.0
    >>> Re(rho=1., u=np.arange(-2, 3), L=1., mu=1.)
    array([2., 1., 0., 1., 2.])
    >>> Re(rho=1., u=1., L=1., mu=np.inf)
    0.0
    """
    return rho * np.abs(u) * L / mu


@njit
def Re_mdot(mdot: KgPerS, A: Meter2, L: Meter, mu: PaS) -> Value:
    r"""
    The Reynolds number (Re) is defined as the ratio between inertial forces
    and viscous forces. If :math:`\dot{m} = \rho uA` is known,
    one may write :math:`\text{Re} = \dot{m}(L / A) / \mu`

    Parameters
    ----------
    mdot: KgPerS
        fluid mass current
    A: Meter2
        flow area
    L: Meter
        characteristic or equivalent length
    mu: PaS
        dynamic viscosity of the fluid

    Returns
    -------
    Re: Value
        The Reynolds number

    Examples
    --------
    >>> Re_mdot(mdot=1., A=1., L=1., mu=1.)
    1.0
    >>> Re_mdot(mdot=-1., A=1., L=1., mu=1.)
    1.0
    >>> Re_mdot(mdot=0., A=1., L=1., mu=1.)
    0.0
    """
    return np.abs(mdot) * L / (A * mu)


@njit
def Pr(cp: JPerKgK, mu: PaS, k: WPerMK) -> Value:
    r"""
    The Prandtl number (Pr) is defined as the ratio between viscous
    diffusivity and thermal diffusivity. :math:`\text{Pr} = c_p \mu / k`

    Parameters
    ----------
    cp: JPerKgK
        specific heat of the fluid
    mu: PaS
        dynamic viscosity of the fluid
    k: WPerMK
        thermal conductivity of the fluid

    Returns
    -------
    Pr: Value
        The Prandtl number

    Examples
    --------
    >>> Pr(cp=0.5, mu=0.1, k=50)
    0.001
    """
    return cp * mu / k


@njit
def Nu(h: WPerM2K, L: Meter, k: WPerMK) -> Value:
    r"""
    The Nusselt number (Nu) is defined as the ratio between convective and
    conductive heat transfer across the boundary. :math:`\text{Nu} = hL/k`

    Parameters
    ----------
    h: WPerM2K
        heat transfer coefficient
    L: Meter
        characteristic or equivalent length
    k: WPerMK
        thermal conductivity of the fluid

    Returns
    -------
    Nu: Value
        The Nusselt number

    Examples
    --------
    >>> Nu(h=1., L=1., k=1.)
    1.0
    """
    return h * L / k


@njit
def Pe(rho: KgPerM3, v: MPerS, L: Meter, cp: JPerKgK, k: WPerMK) -> Value:
    r"""
    The Peclet number (Pe) is defined as the ratio between advective and
    diffusive transport rates. For heat transfer, it is the product of
    :math:`\text{Pe} = \text{Re}\text{Pr}`

    Parameters
    ----------
    rho: KgPerM3
        fluid density
    v: MPerS
        characteristic or equivalent velocity
    L: Meter
        characteristic or equivalent length
    cp: JPerKgK
        specific heat of the fluid
    k: WPerMK
        thermal conductivity of the fluid

    Returns
    -------
    Pe: Value
        The Peclet Number

    Examples
    --------
    >>> Pe(rho=1., v=1., L=1., cp=1., k=1.)
    1.0
    """
    return Re(rho=rho, u=v, L=L, mu=1.0) * Pr(cp=cp, k=k, mu=1.0)


@njit
def Gr(
    rho: KgPerM3,
    mu: PaS,
    beta: PerC,
    T: Celsius,
    Twall: Celsius,
    Dh: Meter,
    g: MPerS2 = local_gravity,
) -> Value:
    r"""The Grashof number (Gr) is an approximation of the ratio of buoyancy to viscous
    forces.

    For vertical flat plates:

    .. math:: \text{Gr} = \rho^2 g\beta (T_\text{wall}-T_\text{bulk})\frac{L_h^3}{\mu^2}

    Parameters
    ----------
    rho: KgPerM3
        fluid density
    mu: PaS
        dynamic viscosity of the fluid
    beta: PerC
        fluid thermal expansion coefficient
    T: Celsius
        fluid bulk temperature
    Twall: Celsius
        wall\surface temperature
    Dh: Meter
        Hydraulic diameter
    g: MPerS2
        Gravitational acceleration constant

    Returns
    -------
    Gr: Value
        The Grashof Number

    Examples
    --------
    >>> Gr(rho=1, mu=1, beta=1, T=50, Twall=50, Dh=1)
    0.0
    """
    return rho**2 * g * beta * (Twall - T) * Dh**3 / mu**2


@njit
def Ra(
    rho: KgPerM3,
    mu: PaS,
    cp: JPerKgK,
    k: WPerMK,
    beta: PerC,
    T: Celsius,
    Twall: Celsius,
    Dh: Meter,
    g: MPerS2 = local_gravity,
) -> Value:
    r"""The Rayleigh number (Ra) is associated with heat transfer for natural convection.
    At low values, heat transfer is primarily conductive, and at high values it is
    primarily convective. :math:`\text{Ra} = \text{Gr}\text{Pr}`

    Parameters
    ----------
    rho: KgPerM3
        fluid density
    mu: PaS
        dynamic viscosity of the fluid
    cp: JPerKgK
        specific heat of the fluid
    k: WPerMK
        thermal conductivity of the fluid
    beta: PerC
        fluid thermal expansion coefficient
    T: Celsius
        fluid bulk temperature
    Twall: Celsius
        wall\surface temperature
    Dh: Meter
        Hydraulic diameter
    g: MPerS2
        Gravitational acceleration constant

    Returns
    -------
    Ra: Value
        The Rayleigh Number

    Examples
    --------
    >>> Ra(rho=1, mu=1, cp=1, k=1, beta=1, T=50, Twall=50, Dh=1)
    0.0
    """
    return Gr(rho, mu, beta, T, Twall, Dh, g) * Pr(cp, mu, k)


@njit
def flow_regimes(re: np.ndarray, bounds: tuple[Value, Value]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turbulent, Laminar and Interim regimes are determined by the Reynolds No.

    Parameters
    ----------
    re: np.ndarray
        Reynolds numbers vector
    bounds: tuple[Value, Value]
        boundaries depicting transition between the aforementioned regimes,
        such that:

         - Re <= bounds[0] is considered laminar.
         - bounds[0] < Re <= bounds[1] is considered interim.
         - bounds[1] < Re is considered turbulent.

    Returns
    -------
    Numpy masks: tuple[np.ndarray, np.ndarray, np.ndarray]
        Laminar, Interim, Turbulent

    Examples
    --------
    >>> a, b, c = flow_regimes(np.arange(5), (2, 3))
    >>> a
    array([ True,  True,  True, False, False])
    >>> b
    array([False, False, False,  True, False])
    >>> c
    array([False, False, False, False,  True])
    """
    laminar = re <= bounds[0]
    interim = (bounds[0] < re) * (re <= bounds[1])
    turbulent = bounds[1] < re
    return laminar, interim, turbulent
