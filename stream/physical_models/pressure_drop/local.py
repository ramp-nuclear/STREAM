"""
References
----------
    .. [#Idelchik] Handbook of Hydraulic Resistance, I. E. Idelchik, ????
"""

from functools import partial
from typing import Callable

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator

from stream.units import (
    Array1D,
    Array2D,
    KgPerM3,
    KgPerS,
    Meter2,
    Pascal,
    Radians,
    Value,
)
from stream.utilities import lin_interp


def _table_interp(
    re_numbers: Array1D,
    area_ratios: Array1D,
    f_factors: Array2D,
    area_ratio: float,
    re: float,
) -> float:
    if np.min(area_ratios) <= area_ratio <= 1:
        interp = RegularGridInterpolator((re_numbers, area_ratios), f_factors)
        return interp([float(re), area_ratio]).item()
    elif area_ratio < 0:
        raise ValueError(f"Area ratio cannot be negative. {area_ratio=} is.")
    else:
        raise ValueError(f"Smaller area must be smaller than the larger area.{area_ratio=} is not <=1!")


_tabulated_area_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
_idelchik_4_2_re = np.array([10, 15, 20, 30, 40, 50, 100, 200, 500, 1e3, 2e3, 3e3, 3300])

_idelchik_4_2_f = np.array(
    [
        [3.10, 3.20, 3.00, 2.40, 2.15, 1.95, 1.70, 1.65, 1.70, 2.00, 1.60, 1.00, 0.81],
        [3.10, 3.20, 2.80, 2.20, 1.85, 1.65, 1.40, 1.30, 1.30, 1.60, 1.25, 0.70, 0.64],
        [3.10, 3.10, 2.60, 2.00, 1.60, 1.40, 1.20, 1.10, 1.10, 1.30, 0.95, 0.60, 0.50],
        [3.10, 3.00, 2.40, 1.80, 1.50, 1.30, 1.10, 1.00, 0.85, 1.05, 0.80, 0.40, 0.36],
        [3.10, 2.80, 2.30, 1.65, 1.35, 1.15, 0.90, 0.75, 0.65, 0.90, 0.65, 0.30, 0.25],
        [3.10, 2.70, 2.15, 1.55, 1.25, 1.05, 0.80, 0.60, 0.40, 0.60, 0.50, 0.20, 0.16],
    ]
)

_idelchik_local_expansion_interp = partial(_table_interp, _idelchik_4_2_re, _tabulated_area_ratios, _idelchik_4_2_f.T)


@njit
def expansion_factor(aratio) -> float:
    r"""
    Compute *K*, the pressure drop factor (Borda, Carnot), for

    .. math:: \Delta p = K\rho v^2/2

    Where

    .. math:: K = \left(1 - \frac{A_1}{A_2}\right)^2

    According to Idelchik Table 4.2 [#idelchik]_, this formula is correct for large Re numbers, above 3300.

    Parameters
    ----------
    aratio: float
        Area ratio in [0,1]

    Returns
    -------
    K: float
        Pressure drop factor

    Examples
    --------
    >>> expansion_factor(0. / 1.)
    1.0
    >>> expansion_factor(1. / np.inf)
    1.0
    >>> expansion_factor(1. / 2.)
    0.25
    """
    return (1 - aratio) ** 2


def _sudden_area_factor(
    max_re: float,
    min_re: float,
    min_re_val: float,
    max_ar: float,
    min_ar: float,
    infval: float,
    analytic: Callable[[float], float],
    table_interp: Callable[[float, float], float],
    aratio: float,
    re: float,
):
    if re >= max_re:
        return analytic(aratio)
    elif re >= min_re and min_ar <= aratio <= max_ar:
        return table_interp(aratio, re)
    else:
        if np.isclose(re, 0):
            return 0.0  # In this case v=0 and there would be no pressure drop
        v1 = min_re_val * min_re / re
        if aratio > max_ar:
            v_at_ratio = lin_interp(max_ar, 1, v1, 0, aratio)
            return lin_interp(min_re, max_re, v_at_ratio, analytic(aratio), re)
        elif aratio < min_ar:
            v_at_ratio = lin_interp(min_ar, 0, v1, infval, aratio)
            return lin_interp(min_re, max_re, v_at_ratio, analytic(aratio), re)
        return v1


sudden_expansion_factor = partial(
    _sudden_area_factor,
    np.max(_idelchik_4_2_re),
    np.min(_idelchik_4_2_re),
    _idelchik_4_2_f[0, 0],
    np.max(_tabulated_area_ratios),
    np.min(_tabulated_area_ratios),
    1.0,
    expansion_factor,
    _idelchik_local_expansion_interp,
)
sudden_expansion_factor.__doc__ = r"""The Idelchik Table 4.2 interpolation of the local sudden expansion pressure drop coefficient. [#idelchik]_
    
    The table used is bounded by :math:`A_1/A_2 \in [0.1, 0.6]` and :math:`\text{Re} \in [10, 3300]`. 
    Within those boundaries, linear interpolation is performed.
    
    For larger Reynolds numbers, an analytic solution is used, :func:`expansion_factor`.
    For smaller Reynolds numbers, :math:`K = 31/\text{Re}` is used.
    
    As for area ratios :math:`A_1/A_2 \notin [0.1, 0.6]`, where the Reynolds no. is within the table values, 
    an interpolation by the Reynolds no. is performed between the limiting analytic solutions, 
    while for the lower end (:math:`\text{Re}=10`) an interpolation by the area ratio is added to match 
    :math:`K(A_1/A_2=1)=0` and :math:`K(A_1/A_2=0)=1` (matching an expansion into an infinite pool).
    
    The above result may not be applicable for such small Re numbers, but it would not 
    have a large effect since these pressure drops zero out as :math:`\mathcal{O}(v)`. 
    
    Parameters
    ----------
    aratio: float
        Area ratio in [0,1]
    re: float
        Reynolds number
    """


@njit
def contraction_factor(aratio: Value) -> Value:
    r"""Closed form approximation for sudden contraction at high Re numbers.

    According to Idelchik Table 4.10, there is a closed form approximation for
    sudden contraction of the form :math:`\frac{1}{2}\left(1-\frac{A_2}{A_1}\right)^{\frac{3}{4}}`.
    This is written to be true for Re > 35000, but the table for lower Re numbers
    stops at 10000, so it is unclear what to do for interim values of Re.

    Parameters
    ----------
    aratio: Value
        Area ratio in [0,1]

    """
    return 0.5 * (1.0 - aratio) ** 0.75


_idelchik_4_10_re = np.array([10, 20, 30, 40, 50, 100, 200, 500, 1e3, 2e3, 4e3, 5e3, 1e4])

# This is the original data
_idelchik_4_10_f = np.array(
    [
        [5.00, 3.20, 2.40, 2.00, 1.80, 1.30, 1.04, 0.82, 0.64, 0.50, 0.80, 0.75, 0.50],
        [5.00, 3.10, 2.30, 1.84, 1.62, 1.20, 0.95, 0.70, 0.50, 0.40, 0.60, 0.60, 0.40],
        [5.00, 2.95, 2.15, 1.70, 1.50, 1.10, 0.85, 0.60, 0.44, 0.30, 0.55, 0.55, 0.35],
        [5.00, 2.80, 2.00, 1.60, 1.40, 1.00, 0.78, 0.50, 0.35, 0.25, 0.45, 0.50, 0.30],
        [5.00, 2.70, 1.80, 1.46, 1.30, 0.90, 0.65, 0.42, 0.30, 0.20, 0.40, 0.42, 0.25],
        [5.00, 2.60, 1.70, 1.35, 1.20, 0.80, 0.56, 0.35, 0.24, 0.15, 0.35, 0.35, 0.20],
    ]
)
# For Re=1e4, we change the original values, so they match `contraction_factor`
_idelchik_4_10_f[:, -1] = contraction_factor(_tabulated_area_ratios)

_idelchik_local_contraction_interp = partial(
    _table_interp, _idelchik_4_10_re, _tabulated_area_ratios, _idelchik_4_10_f.T
)

sudden_contraction_factor = partial(
    _sudden_area_factor,
    np.max(_idelchik_4_10_re),
    np.min(_idelchik_4_10_re),
    _idelchik_4_10_f[0, 0],
    np.max(_tabulated_area_ratios),
    np.min(_tabulated_area_ratios),
    0.5,
    contraction_factor,
    _idelchik_local_contraction_interp,
)
sudden_contraction_factor.__doc__ = r"""The Idelchik Table 4.10 interpolation of the local sudden expansion pressure drop coefficient. [#idelchik]_
    
    The table used is bounded by :math:`A_2/A_1 \in [0.1, 0.6]` and :math:`\text{Re} \in [10, 10000]`. 
    Within those boundaries, linear interpolation is performed.
    
    For larger Reynolds numbers, an analytic solution is used, :func:`contraction_factor`.
    For smaller Reynolds numbers, :math:`K = 50/\text{Re}` is used.
    
    As for area ratios :math:`A_2/A_1 \notin [0.1, 0.6]`, where the Reynolds no. is within the table values, 
    an interpolation by the Reynolds no. is performed between the limiting analytic solutions, 
    while for the lower end (:math:`\text{Re}=10`) an interpolation by the area ratio is added to match 
    :math:`K(A_2/A_1=1)=0` and :math:`K(A_2/A_1=0)=0.5` (matching a contraction from an infinite pool, [#idelchik]_).
    
    The above result may not be applicable for such small Re numbers, but it would not 
    have a large effect since these pressure drops zero out as :math:`\mathcal{O}(v)`.
    
    Parameters
    ----------
    aratio: float
        Area ratio in [0,1]
    re: float
        Reynolds number
    
    """


def local_pressure_factor(
    *,
    mdot: KgPerS,
    positive_flow: Callable[[...], float],
    negative_flow: Callable[[...], float],
    **kwargs,
):
    r"""Considering :math:`\dot{m}`, returns ``K`` pressure coefficient of
    ``positive_flow(A1, A2)`` or ``negative_flow(A1, A2)``.

    This pressure coefficient is the loss coefficient for the total pressure in the Bernoulli equation,
    i.e. actual energy loss, not energy transfer between static and dynamic pressures.

    Parameters
    ----------
    mdot: KgPerS
        Mass current.
    positive_flow:  LocalPressureFactorFunc
        Expansion/Contraction
    negative_flow : LocalPressureFactorFunc
        Expansion/Contraction

    Returns
    -------
    K: float
        Pressure drop factor

    Examples
    --------
    >>> def pos(a1, a2): return a1+a2
    >>> def neg(a1, a2): return a1*a2
    >>> local_pressure_factor(mdot=1., positive_flow=pos, negative_flow=neg, a1=1., a2=2.)
    3.0
    >>> local_pressure_factor(mdot=-1., positive_flow=pos, negative_flow=neg, a1=1., a2=2.)
    2.0
    """
    return positive_flow(**kwargs) if mdot >= 0 else negative_flow(**kwargs)


@njit
def local_pressure_by_mdot(mdot: KgPerS, rho: KgPerM3, f: Value, A: Meter2) -> Pascal:
    r"""Calculates the local pressure dot according to a known mass flow rate.

    This pressure drop is not the static pressure drop across the local pipe area change,
    but rather the total pressure drop, i.e. the amount of lost energy in the Bernoulli equation terms.

    Following the quadratic velocity relation to local pressure:

    .. math:: \Delta p = f \frac{\rho u^2}{2}

    In this function, the velocity is replaced by :math:`\dot{m}=\rho u A`
    where `u` is the average velocity and `A` the cross-section.
    The pressure drop is sensitive to the flow's direction:

    .. math:: \Delta p = f \frac{\dot{m}|\dot{m}|}{2 \rho A^2}

    If :math:`\dot{m}>0` then the pressure returned is positive.

    Parameters
    ----------
    mdot: KgPerS
        Mass current.
    rho: KgPerM3
        Fluid density.
    f: Value
        The Darcy Friction Factor.
    A: Meter2
        Cross-sectional flow area.

    Returns
    -------
    dp: Pascal
        Pressure drop across the pipe.

    Examples
    --------
    >>> local_pressure_by_mdot(1., 1000., 1., 1.)
    0.0005
    >>> local_pressure_by_mdot(-1., 1000., 1., 1.)
    -0.0005
    >>> local_pressure_by_mdot(0., 1000., 1., 1.)
    0.0
    """
    return f * (mdot * np.abs(mdot) / (2 * rho * A**2))


@njit
def mdot_by_local_pressure(dp: Pascal, rho: KgPerM3, f: Value, A: Meter2) -> KgPerS:
    r"""The inverse function of :func:`local_pressure_by_mdot`, which returns
    :math:`\dot{m}` given a known pressure difference.

    .. math:: \dot{m}
     = \text{sign}(\Delta p)\sqrt{\frac{2\rho A^2 |\Delta p|}{f}}

    If :math:`\Delta p > 0` then the current returned is positive.

    Parameters
    ----------
    dp: Pascal
        Pressure drop across the pipe.
    rho: KgPerM3
        Fluid density.
    f: Value
        The Darcy Friction Factor.
    A: Meter2
        Cross-sectional flow area.

    Returns
    -------
    mdot: KgPerS
        Mass current.

    Examples
    --------
    >>> mdot_by_local_pressure(dp=0., rho=1e3, f=1., A=1.)
    0.0
    >>> mdot_by_local_pressure(dp=1., rho=1., f=2., A=1.)
    1.0
    >>> mdot_by_local_pressure(dp=-1., rho=1., f=2., A=1.)
    -1.0
    """
    return np.sign(dp) * np.sqrt(np.abs(dp) * (2 * rho * A**2) / f)


def bend_factor(angle: Radians, relative_curvature: float, re: float) -> float:
    r"""Compute *K*, the local pressure drop factor for

    .. math:: \Delta p = K\rho v^2/2,

    (where :math:`\Delta p` is the pressure drop over a pipe bend, :math:`\rho` is the fluid's density
    and :math:`v` is the fluid's velocity), according to Idelchik Diagram 6.1 [#idelchik]_ (page 424).

    This correlation applies to the case of a low curvature smooth pipe bend

    .. math:: R_0/D_h < 3

    (where :math:`R_0` is the pipe's axis radius of curvature, measured from the bend center to the
    center of the pipe, and :math:`D_h` is the hydraulic diameter of the pipe), with circular/square cross-sections
    and at large Re numbers (above 10,000).


    Parameters
    ----------
    angle: Radians
        The angle of the bend.
    relative_curvature: float
        The radius of curvature of the pipe's axis divided by the hydraulic diameter.
    re: float
        The Reynolds number.

    Returns
    -------
    K: float
        Local pressure drop factor.
    """
    a1 = _angle_effect(angle)
    b1 = _relative_curvature_effect(relative_curvature)
    c1 = 1.0

    k_re = _reynolds_number_effect(re, relative_curvature)

    return k_re * a1 * b1 * c1


_angles = np.array(np.deg2rad([70, 75, 90, 100]))
_angle_effects = np.array([0.9 * np.sin(np.deg2rad(70)), 0.9, 1.0, 0.7 + 0.35 * (100 / 90)])


def _angle_effect(angle: Radians) -> float:
    degree = np.rad2deg(angle)
    if 0 <= degree < 70:
        a1 = 0.9 * np.sin(angle)
    elif 70 <= degree <= 100:
        a1 = np.interp(angle, _angles, _angle_effects)
    elif 100 < degree <= 180:
        a1 = 0.7 + 0.35 * degree / 90
    else:
        raise ValueError(f"Bend {angle = } out of limits (0 <= angle <= pi).")
    return a1


def _relative_curvature_effect(relative_curvature: float) -> float:
    if relative_curvature < 0.5:
        raise ValueError("Relative curvature must be greater than 0.5.")

    power = -2.5 if (0.5 <= relative_curvature < 1.0) else -0.5
    return 0.21 * (relative_curvature**power)


# Taken from I.E. Idelchik, Handbook of Hydraulic Resistance, 4th Edition, Diagram 6.1(e), Page 426
_reynolds_numbers = np.array(
    [
        0.1e5,
        0.14e5,
        0.2e5,
        0.3e5,
        0.4e5,
        0.6e5,
        0.8e5,
        1.0e5,
        1.4e5,
        2.0e5,
        3.0e5,
        4.0e5,
    ]
)
_reynolds_number_effect_low = np.array([1.40, 1.33, 1.26, 1.19, 1.14, 1.09, 1.06, 1.04, 1.0, 1.0, 1.0, 1.0])
_reynolds_number_effect_mid = np.array([1.67, 1.58, 1.49, 1.40, 1.34, 1.26, 1.21, 1.19, 1.17, 1.14, 1.06, 1.0])
_reynolds_number_effect_high = np.array([2.00, 1.89, 1.77, 1.64, 1.56, 1.46, 1.38, 1.30, 1.15, 1.02, 1.0, 1.0])


def _reynolds_number_effect(re: float, relative_curvature: float) -> float:
    if 0.5 <= relative_curvature < 0.55:
        k_re = np.interp(re, _reynolds_numbers, _reynolds_number_effect_low)
    elif 0.55 <= relative_curvature < 0.70:
        k_re = np.interp(re, _reynolds_numbers, _reynolds_number_effect_mid)
    elif relative_curvature >= 0.70:
        k_re = np.interp(re, _reynolds_numbers, _reynolds_number_effect_high)
    else:
        raise ValueError(f"Relative curvature {relative_curvature:.5f} is not supported")

    return k_re
