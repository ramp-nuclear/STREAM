from typing import Callable, Literal

import numpy as np
from numba import njit

from stream.physical_models.dimensionless import Re_mdot, flow_regimes
from stream.pipe_geometry import EffectivePipe
from stream.substances import LiquidFuncs
from stream.units import (
    Celsius, KgPerS, Meter2, KgPerM3, Meter, Pascal, Value,
    )
from stream.utilities import lin_interp


@njit
def turbulent_friction(re: Value, epsilon: Value = 0) -> Value:
    r"""An approximation for the friction factor of the implicit Colebrook-White equation,
    as written in RELAP and [#KAERI]_ page 3, chapter 2.1.2.

    For very low Reynolds values (approx. <7), 0.0 is returned.

    Parameters
    ----------
    re: Value
        Reynolds Number.
    epsilon: Value.
        Relative roughness (roughness height / hydraulic diameter).

    Returns
    -------
    f: Value
        The Darcy friction factor

    Examples
    --------
    >>> turbulent_friction(4e3)
    0.039804935964641644
    >>> turbulent_friction(4e3, 0.1)
    0.10560870441248855
    >>> turbulent_friction(1e6)
    0.011649393290640643
    >>> turbulent_friction(5.0)
    0.0
    """
    inlog = np.log10(epsilon + 21.25 / re ** 0.9)
    outlog = np.log10(epsilon / 3.7 + (2.51 / re) * (1.14 - 2 * inlog))
    return np.nan_to_num((-2 * outlog) ** -2)


@njit
def Blasius_friction(re: Value) -> Value:
    r"""An early (1913) approximation by Blasius [#Blasius]_ for the turbulent pipe flow Darcy friction factor.

    .. math:: f = \frac{0.3164}{\text{Re}^{0.25}}

    Parameters
    ----------
    re: Value
        The Reynolds number.

    Returns
    -------
    f: Value
        The Darcy friction factor

    Examples
    --------
    >>> Blasius_friction(10_000)
    0.03164
    """
    return 0.3164 / re ** 0.25


@njit
def Darcy_Weisbach_pressure_by_mdot(
        mdot: KgPerS, rho: KgPerM3, f: Value, L: Meter, Dh: Meter, A: Meter2
        ) -> Pascal:
    r"""
    The Darcy-Weisbach equation relates pressure loss due to
    friction in pipe flow to the average velocity of an incompressible fluid:

    .. math:: \Delta p = f \frac{\rho u^2}{2}\frac{L}{D_h}

    In this function, the velocity is replaced by :math:`\dot{m}=\rho u A`
    where `u` is the average velocity and `A` the cross-section.
    The pressure drop is sensitive to the flow's direction:

    .. math:: \Delta p = f \frac{\dot{m}|\dot{m}|}{2\rho A^2}\frac{L}{D_h}

    If :math:`\dot{m}>0` then the pressure returned is positive.

    Parameters
    ----------
    mdot: KgPerS
        Mass current.
    rho: KgPerM3
        Fluid density.
    f: Value
        The Darcy Friction Factor.
    L: Meter
        Pipe length.
    Dh: Meter
        Hydraulic diameter.
    A: Meter2
        Cross-sectional flow area.

    Returns
    -------
    dp: Pascal
        Pressure drop across the pipe.

    Examples
    --------
    >>> Darcy_Weisbach_pressure_by_mdot(mdot=0., rho=1, f=1, L=1, Dh=1, A=1)
    0.0
    >>> Darcy_Weisbach_pressure_by_mdot(mdot=1., rho=1, f=1, L=1, Dh=1, A=1)
    0.5
    >>> Darcy_Weisbach_pressure_by_mdot(mdot=-1., rho=1, f=1, L=1, Dh=1, A=1)
    -0.5
    """
    return f * (mdot * np.abs(mdot) / (2 * rho * A ** 2)) * (L / Dh)


@njit
def viscosity_correction(heat_wet_ratio: Value, mu_ratio: Value) -> Value:
    """Temperature differences yield varying viscosity, which affects friction.
    Thus, a correction must be made.

    In this function such a correction is returned as used in [#KAERI]_.

    Parameters
    ----------
    heat_wet_ratio: Value
        Heated perimeter / Wet perimeter
    mu_ratio: Value
        Viscosity at wall / Viscosity at bulk.

    Returns
    -------
    K_H: Value
        viscosity correction coefficient

    Examples
    --------
    >>> viscosity_correction(1., 1.)
    1.0
    >>> viscosity_correction(1., 0.)
    0.0
    >>> viscosity_correction(1., 2.)
    1.4948492486349383
    """
    return 1 + heat_wet_ratio * (mu_ratio ** 0.58 - 1)


def rectangular_laminar_correction(aspect_ratio: float) -> Value:
    """For a circular pipe, the laminar Darcy friction factor is analytical.
    For a rectangular pipe, corrections must be made.

    In this function such a correction is returned as used in [#KAERI]_.

    Parameters
    ----------
    aspect_ratio: float
        channel_depth / channel_width (less than 1)

    Returns
    -------
    K_R: float
        geometric correction coefficient

    See Also
    --------
    laminar_friction

    Examples
    --------
    >>> rectangular_laminar_correction(1.)
    1.1246190353017915
    >>> rectangular_laminar_correction(0.)
    0.6668484123609236
    >>> rectangular_laminar_correction(0.5)
    1.0363896075094456
    """
    assert 0.0 <= aspect_ratio <= 1.0, f"{aspect_ratio = } must be non-negative and less than 1"
    return (
                   0.88919
                   + 87.656
                   * ((1 + aspect_ratio * (np.sqrt(2) - 1)) / (4 * (1 + aspect_ratio)) - np.sqrt(
               2) / 8) ** 1.9
           ) ** (-1)


@njit
def laminar_friction(re: Value) -> Value:
    r"""Due to the Hagen-Poiseuille law, an incompressible Newtonian fluid in laminar flow through a long constant
    cylindrical pipe yields an analytic Darcy friction factor: :math:`f=64/\text{Re}`

    Parameters
    ----------
    re: Value
        The Reynolds number.

    Returns
    -------
    f: Value
        The Darcy friction factor.
    """
    return 64.0 / re


def regime_dependent_friction(
        T_cool: Celsius,
        T_wall: Celsius,
        mdot: KgPerS,
        fluid: LiquidFuncs,
        pipe: EffectivePipe,
        re_bounds: tuple[float, float],
        k_R: float,
        k_H: Callable[[float, float], float] | None = None,
        turbulent: Callable[[Value], Value] = turbulent_friction,
        laminar: Callable[[Value], Value] = laminar_friction,

        ) -> Value:
    r"""A flow-regime-dependent friction coefficient function.

    Parameters
    ----------
    T_cool: Celsius
        Coolant bulk temperatures
    T_wall: Celsius
        Wall temperatures
    mdot: KgPerS
        Coolant mass flow rate
    fluid: LiquidFuncs
        Coolant properties
    pipe: EffectivePipe
        Pipe geometry
    re_bounds: tuple[float, float]
        Flow regime (Re values) boundaries, see :func:`.flow_regimes`.
    k_R: float
        Geometrical factor. At laminar flow the Darcy coefficient is analytical (see :func:`laminar_friction`)
        for a circular duct. ``k_R`` is a multiplicative factor such that
        :math:`f_\text{laminar} = \frac{64}{\text{Re} k_R}`.
        See for example, :func:`rectangular_laminar_correction`.
        This correction is applied to all regimes.
    k_H: Callable[[float, float], float] | None
        Viscosity factor function, which takes as input
        :math:`k_H(P_\text{heat}/P_\text{wet}, \mu_\text{wall} / \mu_\text{bulk})`.
        By default, it is omitted.
        See :func:`viscosity_correction`.
    turbulent: Callable[[Value], Value]
        Strategy for calculating the friction coefficient for turbulent flow.
    laminar: Callable[[Value], Value]
        Strategy for calculation the friction coefficient for laminar flow.

    Returns
    -------
    f: Value
        The Darcy friction factor.
    """
    if float(mdot) == 0.0:
        return 0.0
    Dh = pipe.hydraulic_diameter
    A = pipe.area
    mu_bulk = fluid.viscosity(T_cool)
    re_bulk = np.atleast_1d(Re_mdot(mdot=mdot, A=A, L=Dh, mu=mu_bulk))

    f = np.empty(len(re_bulk))
    lam, inter, turb = flow_regimes(re_bulk, re_bounds)
    f_turb = turbulent(re_bulk * k_R)
    f_lam = laminar(re_bulk * k_R)

    f[lam] = f_lam[lam]
    f[inter] = lin_interp(
        x1=re_bounds[0],
        x2=re_bounds[1],
        y1=f_lam[inter],
        y2=f_turb[inter],
        x=re_bulk[inter],
        )
    f[turb] = f_turb[turb]

    heat_wet_ratio = pipe.heated_perimeter / pipe.wet_perimeter
    mu_ratio = fluid.viscosity(T_wall) / mu_bulk

    return f * (1.0 if k_H is None else k_H(heat_wet_ratio, mu_ratio))


GeneralDarcyFactor = Callable[[Celsius, Celsius, KgPerS,
                               LiquidFuncs, EffectivePipe], Value]
_DARCY_NAMES = {
    (_REGIME := "regime_dependent"): None,
    "laminar": laminar_friction,
    "turbulent": turbulent_friction,
    "Blasius": Blasius_friction,
    }


def _re_friction(f: Callable[[Value, dict], Value], **kwargs) -> GeneralDarcyFactor:
    def _f(
            T_cool: Celsius,
            T_wall: Celsius,
            mdot: KgPerS,
            fluid: LiquidFuncs,
            pipe: EffectivePipe,
            ) -> Value:
        mu = fluid.viscosity(T_cool)
        re = Re_mdot(mdot=mdot, A=pipe.area, L=pipe.hydraulic_diameter, mu=mu)
        return f(re, **kwargs)

    _f.__name__ = f.__name__
    _f.__doc__ = f.__doc__ + (
        """

    This function has been wrapped so that it has a
    general friction factor signature:

    Parameters
    ----------
    T_cool: Celsius
        Coolant bulk temperatures
    T_wall: Celsius
        Wall temperatures
    mdot: KgPerS
        Coolant mass flow rate
    fluid: LiquidFuncs
        Coolant properties
    pipe: EffectivePipe
        Pipe geometry
    """
    )
    return _f


def friction_factor(
        name: Literal["regime_dependent", "laminar", "turbulent", "Blasius"], **kwargs
        ) -> GeneralDarcyFactor:
    r"""Create a Darcy friction factor chosen from the list below with uniform signatures.
    The main usage of this function is as input for :func:`pressure_diff`.

    Available functions:

    .. list-table::
        :widths: 20, 80

        * - **regime_dependent**
          - :func:`regime_dependent_friction`, which depends on the :func:`~.Re` No., given ``re_bounds``.
            Importantly, employs :func:`viscosity <viscosity_correction>` and geometry (e.g. :func:`rectangular
            <rectangular_laminar_correction>`) corrections upon the laminar and turbulent factors.
        * - **laminar**
          - :func:`laminar_friction`. Does not employ corrections as described
            above.
        * - **turbulent**
          - :func:`turbulent_friction`
        * - **Blasius**
          - :func:`Blasius_friction`


    Parameters
    ----------
    name: Literal['regime_dependent', 'laminar', 'turbulent', 'Blasius']
        Method name
    kwargs: Dict
        Options to pass onto the given method

    Returns
    -------
    f: GeneralDarcyFactor
        Darcy friction factor
    """
    if name == _REGIME:
        def _regime_dependent(
                T_cool: Celsius,
                T_wall: Celsius,
                mdot: KgPerS,
                fluid: LiquidFuncs,
                pipe: EffectivePipe,
                ) -> Value:
            return regime_dependent_friction(
                T_cool, T_wall, mdot, fluid, pipe, **kwargs
                )

        _regime_dependent.__name__ = regime_dependent_friction.__name__
        _regime_dependent.__doc__ = regime_dependent_friction.__doc__

        return _regime_dependent
    try:
        return _re_friction(_DARCY_NAMES[name], **kwargs)
    except KeyError as e:
        raise ValueError(f"{name=} not found in {list(_DARCY_NAMES.keys())}") from e
