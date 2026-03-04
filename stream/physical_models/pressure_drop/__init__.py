from numba import njit

from stream.physical_models.pressure_drop.friction import (
    Darcy_Weisbach_pressure_by_mdot,
    GeneralDarcyFactor,
    friction_factor,
    Blasius_friction,
    laminar_friction,
    regime_dependent_friction,
    rectangular_laminar_correction,
    turbulent_friction,
    viscosity_correction,
)
from stream.physical_models.pressure_drop.local import (
    contraction_factor,
    expansion_factor,
    local_pressure_by_mdot,
    local_pressure_factor,
    mdot_by_local_pressure,
)
from stream.pipe_geometry import EffectivePipe
from stream.substances import LiquidFuncs
from stream.units import (
    Celsius,
    KgPerS,
    KgPerS2,
    Meter2,
    KgPerM3,
    Meter,
    MPerS2,
    g as local_gravity,
    Pascal,
)


__all__ = [
    "Blasius_friction",
    "contraction_factor",
    "Darcy_Weisbach_pressure_by_mdot",
    "expansion_factor",
    "friction_factor",
    "gravity_pressure",
    "inertia_pressure",
    "laminar_friction",
    "local_pressure_by_mdot",
    "local_pressure_factor",
    "mdot_by_local_pressure",
    "pressure_diff",
    "rectangular_laminar_correction",
    "regime_dependent_friction",
    "turbulent_friction",
    "viscosity_correction",
    "static_pressure",
]


@njit
def gravity_pressure(rho: KgPerM3, dh: Meter, g: MPerS2 = local_gravity) -> Pascal:
    r"""
    In the case of an incompressible barotropic fluid at rest, the force exerted by gravity over a fluid column is
    :math:`\Delta p=\rho g\Delta h`

    Parameters
    ----------
    rho: KgPerM3
        Fluid density.
    g: MPerS2
        Gravitational acceleration.
    dh: Meter
        Height difference.

    Returns
    -------
    dp: Pascal
        Pressure drop due to gravity.

    Examples
    --------
    >>> gravity_pressure(1., 1.)
    9.80665
    >>> gravity_pressure(1000, 0.)
    0.0
    >>> # What is the pressure head (for water) on the moon?
    >>> gravity_pressure(1000, 1, g=(moon := 1.62))
    1620.0
    """
    return rho * g * dh


def pressure_diff(
    T: Celsius,
    Tw: Celsius,
    fluid: LiquidFuncs,
    mdot: KgPerS,
    pipe: EffectivePipe,
    dz: Meter,
    f: GeneralDarcyFactor = friction_factor("Blasius"),
    g: MPerS2 = local_gravity,
    mdot2: KgPerS2 | None = None,
    **_,
) -> Pascal:
    r"""
    Returns pressure difference. Positive direction is assumed downward.
    This function calculates frictional, inertial, and gravitational pressure drops.

    Parameters
    ----------
    T: Celsius
        Bulk Coolant temperature
    Tw: Celsius
        Wall temperature
    fluid: LiquidFuncs
        Coolant properties
    mdot: KgPerS
        Coolant mass current
    pipe: EffectivePipe
        Channel geometry
    dz: Meter
        Cell sizes in the flow's direction
    f: GeneralDarcyFactor | None
        Darcy-Weisbach friction factor, default being :func:`friction.Blasius_friction`.
    g: MPerS2
        Gravitational acceleration.
    mdot2: KgPerS2 | None
        :math:`\ddot{m}`, time derivative of ``mdot``

    Returns
    -------
    dp: Pascal
        Pressure drop at each cell

    See Also
    --------
    friction.Darcy_Weisbach_pressure_by_mdot, gravity_pressure
    """
    gravity = gravity_pressure(rho=(rho := fluid.density(T)), dh=dz, g=g)
    _friction = (
        0
        if mdot == 0
        else -Darcy_Weisbach_pressure_by_mdot(
            mdot=mdot,
            rho=rho,
            L=dz,
            Dh=pipe.hydraulic_diameter,
            A=pipe.area,
            f=f(T, Tw, mdot, fluid, pipe),
        )
    )
    inertia = 0 if mdot2 is None else -inertia_pressure(mdot2, dz, pipe.area)
    return _friction + gravity + inertia


@njit
def inertia_pressure(mdot2: KgPerS2, dl: Meter, A: Meter2) -> Pascal:
    r"""A change in flow along a streamline incurs a pressure drop.
    See further explanation in :mod:`~stream.calculations.ideal.inertia`.

    .. math:: \Delta p = \frac{dl}{A}\frac{d\dot{m}}{dt}


    Parameters
    ----------
    mdot2: KgPerS2
         :math:`\ddot{m}`, time derivative of mass flow
    dl: Meter
        Length element along the streamline
    A: Meter2
        Area through which the liquid flows in ``dl``.

    Returns
    -------
    dp: Pascal
        Pressure drop (positive).
    """
    return mdot2 * dl / A


@njit
def static_pressure(pressure: Pascal, mdot: KgPerS, area: Meter2, density: KgPerM3) -> Pascal:
    """Calculates the static pressure from the total (static+dynamic) Bernoulli term.

    Parameters
    ----------
    pressure: Pascal
        The total pressure (static+dynamic)
    mdot: KgPerS
        The mass flow rate
    area: Meter2
        The cross-section area of the pipe
    density: KgPerM3
        The fluid density

    Examples
    --------
    >>> static_pressure(1., 1., 1., 1.)
    0.5
    >>> static_pressure(1., 0., 1., 1.)
    1.0
    >>> static_pressure(1., -1., 1., 1.)
    0.5

    """
    return pressure - 0.5 * (mdot**2) / (density * area**2)
