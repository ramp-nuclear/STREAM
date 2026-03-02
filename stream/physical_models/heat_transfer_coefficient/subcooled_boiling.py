import numpy as np
from numba import njit

from stream.physical_models.dimensionless import Pr, flow_regimes
from stream.substances import Liquid
from stream.units import Celsius, Value, WPerM2, g
from stream.utilities import lin_interp


def Bergles_Rohsenhow_SCB_heat_flux(
    T_wall: Celsius, coolant: Liquid, n: float = 1.26, csf: float = 0.011, **_
) -> WPerM2:
    r"""
    Given the wall initial boiling temperature, the heat flux for subcooled
    boiling may be given by [#BR]_

    .. math:: q_\text{SCB}''(T_{wall})
        = \left[\frac{c_p(T_{wall,inc}-T_{sat})}{h_{fg}}
        \frac{\text{Pr}^{-n}}{C_{sf}}\right]^{1/0.33}
        \frac{\mu_l h_{fg}}{\sqrt{\sigma/(g(\rho_l - \rho_v))}}

    Where the coolant properties are those in saturation temperature.

    Parameters
    ----------
    T_wall: Celsius
        Wall temperatures
    coolant: Liquid
        Coolant properties
    n, csf: float
        Dimensionless params. See equation above.

    Returns
    -------
    q: WPerM2
        Appropriate heat flux

    Examples
    --------

    >>> from stream.substances import light_water
    >>> liquid = light_water.to_properties(10.)
    >>> Bergles_Rohsenhow_SCB_heat_flux(liquid.sat_temperature, liquid)
    0.0
    >>> Bergles_Rohsenhow_SCB_heat_flux(120., liquid, n=np.inf)
    0.0
    >>> Bergles_Rohsenhow_SCB_heat_flux(120., liquid, csf=np.inf)
    0.0
    """
    return _BR_q_scb_inner(
        cp=coolant.specific_heat,
        T_wall=T_wall,
        T_sat=coolant.sat_temperature,
        hfg=coolant.latent_heat,
        mu=coolant.viscosity,
        k=coolant.conductivity,
        n=n,
        csf=csf,
        rho=coolant.density,
        steam_rho=coolant.vapor_density,
        sigma=coolant.surface_tension,
    )


@njit
def _BR_q_scb_inner(cp, T_wall, T_sat, hfg, mu, k, n, csf, rho, steam_rho, sigma):
    a = cp * (T_wall - T_sat) / hfg
    b = Pr(cp, mu, k) ** (-n) / csf
    d_rho = rho - steam_rho
    c = mu * hfg / np.sqrt(sigma / (g * d_rho))
    return c * (a * b) ** (1.0 / 0.33)


@njit
def McAdams_SCB_heat_flux(T_sat: Celsius, T_wall: Celsius) -> WPerM2:
    r"""
    A correlation for heat flux at the Subcooled Boiling (SCB) regime. [#tecdoc]_

    Parameters
    ----------
    T_sat: Celsius
        Coolant saturation temperature.
    T_wall: Celsius
        Wall (or interface) temperature.

    Returns
    -------
    q: WPerM2
        heat flux at the Subcooled Boiling (SCB) regime

    Examples
    --------
    >>> McAdams_SCB_heat_flux(T_sat=100., T_wall=100.)
    0.0
    >>> McAdams_SCB_heat_flux(T_sat=100., T_wall=120.)
    237730.12702161702

    References
    ----------
    .. [#tecdoc] IAEA TECDOC-233
    """
    return 2.26 * (T_wall - T_sat) ** 3.86


def regime_dependent_q_scb(
    T_wall: Celsius,
    coolant: Liquid,
    re: np.ndarray,
    re_bounds: tuple[float, float],
    **_,
) -> WPerM2:
    r"""A flow-regime-dependent Subcooled Boiling heat flux function.

    Different flow regimes (different ``re`` values, partitioned by ``re_bounds``)
    use different subcooled boiling heat flux `correlations`:

    .. list-table::
        :widths: 20,80

        * - **Turbulent**
          - :func:`McAdams <McAdams_SCB_heat_flux>`
        * - **Laminar**
          - :func:`Bergles-Rosenhow <Bergles_Rohsenhow_SCB_heat_flux>`
        * - **Interim**
          - Interpolation between the limit values, see :func:`.lin_interp`

    Parameters
    ----------
    T_wall: Celsius
        Wall temperature
    coolant: Liquid
        Coolant properties
    re: Value
        Reynolds No., see :func:`~stream.physical_models.dimensionless.Re`
    re_bounds: tuple[float, float]
        Regime boundaries

    Returns
    -------
    q_SCB: WPerM2
        Subcooled boiling heat flux
    """
    q = np.empty(len(T_wall))
    T_sat = coolant.sat_temperature

    lam, inter, turb = flow_regimes(re, re_bounds)

    q[lam] = Bergles_Rohsenhow_SCB_heat_flux(T_wall[lam], coolant[lam])
    q[inter] = lin_interp(
        x1=re_bounds[0],
        x2=re_bounds[1],
        x=re[inter],
        y2=McAdams_SCB_heat_flux(T_sat[inter], T_wall[inter]),
        y1=Bergles_Rohsenhow_SCB_heat_flux(T_wall[inter], coolant[inter]),
    )
    q[turb] = McAdams_SCB_heat_flux(T_sat[turb], T_wall[turb])

    return q


@njit
def Bergles_Rohsenhow_partial_SCB(q_spl: WPerM2, q_scb: WPerM2, q_scb_inc: WPerM2) -> Value:
    r"""
    Between INC and the fully developed boiling curve, the heat flux for some T_wall is deduced by using the SPL curve
    and the SCB curve.

    .. math:: q_{partial,scb}''=q_{spl}''\left[1+\left\{ \frac{q_{scb}''}
        {q_{spl}''}\left(1-\frac{q_{scb,inc}''}{q_{scb}''}\right)\right\}
        ^{2}\right]^{0.5}

    Parameters
    ----------
    q_spl: WPerM2
        Single Phase heat flux at T_wall.
    q_scb: WPerM2
        SCB heat flux at T_wall.
    q_scb_inc: WPerM2
        SCB heat flux at INC temperature.

    Returns
    -------
    multiplicative_factor: Value
        Correction for the heat transfer coefficient (h) or equivalently the heat flux (q'').

    Examples
    --------
    >>> Bergles_Rohsenhow_partial_SCB(1., 1., 1.)
    1.0
    """
    return (1 + (q_scb / q_spl * (1 - q_scb_inc / q_scb)) ** 2) ** 0.5
