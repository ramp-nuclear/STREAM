r"""Single-phase liquid discharge through a hole, orifice or broken pipe stub.

The correlations here describe a subcooled liquid jet leaving a vessel or a severed pipe
into a lower-pressure space, and the gravity draining that follows. They are valid while
the throat stays liquid: the static pressure at the vena contracta must remain above the
local saturation pressure, otherwise the flow flashes and a critical-flow model is needed
instead.

References
----------
    .. [#LDM] A. Lichtarowicz, R. K. Duggins, E. Markland, "Discharge coefficients for
       incompressible non-cavitating flow through long orifices",
       Journal of Mechanical Engineering Science 7(2), 1965.
"""

import numpy as np
from numba import njit

from stream.units import (
    KgPerM3,
    KgPerS,
    Meter,
    Meter2,
    Pascal,
    Second,
    Value,
    g as local_gravity,
)

DISCHARGE_CD: dict[str, float] = {
    "sharp": 0.61,
    "rounded": 0.98,
    "short_tube": 0.81,
    "borda": 0.51,
    "pipe_stub": 0.6,
}


def discharge_cd(name: str) -> float:
    r"""Look up a fully-turbulent discharge coefficient :math:`C_d = C_c C_v` by hole geometry.

    .. list-table::
        :widths: 20, 15, 65

        * - **sharp**
          - 0.61
          - Sharp-edged thin plate; the coefficient is set by the vena contracta
            (:math:`C_c \approx 0.611`).
        * - **rounded**
          - 0.98
          - Rounded, bellmouth or nozzle inlet: no contraction, friction only.
        * - **short_tube**
          - 0.81
          - Thick hole with :math:`L/d` of 2 to 4, where the jet reattaches and fills the bore.
        * - **borda**
          - 0.51
          - Re-entrant tube protruding into the vessel: maximal contraction, no reattachment.
        * - **pipe_stub**
          - 0.6
          - Clean severed pipe end, i.e. a sharp entrance into the remaining stub.

    These values hold for a throat Reynolds number above roughly :math:`10^4`; below that use
    :func:`lichtarowicz_cd`.

    Parameters
    ----------
    name: str
        Geometry key, one of the names in :data:`DISCHARGE_CD`.

    Returns
    -------
    cd: float
        Discharge coefficient.

    Examples
    --------
    >>> discharge_cd("sharp")
    0.61
    >>> discharge_cd("borda")
    0.51
    """
    try:
        return DISCHARGE_CD[name]
    except KeyError as e:
        raise ValueError(f"{name=} not found in {list(DISCHARGE_CD.keys())}") from e


@njit
def lichtarowicz_cd(re: Value, L_over_d: Value) -> Value:
    r"""The discharge coefficient of a parallel-bore orifice at finite Reynolds number [#LDM]_.

    .. math::

        C_{du} = 0.827 - 0.0085 \frac{L}{d}

    .. math::

        \frac{1}{C_d} = \frac{1}{C_{du}} + \frac{20}{\text{Re}}\left(1 + 2.25\frac{L}{d}\right)
        - \frac{0.005 L/d}{1 + 7.5\left(\log_{10}\left(1.5\cdot 10^{-4}\text{Re}\right)\right)^2}

    :math:`C_{du}` is the ultimate (high Reynolds) value, which :math:`C_d` approaches from below
    as the viscous :math:`20/\text{Re}` term dies out. The correlation was fitted for
    :math:`10 \le \text{Re} \le 2\cdot 10^4` and :math:`L/d \le 10`; outside that box it is an
    extrapolation. For a thin plate rather than a bore, use ``discharge_cd("sharp")``.

    Parameters
    ----------
    re: Value
        Reynolds number at the throat, built on the bore diameter.
    L_over_d: Value
        Bore length over bore diameter.

    Returns
    -------
    cd: Value
        Discharge coefficient.

    See Also
    --------
    discharge_cd

    Examples
    --------
    >>> lichtarowicz_cd(2e4, 2.0)
    0.8088165976866
    >>> lichtarowicz_cd(100.0, 2.0)
    0.4284155081876958
    """
    cdu = 0.827 - 0.0085 * L_over_d
    inverse = (
        1.0 / cdu
        + (20.0 / re) * (1.0 + 2.25 * L_over_d)
        - 0.005 * L_over_d / (1.0 + 7.5 * np.log10(0.00015 * re) ** 2)
    )
    return 1.0 / inverse


@njit
def stub_discharge_mdot(dp: Pascal, rho: KgPerM3, area: Meter2, k_total: Value) -> KgPerS:
    r"""Mass flow discharged through a resistance sum, e.g. a broken pipe stub.

    .. math:: \dot{m} = A\sqrt{\frac{2\rho\Delta p}{K_\text{total}}}

    :math:`K_\text{total}` collects every loss between the intact system and the break:
    entrance, the stub's own friction :math:`fL/d`, and exit. For a bare hole,
    :math:`K_\text{total} = 1/C_d^2` recovers the Torricelli form. Friction takes over from the
    entrance loss at roughly :math:`L/d > 40`.

    This is a bare square root, undefined for ``dp < 0`` and with an infinite derivative at
    ``dp = 0``. Callers that hand the result to a solver should regularize it, e.g. with
    :func:`stream.smoothing.smooth_signed_sqrt`.

    Parameters
    ----------
    dp: Pascal
        Driving pressure difference, upstream stagnation minus back pressure.
    rho: KgPerM3
        Liquid density.
    area: Meter2
        Break flow area.
    k_total: Value
        Sum of the loss coefficients along the discharge path.

    Returns
    -------
    mdot: KgPerS
        Discharged mass flow rate.

    Examples
    --------
    >>> stub_discharge_mdot(1e5, 1000.0, 1e-4, 1.0)
    1.4142135623730951
    >>> stub_discharge_mdot(0.0, 1000.0, 1e-4, 2.5)
    0.0
    """
    return area * np.sqrt(2.0 * rho * dp / k_total)


@njit
def drain_time(h0: Meter, h1: Meter, area_tank: Meter2, area_hole: Meter2, cd: Value) -> Second:
    r"""Time for a constant cross-section tank to drain by gravity from level ``h0`` to ``h1``.

    .. math:: t = \frac{A_\text{tank}}{C_d A_\text{hole}}\sqrt{\frac{2}{g}}
              \left(\sqrt{h_0} - \sqrt{h_1}\right)

    Both levels are measured above the hole, and the discharge is quasi-steady and unsubmerged.
    A tank whose cross-section varies with level needs the underlying ODE
    :math:`A_\text{tank}(h)\,dh/dt = -C_d A_\text{hole}\sqrt{2gh}` integrated instead.

    Parameters
    ----------
    h0: Meter
        Initial liquid level above the hole.
    h1: Meter
        Final liquid level above the hole.
    area_tank: Meter2
        Tank free-surface area.
    area_hole: Meter2
        Hole area.
    cd: Value
        Discharge coefficient.

    Returns
    -------
    t: Second
        Elapsed draining time.

    See Also
    --------
    drain_level

    Examples
    --------
    >>> drain_time(4.0, 1.0, 2.0, 5e-4, 0.61)
    2961.3164311592627
    >>> drain_time(4.0, 4.0, 2.0, 5e-4, 0.61)
    0.0
    """
    return (area_tank / (cd * area_hole)) * np.sqrt(2.0 / local_gravity) * (np.sqrt(h0) - np.sqrt(h1))


@njit
def drain_level(t: Second, h0: Meter, area_tank: Meter2, area_hole: Meter2, cd: Value) -> Meter:
    r"""Liquid level above the hole after gravity draining for a time ``t``, the inverse of
    :func:`drain_time`.

    .. math:: h(t) = \left(\sqrt{h_0}
              - \frac{C_d A_\text{hole}}{A_\text{tank}}\sqrt{\frac{g}{2}}\,t\right)^2

    The square root is floored at zero, so the level stays empty once the tank has run dry
    instead of following the parabola back up.

    Parameters
    ----------
    t: Second
        Time since the level was ``h0``.
    h0: Meter
        Initial liquid level above the hole.
    area_tank: Meter2
        Tank free-surface area.
    area_hole: Meter2
        Hole area.
    cd: Value
        Discharge coefficient.

    Returns
    -------
    h: Meter
        Liquid level above the hole.

    See Also
    --------
    drain_time

    Examples
    --------
    >>> drain_level(0.0, 4.0, 2.0, 5e-4, 0.61)
    4.0
    >>> drain_level(2961.3164311592627, 4.0, 2.0, 5e-4, 0.61)
    1.0
    >>> drain_level(1e5, 4.0, 2.0, 5e-4, 0.61)
    0.0
    """
    root = np.sqrt(h0) - (cd * area_hole / area_tank) * np.sqrt(local_gravity / 2.0) * t
    return np.maximum(root, 0.0) ** 2
