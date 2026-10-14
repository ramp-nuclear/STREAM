"""Smooth (continuously differentiable or C-infinity) primitives with explicit, physically scaled widths.

These primitives replace hard sign branches and zero-flow divisions that make
``F(y, t)`` non-smooth on surfaces a transient must cross, such as
``mdot = 0`` at a junction or ``Gr/Re**2 = 1`` in a channel cell.

Widths are always arguments; functions here never read module globals inside
jitted code.
"""

import numpy as np
from numba import njit

from stream.units import KgPerS

DEFAULT_MDOT_EPS: KgPerS = 1e-3


@njit
def smooth_abs(x, eps):
    r"""C-infinity absolute value :math:`\sqrt{x^2 + \varepsilon^2}`.

    Equals ``eps`` at ``x = 0``, is ``>= |x|`` and ``<= |x| + eps`` everywhere.

    Examples
    --------
    >>> smooth_abs(0.0, 1e-3)
    0.001
    >>> float(round(smooth_abs(3.0, 4.0), 10))
    5.0
    """
    return np.sqrt(x * x + eps * eps)


@njit
def smooth_sign(x, eps):
    r"""C-infinity sign :math:`x / \sqrt{x^2 + \varepsilon^2}`.

    Slope ``1/eps`` at the origin, tends to ``+-1`` for ``|x| >> eps``.

    Examples
    --------
    >>> smooth_sign(0.0, 1e-3)
    0.0
    >>> float(round(smooth_sign(1e6, 1e-3), 9))
    1.0
    """
    return x / smooth_abs(x, eps)


@njit
def soft_pos(x, eps):
    r"""Strictly positive C-infinity positive-part
    :math:`\tfrac12 (x + \sqrt{x^2 + \varepsilon^2})`.

    ``soft_pos(0) = eps/2``; ``> 0`` everywhere (including ``x <= -eps``, where
    it decays as ``eps**2 / (4|x|)``); tends to ``x`` for ``x >> eps``; its
    derivative lies in ``(0, 1)``.

    The ``x < 0`` branch uses the algebraically-identical, cancellation-free
    form ``eps**2 / (sqrt(x**2 + eps**2) - x)``: the direct ``x + sqrt(...)``
    underflows to exactly ``0`` for large negative ``x`` (e.g. ``x = -1e6``),
    which would produce a zero denominator when the result is used as a mixing
    weight. The guarded denominator
    ``sqrt(x**2 + eps**2) + |x|`` equals ``sqrt(...) - x`` on that branch and is
    always ``>= eps > 0``.

    Examples
    --------
    >>> soft_pos(0.0, 1e-3)
    0.0005
    >>> float(round(soft_pos(1000.0, 1e-3), 6))
    1000.0
    >>> soft_pos(-1e6, 1e-3) > 0.0
    True
    """
    s = smooth_abs(x, eps)
    return 0.5 * np.where(x >= 0.0, x + s, eps * eps / (s + np.abs(x)))


@njit
def smooth_step(x, x0, x1):
    r"""Cubic smoothstep: ``0`` for ``x <= x0``, ``1`` for ``x >= x1``, and
    :math:`s^2 (3 - 2 s)` between, with :math:`s = (x - x_0)/(x_1 - x_0)`.

    Continuously differentiable with zero slope at both ends; compact support (exactly 0 / 1 outside).

    Examples
    --------
    >>> smooth_step(-1.0, 0.0, 1.0)
    0.0
    >>> smooth_step(0.5, 0.0, 1.0)
    0.5
    >>> smooth_step(2.0, 0.0, 1.0)
    1.0
    """
    s = np.minimum(1.0, np.maximum(0.0, (x - x0) / (x1 - x0)))
    return s * s * (3.0 - 2.0 * s)


@njit
def smooth_pos_weight(x, eps):
    r"""Continuously differentiable direction weight in ``[0, 1]``: ``smooth_step(x, -eps, +eps)``.

    Exactly 0 / 1 outside the band (compact support); ``1/2`` at ``x = 0``;
    maximum slope ``0.75 / eps``.

    Examples
    --------
    >>> smooth_pos_weight(-1.0, 1e-3)
    0.0
    >>> smooth_pos_weight(0.0, 1e-3)
    0.5
    >>> smooth_pos_weight(1.0, 1e-3)
    1.0
    """
    return smooth_step(x, -eps, eps)
