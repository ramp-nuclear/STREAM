"""C1/C-infinity smoothing primitives with explicit, physically scaled widths.

These primitives replace the hard sign branches, zero-flow divisions and
square-root singularities that make ``F(y, t)`` non-smooth exactly on the
surfaces a LOFA trajectory must cross (``mdot = 0`` at every junction and
component, ``Gr/Re**2 = 1`` in every channel cell, ``dp = 0`` across the
flapper).

Module rule (binding): functions here never read module globals inside jitted
code; widths are always arguments. Callers that want the module default must
read ``smoothing.DEFAULT_MDOT_EPS`` via attribute access at call time -- never
``from stream.smoothing import DEFAULT_MDOT_EPS``, which would freeze the value
at import time and silently ignore a later runtime override (numba freezes such
a global at first jitted call).

This module imports only ``numpy``, ``numba`` and ``stream.units`` so that
``stream.utilities`` may import it without a circular import through
``stream.physical_models``.
"""

import numpy as np
from numba import njit

from stream.units import KgPerS

DEFAULT_MDOT_EPS: KgPerS = 1e-3  # advection direction-blending width


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
    return x / np.sqrt(x * x + eps * eps)


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
    which would reintroduce a zero denominator in :class:`~.Junction` mixing on
    all-outflow Newton iterates. The guarded denominator
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
    s = np.sqrt(x * x + eps * eps)
    return 0.5 * np.where(x >= 0.0, x + s, eps * eps / (s + np.abs(x)))


@njit
def smooth_step(x, x0, x1):
    r"""Cubic smoothstep: ``0`` for ``x <= x0``, ``1`` for ``x >= x1``, and
    :math:`s^2 (3 - 2 s)` between, with :math:`s = (x - x_0)/(x_1 - x_0)`.

    C1 with zero slope at both ends; compact support (exactly 0 / 1 outside).

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
    r"""C1 direction weight in ``[0, 1]``: ``smooth_step(x, -eps, +eps)``.

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


@njit
def smooth_max(a, b, eps):
    r"""C-infinity maximum :math:`\tfrac12 (a + b + \mathrm{smooth\_abs}(a-b))`.

    Bounded above by ``max(a, b) + eps/2``.

    Examples
    --------
    >>> float(round(smooth_max(3.0, -1.0, 1e-3), 6))
    3.0
    """
    return 0.5 * (a + b + smooth_abs(a - b, eps))


@njit
def smooth_min(a, b, eps):
    r"""C-infinity minimum :math:`\tfrac12 (a + b - \mathrm{smooth\_abs}(a-b))`.

    Bounded below by ``min(a, b) - eps/2``.

    Examples
    --------
    >>> float(round(smooth_min(3.0, -1.0, 1e-3), 6))
    -1.0
    """
    return 0.5 * (a + b - smooth_abs(a - b, eps))


@njit
def smooth_signed_sqrt(x, eps):
    r"""C-infinity regularization of :math:`\mathrm{sign}(x)\sqrt{|x|}`:
    :math:`x / (x^2 + \varepsilon^2)^{1/4}`.

    Slope at the origin is ``eps**(-1/2)`` (finite); for ``|x| >> eps`` it
    matches ``sign(x) sqrt(|x|)`` with relative error ``~ eps**2/(4 x**2)``
    (0.25 % at ``|x| = 10 eps``). Odd in ``x`` (no NaN for ``x < 0``).

    Examples
    --------
    >>> smooth_signed_sqrt(0.0, 1e-3)
    0.0
    >>> float(round(smooth_signed_sqrt(100.0, 1e-3), 6))
    10.0
    >>> float(round(smooth_signed_sqrt(-100.0, 1e-3), 6))
    -10.0
    """
    return x / (x * x + eps * eps) ** 0.25
