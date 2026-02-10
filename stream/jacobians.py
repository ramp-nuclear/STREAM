r"""
:class:`.Aggregator` solvers can greatly benefit from calculated Jacobians. The following
are some implementations for the ``DAE`` (Differential-Algebraic Equations) form and
the ``ALG`` (Algebraic) form.

The main idea behind these implementations is that the graphical structure of an
:class:`.Aggregator` object can be utilized to deduce the Jacobian sparsity, at least
across :class:`.Calculation` boundaries.
"""
from typing import Callable, Sequence

import numpy as np
from stream import Aggregator, Calculation

__all__ = ["DAE_jacobian", "ALG_jacobian"]


def DAE_jacobian(agr: Aggregator, step_strategy=None) -> Callable:
    r"""
    A function for creating a one-sided approximation of the Jacobian of
    an :class:`Aggregator` functional. The Jacobian is defined as follows:

    .. math:: J_{ij} = \frac{dF_i}{dy_j}
        \approx \frac{F_i(y + h_j) - F_i(y)}{h_j}

    Where :math:`h_j` is the ``jth`` step size and :math:`y + h_j` simply means
    the ``jth`` component is :math:`y_j + h_j`. This is however a bit
    simplistic, since the actual required Jacobian is of the full functional:

    .. math:: G(t, y, \dot{y}) = F(y, t) - M\dot{y}

    Meaning the Jacobian is actually

    .. math:: \frac{dG}{dy}=\frac{\partial G}{\partial\dot{y}}
        \frac{\partial\dot{y}}{\partial y} + \frac{\partial G}{\partial y}

    Since :math:`\dot{y}` is approximated linearly,
    :math:`\partial_y\dot{y}\equiv\sigma` does not depend
    on the functions considered above, only on the approximation scheme and
    time step.

    In our case, :math:`\partial_{\dot{y}_j} G_i=-M_{ij} = -m_i \delta_{i,j}`.

    Parameters
    ----------
    agr : Aggregator
        The Aggregator whose functional is desired
    step_strategy: Callable or None
        A function for the step vector :math:`h = f(y, \dot{y})`

    Returns
    -------
    J_func: Callable
        A function whose signature is
        :math:`J(t, y, \dot{y}, G(t, y, \dot{y}), \sigma, \text{result})`,
        matching the required SUNDIALS format.
    """
    step_strategy = step_strategy or (lambda y, yd: 1e-12 + 1e-6 * np.abs(y))
    associated_calculations = _associated_calculations(agr)

    def jac_func(t, y, ydot, Gy, cj, J):
        h = step_strategy(y, ydot)
        Fy = Gy + agr.mass * ydot
        yh = y.copy()
        for j, hj in enumerate(h):
            yh[j] += hj
            Fyh = Fy.copy()
            for c in associated_calculations[j]:
                Fyh[agr.sections[c]] = agr._op("calculate", yh, t, c)

            J[:, j] = (Fyh - Fy) / hj
            J[j, j] -= cj * agr.mass[j]
            yh[j] -= hj
        return 0

    return jac_func


def _associated_calculations(agr: Aggregator) -> dict[int, Sequence[Calculation]]:
    indices = np.arange(len(agr))
    a = {i: [calculation]
         for calculation, section in agr.sections.items()
         for i in indices[section]}
    for v, d in agr.external.items():
        for name, ud in d.items():
            for u, place in ud.items():
                for i in np.atleast_1d(indices[place]):
                    a[i].append(v)
    return a


def ALG_jacobian(agr: Aggregator, step_strategy=None) -> Callable:
    step_strategy = step_strategy or (lambda y: 1e-12 + 1e-6 * np.abs(y))
    associated_calculations = _associated_calculations(agr)
    J = np.empty((len(agr), len(agr)))

    def jac_func(y, t=0):
        h = step_strategy(y)
        yh = y.copy()
        Fy = agr.compute(y, t)
        for j, hj in enumerate(h):
            yh[j] += hj
            Fyh = Fy.copy()
            for c in associated_calculations[j]:
                Fyh[agr.sections[c]] = agr._op("calculate", yh, t, c)

            J[:, j] = (Fyh - Fy) / hj
            yh[j] -= hj
        return J

    return jac_func
