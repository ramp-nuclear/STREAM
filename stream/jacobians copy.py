r"""
:class:`.Aggregator` solvers can greatly benefit from calculated Jacobians. The following
are some implementations for the ``DAE`` (Differential-Algebraic Equations) form and
the ``ALG`` (Algebraic) form.

The main idea behind these implementations is that the graphical structure of an
:class:`.Aggregator` object can be utilized to deduce the Jacobian sparsity, at least
across :class:`.Calculation` boundaries.
"""
from typing import Callable, Sequence, Protocol, Literal, TypeVar

import numpy as np

from stream import Aggregator, Calculation

__all__ = ["DAE_jacobian", "ALG_jacobian"]

from stream.units import Array1D, Array2D, Value

StepStrategyWithydot = Callable[[Array1D, Array1D], Array1D]
StepStrategy = Callable[[Array1D], Array1D]
T = TypeVar("T", bound=Value)

def _default_step_strategy(y: T, *_) -> T: return 1e-12 + 1e-6 * np.abs(y)

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

def _inner(fy, h, t, yh, *, agr, jac, cj=None):
    associated_calculations = _associated_calculations(agr)
    for j, hj in enumerate(h):
        yh[j] += hj
        fyh = fy.copy()
        for c in associated_calculations[j]:
            fyh[agr.sections[c]] = agr._op("calculate", yh, t, c)
        
        jac[:, j] = (fyh - fy) / hj
        if cj is not None:
            jac[j, j] -= cj * agr.mass[j]
        yh[j] -= hj
    return jac if cj is None else 0

class JacFuncDAE(Protocol):
    """Protocol for the signature one should expect from the returned function."""
    def __call__(self, t, y: np.ndarray, ydot: Value, Gy: Value, cj: Value, J: Array2D) -> Literal[0]:
        """A function that edits the Jacibuian J as its output using the current solver values.
        This signature is required by the DAE solver and not of our choice.
        """
        ...


def DAE_jacobian(agr: Aggregator, step_strategy: StepStrategyWithydot = _default_step_strategy) -> JacFuncDAE:
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

    def _jac_func(t, y: np.ndarray, ydot: Value, Gy: Value, cj: Value, J: Array2D) -> Literal[0]:
        h = step_strategy(y, ydot)
        Fy = Gy + agr.mass * ydot
        _inner(Fy, h, t, y.copy(), agr=agr, jac=J, cj=cj)
        return 0 # This function edits J which is how the data actually comes out.
    
    return _jac_func


class JacFuncALG(Protocol):
    """Protocol for the jacobian calculation function used in the algebraic solver.
    It computes the jacobian at a given time and a given state, with the aggregator baked in."""
    def __call__(self, y, t=0) -> Array2D:
        ...


def ALG_jacobian(agr: Aggregator, step_strategy: StepStrategy = _default_step_strategy) -> JacFuncALG:
    jac = np.empty((len(agr), len(agr)))

    def _jac_func(y: np.ndarray, t=0) -> Array2D:
        h = step_strategy(y)
        Fy = agr.compute(y, t)
        return _inner(Fy, h, t, y.copy, agr=agr, jac=jac)

    return _jac_func
