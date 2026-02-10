r"""
The :class:`~stream.aggregator.Aggregator` defines a functional :math:`\vec{F}(\vec{y},t)` which can be
utilized to solve several problems. Below are several backends used for different
classes of problems.

Backends
--------
Each ``eq_type`` in :meth:`~stream.aggregator.Aggregator.solve` relies on a different backend solver,
for which different ``options`` are relevant. For the sake of brevity, these options are not specified here,
and the user is encouraged to look at each backend, as well as the defaults set in :mod:`~stream.aggregator`
to achieve a higher level of proficiency.

.. table::
   :name: Backend Solvers

   +---------+--------------------------+-------------------------------+
   |eq_type  |Solver                    |Function Name                  |
   +=========+==========================+===============================+
   |**ODE**  |Scipy.integrate.solve_ivp |:func:`differential`           |
   +---------+--------------------------+-------------------------------+
   |**DAE**  |Scikits.odes.ida          |:func:`differential_algebraic` |
   +---------+--------------------------+-------------------------------+
   |**ALG**  |Scipy.optimize.root       |:func:`algebraic`              |
   +---------+--------------------------+-------------------------------+

"""
import logging
from typing import Callable, Sequence

import numpy as np
from scikits.odes import dae
from scipy import optimize as opt
from scipy.integrate import solve_ivp

from stream.units import Array, Array1D, Array2D, Functional
from stream.utilities import concat, ignore_warnings, STREAM_DEBUG

logger = logging.getLogger("stream.aggregator")


class TransientRuntimeError(RuntimeError):
    """RuntimeError which occurred during a transient simulation,
    mostly because of solver convergence problems.

    Data for debugging the error may be extracted by::

        try:
            # error raising simulation
        except TransientRuntimeError as e:
            t, y, ydot = e.t, e.y, e.ydot
            results = Solution(t, y)
            raise e
    """

    def __init__(self, t, y, ydot, message: str, *args):
        if t is not None:
            message = f"At t = {t[-1]:.5f}: " + message
        super().__init__(message, *args)
        self.message = message
        self.t = t
        self.y = y
        self.ydot = ydot


def differential_algebraic(
    F: Functional,
    mass: Array1D,
    y0: Array1D,
    time: Sequence[float],
    yp0: Array1D | None = None,
    R: Functional | None = None,
    continuous: bool = False,
    **options,
) -> tuple[Array2D, Array1D]:
    r"""Solving a Differential Algebraic Equation (DAE) :math:`M\dot{y}=F(y, t)`

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`.
    mass: Array1D
        Mass vector, determining which index is differential and which is algebraic.
    y0: Array1D
        Initial values.
    time: Sequence[float]
        Time points for which the simulation values should be returned.
    yp0: Array1D | None
        Initial value derivatives. If None, considered 0. However, when
        ``compute_initcond = 'yp0'`` is set (as is default), this vector is deduced
        from ``y0``.
    R: Functional | None
        A function controlling simulation stop events. Simulation continues unless a
        ``False`` valued index is returned.
    continuous: bool
        If ``True``, simulation continues after ``R`` has yielded a non-True value,
        whereas the simulation stops in that case for ``False``. An added behavior
        is that using ``continuous=True`` forces a restart of the simulation, such that initial
        steps are much smaller, and controlled by ``first_step_size``
    options:
        Other options to be passed to the ``scikits.odes`` solver.


    Returns
    -------
    solution: tuple[Array2D, Array1D]
        The solution matrix: [time, variable], and the vector of times 
        in which it was calculated.
    """

    setup = solve, time, y0, yp0 = _dae_setup(F, mass, y0, time, yp0, R, **options)
    if continuous:
        return _continuous_mode_dae(*setup)
    solution = solve(time, y0, yp0)
    return solution.values.y, solution.values.t


def _ida_post_solution(solution):
    if solution.flag < 0:
        raise TransientRuntimeError(*solution.values, solution.message)


def _dae_setup(
    F: Functional,
    mass: Array1D,
    y0: Array1D,
    time: Sequence[float],
    yp0: Array1D | None = None,
    R: Functional | None = None,
    **options,
) -> tuple[Callable, Array1D, Array1D, Array1D]:
    def residues(t, y, ydot, result):
        result[:] = F(y, t) - mass * ydot

    def root(t, y, _, g, __):
        g[:] = R(y, t)

    time = np.asarray(time)
    yp0 = yp0 if yp0 is not None else np.zeros(len(y0))
    root = None if R is None else root
    defaults = dict(
        compute_initcond="yp0",
        old_api=False,
        algebraic_vars_idx=np.flatnonzero(1 - mass),
        implementation="serial",
        rootfn=root,
    )
    options = defaults | options
    solver = dae("ida", residues, **options)

    def solve(time, y0, yp0):
        with ignore_warnings(DeprecationWarning):
            sol_ = solver.solve(time, y0, yp0)
        _ida_post_solution(sol_)
        return sol_

    return solve, time, y0, yp0


def _continuous_mode_dae(solve: Callable, time: Array1D, y0: Array1D, yp0: Array1D
                         ) -> tuple[Array2D, Array1D]:
    solution = solve(time, y0, yp0)
    t_end = time[-1]
    t, y, ydot = solution.values
    while (t_stopped := t[-1]) < t_end:
        new_time = concat([t_stopped], time[time > t_stopped])
        logger.info(
            "Continuous mode is on, restarted simulation from previous end time"
            f" {t_stopped:.5f}."
        )
        try:
            new_solution = solve(new_time, y[-1], ydot[-1])
        except TransientRuntimeError as e:
            logger.critical(e.message)
            t = concat(t, e.t)
            y = concat(y, e.y)
            break
        t = concat(t, new_solution.values.t[1:])
        y = concat(y, new_solution.values.y[1:])
        ydot = concat(ydot, new_solution.values.ydot[1:])
    return y, t


class AlgRuntimeError(RuntimeError):
    pass


def algebraic(
    F: Functional,
    y0: Array1D,
    time: Sequence[float] | None = None,
    R: Functional = None,
    **options,
) -> Array:
    r"""Solving an Algebraic Equation :math:`0=F(y, t)`

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`.
    y0: Array1D
        Initial Guess.
    time: Sequence[float] | None
        If ``None``, the root of the functional is found. Else, the root is found at the specified
        time points, given sequential initial guesses. It is a quasi-static simulation,
        if one wills it. The first vector is then the initial guess.
    R: Functional | None
        A function controlling transient simulation stop events.
    options:
        Other options to be passed to the ``Scipy.optimize.root`` solver.


    Returns
    -------
    solution: Array
        The solution matrix at requested times: [time, variable].
    """
    def _solve(_vec, _t):
        _sol = opt.root(F, _vec, (_t,), **options)
        if not _sol["success"]:
            timestr = f"At t={_t:.3f}, " if _t is not None else ""
            raise AlgRuntimeError(
                f"{timestr}Root Finding failed with the following message:\n" + _sol["message"]
            )
        return _sol.x

    if time is not None:
        y = np.zeros((len(time), len(y0)))
        y[0] = y0
        for i, t in enumerate(time[1:]):
            sol = _solve(y[i], t)
            if R is not None and not np.all(R(sol, t)):
                return y[: i + 1]
            y[i + 1] = sol
        return y
    else:
        return _solve(y0, 0)


def differential(F: Functional, y0: Array1D, time: Sequence[float], **options) -> Array2D:
    r"""Solving an Ordinary Differential Equation (ODE) :math:`\dot{y}=F(y, t)`

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`.
    y0: Array1D
        Initial values.
    time: Sequence[float]
        Time points for which the simulation values should be returned.
    options:
        Other options to be passed to the ``scipy.integrate.solve_ivp`` solver.

    Returns
    -------
    solution: Array2D
        The solution matrix at requested times: [time, variable].
    """
    time_limits = (time[0], time[-1])
    solution = solve_ivp(lambda t, y: F(y, t), time_limits, y0, t_eval=time, **options)
    return np.transpose(solution.y)
