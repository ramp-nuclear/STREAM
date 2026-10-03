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
   |**ALG**  |Scipy.optimize.root       |:func:`algebraic`,             |
   |         |                          |:func:`quasi_static`           |
   +---------+--------------------------+-------------------------------+

"""

import logging
from typing import Callable, Sequence

import numpy as np
from scikits.odes import dae
from scipy import optimize as opt
from scipy.integrate import solve_ivp

from stream.units import Array1D, Array2D, Functional
from stream.utilities import concat, ignore_warnings

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


def _continuous_mode_dae(solve: Callable, time: Array1D, y0: Array1D, yp0: Array1D) -> tuple[Array2D, Array1D]:
    solution = solve(time, y0, yp0)
    t_end = time[-1]
    t, y, ydot = solution.values
    while (t_stopped := t[-1]) < t_end:
        new_time = concat([t_stopped], time[time > t_stopped])
        logger.info(f"Continuous mode is on, restarted simulation from previous end time {t_stopped:.5f}.")
        try:
            new_solution = solve(new_time, y[-1], ydot[-1])
        except TransientRuntimeError as e:
            logger.critical(e.message)
            if e.t is not None:
                t = concat(t, e.t[1:])
                y = concat(y, e.y[1:])
            break
        t = concat(t, new_solution.values.t[1:])
        y = concat(y, new_solution.values.y[1:])
        ydot = concat(ydot, new_solution.values.ydot[1:])
    return y, t


class AlgRuntimeError(RuntimeError):
    pass


def _root(F: Functional, y0: Array1D, t: float, **options) -> Array1D:
    sol = opt.root(F, y0, (t,), **options)
    if not sol["success"]:
        raise AlgRuntimeError(f"At t={t:.3f}, Root Finding failed with the following message:\n" + sol["message"])
    return sol.x


def algebraic(F: Functional, y0: Array1D, **options) -> Array1D:
    r"""Solving an Algebraic Equation :math:`0=F(y, 0)`

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`, evaluated at :math:`t=0`.
    y0: Array1D
        Initial Guess.
    options:
        Other options to be passed to the ``Scipy.optimize.root`` solver.

    Returns
    -------
    solution: Array1D
        The root of the functional. A failed root find raises :class:`AlgRuntimeError`.
    """
    return _root(F, y0, 0, **options)


def quasi_static(
    F: Functional,
    y0: Array1D,
    time: Sequence[float],
    R: Functional = None,
    **options,
) -> tuple[Array2D, Array1D]:
    r"""Solving an Algebraic Equation :math:`0=F(y, t)` at each of a sequence of time points

    The root at each time point is found from the root at the previous one, so this is a
    quasi-static simulation of the algebraic system.

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`.
    y0: Array1D
        The state at ``time[0]``, which is taken as is.
    time: Sequence[float]
        Time points at which the root is found.
    R: Functional | None
        A function controlling stop events: the simulation stops after the first time point at
        which not all of its values are truthy.
    options:
        Other options to be passed to the ``Scipy.optimize.root`` solver.

    Returns
    -------
    solution: tuple[Array2D, Array1D]
        The solution matrix ([time, variable]) and the times it spans. These are shorter than
        the requested ``time`` when the stop condition ``R`` trips (the stop-triggering row
        included), so callers must take their time axis from the returned times. A failed
        root find raises :class:`AlgRuntimeError`.
    """
    time = np.asarray(time)
    y = np.zeros((len(time), len(y0)))
    y[0] = y0
    for i, t in enumerate(time[1:], start=1):
        y[i] = _root(F, y[i - 1], t, **options)
        if R is not None and not np.all(R(y[i], t)):
            return y[: i + 1], time[: i + 1]
    return y, time


def differential(F: Functional, y0: Array1D, time: Sequence[float], **options) -> tuple[Array2D, Array1D]:
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
    solution: tuple[Array2D, Array1D]
        The solution matrix ([time, variable]) and the times it spans. A solver
        failure raises :class:`TransientRuntimeError` carrying the reached times and
        the partial data, as the DAE path does, instead of returning a truncated
        result.
    """
    time_limits = (time[0], time[-1])
    solution = solve_ivp(lambda t, y: F(y, t), time_limits, y0, t_eval=time, **options)
    data = np.transpose(solution.y)
    if not solution.success:
        reached = solution.t if solution.t is not None and len(solution.t) else None
        raise TransientRuntimeError(reached, data, None, solution.message)
    return data, solution.t
