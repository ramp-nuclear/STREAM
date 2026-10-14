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

Beyond the ``eq_type`` backends above, :func:`scaled_newton` provides a
globalized steady solve, an equilibrated, Armijo-damped Newton, for root finds
from far guesses where :func:`algebraic`'s ``hybr`` stalls.
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


def _equilibrate(J: Array2D) -> tuple[Array2D, Array1D, Array1D]:
    r"""Row-then-column infinity-norm equilibration of a matrix (single pass).

    Scales ``J`` by dividing each row by its infinity norm and then each of the
    row-scaled columns by its infinity norm, so the returned ``J_eq`` has unit
    largest row and column magnitudes. Both scale vectors are floored at
    ``1e-30`` so a vanishing row/column is never divided by zero (and never
    amplified).

    Parameters
    ----------
    J : Array2D
        The matrix to equilibrate (typically a Jacobian).

    Returns
    -------
    J_eq : Array2D
        The equilibrated matrix, ``J / (rs[:, None] * cs[None, :])``.
    rs : Array1D
        Row scales (row infinity norms of ``J``, floored).
    cs : Array1D
        Column scales (column infinity norms of the row-scaled ``J``, floored).

    Notes
    -----
    Single-pass (not Ruiz-iterated). Row scaling scales the residual and column
    scaling scales the variables, so a linear solve on ``J_eq`` recovers the
    original solution as ``dy = dy_eq / cs`` from a right-hand side ``-(F / rs)``.
    """
    rs = np.maximum(np.max(np.abs(J), axis=1), 1e-30)
    Jr = J / rs[:, None]
    cs = np.maximum(np.max(np.abs(Jr), axis=0), 1e-30)
    return Jr / cs[None, :], rs, cs


def _scaled_newton_core(
    F: Functional,
    y0: Array1D,
    jac: Callable[[Array1D, float], Array2D],
    *,
    t: float,
    tol: float,
    maxit: int,
    tikhonov: float,
) -> tuple[Array1D, bool, bool, float, int]:
    """Inner loop of :func:`scaled_newton`.

    Runs equilibrated, Tikhonov-regularized, Armijo-damped Newton iterations and
    reports status flags instead of raising. See :func:`scaled_newton` for the
    references.

    Returns
    -------
    y : Array1D
        The last iterate reached.
    converged : bool
        ``True`` if ``||F(y, t)|| < tol`` was reached.
    exhausted : bool
        ``True`` if an Armijo search found no descent (``a`` fell to ``2**-30``
        with no merit decrease).
    nF : float
        ``||F(y, t)||`` at the returned iterate.
    it : int
        Number of completed iterations.
    """
    y = np.array(y0, float)
    for it in range(maxit):
        Fy = F(y, t)
        nF = float(np.linalg.norm(Fy))
        if nF < tol:
            return y, True, False, nF, it
        J = np.array(jac(y, t))  # copy: ALG_jacobian returns a shared closure buffer
        Jeq, rs, cs = _equilibrate(J)
        A, b = Jeq + tikhonov * np.eye(len(Jeq)), -(Fy / rs)
        try:
            dy = np.linalg.solve(A, b) / cs
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(A, b)[0] / cs
        a = 1.0
        merit = float(np.linalg.norm(F(y + a * dy, t)))
        while merit >= nF and a > 2**-30:
            a *= 0.5
            merit = float(np.linalg.norm(F(y + a * dy, t)))
        if merit >= nF:
            return y, False, True, nF, it
        y = y + a * dy
    nF = float(np.linalg.norm(F(y, t)))
    return y, nF < tol, False, nF, maxit


def scaled_newton(
    F: Functional,
    y0: Array1D,
    jac: Callable[[Array1D, float], Array2D],
    *,
    t: float = 0.0,
    tol: float = 1e-6,
    maxit: int = 100,
    tikhonov: float = 1e-10,
) -> Array1D:
    r"""Globalized damped Newton steady solve with equilibrated linear solves.

    Each iteration equilibrates the Jacobian (row-then-column infinity norm)
    [#Higham]_, solves the Tikhonov-regularized [#Tikhonov]_, equilibrated
    linear system for a Newton step, and backtracks the step with a merit line
    search [#NocedalWright]_ on ``||F(y + a*dy, t)||``. It is meant
    for steady solves from far guesses where the globalization in
    :func:`scipy.optimize.root` (``hybr``) stalls; once the damped steps bring
    the iterate near a root, full Newton steps resume.

    Parameters
    ----------
    F : Functional
        Residual ``F(y, t) -> Array1D`` (the aggregator ``compute`` signature).
        The root ``F = 0`` is sought.
    y0 : Array1D
        Initial guess.
    jac : Callable
        Analytic/approximate Jacobian ``jac(y, t) -> Array2D`` of ``F``.
    t : float, optional
        Time argument threaded into ``F`` and ``jac`` (default ``0.0``).
    tol : float, optional
        Convergence tolerance on ``||F||`` (default ``1e-6``).
    maxit : int, optional
        Maximum Newton iterations (default ``100``).
    tikhonov : float, optional
        Tikhonov regularization [#Tikhonov]_: a small multiple of the identity
        added to the *equilibrated* (``O(1)``) Jacobian so a rank-deficient step
        stays finite (default ``1e-10``). If the shifted matrix is still
        singular, the step is the minimum-norm least-squares solution.

    Returns
    -------
    Array1D
        The converged state ``y`` with ``||F(y, t)|| < tol``.

    Raises
    ------
    AlgRuntimeError
        If the backtracking line search [#NocedalWright]_ finds no descent
        (raised immediately, not spun out to ``maxit``) or ``maxit`` is
        exhausted. The best iterate reached is attached as the ``y`` attribute
        for diagnosis.

    Notes
    -----
    The equilibration serves conditioning diagnostics and insurance for
    badly-scaled systems (making ``cond``/rank meaningful and giving Tikhonov a
    sane ``O(1)`` metric); the *damping* provides the convergence. The Tikhonov
    term is a defensive default with no measurable effect on well-posed systems.

    References
    ----------
    .. [#NocedalWright] J. Nocedal, S. J. Wright, "Numerical Optimization", 2nd
       edition, Springer, 2006, chapter 3 (backtracking line search with the
       Armijo sufficient decrease condition, Algorithm 3.1).
    .. [#Tikhonov] A. N. Tikhonov, V. Y. Arsenin, "Solutions of Ill-Posed
       Problems", Winston, Washington DC, 1977.
    .. [#Higham] N. J. Higham, "Accuracy and Stability of Numerical Algorithms",
       2nd edition, SIAM, 2002, section 7.3 (row and column scaling); LAPACK
       routine ``dgeequ``.
    """
    y, converged, exhausted, nF, it = _scaled_newton_core(
        F, y0, jac, t=t, tol=tol, maxit=maxit, tikhonov=tikhonov
    )
    if converged:
        return y
    reason = "Armijo line search found no descent direction" if exhausted else f"exceeded maxit={maxit}"
    err = AlgRuntimeError(f"scaled_newton failed at iteration {it} ({reason}); ||F||={nF:.3e}.")
    err.y = y
    raise err


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
