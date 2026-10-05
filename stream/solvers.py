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

Beyond the ``eq_type`` backends above, :func:`scaled_newton` and
:func:`pseudo_transient` provide globalized steady-solve machinery — an
equilibrated, Armijo-damped Newton and an implicit-Euler pseudo-transient
continuation fallback — for stubborn root finds from far guesses where
:func:`algebraic`'s ``hybr`` stalls.
"""

import logging
from typing import Callable, Sequence

import numpy as np
from scikits.odes import dae
from scipy import optimize as opt
from scipy.integrate import solve_ivp

from stream.errors import StreamError, hint_block
from stream.units import Array, Array1D, Array2D, Functional
from stream.utilities import STREAM_DEBUG, concat, ignore_warnings

logger = logging.getLogger("stream.aggregator")


# IDA status code -> (IDA_* symbol, plain-language meaning); undescribed flags carry an empty meaning.
_IDA_STATUS: dict[int, tuple[str, str]] = {
    -1: ("IDA_TOO_MUCH_WORK", "mxsteps internal steps taken before an output time — the step collapsed as the system stiffened (e.g. approaching bulk Tsat / an SCB switch)"),
    -2: ("IDA_TOO_MUCH_ACC", "the requested tolerance is unreachable at the system scale (atol/rtol too tight)"),
    -3: ("IDA_ERR_FAIL", "repeated local error-test failures drove the step down to hmin (a stiff transient)"),
    -4: ("IDA_CONV_FAIL", "the modified-Newton corrector could not converge and the step fell to hmin (a stiff transient; typical when a channel approaches bulk saturation)"),
    -5: ("IDA_LINIT_FAIL", ""),
    -6: ("IDA_LSETUP_FAIL", "the linear solver's setup (Jacobian factorization) failed unrecoverably (a near-singular reversal Jacobian)"),
    -7: ("IDA_LSOLVE_FAIL", "the linear solver's solve stage failed unrecoverably"),
    -8: ("IDA_RES_FAIL", "the residual function (compute) raised or returned NaN inside IDA (a property-domain excursion)"),
    -9: ("IDA_REP_RES_ERR", "the residual was repeatedly non-finite near a domain edge"),
    -10: ("IDA_RTFUNC_FAIL", "a user event_margin/rootfn failed (but this usually surfaces as SystemError, not this flag)"),
    -11: ("IDA_CONSTR_FAIL", "the inequality constraints option could not be met"),
    -12: ("IDA_FIRST_RES_FAIL", "compute(y0, t0) was non-finite — a bad guess that is already unphysical"),
    -13: ("IDA_LINESEARCH_FAIL", ""),
    -14: ("IDA_NO_RECOVERY", "the consistent-IC solve (IDACalcIC) could not recover — an IC-stage failure"),
    -15: ("IDA_NLS_INIT_FAIL", ""),
    -16: ("IDA_NLS_SETUP_FAIL", ""),
    -17: ("IDA_NLS_FAIL", ""),
    -20: ("IDA_MEM_NULL", ""),
    -21: ("IDA_MEM_FAIL", ""),
    -22: ("IDA_ILL_INPUT", "malformed options / mass / algebraic_vars_idx mismatch"),
    -23: ("IDA_NO_MALLOC", ""),
    -24: ("IDA_BAD_EWT", "a zero in the error-weight vector (a scale or atol of 0)"),
    -25: ("IDA_BAD_K", ""),
    -26: ("IDA_BAD_T", ""),
    -27: ("IDA_BAD_DKY", ""),
    -28: ("IDA_VECTOROP_ERR", ""),
    -99: ("IDA_UNRECOGNIZED_ERROR", ""),
}

# scipy.optimize.root (hybr/lm) status -> STREAM-terms meaning (status 1 = success).
_HYBR_STATUS: dict[int, str] = {
    2: "often a false negative at an already-converged root with a flat/near-null Jacobian (reversing mdot, closed flapper)",
    3: "the step fell below xtol; usually genuinely stuck or already at the root",
    4: "an ill-conditioned/near-singular Jacobian (no improvement over the last five Jacobian evaluations)",
    5: "no descent from the current basin — a far/ballpark guess (the case solve_steady's globalize cascade rescues)",
}

# scipy.integrate.solve_ivp status -> STREAM-terms meaning (0 = reached end, 1 = event).
_IVP_STATUS: dict[int, str] = {
    -1: "step size underflow — the explicit integrator's step fell below floating-point spacing (stiff system: consider eq_type='DAE' or looser tolerances)",
}


class TransientRuntimeError(StreamError, RuntimeError):
    """RuntimeError which occurred during a transient simulation,
    mostly because of solver convergence problems.

    The failure state is attached for post-mortem: ``e.y`` (the reached
    trajectory, or a single failing state), ``e.t`` (its times) and ``e.ydot``.
    Bridge it into a readable state with
    :meth:`~stream.aggregator.Aggregator.state_from`::

        try:
            ...                                    # error-raising simulation
        except TransientRuntimeError as e:
            history = agr.state_from((e.t, e.y))   # 2-D trajectory -> StateTimeseries
            last = history[max(history)]           # pick the failing row
            raise

    On an IC-stage (consistent-initial-condition) DAE failure there is no
    trajectory, so ``e.y`` is instead the *single* failing state recovered from
    IDA's error record (a note on the exception says so); bridge that 1-D vector
    with ``agr.state_from(e.y)`` (equivalently ``agr.save(e.y)``) to get one
    :class:`~stream.state.State`. ``agr.state_from((e.t, e.y))`` is the same as
    ``agr.save(Solution(e.t, e.y))`` for the 2-D case.

    When the failure comes from the IDA (DAE) backend, ``flag`` holds the numeric
    SUNDIALS status (e.g. ``-4``) and ``symbol`` the ``IDA_*`` name (e.g.
    ``IDA_CONV_FAIL``) — both are how the SUNDIALS troubleshooting docs are
    indexed. They are ``None`` for failures from other backends.
    """

    def __init__(self, t, y, ydot, message: str, *args, flag: int | None = None, symbol: str | None = None):
        if t is not None:
            message = f"At t = {t[-1]:.5f}: " + message
        super().__init__(message, *args)
        self.message = message
        self.t = t
        self.y = y
        self.ydot = ydot
        self.flag = flag
        self.symbol = symbol


def differential_algebraic(
    F: Functional,
    mass: Array1D,
    y0: Array1D,
    time: Sequence[float],
    yp0: Array1D | None = None,
    R: Functional | None = None,
    on_event: Callable | None = None,
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
        The continuous event-margin function ``R(y, t) -> float[]`` (IDA's rootfn).
        Each component is positive while its event is not pending and sign-changes
        at the event, so IDA localizes the true event time. When ``None`` there are
        no events and the equation is integrated straight through.
    on_event: Callable | None
        ``on_event(t_root, y_root, ydot_root) -> bool`` applied at each confirmed
        root: it performs the state transition and returns whether to continue.
    continuous: bool
        Controls what a stop request (``on_event`` returning ``False``) does. With
        ``continuous=False`` (default) it stops the run; with ``True`` the run
        restarts past it. In either case a *non-terminal* transition (``on_event``
        returns ``True``) restarts from the root with fresh, small initial steps.
    options:
        Other options to be passed to the ``scikits.odes`` solver.


    Returns
    -------
    solution: tuple[Array2D, Array1D]
        The solution matrix: [time, variable], and the vector of times
        in which it was calculated.
    """

    solve, time, y0, yp0 = _dae_setup(F, mass, y0, time, yp0, R, **options)
    if R is None or (on_event is None and not continuous):
        solution = solve(time, y0, yp0)
        return solution.values.y, solution.values.t
    return _event_loop_dae(solve, time, y0, yp0, on_event, continuous)


def _ida_post_solution(solution):
    flag = solution.flag
    if flag >= 0:
        return
    symbol, meaning = _IDA_STATUS.get(flag, (f"flag {flag}", ""))
    message = f"IDA {symbol}({flag}): {solution.message}"
    if meaning:
        message += f" — {meaning}"
    if solution.values.t is None and solution.errors.t is not None:
        # IC-stage failure: the failing state is in solution.errors, whose scalar t must be lifted to 1-D.
        errors = solution.errors
        err = TransientRuntimeError(
            np.atleast_1d(errors.t), errors.y, errors.ydot, message, flag=flag, symbol=symbol
        )
        err.add_note("state at failure recovered from IDA's error record (IC-stage failure)")
        raise err
    raise TransientRuntimeError(*solution.values, message, flag=flag, symbol=symbol)


def _dae_setup(
    F: Functional,
    mass: Array1D,
    y0: Array1D,
    time: Sequence[float],
    yp0: Array1D | None = None,
    R: Functional | None = None,
    **options,
) -> tuple[Callable, Array1D, Array1D, Array1D]:
    # A callback exception would cross the scikits.odes Cython boundary as an opaque SystemError; each closure parks it in `pending` and aborts the solve so solve() can re-raise the original.
    pending: list[BaseException] = []

    def residues(t, y, ydot, result):
        try:
            result[:] = F(y, t) - mass * ydot
        except BaseException as e:
            e.add_note(f"raised inside the residual function at t={t}")
            pending.append(e)
            result[:] = 1.0
            return -1  # IDA convention: negative residual return = unrecoverable failure

    def root(t, y, _, g, __):
        try:
            g[:] = R(y, t)
        except BaseException as e:
            e.add_note(f"raised inside the event/root function at t={t}")
            pending.append(e)
            g[:] = 1.0  # +1.0 exactly, NO sign change: a 0/negative fill would fire a spurious root

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
        try:
            with ignore_warnings(DeprecationWarning):
                sol_ = solver.solve(time, y0, yp0)
        except BaseException as cython_err:
            if pending:
                err = pending[0]
                pending.clear()
                raise err from cython_err
            raise
        if pending:
            err = pending[0]
            pending.clear()  # clear before raise: _event_loop_dae reuses solve() across restart segments
            raise err
        _ida_post_solution(sol_)
        return sol_

    return solve, time, y0, yp0


def _attach_trajectory(err: BaseException, t: Array1D, y: Array2D) -> None:
    """Attach a ``(t, y)`` trajectory to an exception for post-mortem, the way a
    solver failure carries its own last state. This lets an event handler raise a
    domain error while the solver — which knows nothing of that domain — preserves
    the state reached up to the failure on ``err.t`` / ``err.y``."""
    err.t, err.y = t, y


def _merge_failure_trajectory(e, t_acc, y_acc):
    """Attach the accumulated pre-failure segments plus this failure's partial
    trajectory to ``e.t``/``e.y``, shared by the DAE/ODE/polling drivers."""
    has_segment = e.t is not None and getattr(e, "y", None) is not None and np.ndim(e.y) == 2
    if has_segment and t_acc:
        t_acc.append(e.t[1:])   # row 0 is the restart point already held upstream
        y_acc.append(e.y[1:])
    elif has_segment:
        return                   # first segment: e already carries exactly its partial
    if t_acc:
        if e.t is not None and np.ndim(getattr(e, "y", None)) == 1:
            e.add_note(f"accumulated pre-failure trajectory attached to e.t/e.y; the IC-stage failure point (t={np.atleast_1d(e.t)[-1]}) it displaced remains described by the earlier note")
        e.t, e.y = concat(*t_acc), concat(*y_acc)


def _advance_eps(time: Array1D) -> float:
    """A tiny floor the next event time must exceed the previous restart time by,
    so a root that fails to advance raises instead of looping forever."""
    span = float(time[-1] - time[0])
    return max(1e-12, 1e-9 * span)


def _event_loop_dae(
    solve: Callable,
    time: Array1D,
    y0: Array1D,
    yp0: Array1D,
    on_event: Callable | None = None,
    continuous: bool = False,
) -> tuple[Array2D, Array1D]:
    r"""Integrate a DAE with continuous-margin events. On each ``IDA_ROOT_RETURN``
    the true root is taken from ``solution.roots`` (not the last output point before
    it), the transition is applied there via ``on_event``, and the run either stops
    (terminal) or restarts *from the root* with fresh small steps. The pre-root
    interval is never re-integrated, and a non-advancing root raises rather than
    hanging. Returned times stay aligned to the requested grid; a terminal stop
    additionally ends the series exactly at the event time."""
    time = np.asarray(time)
    eps = _advance_eps(time)
    t_acc: list[Array1D] = []
    y_acc: list[Array2D] = []
    t0, y_cur, yp_cur = float(time[0]), y0, yp0
    # A rootfn/direction crossing cannot detect a margin already <= 0 at t0; apply it explicitly.
    if on_event is not None:
        try:
            keep0 = on_event(t0, y0, yp0)
        except Exception as e:
            _attach_trajectory(e, np.array([t0]), y0[None, :])  # already past the limit at t0
            raise
        if not keep0 and not continuous:
            return y0[None, :], np.array([t0])
    last_t = t0
    remaining = time
    first = True
    while True:
        if first:
            solution = solve(remaining, y_cur, yp_cur)
        else:
            try:
                solution = solve(remaining, y_cur, yp_cur)
            except TransientRuntimeError as e:
                logger.log(STREAM_DEBUG, e.message)
                _merge_failure_trajectory(e, t_acc, y_acc)
                raise
        vt, vy, _ = solution.values
        keep_head = 0 if first else 1  # drop the duplicated restart row on restarts
        seg_t = vt[keep_head:]
        t_acc.append(seg_t)
        y_acc.append(vy[keep_head:])
        if seg_t.size:
            last_t = float(seg_t[-1])
        first = False

        roots = solution.roots
        if roots.t is None or len(roots.t) == 0:
            break

        t_root = float(roots.t[-1])
        y_root, yp_root = roots.y[-1], roots.ydot[-1]
        if not (t_root > t0 + eps):
            raise TransientRuntimeError(
                np.array([t_root]),
                y_root,
                yp_root,
                f"Event at t={t_root:.6g} did not advance past t={t0:.6g}; "
                "aborting to avoid an infinite restart loop.",
            )

        try:
            keep_going = on_event(t_root, y_root, yp_root) if on_event is not None else True
        except Exception as e:
            if t_root > last_t:
                t_acc.append(np.array([t_root]))
                y_acc.append(y_root[None, :])
            if t_acc:
                _attach_trajectory(e, concat(*t_acc), concat(*y_acc))
            raise
        if (not keep_going) and (not continuous):
            if t_root > last_t:
                t_acc.append(np.array([t_root]))
                y_acc.append(y_root[None, :])
            break

        logger.info(f"Event at t = {t_root:.5f}; restarting the integration from the root.")
        t0, y_cur, yp_cur = t_root, y_root, yp_root
        remaining = concat([t_root], time[time > t_root])
        if len(remaining) < 2:
            break
    return concat(*y_acc), concat(*t_acc)


class AlgRuntimeError(StreamError, RuntimeError):
    pass


def algebraic(
    F: Functional,
    y0: Array1D,
    time: Sequence[float] | None = None,
    R: Functional = None,
    **options,
) -> Array | tuple[Array2D, Array1D]:
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
    solution: Array or tuple[Array2D, Array1D]
        When ``time is None`` (steady root find), the bare solution vector. In
        quasi-static time mode, the solution matrix ([time, variable]) **and** the
        vector of times it actually spans — these are shorter than the requested
        ``time`` if a stop event ``R`` tripped, so callers must rebind their time
        axis from the returned times rather than the requested grid.
    """

    def _solve(_vec, _t):
        _sol = opt.root(F, _vec, (_t,), **options)
        if not _sol["success"]:
            timestr = f"At t={_t:.3f}, " if _t is not None else ""
            message = f"{timestr}Root Finding failed with the following message:\n" + _sol["message"]
            status = _sol.get("status")
            meaning = _HYBR_STATUS.get(status)
            if meaning:
                message += f"\n[hybr status {status}] {meaning}"
            # hybr can flag failure at an already-converged root (flat/near-null Jacobian); check ‖F‖ to detect it.
            nF = float(np.linalg.norm(np.atleast_1d(F(_sol.x, _t if _t is not None else 0))))
            if nF < 1e-6:
                message += (
                    f"\nthe returned iterate has ‖F‖={nF:.1e} — likely a MINPACK false negative "
                    "at an already-converged root (flat/near-null Jacobian); err.y holds the iterate"
                )
            err = AlgRuntimeError(message)
            err.y = _sol.x
            err.status = status
            err.backend_message = _sol["message"]  # the raw scipy text, un-wrapped
            raise err
        return _sol.x

    if time is not None:
        time = np.asarray(time)
        y = np.zeros((len(time), len(y0)))
        y[0] = y0
        for i, t in enumerate(time[1:]):
            try:
                sol = _solve(y[i], t)
            except AlgRuntimeError as err:
                err.t = time[: i + 1]
                err.data = y[: i + 1]
                raise
            y[i + 1] = sol
            # Stop after storing the solved row so the stop-triggering state is kept, not dropped.
            if R is not None and not np.all(R(sol, t)):
                return y[: i + 2], time[: i + 2]
        return y, time
    else:
        return _solve(y0, 0)


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
    name: str = "scaled_newton",
) -> tuple[Array1D, bool, bool, float, int]:
    """Shared inner loop of :func:`scaled_newton` and :func:`pseudo_transient`.

    Runs equilibrated, Tikhonov-regularized, Armijo-damped Newton iterations and
    reports status flags instead of raising, so the public solver (which raises)
    and the pseudo-transient inner solve (which tolerates a stalled step) share
    the exact same numerics.

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
        try:
            dy = np.linalg.solve(Jeq + tikhonov * np.eye(len(Jeq)), -(Fy / rs)) / cs
        except np.linalg.LinAlgError as e:
            err = AlgRuntimeError(
                f"{name}: linear solve failed — the (equilibrated, Tikhonov-regularized) "
                "Jacobian is numerically singular; the system may be structurally "
                "under-determined (e.g. point kinetics at exactly rho=0, a floating "
                "pressure datum)"
                + hint_block("check for a rank-deficient/undetermined system",
                             "inspect agr.worst_residuals(err.y)")
            )
            err.y = y
            raise err from e
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

    Each iteration equilibrates the Jacobian (row-then-column infinity norm),
    solves the Tikhonov-regularized, equilibrated linear system for a Newton
    step, and backtracks the step with an Armijo merit line search on
    ``||F(y + a*dy, t)||``. It is the load-bearing cure for steady solves from far
    or ballpark guesses where the globalization in :func:`scipy.optimize.root`
    (``hybr``) stalls — the damping keeps taking full Newton steps after one
    damped step from a far basin.

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
        Regularization added on the diagonal of the *equilibrated* (``O(1)``)
        matrix (default ``1e-10``).

    Returns
    -------
    Array1D
        The converged state ``y`` with ``||F(y, t)|| < tol``.

    Raises
    ------
    AlgRuntimeError
        If the Armijo line search finds no descent (raised immediately, not
        spun out to ``maxit``) or ``maxit`` is exhausted. The best iterate
        reached is attached as the ``y`` attribute for diagnosis, and callers
        may fall back to :func:`pseudo_transient`.

    Notes
    -----
    The equilibration serves conditioning diagnostics and insurance for
    badly-scaled systems (making ``cond``/rank meaningful and giving Tikhonov a
    sane ``O(1)`` metric); the *damping* provides the convergence. The Tikhonov
    term is a defensive default — provably inactive on well-grounded systems.
    """
    y, converged, exhausted, nF, it = _scaled_newton_core(
        F, y0, jac, t=t, tol=tol, maxit=maxit, tikhonov=tikhonov
    )
    if converged:
        return y
    reason = "Armijo line search found no descent direction" if exhausted else f"exceeded maxit={maxit}"
    err = AlgRuntimeError(f"scaled_newton failed at iteration {it} ({reason}); ||F||={nF:.3e}. Fall back to pseudo_transient.")
    err.y = y
    raise err


def pseudo_transient(
    F: Functional,
    mass: Array1D,
    y0: Array1D,
    jac: Callable[[Array1D, float], Array2D],
    *,
    t: float = 0.0,
    tol: float = 1e-2,
    dtau0: float = 1e-2,
    maxsteps: int = 200,
    inner_tol_floor: float = 1e-9,
    **newton_kw,
) -> Array1D:
    r"""Implicit-Euler pseudo-transient continuation to reach a steady basin.

    A last-resort fallback when :func:`scaled_newton` cannot reach the basin
    from a truly bad guess. Each pseudo-step solves the backward-Euler stage
    equation ``G(y) = F(y, t) - mass*(y - y_prev)/dtau = 0`` with an inner
    equilibrated damped Newton (the same machinery as :func:`scaled_newton`; the
    inner Jacobian is ``jac(y, t) - diag(mass)/dtau``). The result is handed back
    once ``||F|| < tol`` so the caller can polish it with :func:`scaled_newton`.

    The **real boolean** ``mass`` vector is essential: on algebraic rows
    (``mass = 0``) the ``mass*(y - y_prev)/dtau`` term vanishes, so those rows
    stay *exact constraints* ``F_a = 0`` at every accepted step. A naive recipe —
    integrate ``dy/dtau = F`` with an all-differential identity mass — is wrong
    because it gives the algebraic rows a spurious self-growth mode and diverges.

    Parameters
    ----------
    F : Functional
        Residual ``F(y, t) -> Array1D`` (the ``M dy/dt = F`` right-hand side).
    mass : Array1D
        The boolean mass vector: ``1`` on differential rows, ``0`` on algebraic
        (constraint) rows.
    y0 : Array1D
        Initial guess.
    jac : Callable
        Jacobian ``jac(y, t) -> Array2D`` of ``F``.
    t : float, optional
        Time argument threaded into ``F`` and ``jac`` (default ``0.0``).
    tol : float, optional
        Basin tolerance on ``||F||`` for termination (default ``1e-2``).
    dtau0 : float, optional
        Initial pseudo-time step (default ``1e-2``).
    maxsteps : int, optional
        Maximum pseudo-transient steps (default ``200``).
    inner_tol_floor : float, optional
        Floor for the inner Newton tolerance ``max(inner_tol_floor, 1e-10*||F||)``
        (default ``1e-9``).
    **newton_kw
        Forwarded to the inner Newton (currently ``tikhonov``).

    Returns
    -------
    Array1D
        A state ``y`` inside the basin (``||F(y, t)|| < tol``).

    Raises
    ------
    AlgRuntimeError
        On divergence (``||F||`` non-finite or ``> 1e3*||F_prev||``) or if
        ``maxsteps`` is exhausted without reaching the basin. The last accepted
        iterate is attached as the ``y`` attribute.

    Notes
    -----
    A Switched-Evolution-Relaxation (SER) controller grows the step:
    ``dtau *= clip(||F_prev||/||F_new||, 0.1, 10)``; ``dtau -> inf`` recovers Newton.
    Pseudo-time is **not** physical — events have no meaning here.
    """
    y_prev = np.array(y0, float)
    mass = np.asarray(mass, float)
    tikhonov = newton_kw.get("tikhonov", 1e-10)
    nF = float(np.linalg.norm(F(y_prev, t)))
    dtau = dtau0
    for step in range(maxsteps):
        if nF < tol:
            return y_prev

        def G(yy, tt, yp=y_prev, dt=dtau):
            return F(yy, tt) - mass * (yy - yp) / dt

        def G_jac(yy, tt, dt=dtau):
            return np.array(jac(yy, tt)) - np.diag(mass) / dt

        inner_tol = max(inner_tol_floor, 1e-10 * nF)
        y, *_ = _scaled_newton_core(
            G, y_prev, G_jac, t=t, tol=inner_tol, maxit=25, tikhonov=tikhonov,
            name="pseudo_transient (inner Newton)",
        )

        nF_new = float(np.linalg.norm(F(y, t)))
        if not np.isfinite(nF_new) or nF_new > 1e3 * nF:
            err = AlgRuntimeError(f"pseudo_transient diverged at step {step}: ||F||={nF_new:.3e}")
            err.y = y_prev
            raise err
        # SER growth, clipped; an exact-zero residual takes the max growth factor.
        dtau *= min(max(nF / nF_new, 0.1), 10.0) if nF_new > 0.0 else 10.0
        y_prev, nF = y, nF_new

    if nF < tol:
        return y_prev
    err = AlgRuntimeError(f"pseudo_transient did not reach the basin in {maxsteps} steps; ||F||={nF:.3e}")
    err.y = y_prev
    raise err


def _ivp_failure_message(solution) -> str:
    """Translate a failed ``solve_ivp`` result into a STREAM-terms message that
    carries the numeric status (so it can be looked up) and its meaning clause."""
    message = f"solve_ivp({solution.status}): {solution.message}"
    meaning = _IVP_STATUS.get(solution.status)
    return f"{message} — {meaning}" if meaning else message


def _annotate_rhs(F: Functional) -> Callable:
    """Wrap the ODE right-hand side so a raise inside it carries the evaluation
    time as a note. ``scipy.solve_ivp`` propagates exceptions from the rhs cleanly
    (no Cython mangling, unlike the DAE path), so annotate-and-reraise is the right
    tool — there is no way to abort ``solve_ivp`` from a poisoned rhs."""

    def rhs(t, y):
        try:
            return F(y, t)
        except BaseException as e:
            e.add_note(f"raised inside the rhs function at t={t}")
            raise

    return rhs


def differential(
    F: Functional,
    y0: Array1D,
    time: Sequence[float],
    events: Sequence[Callable] | None = None,
    on_event: Callable | None = None,
    continuous: bool = False,
    **options,
) -> tuple[Array2D, Array1D]:
    r"""Solving an Ordinary Differential Equation (ODE) :math:`\dot{y}=F(y, t)`

    Parameters
    ----------
    F: Functional
        The main right-hand side function :math:`F(y,t)`.
    y0: Array1D
        Initial values.
    time: Sequence[float]
        Time points for which the simulation values should be returned.
    events: Sequence[Callable] | None
        ``solve_ivp`` terminal event functions (one per continuous margin). When
        given, events are localized and handled with a restart loop mirroring the
        DAE path, so controllers/SCRAMs/aborts fire in ODE mode too.
    on_event: Callable | None
        ``on_event(t_event, y_event) -> bool`` applied at each localized event.
    continuous: bool
        As in :func:`differential_algebraic`: whether a stop request halts or is
        restarted past. Non-terminal transitions always restart from the event.
    options:
        Other options to be passed to the ``scipy.integrate.solve_ivp`` solver.

    Returns
    -------
    solution: tuple[Array2D, Array1D]
        The solution matrix at reached times ([time, variable]) and the vector of
        those times. On a solver failure the reached times are shorter than the
        requested ``time``; a :class:`TransientRuntimeError` is raised in that case
        (mirroring the DAE path) rather than returning a truncated result.
    """
    time = np.asarray(time)
    if events and (on_event is not None or continuous):
        return _event_loop_ode(F, y0, time, events, on_event, continuous, **options)
    time_limits = (time[0], time[-1])
    solution = solve_ivp(_annotate_rhs(F), time_limits, y0, t_eval=time, events=events, **options)
    data = np.transpose(solution.y)
    if not solution.success:
        reached = solution.t if solution.t is not None and len(solution.t) else None
        raise TransientRuntimeError(reached, data, None, _ivp_failure_message(solution))
    return data, solution.t


def _event_loop_ode(
    F: Functional,
    y0: Array1D,
    time: Array1D,
    events: Sequence[Callable],
    on_event: Callable | None,
    continuous: bool,
    **options,
) -> tuple[Array2D, Array1D]:
    """The ODE analogue of :func:`_event_loop_dae`: ``solve_ivp`` localizes the
    terminal events (via Brent), the transition is applied at the true event time,
    and the run stops (terminal) or restarts from the event. Returned times stay
    grid-aligned; a terminal stop ends the series exactly at the event."""
    eps = _advance_eps(time)
    t_acc: list[Array1D] = []
    y_acc: list[Array2D] = []
    t0, y_cur = float(time[0]), y0
    # solve_ivp's direction=-1 cannot detect a margin already non-positive at t0; apply it explicitly.
    if on_event is not None:
        try:
            keep0 = on_event(t0, y0)
        except Exception as e:
            _attach_trajectory(e, np.array([t0]), y0[None, :])
            raise
        if not keep0 and not continuous:
            return y0[None, :], np.array([t0])
    last_t = t0
    remaining = time
    first = True
    while True:
        solution = solve_ivp(
            _annotate_rhs(F), (remaining[0], remaining[-1]), y_cur, t_eval=remaining, events=events, **options
        )
        data = np.transpose(solution.y)
        if not solution.success:
            reached = solution.t if solution.t is not None and len(solution.t) else None
            err = TransientRuntimeError(reached, data, None, _ivp_failure_message(solution))
            _merge_failure_trajectory(err, t_acc, y_acc)
            raise err
        keep_head = 0 if first else 1
        seg_t = solution.t[keep_head:]
        t_acc.append(seg_t)
        y_acc.append(data[keep_head:])
        if seg_t.size:
            last_t = float(seg_t[-1])
        first = False

        if solution.status != 1:  # 1 == terminated by a terminal event
            break

        # The event time/state live in t_events/y_events (t_eval never lands on them).
        fired = [(te[-1], ye[-1]) for te, ye in zip(solution.t_events, solution.y_events) if len(te)]
        t_root, y_root = min(fired, key=lambda p: p[0])
        t_root = float(t_root)
        if not (t_root > t0 + eps):
            raise TransientRuntimeError(
                np.array([t_root]),
                y_root,
                None,
                f"Event at t={t_root:.6g} did not advance past t={t0:.6g}; "
                "aborting to avoid an infinite restart loop.",
            )

        try:
            keep_going = on_event(t_root, y_root) if on_event is not None else True
        except Exception as e:
            if t_root > last_t:
                t_acc.append(np.array([t_root]))
                y_acc.append(y_root[None, :])
            if t_acc:
                _attach_trajectory(e, concat(*t_acc), concat(*y_acc))
            raise
        if (not keep_going) and (not continuous):
            if t_root > last_t:
                t_acc.append(np.array([t_root]))
                y_acc.append(y_root[None, :])
            break

        logger.info(f"ODE event at t = {t_root:.5f}; restarting the integration from the event.")
        t0, y_cur = t_root, y_root
        remaining = concat([t_root], time[time > t_root])
        if len(remaining) < 2:
            break
    return concat(*y_acc), concat(*t_acc)
