r"""Tests for the globalized steady-solve backend :func:`scaled_newton` in
:mod:`stream.solvers` (equilibrated, Tikhonov-regularized, Armijo-damped Newton),
plus the :func:`_equilibrate` helper.

The test problems are small synthetic systems whose variable scales span about
six decades and whose residual scales span several decades. Roots are known
analytically.
"""

import numpy as np
import pytest

from stream.solvers import AlgRuntimeError, _equilibrate, scaled_newton

_SCALE = np.array([1.0, 1e2, 1e4, 1e6])
_ROW = np.array([1.0, 1e2, 1e6, 1e5])
_ROOT = _SCALE.copy()
_K0, _K1, _B0, _B1, _C2, _C3 = 2.0, 3.0, 1.0, 1.0, 0.5, 0.4


def _F(y, t=0.0):
    """Badly-scaled residual whose unique root is ``_SCALE``.

    Rows 0 and 1 are differential; rows 2 and 3 are constraints with large raw,
    positive self-coefficients, on which forward Euler is unstable.
    """
    d = y / _SCALE - 1.0
    return _ROW * np.array(
        [
            -_K0 * d[0] + _B0 * d[1],
            -_K1 * d[1] + _B1 * d[2],
            d[2] + _C2 * d[0],
            d[3] + _C3 * d[1],
        ]
    )


def _jac(y, t=0.0):
    dres = np.array(
        [
            [-_K0, _B0, 0.0, 0.0],
            [0.0, -_K1, _B1, 0.0],
            [_C2, 0.0, 1.0, 0.0],
            [0.0, _C3, 0.0, 1.0],
        ]
    )
    return (_ROW[:, None] * dres) / _SCALE[None, :]


def test_scaled_newton_converges_on_badly_scaled_system_from_far_guess():
    """From a far guess on a 6-decade-scaled system where the Newton direction
    must be damped on the first step, :func:`scaled_newton` reaches the known
    root."""
    far_guess = _SCALE * 8.0
    assert np.linalg.norm(_F(far_guess)) > 1e5

    y = scaled_newton(_F, far_guess, _jac, tol=1e-6, maxit=200)

    assert np.linalg.norm(_F(y)) < 1e-6
    np.testing.assert_allclose(y, _ROOT, rtol=1e-8)


def test_scaled_newton_raises_on_armijo_exhaustion_with_iterate_attached():
    """A constant residual admits no descent direction, so the Armijo search
    exhausts on the first iteration. :func:`scaled_newton` must raise
    immediately (not spin to ``maxit``) and attach the best iterate."""
    y0 = np.array([0.0])

    def const_F(y, t=0.0):
        return np.array([1.0])

    def const_jac(y, t=0.0):
        return np.array([[1.0]])

    with pytest.raises(AlgRuntimeError) as excinfo:
        scaled_newton(const_F, y0, const_jac, tol=1e-6, maxit=50)

    assert hasattr(excinfo.value, "y")
    np.testing.assert_array_equal(excinfo.value.y, y0)


def test_scaled_newton_returns_immediately_when_already_converged():
    """When the guess already satisfies ``‖F‖ < tol``, return it with zero
    iterations (the Jacobian is never evaluated), so a root that another solver
    failed to accept comes back unchanged."""
    jac_calls = {"n": 0}

    def tiny_F(y, t=0.0):
        return np.array([1e-12, 0.0])

    def counting_jac(y, t=0.0):
        jac_calls["n"] += 1
        return np.eye(2)

    y0 = np.array([1.0, 2.0])
    out = scaled_newton(tiny_F, y0, counting_jac, tol=1e-6)

    assert jac_calls["n"] == 0
    np.testing.assert_array_equal(out, y0)


def test_scaled_newton_tikhonov_is_inactive_on_well_posed_problem():
    """On a well-conditioned problem the Tikhonov term is inactive: solutions
    with ``tikhonov=0`` and ``tikhonov=1e-10`` agree to ~1e-12."""

    def F(y, t=0.0):
        x0, x1 = y
        return np.array([x0 + 0.1 * x1 - 1.0, 0.1 * x0 + x1**3 - 8.0])

    def jac(y, t=0.0):
        return np.array([[1.0, 0.1], [0.1, 3 * y[1] ** 2]])

    y_no_reg = scaled_newton(F, np.array([0.0, 0.0]), jac, tol=1e-10, tikhonov=0.0)
    y_reg = scaled_newton(F, np.array([0.0, 0.0]), jac, tol=1e-10, tikhonov=1e-10)

    assert np.max(np.abs(y_no_reg - y_reg)) < 1e-12


def test_scaled_newton_takes_least_squares_step_on_singular_jacobian():
    """A nilpotent Jacobian stays singular after the diagonal shift; the step
    falls back to the minimum-norm least-squares solution and still converges."""
    J = np.array([[-1.0, 1.0], [-1.0, 1.0]])

    def F(y, t=0.0):
        return J @ y - 1.0

    y = scaled_newton(F, np.zeros(2), lambda y, t=0.0: J)

    np.testing.assert_allclose(y, [-0.5, 0.5])


def test_equilibrate_unit_row_and_column_norms_positive_scales_and_reconstruction():
    """:func:`_equilibrate` returns ``J_eq`` whose largest row and column
    magnitudes are both 1, with strictly positive scales, and reconstructs the
    original matrix as ``rs[:, None] * J_eq * cs[None, :]``."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((6, 6)) + 0.5
    J = np.diag([1.0, 1e3, 1e-2, 1e5, 1e-4, 1e1]) @ base @ np.diag([1e2, 1.0, 1e-3, 1e4, 1e6, 1e-1])

    J_eq, rs, cs = _equilibrate(J)

    row_norms = np.max(np.abs(J_eq), axis=1)
    col_norms = np.max(np.abs(J_eq), axis=0)
    assert np.isclose(np.max(row_norms), 1.0)
    assert np.isclose(np.max(col_norms), 1.0)
    assert np.all(np.abs(J_eq) <= 1.0 + 1e-12)
    assert np.all(rs > 0) and np.all(cs > 0)
    np.testing.assert_allclose(rs[:, None] * J_eq * cs[None, :], J, rtol=1e-12)


def test_equilibrate_floors_vanishing_rows_and_columns():
    """A zero row/column is floored (never divided by zero, never amplified)."""
    J = np.array([[0.0, 0.0], [3.0, 4.0]])
    J_eq, rs, cs = _equilibrate(J)
    assert np.all(np.isfinite(J_eq))
    assert np.all(rs >= 1e-30) and np.all(cs >= 1e-30)

