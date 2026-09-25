r"""Tests for the globalized steady-solve backends in :mod:`stream.solvers`:
:func:`scaled_newton` (equilibrated, Tikhonov-regularized, Armijo-damped Newton)
and :func:`pseudo_transient` (implicit-Euler, real-mass pseudo-transient
continuation), plus the :func:`_equilibrate` helper.

The test problems are small synthetic systems that reproduce the LOFA pathology
(variable scales spanning ~6 decades, residual scales spanning several decades)
without importing the benchmark. Roots are known analytically.
"""

import numpy as np
import pytest
from scipy import optimize as opt

from stream.solvers import AlgRuntimeError, _equilibrate, pseudo_transient, scaled_newton

# --- A badly-scaled DAE-form system with a known root and a mixed mass vector ---
# Variables scale as [1, 1e2, 1e4, 1e6] (6 decades); residual rows carry scales
# [1, 1e2, 1e6, 1e5] with the two algebraic (constraint) rows holding large raw
# self-coefficients, mimicking the inertia coefficients that make LOFA's Jacobian
# ill-conditioned (cond(J*) ~ 1.5e10 here). Rows 0,1 are differential (mass 1),
# rows 2,3 are exact constraints (mass 0). The unique root is y = SCALE.
_SCALE = np.array([1.0, 1e2, 1e4, 1e6])
_ROW = np.array([1.0, 1e2, 1e6, 1e5])
_MASS = np.array([1.0, 1.0, 0.0, 0.0])
_ROOT = _SCALE.copy()
_K0, _K1, _B0, _B1, _C2, _C3 = 2.0, 3.0, 1.0, 1.0, 0.5, 0.4


def _F(y, t=0.0):
    d = y / _SCALE - 1.0
    return _ROW * np.array(
        [
            -_K0 * d[0] + _B0 * d[1],
            -_K1 * d[1] + _B1 * d[2],
            d[2] + _C2 * d[0],  # constraint: dF2/dy2 > 0 (forward-Euler unstable)
            d[3] + _C3 * d[1],  # constraint: dF3/dy3 > 0
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
    must be damped on the first step, :func:`scaled_newton` reaches the known root."""
    far_guess = _SCALE * 8.0
    assert np.linalg.norm(_F(far_guess)) > 1e5  # genuinely far from the basin

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
    iterations (the Jacobian is never evaluated). This guards the auto-fallback
    rescue of hybr's false-negative-at-root."""
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
    """On a well-conditioned problem the Tikhonov term is provably inactive:
    solutions with ``tikhonov=0`` and ``tikhonov=1e-10`` agree to ~1e-12."""

    def F(y, t=0.0):
        x0, x1 = y
        return np.array([x0 + 0.1 * x1 - 1.0, 0.1 * x0 + x1**3 - 8.0])

    def jac(y, t=0.0):
        return np.array([[1.0, 0.1], [0.1, 3 * y[1] ** 2]])

    y_no_reg = scaled_newton(F, np.array([0.0, 0.0]), jac, tol=1e-10, tikhonov=0.0)
    y_reg = scaled_newton(F, np.array([0.0, 0.0]), jac, tol=1e-10, tikhonov=1e-10)

    assert np.max(np.abs(y_no_reg - y_reg)) < 1e-12


def test_pseudo_transient_relaxes_far_guess_and_keeps_constraints_exact():
    """PTC on the badly-scaled mixed-mass system relaxes a far guess into the
    basin (``‖F‖ < tol``). Because the real boolean mass leaves the algebraic
    rows as exact constraints, those residual rows are driven to ~0 (well below
    the basin tolerance) — verified at the returned point, where the inner
    Newton has solved ``G = F`` on the mass-0 rows to its inner tolerance."""
    far_guess = _SCALE * 8.0
    assert np.linalg.norm(_F(far_guess)) > 1e5

    y = pseudo_transient(_F, _MASS, far_guess, _jac, tol=1e-2, dtau0=1e-2, maxsteps=200)

    assert np.linalg.norm(_F(y)) < 1e-2
    algebraic_rows = np.abs(_F(y))[_MASS == 0]
    assert np.all(algebraic_rows < 1e-6)  # exact-constraint property

    # A scaled_newton polish reaches the true root from the PTC hand-off.
    y_polished = scaled_newton(_F, y, _jac, tol=1e-6)
    np.testing.assert_allclose(y_polished, _ROOT, rtol=1e-8)


def test_identity_mass_literal_recipe_diverges_where_pseudo_transient_converges():
    """A forward-Euler ``y += dtau*F`` with an all-differential identity mass blows
    up on the algebraic constraint rows, while :func:`pseudo_transient` with the true
    boolean mass converges on the same system. The identity-mass recipe is implemented
    inline here and must NOT exist in ``stream``."""
    far_guess = _SCALE * 8.0
    n0 = np.linalg.norm(_F(far_guess))

    # Identity-mass forward-Euler integration.
    y = far_guess.copy()
    worst = n0
    dtau = 1e-2
    for _ in range(100):
        y = y + dtau * _F(y)
        n = np.linalg.norm(_F(y))
        worst = max(worst, n)
        if not np.isfinite(n):
            break
    assert worst >= 1e3 * n0  # diverges

    # The correct real-mass PTC converges on the identical system.
    y_ptc = pseudo_transient(_F, _MASS, far_guess, _jac, tol=1e-2, dtau0=1e-2)
    assert np.linalg.norm(_F(y_ptc)) < 1e-2


def test_pseudo_transient_raises_on_maxsteps_with_iterate_attached():
    """When the basin cannot be reached in ``maxsteps``, PTC raises with the
    last accepted iterate attached."""
    far_guess = _SCALE * 8.0
    with pytest.raises(AlgRuntimeError) as excinfo:
        pseudo_transient(_F, _MASS, far_guess, _jac, tol=1e-12, dtau0=1e-3, maxsteps=1)
    assert hasattr(excinfo.value, "y")


def test_equilibrate_unit_row_and_column_norms_positive_scales_and_reconstruction():
    """:func:`_equilibrate` returns ``J_eq`` whose largest row and column
    magnitudes are both 1, with strictly positive scales, and reconstructs the
    original matrix as ``rs[:, None] * J_eq * cs[None, :]``."""
    rng = np.random.default_rng(0)
    # A badly-scaled random matrix: rows/cols multiplied by wildly different factors.
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


def test_ptc_survives_exact_zero_residual():
    """An inner solve landing on an exact root must not break the SER update.

    A pure-algebraic linear row is solved exactly by one Newton step, so the
    post-step ``||F||`` is exactly 0.0; the SER controller must take its capped
    growth instead of dividing by zero.
    """
    F = lambda y, t: np.array([y[0]])
    jac = lambda y, t: np.array([[1.0]])
    y = pseudo_transient(F, np.array([0.0]), np.array([1.0]), jac)
    assert np.linalg.norm(F(y, 0.0)) < 1e-2


# --- A structurally singular (rank-deficient) system: a line of roots ---
# F(y) = [y0 - y1, y0 - y1] has jac [[1, -1], [1, -1]]; the equilibrated,
# Tikhonov-regularized 2x2 has det ~ 1e-20 -> np.linalg.solve raises LinAlgError.
def _singular_F(y, t=0.0):
    return np.array([y[0] - y[1], y[0] - y[1]])


def _singular_jac(y, t=0.0):
    return np.array([[1.0, -1.0], [1.0, -1.0]])


def test_scaled_newton_converts_singular_linalgerror_to_algruntimeerror():
    """A structurally singular equilibrated+regularized Jacobian makes
    np.linalg.solve raise a bare LinAlgError. scaled_newton must convert it to the
    documented AlgRuntimeError (message names it 'singular'), keep the current iterate
    on err.y, and chain the LinAlgError as __cause__ — no bare numpy error escapes."""
    with pytest.raises(AlgRuntimeError, match="singular") as excinfo:
        scaled_newton(_singular_F, np.array([3.0, 0.0]), _singular_jac)
    e = excinfo.value
    assert getattr(e, "y", None) is not None
    assert isinstance(e.__cause__, np.linalg.LinAlgError)


def test_pseudo_transient_converts_singular_linalgerror_to_algruntimeerror():
    """The same conversion covers pseudo_transient, whose only linear
    solves live in the shared Newton core (name 'pseudo_transient (inner Newton)')."""
    with pytest.raises(AlgRuntimeError, match="singular") as excinfo:
        pseudo_transient(_singular_F, np.array([0.0, 0.0]), np.array([3.0, 0.0]), _singular_jac)
    e = excinfo.value
    assert getattr(e, "y", None) is not None
    assert isinstance(e.__cause__, np.linalg.LinAlgError)
