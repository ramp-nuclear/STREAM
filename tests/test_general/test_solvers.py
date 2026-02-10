import numpy as np
import pytest
from scikits.odes import dae

from stream import Aggregator
from stream.composition import Calculation_factory
from stream.solvers import TransientRuntimeError
from stream.units import Array1D
from stream.utilities import ignore_warnings


def test_dae_solver_using_planar_pendulum_against_scikits_odes_documented_solution():
    """
    This is an example given by the Scikits.Odes developers for a simple
    usage case: A planar pendulum. For further explanation please refer to
    their documentation.
    """
    length = 1
    m = 1
    g = 1
    lambdaval = 0.1
    theta0 = np.pi / 3
    x0 = np.sin(theta0)
    y0 = -((length - x0 ** 2) ** 0.5)
    z0 = np.array([x0, y0, 0.0, 0.0, lambdaval])
    zp0 = np.array([0.0, 0.0, lambdaval * x0 / m, (lambdaval * y0 / m) - g, -g])

    def calculate(variables) -> Array1D:
        x, y, u, v, lamda = variables
        k = lamda / m
        xdot = u
        ydot = v
        udot = k * x
        vdot = k * y - g
        constraint = u ** 2 + v ** 2 + k * (x ** 2 + y ** 2) - y * g
        return np.array([xdot, ydot, udot, vdot, constraint])

    PendulumCalculation = Calculation_factory(
        calculate=calculate,
        mass_vector=4 * [True] + [False],
        variables=dict(x=0, y=1, u=2, v=3, lamda=4)
    )

    agr = Aggregator.from_decoupled(PendulumCalculation())
    solution = agr.solve(z0, [0.0, 1.0, 2.0], zp0)
    assert np.allclose(solution[:, 0], [0.866025, 0.592663, -0.304225])


def test_dae_with_undetermined_var_fails_as_TransientRuntimeError():
    """
    This test constructs a calculation in which one variable is free, meaning
    its algebraic equation returns 0 for any input. It is also disconnected
    from the rest of the equations.

    Such a calculation is expected to fail.
    """
    ExponentWithUndeterminedVar = Calculation_factory(
        calculate=lambda y: np.array([-y[0], 0.0]),
        mass_vector=[True, False],
        variables=dict(x=0, free=1)
    )
    agr = Aggregator.from_decoupled(ExponentWithUndeterminedVar())
    with pytest.raises(TransientRuntimeError):
        agr.solve(np.array([100.0, 5.0]), time=[0.0, 10.0], yp0=np.array([-100, 0.0]))


def test_dae_solver_accepts_all_algebraic_calculations():
    """
    Testing the possibility of working with an all algebraic set of equations
    in scikits.odes.
    """

    def residues(t, y, ydot, result):
        result[:] = [y[0] - y[1], y[0] + y[1]]

    kwargs = dict(
        compute_initcond="yp0", old_api=False, algebraic_vars_idx=np.arange(2)
    )
    solver = dae("ida", residues, **kwargs)
    with ignore_warnings(DeprecationWarning):
        solution = solver.solve(tspan=[0, 0.001], y0=[100, 200], yp0=[-100, 300])
    sol = solution.values.y[-1]
    assert np.allclose(sol, 0.0)
