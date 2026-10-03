import numpy as np
import pytest
from scikits.odes import dae

from stream.aggregator import Aggregator
from stream.calculations import Gravity, Pump, Resistor
from stream.calculations.ideal.resistors import ResistorSum
from stream.composition import Calculation_factory
from stream.composition.cycle import flow_edge, flow_graph
from stream.composition.cycle import flow_graph_to_agr_and_k as agr_k
from stream.jacobians import ALG_jacobian
from stream.solvers import TransientRuntimeError
from stream.state import State
from stream.substances import light_water
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
    y0 = -((length - x0**2) ** 0.5)
    z0 = np.array([x0, y0, 0.0, 0.0, lambdaval])
    zp0 = np.array([0.0, 0.0, lambdaval * x0 / m, (lambdaval * y0 / m) - g, -g])

    def calculate(variables) -> Array1D:
        x, y, u, v, lamda = variables
        k = lamda / m
        xdot = u
        ydot = v
        udot = k * x
        vdot = k * y - g
        constraint = u**2 + v**2 + k * (x**2 + y**2) - y * g
        return np.array([xdot, ydot, udot, vdot, constraint])

    PendulumCalculation = Calculation_factory(
        calculate=calculate,
        mass_vector=4 * [True] + [False],
        variables=dict(x=0, y=1, u=2, v=3, lamda=4),
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
        variables=dict(x=0, free=1),
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

    kwargs = dict(compute_initcond="yp0", old_api=False, algebraic_vars_idx=np.arange(2))
    solver = dae("ida", residues, **kwargs)
    with ignore_warnings(DeprecationWarning):
        solution = solver.solve(tspan=[0, 0.001], y0=[100, 200], yp0=[-100, 300])
    sol = solution.values.y[-1]
    assert np.allclose(sol, 0.0)


def _gravity_resistor_loop(resistance):
    """A minimal Kirchhoff loop whose dp row mixes a large gravity head with a
    small ``resistance * mdot`` slope, exercising flow reversal.

    The R.pressure residual row is ``p_var - dp_out`` with
    ``dp_out = rho*g*h - resistance*mdot``, so its exact derivative w.r.t. the
    loop mdot is ``+resistance`` for every mdot, including mdot = 0.
    """
    T = 50.0
    P = Pump(pressure=1.0e4)
    R = ResistorSum(Gravity(light_water, disposition=1.0), Resistor(resistance=resistance), name="R")
    fg = flow_graph(flow_edge(("A", "B"), P), flow_edge(("B", "A"), R))
    agr, K = agr_k(fg, {P.name: dict(Tin=T), R.name: dict(Tin=T)})

    rho = light_water.density(T)
    head = rho * 9.80665 * 1.0

    def mdot_derivative(mdot):
        st = State.merge(
            {K.name: {K.component_edge(P): mdot, K.component_edge(R): mdot}},
            State.uniform((R, P), T),
            {R.name: dict(pressure=head - resistance * mdot), P.name: dict(pressure=1.0e4)},
        )
        y = agr.load(st)
        J = ALG_jacobian(agr)(y).copy()
        row = agr.sections[R].start + 1  # R.pressure residual row
        cols = range(agr.sections[K].start, agr.sections[K].stop)
        # exactly one loop-mdot column carries the R.pressure derivative
        return max(J[row, c] for c in cols)

    return mdot_derivative


def test_alg_jacobian_step_floor_is_accurate_for_near_zero_mdot():
    """The FD step floor must be ~sqrt(eps), not 1e-12, so that the mdot column of
    a dp row is not quantized to the ulp of the large gravity head at flow reversal
    (mdot -> 0); the derivative stays ~1.0."""
    derivative = _gravity_resistor_loop(resistance=1.0)
    for mdot in (0.0, 1e-6, 1e-4, 1.0):
        assert derivative(mdot) == pytest.approx(1.0, abs=2e-4), f"at mdot={mdot}"


def test_alg_jacobian_does_not_zero_out_subquantum_slopes():
    """A laminar-scale slope below one quantization step of the 1e-12 floor
    (~1.82 for the 9.7 kPa head) must not be returned as 0.0, which would make the
    Jacobian singular in the mdot subspace at the at-rest state."""
    derivative = _gravity_resistor_loop(resistance=0.05)
    value = derivative(0.0)
    assert value == pytest.approx(0.05, abs=1e-3), value


