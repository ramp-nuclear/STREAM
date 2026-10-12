import numpy as np
import pytest
from networkx import DiGraph
from scikits.odes import dae

from stream.aggregator import Aggregator, vars_
from stream.calculations import Gravity, Pump, Resistor
from stream.calculations.ideal.resistors import ResistorSum
from stream.composition import Calculation_factory
from stream.composition.cycle import flow_edge, flow_graph
from stream.composition.cycle import flow_graph_to_agr_and_k as agr_k
from stream.jacobians import ALG_jacobian
from stream.solvers import TransientRuntimeError, _continuous_mode_dae
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


class _StopAfter:
    """Minimal algebraic calculation with root x = 0 whose should_continue turns
    False after ``n`` solved time points, driving the ALG quasi-static stop path."""

    def __init__(self, n):
        self.name = "stopper"
        self.mass_vector = np.array([False])
        self.variables = {"x": 0}
        self._n = n
        self._calls = 0

    def __len__(self):
        return 1

    def __hash__(self):
        return hash(self.name)

    def calculate(self, variables, **_):
        return np.array([-variables[0]])

    def indices(self, variable, asking=None):
        return self.variables[variable]

    def load(self, state):
        return np.array([state["x"]])

    def save(self, vector, t=0):
        return {"x": vector[0]}

    strict_save = save

    def change_state(self, variables, **_):
        pass

    def should_continue(self, variables, **_):
        self._calls += 1
        return self._calls <= self._n


def test_alg_quasi_static_early_stop_keeps_time_and_data_aligned():
    """On an early stop the output must stay aligned (time length == data rows) and
    include the stop-triggering solved row, so save() round-trips without an
    IndexError."""
    stopper = _StopAfter(n=40)
    agr = Aggregator(_single_node_graph(stopper))
    time = np.linspace(0, 100, 101)
    with ignore_warnings(UserWarning):
        sol = agr.solve(np.array([1.0]), time=time, eq_type="ALG")
    assert len(sol.time) == sol.data.shape[0]
    # rows: the initial guess at t=0, 40 continuing points, and the stop-triggering point at t=41
    assert sol.time[-1] == pytest.approx(41.0)
    assert sol.data.shape[0] == 42
    saved = agr.save(sol)
    assert len(saved) == len(sol.time)


def _single_node_graph(node):
    g = DiGraph()
    g.add_node(node)
    return g


class _EchoInput:
    """A 3-variable calculation whose calculate() returns the result it was built with,
    whatever its length."""

    variables = {"a": 0, "b": 1, "c": 2}
    mass_vector = np.array([False, False, False])

    def __init__(self, result, name="wrong"):
        self._result = result
        self.name = name

    def __len__(self):
        return 3

    def __hash__(self):
        return hash(self.name)

    def calculate(self, y, **_):
        return self._result

    def indices(self, var, asking=None):
        return self.variables[var]

    def load(self, state):
        return np.zeros(3)

    def save(self, y, t=0):
        return {"a": y[0], "b": y[1], "c": y[2]}


@pytest.mark.parametrize("result", [np.array([7.0]), 5.0, np.array([1.0, 2.0])])
def test_compute_rejects_wrong_length_calculate_result(result):
    """compute() must reject any size mismatch and name the offending calculation.
    A scalar or length-1 return would silently broadcast across the section, and a
    length-2 return would raise a bare numpy ValueError naming no calculation."""
    agr = Aggregator(_single_node_graph(_EchoInput(result)))
    with pytest.raises(ValueError, match="would silently broadcast") as excinfo:
        agr.compute(np.array([1.0, 2.0, 3.0]), 0.0)
    assert "'wrong': (3, " in str(excinfo.value)


def test_compute_reports_every_wrong_length_result_at_once():
    """All calculations whose result length differs from their section are named in
    one error, each with its expected and returned lengths."""
    graph = DiGraph()
    graph.add_nodes_from([_EchoInput(5.0, name="first"), _EchoInput(np.ones(2), name="second")])
    agr = Aggregator(graph)
    with pytest.raises(ValueError, match="would silently broadcast") as excinfo:
        agr.compute(np.zeros(6), 0.0)
    assert "'first': (3, 1)" in str(excinfo.value)
    assert "'second': (3, 2)" in str(excinfo.value)


def test_compute_accepts_correct_length_result():
    """The guard must not reject a correctly-shaped result."""
    agr = Aggregator(_single_node_graph(_EchoInput(np.array([1.0, 2.0, 3.0]))))
    out = agr.compute(np.array([4.0, 5.0, 6.0]), 0.0)
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def _coupled_algebraic_aggregator():
    """A tiny coupled 2-variable algebraic system with a unique root (y=2, x=1)."""
    Addition = Calculation_factory(lambda y, *, x: y - x - 1.0, [False], dict(y=0))
    Multiplication = Calculation_factory(lambda x, *, y: x - 0.5 * y, [False], dict(x=0))
    add = Addition(name="Add")
    multiply = Multiplication(name="Multiply")
    graph = DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))])
    return Aggregator(graph)


@pytest.mark.parametrize("eq_type", [None, "ODE", "DAE", "ALG"])
def test_solve_with_time_none_points_at_solve_steady(eq_type):
    """solve() integrates or steps through time points; without them it raises and
    names solve_steady, the steady root find."""
    agr = _coupled_algebraic_aggregator()
    with pytest.raises(ValueError, match="solve_steady"):
        agr.solve(np.array([0.0, 0.0]), None, eq_type=eq_type)


def _blowup_aggregator():
    """y' = y^2 with y0 = 1: a finite-time blowup at t = 1 that makes RK45 fail
    (step size underflow / overflow) before reaching the requested horizon."""
    Blowup = Calculation_factory(calculate=lambda y: y**2, mass_vector=[True], variables=dict(y=0))
    return Aggregator.from_decoupled(Blowup())


def test_ode_backend_raises_on_solver_failure_instead_of_truncating():
    """An ODE solver failure must surface as TransientRuntimeError rather than
    returning only the reached t_eval points (a truncated data array misaligned
    against the requested time vector, which crashes save())."""
    agr = _blowup_aggregator()
    time = np.linspace(0.0, 2.0, 101)
    with ignore_warnings(RuntimeWarning):  # overflow in y**2
        with pytest.raises(TransientRuntimeError):
            agr.solve(np.array([1.0]), time)


def test_ode_backend_solution_time_and_data_stay_aligned():
    """On a well-posed ODE the ODE branch must rebind time from the solver so
    len(time) == data rows and save() round-trips without an IndexError."""
    Decay = Calculation_factory(calculate=lambda y: -y, mass_vector=[True], variables=dict(y=0))
    agr = Aggregator.from_decoupled(Decay())
    time = np.linspace(0.0, 1.0, 11)
    sol = agr.solve(np.array([1.0]), time)
    assert len(sol.time) == sol.data.shape[0]
    saved = agr.save(sol)
    assert len(saved) == len(sol.time)


class _FakeVals:
    """A scikits.odes-like `.values`: attribute access AND (t, y, ydot) unpacking."""

    def __init__(self, t, y, ydot):
        self.t, self.y, self.ydot = t, y, ydot

    def __iter__(self):
        return iter((self.t, self.y, self.ydot))


class _FakeSol:
    def __init__(self, t, y, ydot):
        self.values = _FakeVals(t, y, ydot)


def _first_segment_then(raiser):
    """A fake `solve` for _continuous_mode_dae: the first call returns a segment
    ending at t=0.5 (so the continuous loop restarts), the restart calls `raiser`."""
    calls = {"n": 0}

    def solve(time_, y0_, yp0_):
        calls["n"] += 1
        if calls["n"] == 1:
            t_ = np.array([0.0, 0.25, 0.5])
            y_ = np.tile(np.asarray(y0_, float), (3, 1))
            return _FakeSol(t_, y_, np.zeros_like(y_))
        return raiser()

    return solve


def test_continuous_mode_restart_ic_failure_keeps_the_partial_trajectory():
    """When a restarted DAE solve fails in its consistent-initialization step the
    error carries no trajectory; the segments integrated before it must still be
    returned."""

    def raise_ic_failure():
        raise TransientRuntimeError(None, None, None, "IC computation failed")

    solve = _first_segment_then(raise_ic_failure)
    with ignore_warnings(UserWarning):
        y, t = _continuous_mode_dae(solve, np.linspace(0, 1, 5), np.array([0.0, 1.0]), np.zeros(2))
    assert np.array_equal(t, [0.0, 0.25, 0.5])
    assert y.shape[0] == 3


def test_continuous_mode_restart_failure_strips_duplicated_restart_point():
    """A restart that fails mid-integration contributes its partial segment without
    repeating the restart time."""

    def raise_with_partial():
        raise TransientRuntimeError(
            np.array([0.5, 0.55]), np.tile([0.0, 1.0], (2, 1)), np.zeros((2, 2)), "failed mid-restart"
        )

    solve = _first_segment_then(raise_with_partial)
    with ignore_warnings(UserWarning):
        _, t = _continuous_mode_dae(solve, np.linspace(0, 1, 5), np.array([0.0, 1.0]), np.zeros(2))
    assert np.all(np.diff(t) > 0), f"duplicated/!monotone time: {t}"
    assert np.array_equal(t, [0.0, 0.25, 0.5, 0.55])
