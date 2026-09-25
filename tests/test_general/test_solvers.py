import numpy as np
import pytest
from scikits.odes import dae

from stream.aggregator import Aggregator
from stream.composition import Calculation_factory
from stream.jacobians import ALG_jacobian
from stream.solvers import TransientRuntimeError, _event_loop_dae
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
    from stream.calculations import Gravity, Pump, Resistor
    from stream.calculations.ideal.resistors import ResistorSum
    from stream.composition.cycle import flow_edge, flow_graph
    from stream.composition.cycle import flow_graph_to_agr_and_k as agr_k
    from stream.state import State
    from stream.substances import light_water

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
        assert derivative(mdot) == pytest.approx(1.0, abs=1e-3), f"at mdot={mdot}"


def test_alg_jacobian_does_not_zero_out_subquantum_slopes():
    """A laminar-scale slope below one quantization step of the 1e-12 floor
    (~1.82 for the 9.7 kPa head) must not be returned as 0.0, which would make the
    Jacobian singular in the mdot subspace at the at-rest state."""
    derivative = _gravity_resistor_loop(resistance=0.05)
    value = derivative(0.0)
    assert value != 0.0
    assert value == pytest.approx(0.05, abs=1e-3)


class _StopAfter:
    """Minimal algebraic calculation with root x = 0 whose should_continue turns
    False after ``n`` solved time points — drives the ALG quasi-static stop path."""

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
    # 40 continuing points (t=1..40) + the solved stop-triggering point at t=41,
    # plus the initial-guess row at t=0 -> 42 rows ending exactly at t=41.
    assert sol.time[-1] == pytest.approx(41.0)
    assert sol.data.shape[0] == 42
    saved = agr.save(sol)  # must not raise
    assert len(saved) == len(sol.time)


def _single_node_graph(node):
    from networkx import DiGraph

    g = DiGraph()
    g.add_node(node)
    return g


class _WrongLength:
    """A 3-variable calculation whose calculate() returns the wrong length."""

    name = "wrong"
    variables = {"a": 0, "b": 1, "c": 2}
    mass_vector = np.array([False, False, False])

    def __init__(self, result):
        self._result = result

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
    """compute() must reject any size mismatch — a scalar/length-1 return would
    silently broadcast across the section, and a length-2 return would raise a bare
    numpy ValueError naming no calculation — and name the offending calculation."""
    agr = Aggregator(_single_node_graph(_WrongLength(result)))
    with pytest.raises(ValueError, match="wrong"):
        agr.compute(np.array([1.0, 2.0, 3.0]), 0.0)


def test_compute_accepts_correct_length_result():
    """The guard must not reject a correctly-shaped result."""
    agr = Aggregator(_single_node_graph(_WrongLength(np.array([1.0, 2.0, 3.0]))))
    out = agr.compute(np.array([4.0, 5.0, 6.0]), 0.0)
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def _coupled_algebraic_aggregator():
    """A tiny coupled 2-variable algebraic system with a unique root (y=2, x=1)."""
    from networkx import DiGraph

    from stream.aggregator import vars_

    Addition = Calculation_factory(lambda y, *, x: y - x - 1.0, [False], dict(y=0))
    Multiplication = Calculation_factory(lambda x, *, y: x - 0.5 * y, [False], dict(x=0))
    add = Addition(name="Add")
    multiply = Multiplication(name="Multiply")
    graph = DiGraph([(add, multiply, vars_("y")), (multiply, add, vars_("x"))])
    return Aggregator(graph), add


def test_solve_with_time_none_returns_well_formed_solution():
    """The documented `solve(guess, None)` steady path must return a proper
    (time, variable) Solution: a length-1 time axis and 2-D single-row data, so
    save()/at_times()/== all work."""
    agr, add = _coupled_algebraic_aggregator()
    guess = np.array([0.0, 0.0])
    sol = agr.solve(guess, None)

    assert np.ndim(sol.time) == 1 and len(sol.time) == 1
    assert np.ndim(sol.data) == 2 and sol.data.shape[0] == 1
    # matches the bare solve_steady result on the single row
    assert np.allclose(sol.data[0], agr.solve_steady(guess))
    # every documented downstream consumer must work
    saved = agr.save(sol)
    assert len(saved) == 1
    assert agr.at_times(sol, add, "y").shape == (1,)
    assert sol == sol


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
    saved = agr.save(sol)  # must not raise
    assert len(saved) == len(sol.time)


class _FakeVals:
    """A scikits.odes-like `.values`: attribute access AND (t, y, ydot) unpacking."""

    def __init__(self, t, y, ydot):
        self.t, self.y, self.ydot = t, y, ydot

    def __iter__(self):
        return iter((self.t, self.y, self.ydot))


class _FakeRoots:
    def __init__(self, t, y, ydot):
        self.t, self.y, self.ydot = t, y, ydot


class _FakeSol:
    def __init__(self, t, y, ydot, roots=None):
        self.values = _FakeVals(t, y, ydot)
        self.roots = roots if roots is not None else _FakeRoots(None, None, None)


def _first_segment_then(raiser):
    """A fake `solve` for _event_loop_dae: the first call returns a segment ending
    at t=0.5 with a root there (so the loop restarts from the root), and the
    restart call raises via `raiser`."""
    calls = {"n": 0}

    def solve(time_, y0_, yp0_):
        calls["n"] += 1
        if calls["n"] == 1:
            t_ = np.array([0.0, 0.25, 0.5])
            y_ = np.tile(np.asarray(y0_, float), (3, 1))
            root = _FakeRoots(np.array([0.5]), y_[-1:], np.zeros((1, y_.shape[1])))
            return _FakeSol(t_, y_, np.zeros_like(y_), roots=root)
        return raiser()

    return solve


def test_continuous_mode_restart_ic_failure_surfaces_with_partial_trajectory():
    """When a restarted DAE solve fails during IC computation, IDA returns t=y=None;
    concat(t, None) must not raise a bare ValueError that masks the failure. The
    TransientRuntimeError must be surfaced (not silently swallowed), carrying the
    pre-failure segment on e.t/e.y for post-mortem."""

    def raise_ic_failure():
        raise TransientRuntimeError(None, None, None, "IC computation failed")

    solve = _first_segment_then(raise_ic_failure)
    with ignore_warnings(UserWarning):
        with pytest.raises(TransientRuntimeError, match="IC computation failed") as exc:
            _event_loop_dae(solve, np.linspace(0, 1, 5), np.array([0.0, 1.0]), np.zeros(2))
    e = exc.value
    # the pre-failure segment survives on the exception; no bare ValueError masks it
    assert np.array_equal(e.t, [0.0, 0.25, 0.5])
    assert e.y.shape[0] == 3


def test_continuous_mode_restart_failure_surfaces_full_trajectory_no_duplicated_point():
    """On a mid-integration restart failure the accumulated trajectory (pre-failure
    segments + the failed segment's partial output) is attached to the raised
    TransientRuntimeError. Row-stripping still applies — the failed segment's first
    row is the restart time (== prior t[-1]) and must not be duplicated — so e.t
    stays strictly monotone, and the failure is raised, not silently truncated."""

    def raise_with_partial():
        raise TransientRuntimeError(
            np.array([0.5, 0.55]), np.tile([0.0, 1.0], (2, 1)), np.zeros((2, 2)), "failed mid-restart"
        )

    solve = _first_segment_then(raise_with_partial)
    with ignore_warnings(UserWarning):
        with pytest.raises(TransientRuntimeError, match="failed mid-restart") as exc:
            _event_loop_dae(solve, np.linspace(0, 1, 5), np.array([0.0, 1.0]), np.zeros(2))
    t = exc.value.t
    assert np.all(np.diff(t) > 0), f"duplicated/!monotone time: {t}"
    assert np.array_equal(t, [0.0, 0.25, 0.5, 0.55])


def test_event_loop_raises_instead_of_hanging_on_nonadvancing_root():
    """E3: a persistent condition that re-roots at the same time made the old
    continuous loop spin forever (values.t[-1] never advanced past the prior grid
    point). The driver must detect the non-advancing root and raise, not hang."""

    def solve(time_, y0_, yp0_):
        t_ = np.asarray(time_, float)
        y_ = np.tile(np.asarray(y0_, float), (1, 1))
        # Always report a root at the segment start -> t_root never advances.
        root = _FakeRoots(np.array([t_[0]]), y_, np.zeros_like(y_))
        return _FakeSol(t_[:1], y_, np.zeros_like(y_), roots=root)

    with pytest.raises(TransientRuntimeError, match="did not advance"):
        _event_loop_dae(
            solve,
            np.linspace(0, 1, 5),
            np.array([1.0]),
            np.zeros(1),
            on_event=lambda t, y, yp: True,
            continuous=False,
        )


def test_event_loop_no_duplicate_time_when_terminal_event_on_grid():
    """A terminal event landing exactly on a requested grid time is already in
    solution.values (IDA's t_out==t path); the terminal append must not add it
    again and produce two equal final times."""

    def solve(time_, y0_, yp0_):
        t_ = np.asarray(time_, float)
        y_ = np.tile(np.asarray(y0_, float), (len(t_), 1))
        root = _FakeRoots(np.array([t_[-1]]), y_[-1:], np.zeros((1, y_.shape[1])))  # root == last grid time
        return _FakeSol(t_, y_, np.zeros_like(y_), roots=root)

    _, t = _event_loop_dae(
        solve,
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0]),
        np.zeros(1),
        on_event=lambda tr, yr, ypr: tr < 1.0,  # continue at start, terminal at the root t=1.0
        continuous=False,
    )
    assert np.all(np.diff(t) > 0), f"duplicated final time: {t}"
    assert np.array_equal(t, [0.0, 0.5, 1.0])


def test_algebraic_attaches_last_iterate_on_failure():
    """A failed steady root-find keeps its last iterate on the error (err.y),
    like scaled_newton/pseudo_transient — so callers can inspect where it stalled
    (e.g. to attribute a saturation-driven steady failure to the offending state)."""
    from stream.solvers import AlgRuntimeError, algebraic

    with pytest.raises(AlgRuntimeError) as exc:
        algebraic(F=lambda y, t: y**2 + 1.0, y0=np.array([3.0]))  # no real root -> hybr stalls
    e = exc.value
    assert getattr(e, "y", None) is not None
    assert len(np.atleast_1d(e.y)) == 1


class _Boom(RuntimeError):
    pass


def test_event_loop_dae_on_event_raise_carries_pre_event_trajectory():
    """T2: when an event handler raises (e.g. a domain guard stopping the run at a
    physical limit), the valid trajectory up to and including the crossing is
    attached to the exception for post-mortem — the solver never learns what the
    guard is, it just preserves the state."""

    def solve(time_, y0_, yp0_):
        t_ = np.asarray(time_, float)
        y_ = np.tile(np.asarray(y0_, float), (len(t_), 1))
        root = _FakeRoots(np.array([t_[-1]]), y_[-1:], np.zeros((1, y_.shape[1])))
        return _FakeSol(t_, y_, np.zeros_like(y_), roots=root)

    def on_event(t, y, yp):
        if t > 0.0:
            raise _Boom("guard tripped at the crossing")
        return True

    with pytest.raises(_Boom) as exc:
        _event_loop_dae(solve, np.array([0.0, 0.5, 1.0]), np.array([2.0]), np.zeros(1), on_event=on_event)
    e = exc.value
    assert np.array_equal(e.t, [0.0, 0.5, 1.0])  # crossing point included
    assert e.y.shape[0] == len(e.t)


def test_event_loop_dae_on_event_raise_at_t0_carries_initial_state():
    """T2: a handler that raises already at t0 (an initial condition past the limit)
    fails loud with the initial state attached, not a bare error."""

    def solve(*_a):
        raise AssertionError("should not integrate past an already-tripped t0")

    def on_event(t, y, yp):
        raise _Boom("already past the limit at t0")

    with pytest.raises(_Boom) as exc:
        _event_loop_dae(solve, np.array([0.0, 1.0]), np.array([9.0]), np.zeros(1), on_event=on_event)
    e = exc.value
    assert np.array_equal(e.t, [0.0]) and e.y.shape[0] == 1
