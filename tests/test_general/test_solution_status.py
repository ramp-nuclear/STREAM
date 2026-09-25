"""A :class:`Solution` carries its completion status and fired events, and threshold
checkers read a Solution's true absolute times.

The status/events fields are metadata only: two-arg positional construction is
unchanged, ``__bool__``/``__eq__`` semantics are frozen, and a converged run is
unchanged. Only failing/stopping paths gain a marker.
"""

import numpy as np
import pytest

from stream import Aggregator, Calculation, unpacked
from stream.aggregator import Event, Solution, SolveStatus  # re-exported next to Solution
from stream.aggregator.solution import Event as _EventFromModule
from stream.analysis.thresholds import _as_states
from stream.composition import Calculation_factory
from stream.units import Array1D
from stream.utilities import ignore_warnings


def test_reexport_matches_module():
    """SolveStatus/Event are the same objects whether imported from the package or
    the module (the explicit ``as`` re-exports next to Solution)."""
    assert Event is _EventFromModule


# --- the Solution object contract ---------------------------


def test_two_arg_positional_construction_is_bit_neutral():
    """Solution(time, data) — the only form the whole codebase uses — defaults to a
    COMPLETED status with no events; bool()/completed/t_stop follow."""
    t = np.array([0.0, 1.0])
    d = np.array([[1.0], [2.0]])
    sol = Solution(t, d)
    assert sol.status is SolveStatus.COMPLETED
    assert sol.events == ()
    assert sol.completed is True
    assert sol.t_stop is None
    assert bool(sol) is True  # unchanged: data is not None
    assert Solution(t, None).__bool__() is False  # unchanged: None data is falsy


def test_eq_ignores_status_and_events():
    """__eq__ is frozen to time/data only: a STOPPED and a COMPLETED solution with
    identical arrays are equal, even though their status/events fields differ."""
    t = np.array([0.0, 1.0])
    d = np.array([[1.0], [2.0]])
    complete = Solution(t, d)
    stopped = Solution(t, d, status=SolveStatus.STOPPED, events=(Event(1.0, ("x",), True),))
    assert complete == stopped
    assert stopped.status is SolveStatus.STOPPED
    assert stopped.completed is False
    assert stopped.t_stop == 1.0


def test_eq_shape_mismatch_raises_clear_message():
    """A shape mismatch that does not broadcast now raises a domain ValueError naming
    both time shapes and both data shapes instead of numpy's broadcast error."""
    a = Solution(np.array([0.0, 1.0]), np.array([[1.0], [2.0]]))
    b = Solution(np.array([0.0, 1.0, 2.0]), np.array([[1.0], [2.0], [3.0]]))
    with pytest.raises(ValueError) as exc:
        a.__eq__(b)
    msg = str(exc.value)
    assert "(2,)" in msg and "(3,)" in msg  # both time shapes
    assert "(2, 1)" in msg and "(3, 1)" in msg  # both data shapes
    assert "same Aggregator" in msg
    assert "run length" in msg


# --- population from Aggregator.solve ------------------------


class _AlgStopCalc(Calculation):
    """A StopCalc: algebraic x = c(t); asks to stop once solved x >= 10.5."""

    name = "x"

    @unpacked
    def calculate(self, variables, *, c, **_) -> Array1D:
        return np.atleast_1d(np.asarray(variables)[0] - c)

    def should_continue(self, variables, **_) -> bool:
        return float(np.asarray(variables)[0]) < 10.5

    @property
    def mass_vector(self):
        return np.array([False])

    @property
    def variables(self):
        return {"x": 0}


def test_quasistatic_alg_stop_is_stopped_by_length_rule():
    """A quasi-static ALG stop yields fewer rows than requested. ALG records no
    events; STOPPED comes from the length rule, and t_stop is the last solved time."""
    agr = Aggregator.from_decoupled(_AlgStopCalc())
    agr.funcs = {agr["x"]: {"c": lambda t: t + 10.0}}  # c(0)=10, c(1)=11 -> stop after t=1
    requested = [0.0, 1.0, 2.0, 3.0]
    sol = agr.solve(np.array([10.0]), time=requested, eq_type="ALG")
    assert len(sol.time) < len(requested)
    assert sol.status is SolveStatus.STOPPED
    assert sol.t_stop == 1.0
    assert sol.completed is False
    assert sol.events == ()  # ALG paths never record; status via the length rule
    assert bool(sol) is True


class _Trip(Calculation):
    """A linear ramp y = y0 - t with a localizable margin (y - level); latches a stop
    once the state reaches the level. The +1e-3 buffer absorbs root-localization
    epsilon so the latch is deterministic at the confirmed root."""

    name = "trip"

    def __init__(self, level):
        self._level = level
        self._tripped = False

    def calculate(self, variables, **_) -> Array1D:
        return np.array([-1.0])

    def event_margin(self, variables, **_) -> Array1D:
        return np.array([float(np.asarray(variables)[0]) - self._level])

    def change_state(self, variables, **_):
        if float(np.asarray(variables)[0]) <= self._level + 1e-3:
            self._tripped = True

    def should_continue(self, variables, **_) -> bool:
        return not self._tripped

    @property
    def mass_vector(self):
        return np.array([True])

    @property
    def variables(self):
        return {"y": 0}


@pytest.mark.parametrize("eq_type", ["ODE", "DAE"])
def test_margin_stop_records_terminal_event_with_node_name(eq_type):
    """A margin-driven terminal stop records exactly one terminal Event carrying the
    stopping node's name; the run is STOPPED and t_stop is the event time. Covers both
    the ODE (2-arg) and DAE (3-arg) recording wrappers."""
    agr = Aggregator.from_decoupled(_Trip(level=0.5))
    time = np.linspace(0.0, 1.0, 11)  # y = 1 - t crosses 0.5 at t=0.5
    sol = agr.solve(np.array([1.0]), time=time, eq_type=eq_type)
    assert sol.status is SolveStatus.STOPPED
    terminal = [e for e in sol.events if e.terminal]
    assert len(terminal) == 1
    assert terminal[0].stopped == ("trip",)
    assert terminal[0].t == pytest.approx(0.5, abs=1e-3)
    assert sol.t_stop == pytest.approx(0.5, abs=1e-3)
    assert sol.completed is False


class _MarginBounce(Calculation):
    """A ramp whose margin crosses zero (a confirmed root) but never requests a stop.
    At the transition it reverses direction so the margin climbs away and does not
    immediately re-cross — a real non-terminal transition with an intact run."""

    name = "bounce"

    def __init__(self, level):
        self._level = level
        self._dir = -1.0

    def calculate(self, variables, **_) -> Array1D:
        return np.array([self._dir])

    def event_margin(self, variables, **_) -> Array1D:
        return np.array([float(np.asarray(variables)[0]) - self._level])

    def change_state(self, variables, **_):
        if float(np.asarray(variables)[0]) <= self._level + 1e-3:
            self._dir = 1.0  # reverse: climb back up so the margin does not re-cross

    def should_continue(self, variables, **_) -> bool:
        return True  # a transition, never a stop

    @property
    def mass_vector(self):
        return np.array([True])

    @property
    def variables(self):
        return {"y": 0}


def test_non_terminal_root_records_event_and_completes():
    """A confirmed root that does not stop is recorded with stopped=() and
    terminal=False; the run reaches the end of the grid and stays COMPLETED."""
    agr = Aggregator.from_decoupled(_MarginBounce(level=0.55))
    time = np.linspace(0.0, 1.0, 11)  # y = 1 - t crosses 0.55 off-grid at t=0.45
    sol = agr.solve(np.array([1.0]), time=time, eq_type="ODE")
    assert sol.status is SolveStatus.COMPLETED
    assert sol.completed is True
    assert sol.t_stop is None
    assert len(sol.events) == 1
    ev = sol.events[0]
    assert ev.terminal is False
    assert ev.stopped == ()
    assert ev.t == pytest.approx(0.45, abs=1e-3)
    assert sol.time[-1] == pytest.approx(1.0)


def test_t0_precheck_stop_records_one_terminal_event_at_t0():
    """A margin already non-positive at t0 stops before any integration: a one-row
    Solution, STOPPED, with a single terminal event recorded at t0."""
    agr = Aggregator.from_decoupled(_Trip(level=1.5))  # margin at t0 = 1.0 - 1.5 < 0
    time = np.linspace(0.0, 1.0, 11)
    sol = agr.solve(np.array([1.0]), time=time, eq_type="ODE")
    assert sol.data.shape[0] == 1
    assert sol.time.tolist() == [0.0]
    assert sol.status is SolveStatus.STOPPED
    assert len(sol.events) == 1
    assert sol.events[0].terminal is True
    assert sol.events[0].t == 0.0
    assert sol.events[0].stopped == ("trip",)
    assert sol.t_stop == 0.0


class _PollStop(Calculation):
    """An event node with NO localizable margin (empty event_margin) -> routed through
    the polling driver; latches a stop once the ramp drops to/below level."""

    name = "poll"

    def __init__(self, level):
        self._level = level
        self._tripped = False

    def calculate(self, variables, **_) -> Array1D:
        return np.array([-1.0])

    def change_state(self, variables, **_):
        if float(np.asarray(variables)[0]) <= self._level:
            self._tripped = True

    def should_continue(self, variables, **_) -> bool:
        return not self._tripped

    @property
    def mass_vector(self):
        return np.array([True])

    @property
    def variables(self):
        return {"y": 0}


def test_polling_driver_stop_records_single_terminal_event():
    """The polling fallback records ONLY the honored stop (no per-poll spam): exactly
    one terminal Event, and STOPPED."""
    agr = Aggregator.from_decoupled(_PollStop(level=0.55))
    time = np.linspace(0.0, 1.0, 11)  # first grid point with y <= 0.55 is t=0.5
    sol = agr.solve(np.array([1.0]), time=time, eq_type="ODE")
    assert sol.status is SolveStatus.STOPPED
    assert len(sol.events) == 1
    assert sol.events[0].terminal is True
    assert sol.events[0].stopped == ("poll",)
    assert sol.completed is False


# --- _as_states reads a Solution's true absolute times -------


def test_as_states_uses_solution_true_times_not_range():
    """Handed a Solution, _as_states saves it (true absolute times) rather than
    falling through to a raw-array range(len) time axis."""
    Node = Calculation_factory(lambda y: -y, [False], dict(x=0))
    agr = Aggregator.from_decoupled(Node())
    sol = Solution(np.array([0.0, 2.5]), np.array([[1.0], [2.0]]))
    states = _as_states(sol, agr, None)
    assert set(states.keys()) == {0.0, 2.5}  # true absolute times
    assert 2.5 in states.keys()  # not range(len(sol.time)) == {0, 1}
