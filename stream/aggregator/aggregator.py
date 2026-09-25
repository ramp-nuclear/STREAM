import logging
import numbers
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Iterable, Literal, NamedTuple, Protocol, Sequence, overload

import numpy as np
from cytoolz import unique, valmap
from networkx import DiGraph, compose

from stream.calculation import Calculation
from stream.errors import StreamConstructionError, StreamError, hint_block
from stream.scales import scale_vector
from stream.solvers import (
    AlgRuntimeError,
    TransientRuntimeError,
    _merge_failure_trajectory,
    algebraic,
    differential,
    differential_algebraic,
    pseudo_transient,
    scaled_newton,
)
from stream.state import DictState, State, StateTimeseries
from stream.units import Array1D, Array2D, Name, Place, Second
from stream.utilities import STREAM_DEBUG, concat, offset

from .solution import Event, Solution, SolveStatus
from .utils import (
    VARS,
    BaseAgr,
    ExternalFunctions,
    draw_aggregator,
    map_externals,
    non_unique_calculations,
    partition,
)

__all__ = ["Aggregator", "CalculationGraph", "Location", "NonUniqueCalculationNameError"]

logger = logging.getLogger("stream.aggregator")


class NonUniqueCalculationNameError(StreamError, ValueError):
    """Error to signify that calculations in an aggregator were not uniquely named."""

    pass


class Location(NamedTuple):
    """Where a state-vector row lives, in domain terms — the inverse of
    :meth:`Aggregator.var_index`.

    A :class:`Location` names *(Calculation, variable, cell)* for a single row of
    :math:`\\vec{y}`, so failures, checkers and debug tools speak of
    ``T_wall cell 3 of 'CC'`` rather than ``row 27``.

    Attributes
    ----------
    calculation: Calculation
        The section-owning calculation object (identity, not just its name).
    name: str
        ``str(calculation)`` — the calculation's name.
    variable: str
        The key in :attr:`Calculation.variables` covering this row. Kirchhoff's
        per-edge keys are already strings; any non-``str`` key is ``str()``-coerced.
        A row a section owns but no variable covers reports ``'<unmapped>'``.
    cell: int | None
        The 0-based offset within the variable's :data:`~stream.units.Place`, or
        ``None`` when the variable occupies a single scalar (``int``) place.
    """

    calculation: Calculation
    name: str
    variable: str
    cell: int | None


def _place_cells(place: Place):
    """Yield ``(cell, absolute_row)`` for an (already section-offset) ``Place``.

    A scalar ``int`` place yields a single ``(None, row)``; a ``slice`` or array
    place yields one pair per covered row, ``cell`` running 0-based over the
    variable (``int`` -> ``cell=None``; slice/array -> the offset)."""
    if isinstance(place, (int, np.integer)):
        yield None, int(place)
    elif isinstance(place, slice):
        for cell, row in enumerate(range(place.start, place.stop)):
            yield cell, row
    else:  # np.ndarray of indices
        for cell, row in enumerate(np.asarray(place).ravel()):
            yield cell, int(row)


class ProgressBarLike(Protocol):
    """What a progresbar must supply to not crash. Hopefully its methods have something to do with updating a display of a progressbar..."""

    def update(self, s: int) -> None:
        """Method to update the state of the progress bar.
        Parameters
        ----------
        s: int
            The step number to update it.
        """
        ...

    def finish(self) -> None:
        """Method to finalize and close the progressbar."""
        ...


def _resolve_typ(agr: "Aggregator", scales: dict[str, float] | Array1D | None) -> Array1D:
    """Resolve a ``scales`` argument to a length-``N`` nominal-magnitude vector.

    ``None`` uses :data:`~stream.scales.DEFAULT_SCALES` via
    :func:`~stream.scales.scale_vector`; a ``dict`` is treated as a
    name -> scale registry; anything else is taken as a ready length-``N`` ``typ``
    vector (validated against ``len(agr)``).
    """
    if scales is None:
        return scale_vector(agr)
    if isinstance(scales, dict):
        return scale_vector(agr, registry=scales)
    typ = np.asarray(scales, dtype=float)
    if typ.shape != (len(agr),):
        raise ValueError(f"scales array has length {len(typ)}; expected {len(agr)} (the state vector length)")
    return typ


def _rung_nF(agr: "Aggregator", y) -> float:
    """``||F(y)||`` for a cascade record; returns ``nan`` if evaluation fails."""
    try:
        return float(np.linalg.norm(agr.compute(y, 0)))
    except Exception:
        return float("nan")


def _rung_record(name: str, start: Array1D, err: BaseException, agr: "Aggregator") -> tuple:
    """One ``(rung_name, start_vector, outcome_first_line, ||F(iterate)||)`` record
    of a globalize-cascade rung that ran and failed."""
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    outcome = lines[0] if lines else str(err)
    if outcome.endswith(":") and getattr(err, "backend_message", None):
        # scipy wraps the real reason onto the next line after a "...:" preamble; backend_message has it unwrapped.
        outcome = f"{outcome} " + " ".join(err.backend_message.split())
    return (name, start, outcome, _rung_nF(agr, getattr(err, "y", None)))


def _cascade_error(rungs: list[tuple], last_err: BaseException, agr: "Aggregator") -> AlgRuntimeError:
    """Assemble the single aggregate ``AlgRuntimeError`` for an exhausted cascade:
    one line per rung (rung, symbolic start, outcome, ``||F||``), a hint, the
    ``.rungs`` record tuple, and ``.y`` = the last rung's iterate."""
    lines = [f"solve_steady: all {len(rungs)} rungs of the globalize cascade failed"]
    for name, _start, outcome, nF in rungs:
        origin = "from pseudo_transient's iterate" if name == "scaled_newton polish" else "from the guess"
        lines.append(f"  {name} ({origin}): {outcome} — ‖F‖={nF:.3e}")
    err = AlgRuntimeError(
        "\n".join(lines)
        + hint_block(
            "a finite ‖F‖ with a stalled cascade usually means the guess, not the physics — "
            "re-solve from a different/expert guess to discriminate",
            "inspect agr.worst_residuals(err.y)",
        )
    )
    err.rungs = tuple(rungs)
    err.y = getattr(last_err, "y", None)
    # Enrich exactly once, here, so all three solve_steady raise sites are covered
    # (§3.2). Best-effort (P8) and notes-only (P4): never touches the message above.
    agr._enrich_failure(err)
    return err


class Aggregator:
    r"""
    Collects and calls calculations in order to solve a large coupled
    generalized equation (presented in the calculation module). This
    object connects the Solver to the Calculations. It receives input from
    the Solver, which is then distributed to the different Calculations.
    The Calculations then compute their allotted piece of the functional.
    The Aggregator then passes the results back to the Solver.

    Attributes
    ----------
    sections: dict[Calculation, slice]
        A mapping of each :class:`~.Calculation` to its
        given slice of :math:`\vec{y}`.
    mass: Array1D
        The gathered :meth:`~.Calculation.mass_vector`.
    external: dict[Calculation, dict[str, dict[Calculation, Place]]]
        For each calculation, maps the variable names which are passed from
        other calculations, and for each variable name maps the
        calculations which pass such a variable and the places in
        :math:`\vec{y}` from which they are to be taken

    See Also
    --------
    .Calculation

    """

    def __init__(self, graph: DiGraph, funcs: ExternalFunctions | None = None):
        r"""
        Create an instance of Aggregator. The following are required:

        Parameters
        ----------
        graph: DiGraph
            containing Calculations at nodes and variable coupling
            at the edges.
        funcs: ExternalFunctions | None
            time-only-dependent functions, which are user controlled.

        """
        if non_unique := non_unique_calculations(graph):
            raise NonUniqueCalculationNameError(f"Calculations were not uniquely named: {non_unique}")
        self.graph = graph
        self.funcs = funcs or {}
        self.sections, self.vector_length = partition(graph.nodes)
        self.mass = concat(*(node.mass_vector for node in graph))
        self.external = map_externals(graph.edges(data=VARS), self.sections)
        self._nodes_num = len(self.graph)
        logger.log(STREAM_DEBUG, f"New Aggregator of length {len(self)}")

    def __len__(self):
        return self.vector_length

    def __getitem__(self, item: str) -> Calculation:
        d = {node.name: node for node in self.graph}
        return d[item]

    def draw(self, node_options=None, edge_options=None):
        r"""Method equivalent of :func:`draw_aggregator`"""
        draw_aggregator(self.graph, node_options, edge_options)

    @classmethod
    def from_decoupled(cls, *nodes: Calculation, funcs: ExternalFunctions | None = None) -> "Aggregator":
        r"""Instantiate an Aggregator from calculations, which are not connected.

        Parameters
        ----------
        nodes: Calculation
            Calculations to ba added
        funcs: ExternalFunctions or None
            to be passed to the aggregator

        Returns
        -------
        agr: Aggregator
            Which contains a disconnected-nodes-graph
        """
        return CalculationGraph.from_decoupled(*nodes, funcs=funcs).to_aggregator()

    @classmethod
    def connect(
        cls,
        a: BaseAgr,
        b: BaseAgr,
        *edges: tuple[Calculation, Calculation, Iterable[Name]],
    ) -> "Aggregator":
        """
        Connect two Aggregator objects. In case of a clash, the second object
        prevails. If ``edges`` contains an edge already in either ``a.graph``
        or ``b.graph``, it is updated, not overridden.

        .. tip::
            The two inputs may share nodes. This is very useful!

        Parameters
        ----------
        a, b: Aggregator
            First (, second) input
        edges: tuple[Calculation, Calculation, Names]
            Any edges which may connect the two.

        Returns
        -------
        new: Aggregator
            A new Aggregator whose graph and functions are composed out of a,b.
        """
        return CalculationGraph.connect(a, b, *edges).to_aggregator()

    def __add__(self, other: BaseAgr) -> "Aggregator":
        return self.connect(self, other)

    @classmethod
    def from_CalculationGraph(cls, a: "CalculationGraph") -> "Aggregator":
        """Creates an Aggregator using a :class:`~.stream.aggregator.CalculationGraph`
        object.

        Parameters
        ----------
        a: CalculationGraph
            Graph to use for the creation.

        """
        return cls(a.graph, a.funcs)

    def compute(self, y: Sequence[float], t: Second = 0) -> Array1D:
        r"""
        The main function of the Aggregator - bridging between the solver
        and the calculations.

        The ``graph`` attribute of Aggregator contains a set of
        :class:`~.Calculation` objects, each representing different equations
        which come together to form the full Functional
        :math:`\vec{F}(\vec{y},t)`, returned by this method. Therefore, this
        method orchestrates the inputs and outputs to each calculation and
        `aggregates` the result.

        Parameters
        ----------
        y: Sequence[float]
            guess/ result from solver
        t: Second
            time

        Returns
        -------
        Functional: Array1D
            the differential-algebraic functional f(y,t)
            which is provided in parts from the different calculations.
        """
        out = np.empty(self.vector_length)
        for node, section in self.sections.items():
            result = np.asarray(self._op("calculate", y, t, node))
            expected = section.stop - section.start
            if result.size != expected:
                name = getattr(node, "name", node)
                raise ValueError(
                    f"Calculation {name!r} returned {result.size} value(s) for its section of "
                    f"{expected} variable(s). A scalar or length-1 result would silently broadcast "
                    f"across the section and solve a different system."
                )
            out[section] = result
        return out

    def _node_external(self, node: Calculation, y: Sequence[float], t: Second) -> dict[str, dict[Calculation, Any]]:
        """Arrange the external parameters for a node at a given time.
        This includes both parameters for which other nodes are responsible,
        and for which input functions are provided. These input functions are
        evaluated at time ``t``.

        Parameters
        ----------
        node: Calculation
            The given calculation to be called
        y: Sequence[float]
            guess/result from solver
        t: Second
            time

        Returns
        -------
        external arguments: dict[str, dict[Calculation, Any]]
            to be passed as kwargs.
        """
        external = valmap(partial(valmap, y.__getitem__), self.external.get(node, {}))
        evaluated_functions = {name: {node: f(t) if callable(f) else f} for name, f in self.funcs.get(node, {}).items()}
        return external | evaluated_functions

    def _event_margins(self, y: Sequence[float], t: Second = 0) -> Array1D:
        r"""Continuous, signed event margins for the whole graph, concatenated in
        node order. This is the transient rootfn and is a **pure** function of
        ``(y, t)`` — it performs no state change, so the solver may evaluate it at
        speculative bisection midpoints without side effects. Sign changes localize
        event times; the transition is applied by :meth:`_handle_event` at the
        confirmed root, not here.
        """
        parts = [
            np.atleast_1d(np.asarray(self._op("event_margin", y, t, node), dtype=float)) for node in self.graph
        ]
        return concat(*parts) if parts else np.empty(0)

    def _handle_event(self, y: Sequence[float], t: Second) -> bool:
        r"""Apply state transitions at a confirmed event time (or, in the polling
        fallback, at an accepted output point), then decide whether to continue.

        :meth:`Calculation.change_state` is invoked on every node — it is
        idempotent and guarded on its own condition internally, so calling
        non-triggered nodes is a no-op, and we need not (and cannot, from Python)
        identify which specific margin fired. Returns ``all(should_continue)``.
        """
        logger.log(STREAM_DEBUG, f"handling event at t: {t:.8g}")
        for node in self.graph:
            self._op("change_state", y, t, node)
        sc = np.fromiter(
            (self._op("should_continue", y, t, node) for node in self.graph),
            dtype=bool,
            count=self._nodes_num,
        )
        if not all(sc):
            stopped = np.array(self.graph, dtype=object)[~sc]
            logger.warning(f"At t = {t:.5f}, the simulation has been stopped by {list(stopped)}")
        return bool(sc.all())

    def _stopped_names(self, y: Sequence[float], t: Second) -> tuple[str, ...]:
        """Names of nodes whose ``should_continue`` is False at ``(y, t)`` — the ones
        that requested the stop. Pure reads (no ``change_state``, no logging); called
        only at a confirmed event/stop time to label an :class:`.Event`."""
        return tuple(str(node) for node in self.graph if not self._op("should_continue", y, t, node))

    def _ode_event(self, k: int):
        """A ``solve_ivp`` terminal event for margin component ``k`` (positive ->
        non-positive crossing), so ODE-mode integration localizes events."""

        def event(t: Second, y: Sequence[float]) -> float:
            return float(self._event_margins(y, t)[k])

        event.terminal = True
        event.direction = -1
        return event

    def _has_event_overrides(self) -> bool:
        """Whether any node is an active event node (so its events must not be
        silently dropped even when it exposes no localizable margin)."""
        return any(node.has_event() for node in self.graph)

    def _marginless_event_nodes(self, y: Sequence[float]) -> list[Calculation]:
        """Active event nodes that expose no localizable margin at ``y``."""
        return [
            node
            for node in self.graph
            if node.has_event() and np.asarray(self._op("event_margin", y, 0.0, node)).size == 0
        ]

    def _warn_marginless_event_nodes(self, y: Sequence[float]) -> None:
        """Warn when an event-overriding node without a margin shares the graph with
        margin-bearing nodes: it is then serviced only at the other nodes' events,
        not at its own threshold. Give it an event_margin/trip_margin to fix."""
        stragglers = self._marginless_event_nodes(y)
        if stragglers:
            names = [getattr(n, "name", n) for n in stragglers]
            warnings.warn(
                f"Event-bearing calculations {names} expose no event_margin, but the graph "
                "has margin-bearing events; their change_state/should_continue will run only "
                "at the other events' times, not at their own thresholds. Provide an "
                "event_margin (e.g. a trip_margin) so their events localize.",
                stacklevel=2,
            )

    def _integrate_with_polling(
        self, solve_segment: Any, time: Array1D, y0: Array1D, continuous: bool, on_stop: Any = None
    ) -> tuple[Array2D, Array1D]:
        """Fallback driver for event nodes without a localizable margin. It solves,
        then at the first output point where a transition changes ``F`` restarts the
        remainder from there so the mutated state feeds back into the integration
        (which post-hoc polling would miss), and truncates on a stop request. Event
        timing is quantized to the output grid; localizable margins are precise."""
        time = np.asarray(time)
        t_acc: list[Array1D] = []
        y_acc: list[Array2D] = []
        y_cur = y0
        remaining = time
        first = True
        while True:
            try:
                data, seg_t = solve_segment(y_cur, remaining)
            except TransientRuntimeError as e:
                _merge_failure_trajectory(e, t_acc, y_acc)
                raise
            start = 0 if first else 1
            first = False
            cut, stop = None, False
            for i in range(start, len(seg_t)):
                f_before = self.compute(data[i], seg_t[i])
                if not self._handle_event(data[i], seg_t[i]) and not continuous:
                    cut, stop = i, True
                    if on_stop is not None:
                        on_stop(data[i], seg_t[i])
                    break
                if not np.array_equal(self.compute(data[i], seg_t[i]), f_before):
                    cut = i
                    break
            if cut is None:
                t_acc.append(seg_t[start:])
                y_acc.append(data[start:])
                break
            t_acc.append(seg_t[start : cut + 1])
            y_acc.append(data[start : cut + 1])
            if stop:
                break
            y_cur = data[cut]
            remaining = seg_t[cut:]
            if len(remaining) < 2:
                break
        return concat(*y_acc), concat(*t_acc)

    @overload
    def load(self, s: DictState) -> Array1D:
        """Loads a dictionary based state to the functional vector it represents.

        Parameters
        ----------
        s: DictState
            The state from (possibly) a saved solution, or a guess

        Returns
        -------
        Array1D
            The state vector given to the functional in this case"""
        ...

    @overload
    def load(self, s: StateTimeseries) -> Solution:
        """Loads a time-series of named states into a 2D vector solution.

        Parameters
        ----------
        s: StateTimeseries
            The (possibly) saved named-variable solution

        Returns
        -------
        Solution
            The 2D solution that an Aggregator would have given to create such a stateful solution."""
        ...

    def load(self, s):
        """Given a description of the system state either at one time or at many,
        returns a vector or a solution that fits this information.

        Parameters
        ----------
        s:
            The system description to parse.

        """
        # StateTimeseries keys are numeric (incl. np.int64/np.float32 that isinstance(float) misses); DictState keys are str.
        has_time = any(isinstance(key, numbers.Number) for key in s)
        return self._solution_from_states(s) if has_time else self._vector_from_state(s)

    def _vector_from_state(self, s: DictState) -> Array1D:
        """
        Given the state of a system, return the untagged corresponding array
        which may be used to calculate the next step

        Parameters
        ----------
        s: DictState
            Tagged information regarding system state

        Returns
        -------
        variables: Array1D
            Calculation ready array
        """
        y = np.empty(self.vector_length)
        for node, section in self.sections.items():
            try:
                calc_state = s[node.name]
            except KeyError as e:
                e.add_note(
                    f"State has no entry for Calculation '{node.name}'; "
                    f"its calculation keys are {sorted(s)}"
                )
                raise
            try:
                y[section] = node.load(calc_state)
            except Exception as e:
                name = getattr(node, "name", node)
                e.add_note(f"while loading Calculation '{name}' ({type(node).__name__}) from the State")
                raise
        return y

    def _solution_from_states(self, states: StateTimeseries) -> Solution:
        """Make a solution matrix from a mapping of states.

        Parameters
        ----------
        states: StateTimeSeries
            The states to make into a solution

        """

        shape = (len(states.keys()), self.vector_length)
        data = np.empty(shape, float)
        for i, (_, state) in enumerate(sorted(states.items(), key=lambda x: x[0])):
            data[i, :] = self.load(state)
        return Solution(np.array(sorted(states.keys())), data)

    @overload
    def save(self, solution: Solution) -> StateTimeseries:
        """Write a 2D vector time-dependent solution as a human-readable object.

        Parameters
        ----------
        solution: Solution
            The 2D solution to save

        Returns
        -------
        StateTimeseries
            The time-dependent human-readable state object"""
        ...

    @overload
    def save(self, solution: Sequence[float], t: Second = 0, strict: bool = False) -> State:
        """Write a vector solution with proper names that allow human readable solutions

        Parameters
        ----------
        solution: Sequence[float]
            The vector solution in mind. Has the length of this aggregator.
        t: Second
            The absolute time at which the solution is given.
        strict: bool
            Flag for wether information beyond the scope of the solution be added

        Returns
        -------
        State
            The saved human-readable version of the solution
        """
        ...

    def save(self, solution, t=0, strict=False):
        """Given either a vector solution of the system or a Solution object,
        creates a human-readable state description of the solution.

        Parameters
        ----------
        solution:
            The solution to make human-readable.
        t:
            The time at which the solution is given (if it is a single vector)
        strict:
            Whether information beyond the vector state variables should be added.

        """
        return (
            self._parse_solution(solution)
            if isinstance(solution, Solution)
            else self._vector_to_state(solution, t, strict)
        )

    def _vector_to_state(self, solution: Sequence[float], t: Second = 0, strict: bool = False) -> State:
        """
        Given input for calculations (which is a legal state of the system),
        tag the information, i.e. create a "State" and return it

        Parameters
        ----------
        solution: Sequence[float]
            Input from the solver
        t: Second
            time
        strict: bool
            Reports only variables if ``True``. Default is ``False``.

        Returns
        -------
        state: State
            Tagged information regarding system state
        """
        save_func = partial(self._op, "strict_save" if strict else "save", solution, t)
        return State({node.name: save_func(node) for node in self.graph})

    def _parse_solution(self, solution: Solution) -> StateTimeseries:
        """Parse a StateTimeseries from a solution from the `~.solve` method.

        Parameters
        ----------
        solution: Solution
            The solution from this Aggregator's solve method.

        """
        return {t: self.save(solution.data[i, :], t) for i, t in enumerate(solution.time)}

    def _op(self, op: str, y: Sequence[float], t: Second, node: Calculation):
        input_ = y[self.sections[node]]
        external = self._node_external(node, y, t)
        try:
            return getattr(node, op)(input_, **external)
        except Exception as e:
            name = getattr(node, "name", node)
            e.add_note(f"while evaluating Calculation '{name}' ({type(node).__name__}).{op} at t={t}")
            raise

    def var_index(self, node: Calculation, var_name: str) -> Place:
        """Return the Place at which a given variable lies

        Parameters
        ----------
        node: Calculation
            The calculation whose variable is requested
        var_name: str
            Variable name

        Returns
        -------
        Place
            The Place in aggregator vector where this variable resides
        """
        place = self.sections[node].start
        index = node.variables[var_name]
        return offset(index, place)

    def _build_row_map(self) -> list[Location]:
        r"""Build the length-``N`` ``row -> Location`` map by walking
        ``sections`` x ``node.variables``, exactly the iteration
        :meth:`var_index` inverts (local place offset by the section start).

        First-defined wins: if two variables would cover the same row, the one
        earlier in ``node.variables`` order keeps it (never overwritten). Any row
        a section owns but no variable covers is filled with a ``'<unmapped>'``
        :class:`Location`, so :meth:`locate` is total over ``0..N-1``.
        """
        n = self.vector_length
        row_map: list[Location | None] = [None] * n
        for node, section in self.sections.items():
            node_name = str(node)
            for var_name, place in node.variables.items():
                variable = var_name if isinstance(var_name, str) else str(var_name)
                for cell, row in _place_cells(offset(place, section.start)):
                    if 0 <= row < n and row_map[row] is None:
                        row_map[row] = Location(node, node_name, variable, cell)
        for node, section in self.sections.items():
            node_name = str(node)
            for row in range(section.start, section.stop):
                if row_map[row] is None:
                    row_map[row] = Location(node, node_name, "<unmapped>", row - section.start)
        return row_map

    def locate(self, row: int) -> Location:
        """Name a state-vector row in domain terms — the inverse of
        :meth:`var_index`.

        Parameters
        ----------
        row: int
            A row of :math:`\\vec{y}`; ``numpy`` integer types are accepted.

        Returns
        -------
        Location
            The ``(calculation, name, variable, cell)`` owning ``row``.

        Raises
        ------
        IndexError
            If ``row`` is outside ``0..len(self) - 1``.
        """
        try:
            row_map = self._row_map
        except AttributeError:
            row_map = self._row_map = self._build_row_map()
        idx = int(row)
        if not 0 <= idx < len(row_map):
            raise IndexError(f"row {row} out of range for a state vector of length {len(row_map)} (valid rows 0..{len(row_map) - 1})")
        return row_map[idx]

    def locate_nonfinite(
        self, y: Array1D, F: Array1D | None = None
    ) -> list[tuple[Location, float, str]]:
        """Locate every non-finite (NaN / inf) entry of ``y`` and, if given, of ``F``.

        Entries are ordered ``y`` before ``F``, ascending row within each source.

        Parameters
        ----------
        y: Array1D
            The state vector (e.g. a failure iterate ``err.y``).
        F: Array1D or None
            An optional residual vector to scan alongside ``y``.

        Returns
        -------
        list[tuple[Location, float, str]]
            One ``(location, value, source)`` per non-finite entry, ``source``
            being ``'y'`` or ``'F'``.
        """
        out: list[tuple[Location, float, str]] = []
        y = np.asarray(y, dtype=float)
        for row in np.flatnonzero(~np.isfinite(y)):
            out.append((self.locate(int(row)), float(y[row]), "y"))
        if F is not None:
            F = np.asarray(F, dtype=float)
            for row in np.flatnonzero(~np.isfinite(F)):
                out.append((self.locate(int(row)), float(F[row]), "F"))
        return out

    def worst_residuals(
        self,
        y: Array1D,
        t: Second = 0.0,
        *,
        n: int = 5,
        scales: dict[str, float] | Array1D | None = None,
    ) -> list[tuple[Location, float]]:
        r"""The ``n`` least-converged rows of :math:`F(y)`, ranked by the *scaled*
        residual :math:`|F/\text{typ}|`.

        Ranking raw ``F`` is unit-dominated — a pressure residual (Pa) dwarfs a
        temperature residual (K) by scale alone, so the "worst" is a units
        artifact, not the variable the solver fights. Dividing by ``typ`` (the
        same nominal magnitudes :meth:`solve_steady` scales with) fixes this.
        Non-finite scaled entries are maximally worrying, so they rank first.

        Parameters
        ----------
        y: Array1D
            The iterate to evaluate (e.g. a failure ``err.y``).
        t: Second, default 0.0
            The time at which to evaluate :math:`F`.
        n: int, default 5
            How many rows to return.
        scales: None, dict[str, float] or Array1D, default None
            Nominal magnitudes ``typ`` resolved exactly as in
            :meth:`solve_steady`: ``None`` -> :data:`~stream.scales.DEFAULT_SCALES`;
            a ``dict`` -> name registry; a length-``N`` array -> used directly.

        Returns
        -------
        list[tuple[Location, float]]
            Up to ``n`` ``(location, signed_scaled_residual)`` pairs, most
            worrying first.
        """
        y = np.asarray(y, dtype=float)
        F = self.compute(y, t)
        typ = _resolve_typ(self, scales)
        scaled = F / typ
        magnitude = np.abs(scaled)
        key = np.where(np.isfinite(magnitude), magnitude, np.inf)
        order = np.lexsort((np.arange(len(key)), -key))
        return [(self.locate(int(row)), float(scaled[row])) for row in order[:n]]

    def state_from(self, obj) -> State | StateTimeseries:
        """Bridge a failure object straight into a human-readable state.

        Debug tools and the post-mortem workflow need a
        :class:`~stream.state.State` / :class:`~stream.state.StateTimeseries`, but
        a failure hands you a raw iterate or an attached trajectory.

        Parameters
        ----------
        obj:
            Exactly one of:

            - a 1-D state vector (e.g. ``err.y``) -> a :class:`~stream.state.State`;
            - a ``(t, y2d)`` pair (e.g. ``(e.t, e.y)``) -> a
              :class:`~stream.state.StateTimeseries` (same path as
              ``save(Solution(...))``);
            - a :class:`.Solution` -> a :class:`~stream.state.StateTimeseries`.

        Returns
        -------
        State or StateTimeseries

        Raises
        ------
        TypeError
            For anything other than the three forms above.

        Examples
        --------
        >>> agr.state_from(err.y)          # doctest: +SKIP
        >>> agr.state_from((e.t, e.y))     # doctest: +SKIP
        """
        if isinstance(obj, Solution):
            return self.save(obj)
        if isinstance(obj, tuple) and len(obj) == 2 and np.ndim(obj[1]) == 2:
            t, y2d = obj
            return self.save(Solution(np.asarray(t, dtype=float), np.asarray(y2d, dtype=float)))
        try:
            arr = np.asarray(obj, dtype=float)
        except (ValueError, TypeError):
            arr = None
        if arr is not None and arr.ndim == 1:
            return self.save(arr)
        raise TypeError(
            "state_from accepts exactly a 1-D state vector (-> State), a (t, y2d) "
            "pair (-> StateTimeseries), or a Solution (-> StateTimeseries); e.g. "
            "agr.state_from(err.y) or agr.state_from((e.t, e.y)). "
            f"Got {type(obj).__name__}."
        )

    def at_times(self, solution: Solution, node: Calculation, var_name: str) -> Array2D:
        """Given a transient solution and a variable, returns the variable at
        the calculated times.

        Parameters
        ----------
        solution: Array2D
            The solution matrix, i.e. the variable vector (columns) at
            different times (rows)
        node: Calculation
            The inquired calculation
        var_name: str
            the inquired variable of node.

        Returns
        -------
        output: Array2D
            A slice of the solution at the correct var_index.
        """
        return solution.data[:, self.var_index(node, var_name)]

    def _enrich_failure(self, err: BaseException) -> None:
        r"""Decorate an in-flight solver failure with domain-term notes (§3.2).

        Runs **only** on the failure path and entirely inside ``try/except``,
        with each probe guarded on its own (P8): a probe that itself fails can
        never mask or suppress the error it explains, and the error always
        propagates with whatever notes did attach. Notes only — the message and
        ``args`` are never touched (P4). Best-effort, it appends:

        1. non-finite locations of the failure state (and, when computable, its
           residual) — ``T_wall=nan at cell 3 of 'CC' (source: y)`` (B3);
        2. the top-3 *scaled* worst residuals (kills the G5-09 unit-domination);
        3. a saturation crossing plus the ``stop_at_saturation`` hint — this is
           what closes the B5/G5-02 discoverability gap.

        A domain-violation note (``domain_report``) is wired in W4.
        """
        try:
            y_attr = getattr(err, "y", None)
            if y_attr is None:
                return
            y = np.asarray(y_attr, dtype=float)
            row = y[-1] if y.ndim == 2 else y  # 1-D IC-recovery payloads exist (W1d)
            t_attr = getattr(err, "t", None)
            t_fail = float(np.atleast_1d(t_attr)[-1]) if t_attr is not None else 0.0

            # Note 1 — non-finite locations (source y and, when computable, F).
            try:
                try:
                    F = self.compute(row, t_fail)
                except Exception:
                    F = None
                locs = self.locate_nonfinite(row, F)
                if locs:
                    entries = []
                    for loc, value, source in locs:
                        where = f"at cell {loc.cell} " if loc.cell is not None else ""
                        entries.append(f"{loc.variable}={value:g} {where}of {loc.name!r} (source: {source})")
                    shown = entries[:8]
                    if len(entries) > 8:
                        shown.append(f"... and {len(entries) - 8} more")
                    err.add_note("non-finite values: " + "; ".join(shown))
            except Exception:
                pass

            try:
                worst = self.worst_residuals(row, t_fail, n=3)
                if worst:
                    items = []
                    for loc, val in worst:
                        cell = f"[cell {loc.cell}]" if loc.cell is not None else ""
                        items.append(f"{loc.variable}{cell} of {loc.name!r}: {val:.3e}")
                    err.add_note("worst scaled residuals: " + ", ".join(items))
            except Exception:
                pass

            # Lazy import: stream.analysis imports aggregator, so a module-level import would cycle.
            try:
                from stream.analysis.thresholds import first_saturation_crossing

                if y.ndim == 2 and t_attr is not None:
                    found = first_saturation_crossing(y, self, times=np.asarray(t_attr))
                else:
                    found = first_saturation_crossing(row, self)
                if found is not None:
                    t_cross, crossings = found
                    c = crossings[0]
                    when = f" at t={t_cross}" if t_cross is not None else ""
                    err.add_note(
                        f"Channel {c.channel!r} crossed Tsat{when} (cells {c.cells}) — the "
                        "single-phase model is invalid past bulk saturation"
                        + hint_block(
                            "stop_at_saturation=True on the channel",
                            "analysis.first_saturation_crossing / raise_on_saturation for the full picture",
                        )
                    )
            except Exception:
                pass

            # Note 4 — domain violations (domain_report): wired in W4.
        except Exception:
            pass

    def solve(
        self,
        y0: Array1D | DictState,
        time: Sequence[float] | None,
        yp0: Array1D = None,
        eq_type: Literal["ODE", "DAE", "ALG"] | None = None,
        *,
        progressbar: ProgressBarLike | bool = False,
        **options,
    ) -> Solution:
        """
        For a Differential Algebraic set of eqs. (DAE), the chosen solver is
        IDA from the LLNL SUNDIALS suite, which is kindly wrapped by
        Scikits.Odes, originally written in C with DASPK (Fortran) usages.
        This solver performs (among many other capabilities) integration by
        variable-order, variable-coefficient BDF. Newton iteration is used to
        find a solution.

        Parameters
        ----------
        y0: Array1D or DictState
            Initial values or guess. Can either be an array or a State, in the
            latter case :meth:`load` will be used to obtain the desired array.
        time: Sequence[float]
            Return results at these time points.
        yp0: Array1D or None
            Initial derivatives. It helps if they're known (in the DAE case),
            but by default the consistent yp0 is found from y0.
        eq_type: 'ODE', 'DAE', 'ALG' or None
            A solver may be chosen deliberately from [ODE, DAE, ALG].
            If None, the method is set by looking at the mass matrix and
            whether time is none.
        progressbar: ProgressBarLike or bool
            Whether to use a progressbar, and if so, which one. If ``True``, use ``use progressbar.ProgressBar``
        options:
            Other options

        Returns
        -------
        solution: Solution
            Calculated vector at requested times: [time, variable].

        References
        ----------
        Scikits.Odes documentation
        """
        if eq_type is None:
            if all(self.mass) and time is not None:
                eq_type = "ODE"
                logger.log(STREAM_DEBUG, "Solving TRANSIENT (ODE)")
            elif any(self.mass) and time is not None:
                eq_type = "DAE"
                logger.log(STREAM_DEBUG, "Solving TRANSIENT")
            else:
                eq_type = "ALG"
                logger.log(STREAM_DEBUG, "Solving STEADY STATE")

        # Backends below rebind the local `time`; keep the requested grid to detect a truncated (STOPPED) run.
        requested = None if time is None else np.asarray(time)
        records: list[Event] = []
        continuous = options.get("continuous", False)

        if not isinstance(y0, np.ndarray):
            y0 = self.load(y0)

        try:
            if eq_type == "ODE":
                width = len(self._event_margins(y0, 0.0))
                if width:
                    self._warn_marginless_event_nodes(y0)
                    t_start = float(time[0])

                    def on_event(t_root, y_root):
                        keep = self._handle_event(y_root, t_root)
                        if (not keep) or (t_root > t_start):
                            records.append(
                                Event(
                                    float(t_root),
                                    self._stopped_names(y_root, t_root) if not keep else (),
                                    terminal=(not keep) and not continuous,
                                )
                            )
                        return keep

                    data, time = differential(
                        F=self.compute,
                        y0=y0,
                        time=time,
                        events=[self._ode_event(k) for k in range(width)],
                        on_event=on_event,
                        **options,
                    )
                elif self._has_event_overrides():
                    data, time = self._integrate_with_polling(
                        lambda y, rem: differential(F=self.compute, y0=y, time=rem, **options),
                        time,
                        y0,
                        continuous,
                        on_stop=lambda y, t: records.append(Event(float(t), self._stopped_names(y, t), terminal=True)),
                    )
                else:
                    data, time = differential(F=self.compute, y0=y0, time=time, **options)
            elif eq_type == "DAE":
                if progressbar and isinstance(progressbar, bool):
                    try:
                        from progressbar import ProgressBar
                    except ImportError as e:
                        e.msg = "User asked for a progressbar without supplying one, and the optional dependency on progressbar2 was not satisfied, so importing it failed"
                        raise
                    progressbar = ProgressBar().start(max_value=int(1e3 * max(time)))
                elif not progressbar:
                    progressbar = None

                width = len(self._event_margins(y0, 0.0))
                if width:
                    reserved = {"rootfn", "nr_rootfns"} & options.keys()
                    if reserved:
                        keys = " and ".join(f"'{k}'" for k in sorted(reserved))
                        raise StreamConstructionError(
                            f"{keys} is managed by the Aggregator's event system — supply events via "
                            "event_margin/trip_margin, or call stream.solvers.differential_algebraic "
                            "directly for a raw rootfn"
                        )
                    self._warn_marginless_event_nodes(y0)

                    def rootfn(y, t, _bar=progressbar):
                        if _bar is not None:
                            _bar.update(int(1e3 * t))
                        return self._event_margins(y, t)

                    t_start = float(time[0])

                    def on_event(t_root, y_root, yp_root):
                        keep = self._handle_event(y_root, t_root)
                        if (not keep) or (t_root > t_start):
                            records.append(
                                Event(
                                    float(t_root),
                                    self._stopped_names(y_root, t_root) if not keep else (),
                                    terminal=(not keep) and not continuous,
                                )
                            )
                        return keep

                    data, time = differential_algebraic(
                        F=self.compute,
                        mass=self.mass,
                        R=rootfn,
                        on_event=on_event,
                        y0=y0,
                        time=time,
                        yp0=yp0,
                        nr_rootfns=width,
                        **options,
                    )
                elif self._has_event_overrides():
                    data, time = self._integrate_with_polling(
                        lambda y, rem: differential_algebraic(
                            F=self.compute, mass=self.mass, R=None, y0=y, time=rem, yp0=None, **options
                        ),
                        time,
                        y0,
                        continuous,
                        on_stop=lambda y, t: records.append(Event(float(t), self._stopped_names(y, t), terminal=True)),
                    )
                else:
                    data, time = differential_algebraic(
                        F=self.compute, mass=self.mass, R=None, y0=y0, time=time, yp0=yp0, **options
                    )
                if progressbar is not None:
                    progressbar.finish()
            elif eq_type == "ALG":
                if time is None:
                    vector = algebraic(F=self.compute, y0=y0, time=None, R=self._handle_event, **options)
                    data, time = vector[None, :], np.array([0.0])
                else:
                    data, time = algebraic(F=self.compute, y0=y0, time=time, R=self._handle_event, **options)
            else:
                raise ValueError(f"Unknown method {eq_type}, choose from [ODE, DAE, ALG]")
        except TransientRuntimeError as e:
            self._enrich_failure(e)
            raise
        status = (
            SolveStatus.STOPPED
            if (any(ev.terminal for ev in records) or (requested is not None and len(time) < len(requested)))
            else SolveStatus.COMPLETED
        )
        return Solution(np.asarray(time), data, status=status, events=tuple(records))

    def solve_steady(
        self,
        guess: Array1D | DictState,
        *,
        globalize: Literal["auto", True, False] = "auto",
        scales: dict[str, float] | Array1D | None = None,
        fallback_ptc: bool = True,
        **options,
    ) -> Array1D:
        r"""Solving an Algebraic Equation :math:`0=F(y)` using
        :func:`~stream.solvers.algebraic`, with an optional globalized fallback.

        Parameters
        ----------
        guess: Array1D or DictState
            Initial guess. Can either be an array or a State, in the
            latter case :meth:`load` will be used to obtain the desired array.
        globalize: {'auto', True, False}, default 'auto'
            Selects the steady-solve strategy (keyword-only):

            - ``'auto'`` — run the :func:`scipy.optimize.root` path; only if it
              raises :class:`~stream.solvers.AlgRuntimeError` fall back to
              :func:`~stream.solvers.scaled_newton`.
            - ``True`` — skip scipy entirely and go straight to
              :func:`~stream.solvers.scaled_newton`.
            - ``False`` — no fallback; an :class:`~stream.solvers.AlgRuntimeError`
              from scipy propagates.
        scales: None, dict[str, float] or Array1D, default None
            Source of the per-variable nominal magnitudes (``typ``) used to build
            the fallback's scaled finite-difference step. ``None`` uses
            :data:`~stream.scales.DEFAULT_SCALES`; a ``dict`` is a
            name -> scale registry; a length-``N`` array is used directly as the
            ``typ`` vector. Ignored on the scipy success path.
        fallback_ptc: bool, default True
            When the ``scaled_newton`` rung also raises, run
            :func:`~stream.solvers.pseudo_transient` and polish the result with a
            second ``scaled_newton``. Set ``False`` to let the ``scaled_newton``
            failure propagate.
        options:
            Solver options forwarded verbatim to :func:`~stream.solvers.algebraic`
            on the scipy path (the keyword-only arguments above never reach
            ``scipy.optimize.root``). A caller-supplied ``jac`` is honored
            by the fallback too; only when absent does the fallback build
            ``ALG_jacobian(self, scaled_step(typ))``.

        Returns
        -------
        solution: Array1D
            Calculated vector.
        """
        if not isinstance(guess, np.ndarray):
            guess = self.load(guess)

        rungs: list[tuple] = []

        if globalize is not True:
            try:
                return algebraic(F=self.compute, y0=guess, R=self._handle_event, **options)
            except AlgRuntimeError as err:
                if globalize is False:
                    raise
                rungs.append(_rung_record("scipy hybr", guess, err, self))

        jac = options.get("jac")
        if jac is None:
            from stream.jacobians import ALG_jacobian, scaled_step  # lazy: jacobians imports Aggregator

            jac = ALG_jacobian(self, scaled_step(_resolve_typ(self, scales)))
        try:
            return scaled_newton(self.compute, guess, jac)
        except AlgRuntimeError as err:
            rungs.append(_rung_record("scaled_newton", guess, err, self))
            if not fallback_ptc:
                raise _cascade_error(rungs, err, self) from err

        try:
            relaxed = pseudo_transient(self.compute, np.asarray(self.mass, float), guess, jac)
        except AlgRuntimeError as err:
            rungs.append(_rung_record("pseudo_transient", guess, err, self))
            raise _cascade_error(rungs, err, self) from err
        try:
            return scaled_newton(self.compute, relaxed, jac)
        except AlgRuntimeError as err:
            rungs.append(("pseudo_transient", guess, "reached the basin", _rung_nF(self, relaxed)))
            rungs.append(_rung_record("scaled_newton polish", relaxed, err, self))
            raise _cascade_error(rungs, err, self) from err

    def scaled_atol(self, rel: float = 1e-6, scales: dict[str, float] | Array1D | None = None) -> Array1D:
        r"""Per-variable absolute tolerance ``rel * typ_j`` for :meth:`solve`.

        Builds a length-``N`` ``atol`` vector aligned to the state vector, for
        threading into DAE/ODE error control as ``agr.solve(..., atol=...)`` (both
        IDA and ``solve_ivp`` accept a per-variable array). Opt-in: nothing
        consumes it implicitly. The ALG steady path has no ``atol`` knob.

        Parameters
        ----------
        rel: float, default 1e-6
            Relative factor multiplying each nominal magnitude.
        scales: None, dict[str, float] or Array1D, default None
            Nominal magnitudes, resolved exactly as in :meth:`solve_steady`:
            ``None`` -> :data:`~stream.scales.DEFAULT_SCALES`; ``dict`` -> registry;
            length-``N`` array -> used directly (validated).

        Returns
        -------
        Array1D
            The vector ``rel * typ``.
        """
        return rel * _resolve_typ(self, scales)


@dataclass
class CalculationGraph:
    """
    A container for an Aggregator input - the functional graph (hence its name),
    it has the same initialization signature, but does not perform any of the buildup an
    Aggregator object does. This fact makes it easy to connect several of these
    objects together, which is very useful.

    Parameters
    ----------
    graph: DiGraph
        containing Calculations at nodes and variable coupling
        at the edges.
    funcs: ExternalFunctions or None
        time-only-dependent functions, which are user controlled.
    """

    graph: DiGraph
    funcs: ExternalFunctions | None = None

    @classmethod
    def connect(
        cls,
        a: BaseAgr,
        b: BaseAgr,
        *edges: tuple[Calculation, Calculation, Iterable[Name]],
    ) -> "CalculationGraph":
        """
        Connect two CalculationGraph objects. In case of a clash, the second object
        prevails. If ``edges`` contains an edge already in either ``a.graph``
        or ``b.graph``, it is updated, not overridden.

        .. tip::
            The two inputs may share nodes. This is very useful!

        Parameters
        ----------
        a: CalculationGraph
            First input
        b: CalculationGraph
            Second input
        edges: tuple[Calculation, Calculation, Names]
            Any edges which may connect the two.

        Returns
        -------
        new: CalculationGraph
            A new CalculationGraph whose graph and functions are composed out of a,b.
        """
        g = compose(a.graph, b.graph)
        # networkx compose lets b's edge data win; re-union routings for edges present in both graphs.
        for e in set(a.graph.edges) & set(b.graph.edges):
            merged_vars = chain(a.graph.edges[e].get(VARS, ()), b.graph.edges[e].get(VARS, ()))
            g.edges[e][VARS] = tuple(unique(merged_vars))
        for edge in edges:
            u, v, d = edge
            if (e := (u, v)) in g.edges:
                g.edges[e][VARS] = tuple(unique(chain(g.edges[e][VARS], d)))
            else:
                g.add_edge(u, v, variables=d)

        af, bf = a.funcs or {}, b.funcs or {}
        # Union the inner name dicts so a shared calculation's bindings are not wholesale replaced.
        merged = {c: {**af.get(c, {}), **bf.get(c, {})} for c in af.keys() | bf.keys()}
        return CalculationGraph(graph=g, funcs=merged or None)

    def __add__(self, other) -> "CalculationGraph":
        return self.connect(self, other)

    @classmethod
    def from_decoupled(cls, *nodes: Calculation, funcs: ExternalFunctions | None = None) -> "CalculationGraph":
        r"""Instantiate an Aggregator from calculations, which are not connected.

        Parameters
        ----------
        nodes: Calculation
            Calculations to ba added
        funcs: ExternalFunctions or None
            to be passed to the aggregator

        Returns
        -------
        agr: CalculationGraph
            Which contains a disconnected-nodes-graph
        """
        g = DiGraph()
        g.add_nodes_from(nodes)
        return cls(g, funcs)

    def to_aggregator(self) -> Aggregator:
        """Initialize an Aggregator from input in self"""
        return Aggregator(self.graph, self.funcs)

    def draw(self, node_options=None, edge_options=None):
        r"""Method equivalent of :func:`draw_aggregator`"""
        draw_aggregator(self.graph, node_options, edge_options)
