import logging
import numbers
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Iterable, Literal, Protocol, Sequence, overload

import numpy as np
from cytoolz import unique, valmap
from networkx import DiGraph, compose

from stream.calculation import Calculation
from stream.solvers import algebraic, differential, differential_algebraic
from stream.state import DictState, State, StateTimeseries
from stream.units import Array1D, Array2D, Name, Place, Second
from stream.utilities import STREAM_DEBUG, concat, offset

from .solution import Solution
from .utils import (
    VARS,
    BaseAgr,
    ExternalFunctions,
    draw_aggregator,
    map_externals,
    non_unique_calculations,
    partition,
)

__all__ = ["Aggregator", "CalculationGraph", "NonUniqueCalculationNameError"]

logger = logging.getLogger("stream.aggregator")


class NonUniqueCalculationNameError(ValueError):
    """Error to signify that calculations in an aggregator were not uniquely named."""

    pass


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

    def _ode_event(self, k: int):
        """A ``solve_ivp`` terminal event for margin component ``k`` (positive ->
        non-positive crossing), so ODE-mode integration localizes events."""

        def event(t: Second, y: Sequence[float]) -> float:
            return float(self._event_margins(y, t)[k])

        event.terminal = True
        event.direction = -1
        return event

    def _has_event_overrides(self) -> bool:
        """Whether any node overrides the default event hooks (so its events must
        not be silently dropped even when it exposes no localizable margin)."""
        return any(
            type(node).change_state is not Calculation.change_state
            or type(node).should_continue is not Calculation.should_continue
            for node in self.graph
        )

    def _marginless_event_nodes(self, y: Sequence[float]) -> list[Calculation]:
        """Event-overriding nodes that expose no localizable margin at ``y``."""
        return [
            node
            for node in self.graph
            if (
                type(node).change_state is not Calculation.change_state
                or type(node).should_continue is not Calculation.should_continue
            )
            and np.asarray(self._op("event_margin", y, 0.0, node)).size == 0
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
        self, solve_segment: Any, time: Array1D, y0: Array1D, continuous: bool
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
            data, seg_t = solve_segment(y_cur, remaining)
            start = 0 if first else 1
            first = False
            cut, stop = None, False
            for i in range(start, len(seg_t)):
                f_before = self.compute(data[i], seg_t[i])
                if not self._handle_event(data[i], seg_t[i]) and not continuous:
                    cut, stop = i, True
                    break
                if not np.array_equal(self.compute(data[i], seg_t[i]), f_before):
                    cut = i  # a transition changed F: restart so it feeds back
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
            y[section] = node.load(s[node.name])
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
        return getattr(node, op)(input_, **external)

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

        if not isinstance(y0, np.ndarray):
            y0 = self.load(y0)

        if eq_type == "ODE":
            width = len(self._event_margins(y0, 0.0))
            if width:
                self._warn_marginless_event_nodes(y0)
                data, time = differential(
                    F=self.compute,
                    y0=y0,
                    time=time,
                    events=[self._ode_event(k) for k in range(width)],
                    on_event=lambda t_root, y_root: self._handle_event(y_root, t_root),
                    **options,
                )
            elif self._has_event_overrides():
                data, time = self._integrate_with_polling(
                    lambda y, rem: differential(F=self.compute, y0=y, time=rem, **options),
                    time,
                    y0,
                    options.get("continuous", False),
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
                self._warn_marginless_event_nodes(y0)

                def rootfn(y, t, _bar=progressbar):
                    if _bar is not None:
                        _bar.update(int(1e3 * t))
                    return self._event_margins(y, t)

                data, time = differential_algebraic(
                    F=self.compute,
                    mass=self.mass,
                    R=rootfn,
                    on_event=lambda t_root, y_root, yp_root: self._handle_event(y_root, t_root),
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
                    options.get("continuous", False),
                )
            else:
                data, time = differential_algebraic(
                    F=self.compute, mass=self.mass, R=None, y0=y0, time=time, yp0=yp0, **options
                )
            if progressbar is not None:
                progressbar.finish()
        elif eq_type == "ALG":
            if time is None:
                # Steady root find: wrap the bare vector as a single-row Solution.
                vector = algebraic(F=self.compute, y0=y0, time=None, R=self._handle_event, **options)
                data, time = vector[None, :], np.array([0.0])
            else:
                data, time = algebraic(F=self.compute, y0=y0, time=time, R=self._handle_event, **options)
        else:
            raise ValueError(f"Unknown method {eq_type}, choose from [ODE, DAE, ALG]")
        return Solution(np.asarray(time), data)

    def solve_steady(self, guess: Array1D | DictState, **options) -> Array1D:
        """Solving an Algebraic Equation :math:`0=F(y)` using
        :func:`~stream.solvers.algebraic`

        Parameters
        ----------
        guess: Array1D or DictState
            Initial guess. Can either be an array or a State, in the
            latter case :meth:`load` will be used to obtain the desired array.
        options:
            Solver options

        Returns
        -------
        solution: Array1D
            Calculated vector.
        """
        if not isinstance(guess, np.ndarray):
            guess = self.load(guess)
        return algebraic(F=self.compute, y0=guess, R=self._handle_event, **options)


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
