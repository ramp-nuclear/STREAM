"""Tools for debugging Aggregators and States."""

import operator
from functools import partial, reduce
from typing import Container

import numpy as np

from stream.aggregator import Aggregator
from stream.aggregator.aggregator import _resolve_typ
from stream.calculations import Kirchhoff
from stream.state import State
from stream.units import Array1D, Value


def debug_derivatives(
    agr: Aggregator, guess: State | Array1D, scales: dict[str, float] | Array1D | str | None = None
) -> State:
    """Return the application of the Aggregator's functional on a guess, tagged.

    For differential equations, this would be the derivative of that variable
    given this guess.
    For algebraic equations, this would be the residual of the appropriate equation.
    Since some variable's algebraic equations aren't actually equations on those
    variables (Mostly because of Kirchhoff's unknown cycle base this would happen
    for some of the mass flow rate variables, but other cases may exist as well).

    Parameters
    ----------
    agr: Aggregator
        The Aggregator to test.
    guess: State or Array1D
        The state to use as a guess for steady state. A raw vector (e.g. a caught
        failure's ``err.y``) is accepted directly and tagged via ``agr.save``.
    scales: None, 'default', dict[str, float] or Array1D
        ``None`` (the default) returns the raw residuals — **unscaled and therefore
        unit-dominated**: a pressure residual (Pa) dwarfs a temperature residual (K)
        by scale alone, so ranking the raw view misleads. Pass ``'default'``
        for :data:`~stream.scales.DEFAULT_SCALES`, or a name registry / length-``N``
        ``typ`` vector (the :meth:`~stream.aggregator.Aggregator.scaled_atol`
        convention), to get residuals divided by ``typ`` — the solver's view.

    """
    if isinstance(guess, np.ndarray):
        guess = agr.save(guess)
    F = agr.compute(agr.load(guess))
    if scales is not None:
        F = F / _resolve_typ(agr, None if isinstance(scales, str) else scales)
    return agr.save(F, strict=True)


def debug_guess_variables(agr: Aggregator, guess: State | Array1D, variables: Container[str]) -> dict[str, Value]:
    """Show the errors in a variable's guesstimate across all calculations.

    This is a subset of debug_derivatives, since that debug tool shows a lot of
    data all at once.

    Parameters
    ----------
    agr: Aggregator
        The Aggregator to debug.
    guess: State or Array1D
        The State we guesstimate as the solution (a raw vector is accepted too).
    variables: Container[str]
        The variables we want to debug for.

    """
    return debug_derivatives(agr, guess).filter_var_names(lambda x: x in variables)


debug_guess_pressures = partial(debug_guess_variables, variables={"pressure"})


def debug_guess_flows(agr: Aggregator, guess: State | Array1D) -> dict[str, Value]:
    """Shows the errors in flows from all Kirchhoffs for a guesstimate.

    Parameters
    ----------
    agr: Aggregator
        The Aggregator to debug.
    guess: State or Array1D
        The State we guesstimate as the solution (a raw vector is accepted too).

    """
    kirchhoffs = {c.name for c in agr.graph if isinstance(c, Kirchhoff)}
    dk = debug_derivatives(agr, guess).filter_calculations(lambda c: c in kirchhoffs)
    dflow = dk.filter_var_names(lambda s: "->" in s)
    return reduce(operator.or_, dflow.values(), {})
