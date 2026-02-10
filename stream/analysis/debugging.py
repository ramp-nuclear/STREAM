"""Tools for debugging Aggregators and States.

"""
import operator
from functools import partial, reduce
from typing import Container

from stream.aggregator import Aggregator
from stream.state import State
from stream.calculations import Kirchhoff
from stream.units import Value


def debug_derivatives(agr: Aggregator, guess: State) -> State:
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
    guess: State
        The state to use as a guess for steady state.

    """
    return agr.save(agr.compute(agr.load(guess)), strict=True)


def debug_guess_variables(agr: Aggregator, guess: State, variables: Container[str]
                          ) -> dict[str, Value]:
    """Show the errors in a variable's guesstimate across all calculations.

    This is a subset of debug_derivatives, since that debug tool shows a lot of
    data all at once.

    Parameters
    ----------
    agr: Aggregator
        The Aggregator to debug.
    guess: State
        The State we guesstimate as the solution.
    variables: Container[str]
        The variables we want to debug for.

    """
    return debug_derivatives(agr, guess).filter_var_names(lambda x: x in variables)


debug_guess_pressures = partial(debug_guess_variables, variables={'pressure'})


def debug_guess_flows(agr: Aggregator, guess: State) -> dict[str, Value]:
    """Shows the errors in flows from all Kirchhoffs for a guesstimate.

    Parameters
    ----------
    agr: Aggregator
        The Aggregator to debug.
    guess: State
        The State we guesstimate as the solution.

    """
    kirchhoffs = {c.name for c in agr.graph if isinstance(c, Kirchhoff)}
    dk = debug_derivatives(agr, guess).filter_calculations(lambda c: c in kirchhoffs)
    dflow = dk.filter_var_names(lambda s: '->' in s)
    return reduce(operator.or_, dflow.values(), {})
