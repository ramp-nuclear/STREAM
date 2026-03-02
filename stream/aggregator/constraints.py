from enum import Enum
from typing import Iterable, TypedDict, Unpack

import numpy as np

from stream.state import State
from stream.utilities import mutually_exclusive

from .aggregator import Aggregator

__all__ = ["CONSTRAINT", "create_constraints"]


class CONSTRAINT(Enum):
    """Possible values for IDA for sign constraints.
    See the sundials documentation for explanation of these values
    """

    negative = -2.0
    non_positive = -1.0
    none = 0.0
    non_negative = 1.0
    positive = 2.0


_VARTYPE = Iterable[str] | None


class _ConstraintTypes(TypedDict, total=False):
    negative: _VARTYPE
    non_positive: _VARTYPE
    none: _VARTYPE
    non_negative: _VARTYPE
    positive: _VARTYPE


def create_constraints(
    agr: Aggregator,
    default_sign: CONSTRAINT = CONSTRAINT.none,
    **kwargs: Unpack[_ConstraintTypes],
) -> np.ndarray:
    """
    Create a constraint array, as expected by IDA
    Currently, we support sign constraints in DAE mode only.
    Defaults to no sign constraint for all variables.

    Meant to be used as the `contraints_type` option for `differential_algebraic`.

    Parameters
    ----------
    agr: Aggregator
        The aggregator for which to create the constraints array.
    default_sign: CONSTRAINT
        The default option to set all variables to, if not specified in kwargs.
    kwargs: _ConstraintTypes
        The variables to set to each of the possible CONSTRAINT values.

    Returns
    -------
    np.ndarray
        Array with the same shape as `agr.graph`, with sign contraints.
    """
    assert mutually_exclusive(list(kwargs.values())), (
        "Keyword list must be mutually exclusive - a variable cannot be in more than one category"
    )

    constraint_state = State.uniform(agr.graph, default_sign.value)

    for sign, variables in kwargs.items():
        if sign not in {item.name for item in CONSTRAINT}:
            raise KeyError(f"Invalid constraint type {sign}. Must be one of {CONSTRAINT.__members__}")
        if variables is not None:
            state = State.uniform(agr.graph, CONSTRAINT.__getitem__(sign).value, *variables)
            constraint_state = State.merge(constraint_state, state)

    return np.array(agr.load(constraint_state))
