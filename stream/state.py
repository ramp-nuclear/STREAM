"""The concept of a human-readable state of the system.

Basically a 2-deep nested dict with str keys and the inner values are Value
objects. We provide additional useful methods in the same namespace.

"""
from functools import reduce
from itertools import chain
from operator import or_
from typing import Iterable, Any, IO, Callable, TypeVar

import numpy as np
import yaml
from pandas import DataFrame

from cytoolz.dicttoolz import keyfilter, valfilter, valmap
from stream.calculation import CalcState, Calculation
from stream.units import Value


def parse_value(records: DataFrame) -> Value:
    """Parse records in a dataframe into a float/array, depending on the shape they represent.

    Parameters
    ----------
    records: DataFrame
        The DataFrame that contains only records that belong to one variable.

    Returns
    -------
    Value
        This is a float if the array is a scalar, or a numpy array if it is dimensional.

    """
    records = records.sort_values(by=['i', 'j'])
    match (len(records.i.values), len(records.j.values)):
        case (1, 1):
            return records.value.values.item()
        case (1, _):
            return np.array(records.value.values)
        case _:
            vector = np.array(records.value.values)
            shape = (records.i.max() + 1, records.j.max() + 1)
            return vector.reshape(shape)


_T = TypeVar('_T')


def _two_deep_value_switch(d: dict[_T, dict],
                           switch: Callable[[...], Any]) -> dict[_T, dict]:
    return {k: {k2: switch(v2) for k2, v2 in v.items()}
            for k, v in d.items()}


ListState = dict[str, dict[str, float | list]]
DictState = dict[str, CalcState]


class State(dict):
    """A nested dictionary with str keys that connects calculations to a dictionary of the values of their variables.

    """

    @classmethod
    def merge(cls, *st: DictState) -> "State":
        """Merge states together. Later states have precedence."""
        return cls({
            k: reduce(or_, (s.get(k, {}) for s in st))
            for k in set(chain(*(s.keys() for s in st)))
            })

    @classmethod
    def uniform(cls, calculations: Iterable[Calculation], value: Value, *variables: str
                ) -> "State":
        r"""Create a state, or a partial state for given calculations and given
        target variables, which are assigned a uniform value.

        Parameters
        ----------
        calculations: Iterable[Calculation]
            Desired calculations to assign a partial state.
        value: Value
            Value to assign to the variable states.
        variables: str
            Variables to have states. If none are given, all variables in ``calculations`` are included.

        Returns
        -------
        state: State
            A partially legal state
        """
        if variables:
            return cls({
                c.name: {var: value for var in c.variables if var in variables}
                for c in calculations
                })
        return cls({c.name: {var: value for var in c.variables} for c in calculations})

    def filter_values(self, f: Callable[[Value], bool]) -> "State":
        """Filter out state variables predicated upon a function.

        Parameters
        ----------
        f: Callable
            Predicate function. False values are filtered out.

        Returns
        -------
        state': State
            A filtered partial state

        Examples
        --------
        >>> State(A=dict(b=1), a=dict(B=2)).filter_values(lambda x: x == 2)
        {'a': {'B': 2}}
        """
        return _filter_vars(f, self, valfilter)

    def filter_var_names(self, f: Callable[[str], bool]) -> "State":
        """Filter out state variable names predicated upon a function.

        Parameters
        ----------
        f: Callable
            Predicate function. False values are filtered out.

        Returns
        -------
        state': State
            A filtered partial state

        Examples
        --------
        >>> State(A=dict(b=1), a=dict(B=2)).filter_var_names(str.isupper)
        {'a': {'B': 2}}
        """
        return _filter_vars(f, self, keyfilter)

    def filter_calculations(self, f: Callable[[str], bool]) -> "State":
        """Filter out state calculations predicated upon a function.

        Parameters
        ----------
        f: Callable
            Predicate function. False values are filtered out.

        Returns
        -------
        state': State
            A filtered partial state

        Examples
        --------
        >>> State(A=dict(b=1), a=dict(B=2)).filter_calculations(str.isupper)
        {'A': {'b': 1}}
        """
        return type(self)(keyfilter(f, self))

    def value_map(self, f: Callable[[Value], Value]) -> "State":
        return type(self)({c: valmap(f, d) for c, d in self.items()})

    def records(self) -> Iterable[dict[str, Any]]:
        """Generate records from this state.

        """
        for calc, variables in self.items():
            for var_name, values in variables.items():
                if isinstance(values, float):
                    yield {'calculation': calc, 'variable': var_name,
                           'i': 0, 'j': 0, 'value': values}
                else:
                    vals = np.atleast_2d(values)
                    for i, oned in enumerate(vals):
                        for j, v in enumerate(oned):
                            yield {'calculation': calc, 'variable': var_name,
                                   'i': i, 'j': j, 'value': v}

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> "State":
        """Creates a State from a DataFrame representation.

        Parameters
        ----------
        df: DataFrame
            Data to turn into a State

        """
        state = cls()
        for c in set(df.calculation):
            state[c] = {}
            calc_df = df[df.calculation == c]
            for var in set(calc_df.variable):
                state[c][var] = parse_value(calc_df[calc_df.variable == var])
        return state

    @classmethod
    def from_liststate(cls, s: ListState) -> "State":
        """Make a new State from its serializable list version.
        """
        d = _two_deep_value_switch(s, lambda x: x if isinstance(x, float) else np.array(x))
        return cls(d)

    @classmethod
    def load(cls, f: IO) -> "State":
        """Load a State from a YAML IO.

        Parameters
        ----------
        f: IO
            Stream to read from

        """
        d = yaml.full_load(f)
        return cls.from_liststate(d)

    def to_dataframe(self) -> DataFrame:
        """Represents a State as a DataFrame

        """
        return (
            DataFrame
            .from_records(self.records())
            .astype(dict(calculation="category", variable="category", i="uint16", j="uint16"))
            )

    def listify(self) -> ListState:
        """Replace all arrays in self with serializable lists

        """
        return _two_deep_value_switch(self, lambda x: x if isinstance(x, float) else x.tolist())

    def dump(self, f=None) -> str | None:
        """Dump this State as YML.

        Parameters
        ----------
        f: Stream to dump to.

        """
        return yaml.dump(self.listify(), f)


#: A mapping from time points to Aggregator :class:`.State`
StateTimeseries = dict[float, State]


def _gen_records_from_timeseries(s: StateTimeseries) -> Iterable[dict[str, Any]]:
    for t, state in s.items():
        for record in state.records():
            record['time'] = t
            yield record


def to_dataframe(s: DictState | StateTimeseries) -> DataFrame:
    """Transforms the state(s) into a DataFrame.

    Parameters
    ----------
    s: State | StateTimeseries
        State or a time keyed dictionary of states.

    Returns
    -------
    DataFrame
        A Pandas DataFrame.

    """
    has_float_keys = any(isinstance(key, float) for key in s)
    return (_state_timeseries_to_dataframe(s) if has_float_keys
            else State(s).to_dataframe())


def _state_timeseries_to_dataframe(s: StateTimeseries) -> DataFrame:
    """Turns a multi-time solution from an Aggregator into a DataFrame.

    Parameters
    ----------
    s: StateTimeseries
        The solution at different times to turn to a DataFrame

    """
    return (
        DataFrame
        .from_records(_gen_records_from_timeseries(s))
        .astype(dict(calculation="category", variable="category", i="uint16", j="uint16"))
        )


def from_dataframe(df: DataFrame) -> State | StateTimeseries:
    """Creates a State or a dictionary of {time: State} from a DataFrame representation.

    Parameters
    ----------
    df: DataFrame
        Data to turn into a State

    """
    return (state_timeseries_from_dataframe(df) if 'time' in df.columns
            else State.from_dataframe(df))


def state_timeseries_from_dataframe(df: DataFrame) -> StateTimeseries:
    """Reads a DataFrame into an Aggregator-readable StateTimeseries.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to parse.

    Raises
    ------
    ValueError
        If the given DataFrame has no column named 'time'.

    See Also
    --------
    .state_timeseries_to_dataframe

    """
    if 'time' not in df.columns:
        raise ValueError("The DataFrame given isn't time-related.")
    return {tv: State.from_dataframe(df[df.time == tv])
            for tv in set(df.time.values)}


def _filter_vars(f: Callable[[...], bool], state: State, filterer) -> State:
    return State(valfilter(lambda d: d != {}, {c: filterer(f, d) for c, d in state.items()}))
