"""Tests for states and dataframes.

"""
import hypothesis.extra.numpy as stnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from networkx import DiGraph
from pandas import DataFrame

from stream.aggregator import State, StateTimeseries, Solution, Aggregator, vars_
from stream.state import to_dataframe, from_dataframe
from .test_calculation import add, multiply


@pytest.fixture(scope="module")
def mock_agr():
    """Simple aggregator example

    """
    return Aggregator(DiGraph([
        (add, multiply, vars_('y')),
        (multiply, add, vars_('x'))
        ]))


def _state_equal(s1: State, s2: State) -> bool:
    if set(s1.keys()) != set(s2.keys()):
        return False
    if any(s1[k].keys() != s2[k].keys() for k in s1):
        return False
    for key in s1:
        for key2 in s1[key]:
            if not np.allclose(s1[key][key2], s2[key][key2]):
                return False
    return True


def _timeseries_equal(s1: StateTimeseries, s2: StateTimeseries) -> bool:
    if set(s1.keys()) != set(s2.keys()):
        return False
    return all(_state_equal(s1[key], s2[key]) for key in s1)


def _vertical_records(v: np.array):
    return [{'calculation': '', 'variable': '', 'j': 0, 'i': i, 'value': vv}
            for i, vv in enumerate(v)]


keys = st.text(alphabet='abcdefghijklmnopqrstuvwxyz')
finites = st.floats(allow_nan=False, allow_subnormal=False, allow_infinity=False,
                    min_value=1e-10, max_value=1e10)
vectors = stnp.arrays(float, elements=finites, shape=st.integers(1, 20))
vertical_vectors = stnp.arrays(float, elements=finites,
                               shape=st.tuples(st.integers(1, 20), st.just(1)))
vertical_dfs = vectors.map(_vertical_records).map(DataFrame.from_records)
matrices = stnp.arrays(float, elements=finites,
                       shape=st.tuples(st.integers(1, 20), st.integers(1, 20)))
values = st.one_of(finites, vectors, matrices)
vardicts = st.dictionaries(keys, values, min_size=1, max_size=10)
states = st.dictionaries(keys, vardicts, min_size=1, max_size=10).map(State)
state_timeseries = st.dictionaries(finites, states, min_size=1, max_size=10)


@settings(deadline=None)
@given(vertical_vectors.filter(lambda x: len(x) > 1))
def test_dataframe_of_vertical_vector_has_multiple_i_single_j(v):
    """Tests that a dataframe with vertical vectors has multiple values of i and
    only 1 value of j.

    """
    s = State({'': {'': v}})
    df = s.to_dataframe()
    assert len(set(df.i)) > 1
    assert len(set(df.j)) == 1


@given(vertical_dfs.filter(lambda x: len(x.i) > 1))
def test_state_of_vertical_vector_is_vertical(df: DataFrame):
    """Tests that a vertically created dataframe is read as vertical vectors.

    """
    vector = State.from_dataframe(df)['']['']
    assert len(vector.shape) == 2
    assert vector.shape[1] == 1


@pytest.mark.slow
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(states)
def test_state_to_dataframe_and_back_is_same(s: State):
    """Tests that going to dataframe and back is unity.

    """
    assert _state_equal(from_dataframe(to_dataframe(s)), s)


@pytest.mark.slow
@settings(deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(state_timeseries)
def test_statetimeseries_to_dataframe_and_back_is_same(s: StateTimeseries):
    """Tests that :func:`~stream.state.state_timeseries_from_dataframe`
    is the inverse of :func:`~stream.state.state_timeseries_to_dataframe`

    """
    assert _timeseries_equal(from_dataframe((to_dataframe(s))), s)


num_times = st.shared(st.integers(1, 20), key='time_length')
times = stnp.arrays(float, num_times, elements=finites, unique=True).map(np.sort)
soldata = stnp.arrays(float, num_times.map(lambda x: (x, 2)),
                      elements=finites)
solutions = st.builds(Solution, times, soldata)


@given(sol=solutions)
def test_solution_to_statetimeseries_and_back_is_same_by_example(mock_agr, sol: Solution):
    """Tests that :meth:`~stream.aggregator.Aggregator.solution_from_states` is the
    inverse of :meth:`~stream.aggregator.Aggregator.solution_from_states`.

    """
    assert mock_agr.load(mock_agr.save(sol)) == sol
