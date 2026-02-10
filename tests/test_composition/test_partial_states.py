import operator
from functools import reduce
from itertools import chain

import hypothesis.strategies as st
from cytoolz import unique
from hypothesis import given

from stream.state import State


class HasVariables:
    def __init__(self, name, *vs):
        self.variables = vs
        self.name = name

    def __repr__(self):
        return f"HasVariables({self.name}, {', '.join(map(str, self.variables))})"


def test_uniform_with_an_example():
    a = HasVariables('a', 1, 2, 3)
    b = HasVariables('b', "a", "b", "c")
    c = HasVariables('c', "hello")

    # noinspection PyTypeChecker
    d = State.uniform((a, b, c), 0.0, 1, 2, "a", "hello")
    assert d == {a.name: {1: 0.0, 2: 0.0}, b.name: {"a": 0.0}, c.name: {"hello": 0.0}}


keys = st.characters(whitelist_categories='L')
calculations = st.integers(1, 5).flatmap(
    lambda n: st.builds(HasVariables, st.just(str(n)), *(n * [keys])))
calc_lists = (st.lists(calculations, min_size=1)
              .filter(lambda lst: len(set(c.name for c in lst)) == len(lst)))
vars_indices = st.lists(st.integers(min_value=0), min_size=1)


@given(calc_lists, st.floats(allow_nan=False, allow_infinity=False), vars_indices)
def test_uniform_sets_only_to_given_value(calcs, value, indices):
    varlist = list(unique(chain(*[c.variables for c in calcs])))
    vlen = len(varlist)
    picked = [varlist[i] for i in indices if i < vlen]

    d = State.uniform(calcs, value, *picked)
    calculated = reduce(operator.__or__,
                        (set(subd.values()) for subd in d.values())
                        )
    assert calculated == {value}, calculated


@given(calc_lists, st.floats(allow_nan=False, allow_infinity=False), vars_indices)
def test_uniform_sets_variables_it_is_told_to(calcs, value, indices):
    varlist = list(unique(chain(*[c.variables for c in calcs])))
    vlen = len(varlist)
    picked = [varlist[i] for i in indices if i < vlen]

    d = State.uniform(calcs, value, *picked)
    calculated = reduce(operator.__or__,
                        (set(subd.keys()) for subd in d.values())
                        )
    assumed = set(picked) if picked else set(varlist)
    assert calculated == assumed, (calculated, assumed)


def test_merge_with_an_example():
    a = {1: {2: 3, 3: 4}, 2: {"a": 1, "b": 2}}
    b = {1: {2: 4, 4: 5}, 2: {}, 3: {1: 2}}

    # noinspection PyTypeChecker
    assert State.merge(a, b) == {1: {2: 4, 3: 4, 4: 5},
                                 2: {"a": 1, "b": 2},
                                 3: {1: 2}}


dicts = st.dictionaries(keys, st.dictionaries(keys, st.integers()))


@given(dicts, dicts)
def test_merge_has_union_keys(a: dict, b: dict):
    c = State.merge(a, b)
    assert set(c.keys()) == (set(a.keys()) | set(b.keys()))
    for key, d in c.items():
        assert set(d.keys()) == (set(a.get(key, {})) | set(b.get(key, {})))


@given(dicts, dicts)
def test_merge_values_appear_in_originals(a: dict, b: dict):
    for key, d in State.merge(a, b).items():
        assert set(d.values()) <= (set(a.get(key, {}).values())
                                   | set(b.get(key, {}).values()))


def test_filter_values_with_an_example():
    a = State({1: {2: 3, 3: 4}, 2: {"a": 1, "b": 2}, 3: {1: 1}})

    def iseven(x): return x % 2 == 0

    assert a.filter_values(f=iseven) == {1: {3: 4}, 2: {"b": 2}}


def test_filter_calculations_with_an_example():
    a = State({1: {2: 3, 3: 4}, 2: {"a": 1, "b": 2}, 3: {1: 1}})

    def iseven(x): return x % 2 == 0

    assert a.filter_calculations(f=iseven) == {2: {"a": 1, "b": 2}}
