from typing import Sequence

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

from stream import unpacked
from stream.calculation import _concat, Calculation
from stream.composition import Calculation_factory


Addition = Calculation_factory(lambda y, *, x: y + x, [False], dict(y=0))
Multiplication = Calculation_factory(lambda x, *, y: x * y, [False], dict(x=0))
Division = Calculation_factory(lambda z, *, x: z / x, [False], dict(z=0))

add = Addition(name="Add")
multiply = Multiplication(name="Multiply")
divide = Division(name='Divide')


@given(st.lists(st.floats(allow_nan=False)))
def test_unpack_correctly_unpacks_data(lst):
    # noinspection PyTypeChecker
    kwargs = dict(
        some_input=dict(enumerate(lst)),
        more_input=dict(enumerate(map(np.array, lst))),
        )

    def give_me_values(*, some_input, more_input):
        return some_input, more_input

    output, more_output = unpacked(give_me_values)(**kwargs)
    assert np.allclose(output, np.array(lst))
    assert np.allclose(more_output, np.array(lst))


dictvals = st.dictionaries(
    st.integers(),
    st.one_of(arrays(dtype=float, shape=st.integers(1, 10),
                     elements=st.floats(allow_nan=False)),
              st.floats(allow_nan=False))
    )


@given(dictvals)
def test_concat_is_at_most_1d(d):
    assert np.ndim(_concat(d)) <= 1


list_arrays = st.lists(arrays(dtype=float, shape=st.integers(1, 10),
                              elements=st.floats(allow_nan=False)))


@given(list_arrays)
def test_concat_of_dictionaried_arrays_is_the_same_as_their_numpy_concat(lst):
    d = dict(zip(range(len(lst)), lst))
    if lst:
        assert np.allclose(_concat(d), np.concatenate(lst))
    else:
        assert not len(_concat(d))


@given(st.floats(allow_nan=False))
def test_default_save_has_correct_output_for_one_structure(val):
    assert add.save([val], x=None) == {"y": val}
    assert multiply.save([val], y=None) == {"x": val}


def _vardict(arr: np.ndarray) -> Sequence[slice]:
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    indices = set(np.argwhere(arr).flatten())
    starts = sorted({0} | indices)
    ends = sorted({len(arr)} | indices)
    slices = [slice(s, e) for s, e in zip(starts, ends)]
    return dict(zip(alphabet, slices))


arrlengths = st.shared(st.integers(1, 20), key="length")
boolarrs = arrays(bool, arrlengths, elements=st.booleans())
valarrs = arrays(float, arrlengths, elements=st.floats(0., 10., allow_nan=False))
vardicts = boolarrs.map(_vardict)
emptyfunc = st.just(lambda y, **_: np.zeros(y.shape))
calctypes = st.builds(Calculation_factory, emptyfunc, boolarrs, vardicts)
calcs = calctypes.map(lambda x: x())


@given(calcs, valarrs)
def test_default_save_is_compatible_with_calc_variables(
        calc: Calculation, arr: np.ndarray):
    state = calc.save(arr)
    for v, place in calc.variables.items():
        assert np.allclose(arr[place], state[v])


@given(st.floats(allow_nan=False))
def test_default_load_for_one_structure(val):
    assert add.load({"y": val}) == [val]
    assert multiply.load({"x": val}) == [val]


@given(calcs, valarrs)
def test_default_load_is_inverse_of_default_save(
        calc: Calculation, arr: np.ndarray):
    assert np.allclose(calc.load(calc.save(arr)), arr)
