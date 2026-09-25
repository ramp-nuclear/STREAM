from typing import Sequence

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

from stream import unpacked
from stream.calculation import Calculation, _concat
from stream.composition import Calculation_factory

Addition = Calculation_factory(lambda y, *, x: y + x, [False], dict(y=0))
Multiplication = Calculation_factory(lambda x, *, y: x * y, [False], dict(x=0))
Division = Calculation_factory(lambda z, *, x: z / x, [False], dict(z=0))

add = Addition(name="Add")
multiply = Multiplication(name="Multiply")
divide = Division(name="Divide")


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


def _give_me_values(*, some_input, more_input):
    return some_input, more_input


@given(st.lists(st.floats(allow_nan=False)))
def test_unpack_correctly_excludes_parameters(lst):
    input_dict = dict(enumerate(lst))
    kwargs = dict(some_input=input_dict, more_input=input_dict)

    output, more_output = unpacked(_give_me_values, exclude=["more_input"])(**kwargs)
    assert np.array_equal(np.atleast_1d(output), list(more_output.values()))
    assert more_output == input_dict


def test_unpack_excluded_but_absent_variable_falls_back_to_default():
    """An excluded variable that was never routed must fall back to the wrapped
    function's own default rather than crashing the decorator."""

    def f(*, some_input, maybe=None):
        return some_input, maybe

    output, maybe = unpacked(f, exclude=["maybe"])(some_input={0: 1.0})
    assert np.array_equal(np.atleast_1d(output), [1.0])
    assert maybe is None


def test_feedbackless_point_kinetics_calculate_without_T():
    """PointKinetics.calculate is @unpacked(exclude=("T",)); with default
    temp_worth/ref_temp (no thermal feedback) T is legitimately absent and must
    fall back to its None default instead of the decorator crashing on the pop."""
    from stream.calculations.point_kinetics import PointKinetics

    pk = PointKinetics(
        generation_time=1e-4,
        delayed_neutron_fractions=np.array([0.0065]),
        delayed_groups_decay_rates=np.array([0.08]),
    )
    out = pk.calculate(np.array([1.0, 1.0]), t=0.0)  # T absent -> default None
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_unpack_does_not_mislabel_a_user_keyerror():
    """A KeyError raised inside the wrapped body (e.g. a correlation dict miss)
    must propagate as itself, not be rewritten as a decorator misconfiguration."""

    @unpacked
    def calc(self=None, **kw):
        return {"nucleate": 1}["subcooled"]

    with pytest.raises(KeyError) as excinfo:
        calc(x={0: 1.0})
    assert "subcooled" in str(excinfo.value)
    assert "not recieved" not in str(excinfo.value)  # the old misdirecting message


def test_unpack_preserves_exception_type_when_args_are_empty():
    """An exception constructed with no args (e.g. raise RuntimeError()) must
    surface as its own type, not an IndexError from e.args[0] in the decorator."""

    @unpacked
    def calc(self=None, **kw):
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        calc(x={0: 1.0})


dictvals = st.dictionaries(
    st.integers(),
    st.one_of(
        arrays(dtype=float, shape=st.integers(1, 10), elements=st.floats(allow_nan=False)),
        st.floats(allow_nan=False),
    ),
)


@given(dictvals)
def test_concat_is_at_most_1d(d):
    assert np.ndim(_concat(d)) <= 1


list_arrays = st.lists(arrays(dtype=float, shape=st.integers(1, 10), elements=st.floats(allow_nan=False)))


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
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    indices = set(np.argwhere(arr).flatten())
    starts = sorted({0} | indices)
    ends = sorted({len(arr)} | indices)
    slices = [slice(s, e) for s, e in zip(starts, ends)]
    return dict(zip(alphabet, slices))


arrlengths = st.shared(st.integers(1, 20), key="length")
boolarrs = arrays(bool, arrlengths, elements=st.booleans())
valarrs = arrays(float, arrlengths, elements=st.floats(0.0, 10.0, allow_nan=False))
vardicts = boolarrs.map(_vardict)
emptyfunc = st.just(lambda y, **_: np.zeros(y.shape))
calctypes = st.builds(Calculation_factory, emptyfunc, boolarrs, vardicts)
calcs = calctypes.map(lambda x: x())


@given(calcs, valarrs)
def test_default_save_is_compatible_with_calc_variables(calc: Calculation, arr: np.ndarray):
    state = calc.save(arr)
    for v, place in calc.variables.items():
        assert np.allclose(arr[place], state[v])


@given(st.floats(allow_nan=False))
def test_default_load_for_one_structure(val):
    assert add.load({"y": val}) == [val]
    assert multiply.load({"x": val}) == [val]


@given(calcs, valarrs)
def test_default_load_is_inverse_of_default_save(calc: Calculation, arr: np.ndarray):
    assert np.allclose(calc.load(calc.save(arr)), arr)


# --- attribution: @unpacked, compute context, load ---

from stream import Aggregator
from stream.state import State


def _err_notes(err):
    return "\n".join(getattr(err, "__notes__", []))


class _Bomb(Calculation):
    """A Calculation whose class-level @unpacked calculate raises a chosen exception,
    so args[0] is the instance (a bound method)."""

    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    @unpacked
    def calculate(self, variables, **_):
        raise self._exc

    @property
    def mass_vector(self):
        return np.array([False])

    @property
    def variables(self):
        return {"x": 0}


def test_unpacked_note_names_instance_with_args():
    """An exception WITH args is attributed to the instance name via a note
    (not the unbound function + address); message/args stay intact."""
    b = _Bomb("primary_pump", ValueError("boom"))
    with pytest.raises(ValueError) as exc:
        b.calculate({0: 0.0})
    assert "primary_pump" in _err_notes(exc.value)
    assert str(exc.value) == "boom"  # message untouched


def test_unpacked_zero_args_exception_still_attributed():
    """A zero-args exception keeps its type AND gains an attribution note."""
    b = _Bomb("primary_pump", RuntimeError())
    with pytest.raises(RuntimeError) as exc:
        b.calculate({0: 0.0})
    assert exc.value.args == ()  # args untouched
    assert "primary_pump" in _err_notes(exc.value)


def test_unpacked_non_calculation_first_arg_falls_back_to_qualname():
    """A wrapped function whose first arg is not a Calculation (no `name`) is
    attributed to the qualname and never crashes the decorator."""

    @unpacked
    def calc(self=None, **kw):
        raise ValueError("inner")

    with pytest.raises(ValueError) as exc:
        calc(np.array([1.0]), x={0: 1.0})  # first arg has no .name
    assert "calc" in _err_notes(exc.value)  # qualname fallback
    assert str(exc.value) == "inner"  # message untouched


def test_compute_context_note_names_calculation_and_op():
    """A raising calculate gains a compute-level note naming the Calculation, the op,
    and the time — via the _op chokepoint."""

    def kaboom(y):
        raise ValueError("kaboom")

    A = Calculation_factory(kaboom, [False], {"x": 0})("reactor_core")
    agr = Aggregator.from_decoupled(A)
    with pytest.raises(ValueError) as exc:
        agr.compute(np.array([0.0]), 1.5)
    notes = _err_notes(exc.value)
    assert "while evaluating Calculation 'reactor_core'" in notes
    assert ".calculate" in notes
    assert "t=1.5" in notes


def _core_agr():
    A = Calculation_factory(lambda y: -np.asarray(y, dtype=float), [True] * 3, {"T": slice(0, 3)})("core")
    return Aggregator.from_decoupled(A)


def test_load_attribution_wrong_calc_name_lists_state_keys():
    """A wrong State key stays a KeyError; the note names the missing Calculation
    and lists the State's actual keys."""
    agr = _core_agr()
    with pytest.raises(KeyError) as exc:
        agr.load(State({"kore": {"T": np.zeros(3)}}))
    notes = _err_notes(exc.value)
    assert "core" in notes and "kore" in notes


def test_load_attribution_wrong_variable_names_calc_and_var():
    """A wrong variable name stays a KeyError; the notes name the calculation, the
    expected variable, and the provided keys."""
    agr = _core_agr()
    with pytest.raises(KeyError) as exc:
        agr.load(State({"core": {"temp": np.zeros(3)}}))
    notes = _err_notes(exc.value)
    assert "core" in notes and "T" in notes and "temp" in notes


def test_load_attribution_shape_mismatch_names_var_and_length():
    """A shape mismatch stays a ValueError; the notes name the calculation, the
    variable, and its expected length."""
    agr = _core_agr()
    with pytest.raises(ValueError) as exc:
        agr.load(State({"core": {"T": np.zeros(2)}}))
    notes = _err_notes(exc.value)
    assert "core" in notes and "T" in notes and "3" in notes
