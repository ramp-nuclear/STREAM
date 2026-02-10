"""
Here lie some unspoken heroes...

As well as some helper functions regarding mixing, conversions, flux shapes
and more.

"""
import warnings
from contextlib import contextmanager
from functools import partial, reduce, singledispatch
from operator import add
from typing import Any, Callable, Iterable, Sequence, Type, TypeVar, Protocol, ParamSpec

import numpy as np
from cytoolz import valmap
from numba import njit
# noinspection PyProtectedMember
from numpy._core._multiarray_umath import normalize_axis_index
# noinspection PyProtectedMember
from numpy.lib._function_base_impl import _diff_dispatcher, array_function_dispatch
from scipy.optimize import fsolve

from stream.units import Array, Array1D, Celsius, Fahrenheit, KgPerS, Place, Value


STREAM_DEBUG = 11


def harmonic_mean(*a: Value, axis=1):
    m = np.column_stack(a)
    return 1 / np.sum((1 / m), axis=axis)


def normalize(a: Sequence) -> Array1D:
    return np.asarray(a) / np.sum(a)


@njit
def to_Fahrenheit(T: Celsius) -> Fahrenheit:
    return 1.8 * T + 32


def _shifted_sin(a: float, mid: float, x: Value):
    return np.sin(a * (x - mid))


def _integrated_cell_cosine_value(a: float, b: float, mid: float, x: Array1D
                                  ) -> Array1D:
    return (b / a) * np.diff(_shifted_sin(a, mid, x))


def cosine_shape(x: Array1D, ppf: float = np.pi / 2, *,
                 xmax: float = None,
                 ) -> Array1D:
    r"""
    Creates a normalized cosine profile for cells whose boundaries are `x`,
    assuming its maximum is achieved at :math:`\ell/2 = (x[-1] + x[0])/2`
    unless stated otherwise.

    Parameters
    ----------
    x: Array1D
        Linear array, 1D spatial ticks of cells boundaries
    ppf: float
        Power Peaking Factor. The default value is :math:`\pi/2`.
    xmax: float
        The point where the cosine shape is at its theoretical maximum.
        Defaults to the midpoint between x[0] and x[-1].

    Returns
    -------
    phi: Array1D
        a normalized array whose profile is cosine which is zero at the
        outer boundaries, and is normalized to unity.

    Notes
    -----
    Here is the derivation. The flux profile is assumed to take the following form:

    .. math:: \phi = \cos\left(\frac{\pi (x - \ell/2)}{L}\right)

    Where :math:`L` is the extrapolated length (half of the period),
    which is set by the ``PPF``:

    .. math:: \text{PPF} \equiv \frac{\max{\phi}}{\bar{\phi}}

    That is

    .. math:: 1/\text{PPF} = \bar{\phi}
     = \frac{1}{\ell}\int^{\ell/2}_{-\ell/2}dx\cos\left(\frac{\pi (x-\ell/2)}{L}\right)
     = \frac{2L}{\ell\pi}\sin\left(\frac{\ell\pi}{2L}\right)

    Then, :math:`L` is found by solving for
    :math:`h = \ell\pi/2L \Rightarrow h/\text{PPF} = \sin(h)`.

    Finally, the cosine shape is integrated for each cell:

    .. math:: \int^{x_{i+1}}_{x_i} dx' \cos\left(\frac{\pi (x' - \ell/2)}{L}\right)
        = \frac{L}{\pi}\left[\sin\left(\frac{\pi (x_{i+1} - \ell/2)}{L}\right)
        - \sin\left(\frac{\pi (x_i - \ell/2)}{L}\right)\right]
    """
    if not (1 <= ppf <= np.pi / 2):
        raise ValueError(f"PPF must be in [1, π/2], but was {ppf:.5f}")
    xmax = (x[-1] + x[0]) / 2 if xmax is None else xmax
    ll = x[-1] - x[0]
    # Safe because we know fsolve returns a float-able for this input deck.
    # noinspection PyTypeChecker
    h: float = float(fsolve(lambda _x: np.sinc(_x / np.pi) - 1 / ppf, x0=1e-3))
    a = 2 * h / ll
    b = ppf / ll
    return _integrated_cell_cosine_value(a, b, xmax, x)


def cosine_shape_by_zero_endpoints(xi: float, xe: float, x: Array1D) -> Array1D:
    r"""A cosine shape which is 0 at the extrapolation values, and integrates
    to 1 over the unextrapolated values.

    Parameters
    ----------
    xi: Low extrapolation boundary.
    xe: High extrapolation boundary.
    x: Boundaries over which to bin the cosine shape

    Notes
    -----
    The general cosine shape is :math:`b\cos\left(a(x-x_{max})\right)`.

    :math:`a` is determined from the extrapolation, where the cosine nullifies
    at :math:`-\frac{\pi}{2},\frac{\pi}{2}`. This means that at xi and xe we
    want :math:`a(xe-x_{max})=\frac{\pi}{2}` and :math:`a(xi-x_{max})=-\frac{\pi}{2}`.
    Since the max point of a cosine shape is exactly midway between those points,
    :math:`x_{max}=\frac{xi+xe}{2}`, which means that if we set :math:`ll=xe-xi`
    we can write :math:`a=\frac{\pi}{ll}`.

    :math:`b` is determined from the normalization condition. We desire that
    the integral inside the boundaries in the `x` vector be strictly 1.
    If we mark the external boundaries as :math:`x_0` and :math:`x_1`, we get:

    .. math:: 1 = b\int_{x_0}^{x_1}{\cos(a(x-x_{max}))dx}
                = \frac{b}{a}\left[\sin(a(x_1-x_{max}))-\sin(a(x_0-x_{max}))\right]

    and thus :math:`b=a\left[\sin(a(x_1-x_{max}))-\sin(a(x_0-x_{max}))\right]`.

    Now that we know what the analytical shape is, we can do the discretization.
    We integrate over each cell in the boundaries of `x`, which have the same
    shape as the equation above, but for a different integration range:

    .. math:: v_i = b \int_{x_i}^{x_{i+1}}{\cos(a(x-x_{max}))dx}
                  = \frac{b}{a}\left[\sin(a(x_{i+1}-x_{max}))-\sin(a(x_i-x_{max}))\right]

    This guarantees that :math:`\sum_{i}{v_i}=1`.

    See Also
    --------
    cosine_shape:
        A similar function based on knowing the PPF rather than the extrapolation
        points.

    Returns
    -------
    Array1D
        A discretized cell integration of a cosine function which is 0 at xi and xe,
        and integrates to 1 over [xi_in, xe_in].
        The vector is thus normalized to have a sum of 1.

    """
    xi_in = x[0]
    xe_in = x[-1]
    ll = xe - xi
    mid = (xe + xi) / 2
    a = np.pi / ll
    b = a / np.diff(_shifted_sin(a, mid, np.array([xi_in, xe_in])))
    return _integrated_cell_cosine_value(a, b, mid, x)


def uppercase_numeric_only(s: str) -> str:
    """Filters a string to return only uppercase or numeric values.

    Parameters
    ----------
    s: str
        The string to filter.

    Examples
    --------
    >>> uppercase_numeric_only('ThisRunsUnderEvaluations')
    'TRUE'
    >>> uppercase_numeric_only('MOthroOw3')
    'MOO3'


    """
    return "".join(filter(lambda c: str.isupper(c) or str.isnumeric(c), s))


def to_array(d: dict[Any, float], dtype=np.float64) -> Array1D:
    """Turn dictionary values into numpy array of type dtype.

    Parameters
    ----------
    d: dict[Any, float]
        Dictionary to turn into a 1D array.
    dtype:
        NumPy dtype value.

    Returns
    -------
    Array1D
        An array of just the values from the dictionary.
    """
    return np.fromiter(d.values(), dtype=dtype, count=len(d))


def _yield_from(val_list):
    for vals in val_list:
        try:
            yield from vals
        except TypeError:
            yield vals


def _length_if_possible(vals):
    try:
        return len(vals)
    except TypeError:
        return 1


def flatten_values(d: dict[Any, Value],
                   dtype=np.float64) -> Value:
    """Take a dictionary of values or sequences of values and return a long
    array with all the values in order, flattened out.

    Parameters
    ----------
    d: dict[Any, float | Sequence[float]]
        Dictionary to flatten down.
    dtype:
        NumPy dtype value.

    Returns
    -------
    Value
        An array of just the values from the dictionary, unless there is only
        one element, in which case it is returned.

    """
    count = sum(map(_length_if_possible, d.values()))
    if count == 1:  # if there is only one, it's faster to return a float
        return next(iter(d.values()))
    return np.fromiter(_yield_from(d.values()), dtype=dtype, count=count)


@contextmanager
def ignore_warnings(warn_type: Type[Warning]):
    warnings.filterwarnings("ignore", category=warn_type)
    yield
    try:
        # noinspection PyUnresolvedReferences
        warnings.filters.pop()
    except IndexError:
        pass


@array_function_dispatch(_diff_dispatcher)
def pair_mean(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Uses np.diff as a base method,
    but calculates the mean between two consecutive pairs. Inline comments in
    the function itself highlight the changes.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are averaged. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the mean is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the mean.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes. Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    pair : ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    numpy.diff
    """

    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))

    a = np.asanyarray(a)
    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is not np._NoValue:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not np._NoValue:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    # np.add replaced np.subtract
    op = np.not_equal if a.dtype == np.bool_ else np.add
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a / 2  # Divided by 2.


def pair_mean_1d(a, prepend=None, append=None):
    assert a.ndim == 1
    assert prepend is None or append is None
    sl1, sl2 = a[:-1], a[1:]
    mn = (sl1 + sl2) / 2
    if prepend is None and append is None:
        return mn
    n = len(a)
    res = np.empty(n, dtype=a.dtype)
    if prepend is not None:
        res[1:] = mn
        res[0] = (prepend + a[0]) / 2
    elif append is not None:
        res[:-1] = mn
        res[-1] = (append + a[-1]) / 2
    else:
        raise ValueError("This should be very impossible to reach. "
                         "If you are here, abandon all hope")
    return res


@njit
def lin_interp(x1: Value, x2: Value, y1: Value, y2: Value, x: Value) -> Value:
    r"""
    Linearly interpolate between two points: (x1, y1) and (x2, y2).
    The assumption being that :math:`x_1 \neq x_2`.

    Let a line be drawn between the two points. Thus:

    :math:`y_1 = a x_1 + b, y_2 = a x_2 + b \Rightarrow a=(y_2-y_1)/(x_2-x_1)`
    and :math:`b = y_2 - a x_2 = y_1 - a x_1`

    Then:

    .. math::
        y = \frac{(y_2 - y_1)}{(x_2 - x_1)}(x - x_2) + y_2


    Parameters
    ----------
    x: Value
        desired position for the interpolation
    x1, x2: Value
        known points, assuming :math:`x \in [x_1, x_2]`
    y1, y2: Value
        known function values for x1, x2

    Returns
    -------
    The interpolated value, y

    Examples
    --------
    >>> lin_interp(x1=1, x2=3, y1=1, y2=3, x=2)
    2.0
    >>> lin_interp(x1=0, x2=4, y1=0, y2=2, x=3)
    1.5
    """
    return y2 + ((y2 - y1) / (x2 - x1)) * (x - x2)


_P = ParamSpec('_P')
_T = TypeVar('_T')


class _FloatWorthy(Protocol):
    def __add__(self: _T, other: float) -> _T:
        ...

    def __radd__(self: _T, other: float) -> _T:
        ...

    def __mul__(self: _T, other: float) -> _T:
        ...

    def __rmul__(self: _T, other: float) -> _T:
        ...


_G = TypeVar('_G', bound=_FloatWorthy)


def factor(f: Callable[_P, _G], by: float = 1.0, add: float = 0.0) -> Callable[_P, _G]:
    """A functor used to lineraly transform a given function's output by some values.

    Parameters
    ----------
    f: Callable
        The decorated function
    by: float
        The multiplicative value
    add: float
        The additve value

    Returns
    -------
    f': Callable
        A new function

    Examples
    --------
    >>> factor(np.ones, 2)(3)
    array([2., 2., 2.])
    """

    def ff(*args, **kwargs):
        return by * f(*args, **kwargs) + add

    try:
        name = f.__name__
    except AttributeError:
        # noinspection PyUnresolvedReferences
        name = f.func.__name__
    ff.__name__ = f'{name} * {by:.2g} + {add:.2g}'
    ff.__doc__ = f.__doc__

    return ff


def if_is(x: Iterable, if_none: Any = 1.0):
    return x if x is not None else if_none


MDOT_INTER_THRESHOLD = 1e-6


@njit
def directed_Tin(Tin: Celsius | None, Tin_minus: Celsius | None, mdot: KgPerS) -> Celsius:
    r"""Computes the inlet temperature for a point component based on flow
    direction.

    Parameters
    ----------
    Tin, Tin_minus : Celsius or None
        Positive (Negative) flow associated inlet temperature
    mdot : KgPerS
        Fluid mass flow rate

    Returns
    -------
    Tin': Celsius
        Inlet temperature

    Notes
    -----
    For the sake of removing stiffness, for absolute flow
    values under ``MDOT_INTER_THRESHOLD``, a linear interpolation between
    ``Tin`` and ``Tin_minus``. This value may be changed by overriding
    ``stream.utilities.MDOT_INTER_THRESHOLD``.

    Examples
    --------
    >>> directed_Tin(1, 2, 3)
    1.0
    >>> directed_Tin(1, 2, -3)
    2.0
    >>> directed_Tin(None, 2, 3)
    2
    >>> directed_Tin(1, None, 3)
    1
    >>> directed_Tin(1, 2, 0)
    1.5
    >>> directed_Tin(np.array([1., 2.]), np.array([3., 4.]), 0.)
    array([2., 3.])

    Raises
    ------
    ValueError : If both ``Tin`` and ``Tin_minus`` are None.
    """
    a = Tin is None
    b = Tin_minus is None
    if a and b:
        raise ValueError("Couldn't deduce temperature, both Tin and Tin_minus are None")
    elif b:
        return Tin
    elif a:
        return Tin_minus
    if np.abs(mdot) < MDOT_INTER_THRESHOLD:
        return lin_interp(
            -MDOT_INTER_THRESHOLD, MDOT_INTER_THRESHOLD, Tin_minus, Tin, mdot
            )
    return Tin if mdot >= 0 else Tin_minus


@njit
def directed(a: np.ndarray, val) -> np.ndarray:
    """
    Parameters
    ----------
    a: np.ndarray
        array to be traversed
    val:
        determining value for traversal. If negative, the array is flipped

    Returns
    -------
    a*: np.ndarray
        A view of 'a' which is either flipped or not
    """
    return a if val >= 0 else a[::-1]


T = TypeVar("T")


def just(val: T) -> Callable[[...], T]:
    r"""
    Parameters
    ----------
    val: Any
        A value which the returned function `just` returns

    Returns
    -------
    f: Callable
        A function which returns ``val`` for any input.

    Examples
    --------
    >>> a = just(5)
    >>> a(3, b="yes")
    5
    """

    def _val(*_, **__) -> T:
        return val

    return _val


def identity(x: T) -> T:
    return x


def summed(it: Iterable[T], initial=None) -> T:
    """Sum items of an iterable by their ``__add__()`` function

    This differs in behavior from the builtin ``sum`` in that sum requires an
    initial value, which is 0 by default, whereas ``summed`` does not.

    Parameters
    ----------
    it : Iterable[T]
        The items to be summed
    initial:
        Value to start the sum from.

    Returns
    -------
    sum: T
        The summed item

    Examples
    --------
    >>> summed(("Hello, ", "it's ", "nice ", "to ", "meet ", "you!"))
    "Hello, it's nice to meet you!"
    >>> summed(([1, 2], [3, 4]))
    [1, 2, 3, 4]
    """
    if initial is not None:
        return reduce(add, it, initial)
    else:
        return reduce(add, it)


def concat(*arrays: Sequence, **kwargs) -> Array:
    """np.concatenate is cumbersome, this function makes it nicer to use
    >>> concat((False, ), (False, False))
    array([False, False, False])
    >>> concat((False,), (False, False), np.zeros(3, dtype=bool))
    array([False, False, False, False, False, False])

    """
    return np.concatenate(arrays, **kwargs)


def strictly_monotonous(*arrays: Sequence) -> Array:
    r"""Concatenate and sort arrays for strictly monotonously rising time

    Examples
    --------
    >>> strictly_monotonous([1, 2, 3], [1.5, 2, 2.5])
    array([1. , 1.5, 2. , 2.5, 3. ])
    """
    return np.unique(np.sort(concat(*arrays)))


@singledispatch
def offset(s: T, move_by: int) -> T:
    """Move a variable representing a ``place`` by some integer value.

    Parameters
    ----------
    s: one of (slice, int, np.array) or a dict whose values are of those types
        Variable to move by an offset.
    move_by: int
        Offset value.

    Returns
    -------
    s': one of (slice, int, np.array) or a dict whose values are of those types
        The returned type matches ``s``.
    """
    raise TypeError(f"Does not recognize type of first argument: {type(s)}")


@offset.register(slice)
def _offset_slice(s: slice, move_by: int) -> slice:
    return slice(move_by + (s.start or 0), move_by + s.stop)


@offset.register(int)
def _offset_int(s: int, move_by: int) -> int:
    return s + move_by


@offset.register(dict)
def _offset_dict(s: dict[Any, Place], move_by: int) -> dict[Any, Place]:
    return valmap(partial(offset, move_by=move_by), s)


@offset.register(np.ndarray)
def _offset_array(s: np.array, move_by: int) -> np.array:
    return s + move_by


def uniform_dataclass(dc, val):
    r"""Construct an instance of dataclass ``dc`` in which all field
    values are ``val``"""
    return dc(*(val for _ in dc.__dataclass_fields__))


class _DataclassMap(Protocol):
    __name__: str
    __doc__: str

    def __call__(self, *dc: T, **kwargs) -> T:
        pass


_FieldMap = _DataclassMap


def dataclass_map(dc: Type[T], f: _FieldMap) -> _DataclassMap:
    """
    If a function should work in the same manner on all fields of a dataclass,
    create that function and return it.

    Parameters
    ----------
    dc: dataclass
    f: Callable

    Returns
    -------
    A function which operates `f` on all fields of `dc` instances, returning
    a new `dc` instance.
    """

    def _dc_f(*instances, **kwargs):
        return dc(
            *(
                f(*(getattr(inst, fld) for inst in instances), **kwargs)
                for fld in dc.__dataclass_fields__
                )
            )

    _dc_f.__name__ = f.__name__
    _dc_f.__doc__ = f.__doc__

    return _dc_f
