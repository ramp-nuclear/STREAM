import numpy as np
import pandas as pd
import pytest

from stream.viz.envelope import at, mesh_shape, reduce_array, reduce_rows, reducer_name, to_array, with_envelopes
from stream.viz.sweep import Coords

Z = Coords(("z",), (np.array([0.05, 0.15, 0.25, 0.35]),), (np.linspace(0, 0.4, 5),), (np.full(4, 0.1),))
ZX = Coords(
    ("z", "x"),
    (np.array([0.25, 0.75]), np.array([0.5, 1.5, 2.5])),
    (np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 2.0, 3.0])),
    (np.array([0.5, 0.5]), np.array([1.0, 1.0, 1.0])),
)


def test_envelopes_from_bands():
    t = pd.DataFrame(dict(value=[1.0, 2.0], sys=[0.1, 0.0], stat=[0.2, 0.0]))
    e = with_envelopes(t, banded=True)
    np.testing.assert_allclose(e.lower, [0.7, 2.0])
    np.testing.assert_allclose(e.upper, [1.3, 2.0])
    np.testing.assert_allclose(e.inner_lower, [0.9, 2.0])
    np.testing.assert_allclose(e.inner_upper, [1.1, 2.0])
    e2 = with_envelopes(t, banded=True, sigma=2.0)
    np.testing.assert_allclose(e2.lower, [0.5, 2.0])
    e0 = with_envelopes(t, banded=True, uncertainty=False)
    np.testing.assert_allclose(e0.lower, t.value)
    np.testing.assert_allclose(e0.upper, t.value)
    np.testing.assert_allclose(e0.inner_upper, t.value)


def test_envelopes_without_bands_equal_value():
    t = pd.DataFrame(dict(value=[1.0, 2.0]))
    e = with_envelopes(t, banded=False)
    assert (e.lower == e.value).all() and (e.upper == e.value).all()


def test_to_array_shapes():
    r1 = pd.DataFrame(dict(i=[0, 0, 0], j=[2, 0, 1], value=[3.0, 1.0, 2.0]))
    np.testing.assert_array_equal(to_array(r1, 1), [1.0, 2.0, 3.0])
    r2 = pd.DataFrame(dict(i=[1, 0, 0, 1], j=[0, 1, 0, 1], value=[3.0, 2.0, 1.0, 4.0]))
    np.testing.assert_array_equal(to_array(r2, 2), [[1.0, 2.0], [3.0, 4.0]])
    assert to_array(pd.DataFrame(dict(i=[0], j=[0], value=[7.0])), 0) == 7.0


def test_to_array_scatters_rows_to_their_own_cell():
    gapped = pd.DataFrame(dict(i=[0, 0, 0], j=[0, 1, 3], value=[40.0, 45.0, 55.0]))
    np.testing.assert_array_equal(to_array(gapped, 1, "value", Z), [40.0, 45.0, np.nan, 55.0])
    np.testing.assert_array_equal(to_array(gapped, 1), [40.0, 45.0, np.nan, 55.0])
    plate = pd.DataFrame(dict(i=[0, 0, 1, 1, 1], j=[0, 2, 0, 1, 2], value=[1.0, 3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(to_array(plate, 2, "value", ZX), [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])


def test_to_array_of_a_quantity_with_no_rows():
    empty = pd.DataFrame(dict(i=[], j=[], value=[]))
    assert np.isnan(to_array(empty, 1, "value", Z)).all()
    assert to_array(empty, 1, "value", Z).shape == (4,)
    assert to_array(empty, 2, "value", ZX).shape == (2, 3)
    assert np.isnan(to_array(empty, 0))
    with pytest.raises(ValueError, match="rank-1.*coords"):
        to_array(empty, 1)


def test_to_array_past_the_mesh_names_the_cell():
    beyond = pd.DataFrame(dict(i=[0, 0], j=[0, 4], value=[1.0, 2.0]))
    with pytest.raises(ValueError, match="'value'.*cell 4.*4"):
        to_array(beyond, 1, "value", Z)


def test_mesh_shape_from_coords_or_from_the_rows():
    rows = pd.DataFrame(dict(i=[0, 1], j=[0, 2], value=[1.0, 2.0]))
    assert mesh_shape(rows, 1, Z) == (4,)
    assert mesh_shape(rows, 2, ZX) == (2, 3)
    assert mesh_shape(rows, 2) == (2, 3)
    assert mesh_shape(rows, 1) == (3,)


def test_reducers_step_over_a_missing_cell():
    v = np.array([np.nan, 45.0, 50.0, 55.0])
    assert reduce_array(v, Z, "min") == 45.0
    assert reduce_array(v, Z, "max") == 55.0
    assert reduce_array(v, Z, "mean") == pytest.approx(50.0)
    assert reduce_array(v, Z, "where_min") == 0.15
    assert reduce_array(v, Z, "where_max") == 0.35
    assert reduce_array(v, Z, at(z=0.25)) == 50.0
    assert np.isnan(reduce_array(v, Z, at(cell=0)))


def test_reducers_of_an_entirely_missing_quantity_are_missing():
    v = np.full(4, np.nan)
    assert np.isnan(reduce_array(v, Z, "min"))
    assert np.isnan(reduce_array(v, Z, "max"))
    assert np.isnan(reduce_array(v, Z, "mean"))
    assert np.isnan(reduce_array(v, Z, "where_min"))
    m = np.array([[np.nan, np.nan, np.nan], [4.0, 3.0, 6.0]])
    np.testing.assert_array_equal(reduce_array(m, ZX, "max", axis="x"), [np.nan, 6.0])
    np.testing.assert_array_equal(reduce_array(m, ZX, "where_max", axis="x"), [np.nan, 2.5])
    np.testing.assert_allclose(reduce_array(m, ZX, "mean", axis="x"), [np.nan, 13.0 / 3.0])


def test_named_reducers_on_a_profile():
    v = np.array([3.0, 1.0, 2.0, 5.0])
    assert reduce_array(v, Z, "min") == 1.0
    assert reduce_array(v, Z, "max") == 5.0
    assert reduce_array(v, Z, "mean") == pytest.approx(2.75)
    assert reduce_array(v, Z, "where_min") == 0.15
    assert reduce_array(v, Z, "where_max") == 0.35


def test_mean_is_length_weighted():
    coords = Coords(("z",), (np.array([0.1, 0.4]),), (np.array([0.0, 0.2, 0.6]),), (np.array([0.2, 0.4]),))
    assert reduce_array(np.array([1.0, 4.0]), coords, "mean") == pytest.approx(3.0)
    assert reduce_array(np.array([1.0, 4.0]), None, "mean") == pytest.approx(2.5)


def test_at_reducer():
    v = np.array([3.0, 1.0, 2.0, 5.0])
    assert reduce_array(v, Z, at(z=0.26)) == 2.0
    assert reduce_array(v, Z, at(cell=3)) == 5.0
    with pytest.raises(ValueError, match="cell 9.*4"):
        reduce_array(v, Z, at(cell=9))
    with pytest.raises(ValueError, match="z"):
        reduce_array(v, None, at(z=0.1))


def test_callable_reducer():
    v = np.array([3.0, 1.0, 2.0, 5.0])
    assert reduce_array(v, Z, lambda values, coords: float(values[-1] - values[0])) == 2.0


def test_rank2_axis_semantics():
    m = np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])
    np.testing.assert_array_equal(reduce_array(m, ZX, "max", axis="x"), [5.0, 6.0])
    np.testing.assert_array_equal(reduce_array(m, ZX, "min", axis="z"), [1.0, 3.0, 2.0])
    assert reduce_array(m, ZX, "max") == 6.0
    np.testing.assert_array_equal(reduce_array(m, ZX, "where_max", axis="x"), [1.5, 2.5])
    with pytest.raises(ValueError, match="where_max.*axis"):
        reduce_array(m, ZX, "where_max")
    with pytest.raises(ValueError, match="'y'"):
        reduce_array(m, ZX, "max", axis="y")


def test_unknown_reducer_raises():
    with pytest.raises(ValueError, match="'median'.*min, max, mean"):
        reduce_array(np.array([1.0]), Z, "median")


def test_reduce_rows_applies_to_three_envelopes():
    rows = pd.DataFrame(
        dict(
            i=0, j=[0, 1, 2, 3], value=[3.0, 1.0, 2.0, 5.0], lower=[2.0, 0.5, 0.0, 4.0], upper=[4.0, 1.5, 3.0, 6.0],
            inner_lower=[2.5, 0.8, 1.5, 4.5], inner_upper=[3.5, 1.2, 2.5, 5.5],
        )
    )
    out = reduce_rows(rows, 1, Z, "min")
    assert out == dict(lower=0.0, value=1.0, upper=1.5, inner_lower=0.8, inner_upper=1.2)
    where = reduce_rows(rows, 1, Z, "where_min")
    assert where == dict(lower=0.25, value=0.15, upper=0.15, inner_lower=0.15, inner_upper=0.15)


def test_reducer_name():
    assert reducer_name("min") == "min"
    assert reducer_name(at(z=0.35)) == "at z=0.35"
    assert reducer_name(at(cell=2)) == "at cell 2"

    def span(values, coords):
        return 0.0

    assert reducer_name(span) == "span"
