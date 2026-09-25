"""Tests for the ``locate`` attribution family on :class:`.Aggregator`:

- :class:`.Location` and the lazily-built row map (``sections`` x ``variables``);
- :meth:`.Aggregator.locate` — total on ``0..N-1`` (int/np.integer rows);
- :meth:`.Aggregator.locate_nonfinite` — first-non-finite locator, y before F;
- :meth:`.Aggregator.worst_residuals` — scaled ranking (avoids the unit-domination
  trap), with non-finite scaled entries ranked first;
- :meth:`.Aggregator.state_from` — the documented failure-object bridge.
"""

import numpy as np
import pytest
from networkx import MultiDiGraph

from stream.aggregator import Aggregator, Location, Solution
from stream.calculations.kirchhoff import Kirchhoff, KirchhoffWDerivatives
from stream.composition import Calculation_factory

# --- fixtures / small aggregators --------------------------------------------


def _identity(y):
    return np.asarray(y, dtype=float)


def _vec_scalar_nodes():
    """A vector variable ('T', a 3-cell slice) and a scalar ('p'), plus a second
    scalar-only node ('B') so section offsets are exercised."""
    A = Calculation_factory(_identity, [False] * 4, {"T": slice(0, 3), "p": 3})("A")
    B = Calculation_factory(_identity, [False], {"x": 0})("B")
    return A, B


def _two_var_node():
    """A single node whose two scalar variables carry pressure- and temperature-
    scale residuals, for the scaled-vs-raw ranking test."""
    return Calculation_factory(_identity, [False, False], {"pressure": 0, "T": 1})("dev")


def _kirchhoff_agr():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("core",))
    g.add_edge("B", "A", comps=("pump",))
    K = Kirchhoff(g, "core", reference_node=("A", 1.5e7))
    return Aggregator.from_decoupled(K), K


# --- Location exactness -------------------------------------------------------


def test_locate_exact_locations_vector_and_scalar():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)

    # The 3-cell vector variable 'T' spans rows 0..2 with 0-based cells.
    for cell in range(3):
        loc = agr.locate(cell)
        assert loc == Location(A, "A", "T", cell)
        assert loc.calculation is A  # node identity, not just name

    # The scalar 'p' at row 3 has cell None (int place).
    assert agr.locate(3) == Location(A, "A", "p", None)

    # Section offset: 'B' starts at row 4; its scalar 'x' is cell None.
    x = agr.locate(4)
    assert x == Location(B, "B", "x", None)
    assert x.calculation is B


def test_locate_accepts_numpy_integers():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)
    assert agr.locate(np.int64(3)) == agr.locate(3)
    assert agr.locate(np.int32(1)) == agr.locate(1)


def test_locate_out_of_range_raises_indexerror_naming_range():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)  # length 5
    with pytest.raises(IndexError, match="5"):
        agr.locate(5)
    with pytest.raises(IndexError):
        agr.locate(-1)


def test_locate_is_total_uncovered_rows_are_unmapped():
    """A section with a row no variable covers still locates: the gap row reports
    '<unmapped>' with its within-section offset, so locate never fails."""
    gap = Calculation_factory(_identity, [False] * 3, {"a": 0, "b": 1})("G")
    agr = Aggregator.from_decoupled(gap)
    loc = agr.locate(2)
    assert loc.calculation is gap
    assert loc.variable == "<unmapped>"
    assert loc.cell == 2


# --- Kirchhoff (per-edge string keys) ----------------------------------------


def test_locate_kirchhoff_every_row_maps_to_owning_node_with_string_keys():
    agr, K = _kirchhoff_agr()
    assert len(agr) == 3  # 2 edge mdots + 1 abs pressure

    for row in range(len(agr)):
        loc = agr.locate(row)
        assert loc.calculation is K  # the section-owning node
        assert loc.name == "Kirchhoff"
        assert isinstance(loc.variable, str)  # per-edge / p_abs keys are strings
        assert loc.variable != "<unmapped>"  # Kirchhoff covers all its rows

    # The per-edge variable strings come out exactly as Kirchhoff names them.
    assert agr.locate(0).variable == "(A -> B, 0)"
    assert agr.locate(1).variable == "(B -> A, 0)"
    assert agr.locate(2).variable == "(p_abs of core)"
    assert {agr.locate(r).variable for r in range(len(agr))} == set(K.variables)


def test_locate_kirchhoff_w_derivatives_covers_mdot2_rows_as_strings():
    g = MultiDiGraph()
    g.add_edge("A", "B", comps=("core",))
    g.add_edge("B", "A", comps=("pump",))
    KD = KirchhoffWDerivatives(g)
    agr = Aggregator.from_decoupled(KD)
    assert len(agr) == len(KD) == 4  # 2 mdot + 2 mdot2

    for row in range(len(agr)):
        loc = agr.locate(row)
        assert loc.calculation is KD
        assert isinstance(loc.variable, str)
        assert loc.variable != "<unmapped>"
    assert {agr.locate(r).variable for r in range(len(agr))} == set(KD.variables)


# --- Laziness -----------------------------------------------------------------


def test_row_map_is_lazy_absent_until_first_locate():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)
    assert not hasattr(agr, "_row_map")  # construction stays untouched
    agr.locate(0)
    assert hasattr(agr, "_row_map")
    assert isinstance(agr._row_map, list)
    assert len(agr._row_map) == len(agr)


def test_row_map_is_lazy_for_locate_nonfinite_and_worst_residuals():
    agr = Aggregator.from_decoupled(_two_var_node())
    assert not hasattr(agr, "_row_map")
    agr.locate_nonfinite(np.array([np.nan, 0.0]))
    assert hasattr(agr, "_row_map")

    agr2 = Aggregator.from_decoupled(_two_var_node())
    assert not hasattr(agr2, "_row_map")
    agr2.worst_residuals(np.array([1.0, 2.0]))
    assert hasattr(agr2, "_row_map")


# --- locate_nonfinite ---------------------------------------------------------


def test_locate_nonfinite_finds_planted_nan_in_y_and_inf_in_F():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)  # length 5

    y = np.zeros(5)
    y[1] = np.nan  # cell 1 of A's vector 'T'
    F = np.zeros(5)
    F[4] = np.inf  # B's scalar 'x'

    res = agr.locate_nonfinite(y, F)
    assert len(res) == 2

    (loc_y, val_y, src_y), (loc_F, val_F, src_F) = res
    assert loc_y == Location(A, "A", "T", 1)
    assert np.isnan(val_y) and src_y == "y"
    assert loc_F == Location(B, "B", "x", None)
    assert np.isinf(val_F) and src_F == "F"


def test_locate_nonfinite_orders_y_first_then_ascending_rows():
    A, B = _vec_scalar_nodes()
    agr = Aggregator.from_decoupled(A, B)

    y = np.zeros(5)
    y[3] = np.nan
    y[1] = np.nan
    F = np.zeros(5)
    F[0] = np.inf

    res = agr.locate_nonfinite(y, F)
    assert [r[2] for r in res] == ["y", "y", "F"]  # y entries first
    assert res[0][0] == agr.locate(1)  # ascending within y
    assert res[1][0] == agr.locate(3)
    assert res[2][0] == agr.locate(0)


def test_locate_nonfinite_without_F_reports_only_y():
    agr = Aggregator.from_decoupled(_two_var_node())
    res = agr.locate_nonfinite(np.array([np.nan, 5.0]))
    assert len(res) == 1
    assert res[0][2] == "y"
    assert res[0][0].variable == "pressure"


# --- worst_residuals (scaled ranking) ---------------------------------


def test_worst_residuals_scaled_ranking_flips_the_raw_worst():
    """With F = [1000 (pressure, Pa), 5 (T, K)] the raw-magnitude worst is pressure,
    but under the solver's scaling temperature is the least-converged.
    worst_residuals must rank by |F/typ|, so temperature wins."""
    agr = Aggregator.from_decoupled(_two_var_node())
    y = np.array([1000.0, 5.0])  # F(y) == y for this node

    # Raw (unit) ranking: pressure dominates. Array-of-ones scales == unscaled.
    raw = agr.worst_residuals(y, scales=np.ones(2))
    assert raw[0][0].variable == "pressure"

    # Scaled ranking (dict): pressure/1e5 = 0.01 << T/1.0 = 5 -> temperature wins.
    scaled = agr.worst_residuals(y, scales={"pressure": 1e5, "T": 1.0}, n=2)
    assert [loc.variable for loc, _ in scaled] == ["T", "pressure"]
    assert scaled[0][1] == pytest.approx(5.0)  # signed scaled residual
    assert scaled[1][1] == pytest.approx(0.01)


def test_worst_residuals_nonfinite_scaled_entry_ranks_first():
    agr = Aggregator.from_decoupled(_two_var_node())
    # pressure residual is a huge finite 1e6; T residual is nan -> ranks first.
    scaled = agr.worst_residuals(np.array([1e6, np.nan]), scales={"pressure": 1.0, "T": 1.0}, n=2)
    assert scaled[0][0].variable == "T"
    assert not np.isfinite(scaled[0][1])
    assert scaled[1][0].variable == "pressure"


def test_worst_residuals_respects_n_and_default_scales():
    agr = Aggregator.from_decoupled(_two_var_node())
    y = np.array([1000.0, 5.0])

    top1 = agr.worst_residuals(y, scales={"pressure": 1e5, "T": 1.0}, n=1)
    assert len(top1) == 1
    assert top1[0][0].variable == "T"

    # scales=None uses DEFAULT_SCALES; both rows are returned when n exceeds N.
    default = agr.worst_residuals(y, n=5)
    assert len(default) == 2
    assert all(isinstance(loc, Location) for loc, _ in default)


# --- state_from ---------------------------------------------


def _scalar_agr():
    """A scalar-only aggregator so State/StateTimeseries equality is unambiguous."""
    A = Calculation_factory(_identity, [False], {"a": 0})("A")
    B = Calculation_factory(_identity, [False], {"b": 0})("B")
    return Aggregator.from_decoupled(A, B)


def test_state_from_vector_roundtrips_to_a_state():
    agr = _scalar_agr()
    vec = np.array([3.0, 7.0])
    assert agr.state_from(vec) == agr.save(vec)


def test_state_from_t_y_pair_roundtrips_to_a_timeseries():
    agr = _scalar_agr()
    t = np.array([0.0, 1.0])
    y2d = np.array([[3.0, 7.0], [4.0, 8.0]])

    ts = agr.state_from((t, y2d))
    expected = agr.save(Solution(t, y2d))
    assert set(ts) == set(expected)
    for k in ts:
        assert ts[k] == expected[k]


def test_state_from_solution_roundtrips_to_a_timeseries():
    agr = _scalar_agr()
    sol = Solution(np.array([0.0, 1.0]), np.array([[3.0, 7.0], [4.0, 8.0]]))
    assert agr.state_from(sol) == agr.save(sol)


def test_state_from_bad_input_raises_typeerror_naming_the_forms():
    agr = _scalar_agr()
    with pytest.raises(TypeError) as exc:
        agr.state_from({"not": "a state"})
    msg = str(exc.value)
    assert "1-D" in msg or "vector" in msg
    assert "Solution" in msg
    assert "t, y2d" in msg or "pair" in msg


def test_state_from_two_d_array_is_rejected():
    """A bare 2-D array is not one of the three accepted forms (a trajectory must
    come as the (t, y2d) pair so its times are known)."""
    agr = _scalar_agr()
    with pytest.raises(TypeError):
        agr.state_from(np.zeros((2, 2)))
