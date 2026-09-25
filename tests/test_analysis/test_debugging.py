"""The debug tools accept the objects a failure hands you, and offer the solver's
scaled view of the residuals."""

import numpy as np
import pytest

from stream import Aggregator
from stream.analysis.debugging import debug_derivatives, debug_guess_variables
from stream.composition import Calculation_factory
from stream.scales import scale_vector


@pytest.fixture()
def small_agr():
    # 'pressure' carries a large default scale; 'T' a small one.
    # Raw: 2e3 Pa dwarfs 150 K. Scaled (typ: pressure 1e4, T 300): 0.2 < 0.5.
    A = Calculation_factory(
        calculate=lambda y: np.asarray([2.0e3, 150.0, 150.0]),
        mass_vector=[False] * 3,
        variables={"pressure": 0, "T": slice(1, 3)},
    )("core")
    return Aggregator.from_decoupled(A)


def _guess(agr):
    return agr.save(np.zeros(len(agr)))


def test_debug_derivatives_accepts_raw_vector(small_agr):
    from_state = debug_derivatives(small_agr, _guess(small_agr))
    from_vector = debug_derivatives(small_agr, np.zeros(len(small_agr)))
    assert from_vector.keys() == from_state.keys()
    np.testing.assert_array_equal(from_vector["core"]["T"], from_state["core"]["T"])


def test_debug_derivatives_default_output_unchanged(small_agr):
    dd = debug_derivatives(small_agr, _guess(small_agr))
    assert dd["core"]["pressure"] == pytest.approx(2.0e3)
    np.testing.assert_allclose(dd["core"]["T"], [150.0, 150.0])


def test_debug_derivatives_scaled_view_divides_by_typ(small_agr):
    typ = scale_vector(small_agr)
    dd = debug_derivatives(small_agr, _guess(small_agr), scales="default")
    raw = debug_derivatives(small_agr, _guess(small_agr))
    assert dd["core"]["pressure"] == pytest.approx(raw["core"]["pressure"] / typ[0])
    np.testing.assert_allclose(dd["core"]["T"], np.asarray(raw["core"]["T"]) / typ[1:3])


def test_debug_derivatives_scaled_view_fixes_unit_domination(small_agr):
    # Raw: the Pa-sized residual dwarfs the K-sized ones. Scaled: it must not.
    raw = debug_derivatives(small_agr, _guess(small_agr))
    scaled = debug_derivatives(small_agr, _guess(small_agr), scales="default")
    assert raw["core"]["pressure"] > np.max(raw["core"]["T"])
    assert scaled["core"]["pressure"] < np.max(scaled["core"]["T"])


def test_debug_derivatives_scales_dict_and_array(small_agr):
    by_dict = debug_derivatives(small_agr, _guess(small_agr), scales={"pressure": 1e3, "T": 50.0})
    assert by_dict["core"]["pressure"] == pytest.approx(2.0)
    np.testing.assert_allclose(by_dict["core"]["T"], [3.0, 3.0])
    by_array = debug_derivatives(small_agr, _guess(small_agr), scales=np.array([1e3, 50.0, 50.0]))
    assert by_array["core"]["pressure"] == pytest.approx(2.0)
    np.testing.assert_allclose(by_array["core"]["T"], [3.0, 3.0])


def test_debug_guess_variables_accepts_raw_vector(small_agr):
    out = debug_guess_variables(small_agr, np.zeros(len(small_agr)), variables={"T"})
    assert set(out["core"]) == {"T"}
