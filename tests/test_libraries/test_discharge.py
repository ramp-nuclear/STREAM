import numpy as np
import pytest
from hypothesis import given, strategies as st

from stream.physical_models.pressure_drop.discharge import (
    DISCHARGE_CD, discharge_cd, drain_level, drain_time, lichtarowicz_cd, stub_discharge_mdot,
)
from stream.units import g


def test_cd_table_matches_handbook_values():
    assert DISCHARGE_CD == {"sharp": 0.61, "rounded": 0.98, "short_tube": 0.81, "borda": 0.51, "pipe_stub": 0.6}
    assert discharge_cd("sharp") == 0.61
    assert discharge_cd("rounded") == 0.98
    with pytest.raises(ValueError, match="sharp"):
        discharge_cd("no_such_geometry")


def test_lichtarowicz_approaches_ultimate_cd_at_high_re():
    cdu = 0.827 - 0.0085 * 2.0
    assert lichtarowicz_cd(2e4, 2.0) == pytest.approx(cdu, rel=0.02)


@given(re=st.floats(50, 2e4), lod=st.floats(0.5, 10.0))
def test_lichtarowicz_bounded_and_monotone_in_re(re, lod):
    cd = lichtarowicz_cd(re, lod)
    assert 0.0 < cd < 1.0
    assert lichtarowicz_cd(re * 1.5, lod) >= cd - 1e-9


def test_stub_discharge_reduces_to_bernoulli_at_unit_k():
    mdot = stub_discharge_mdot(1e5, 1000.0, 1e-4, 1.0)
    assert mdot == pytest.approx(1e-4 * np.sqrt(2 * 1000.0 * 1e5))


def test_drain_time_closed_form_round_trips_with_level():
    t = drain_time(4.0, 1.0, 2.0, 5e-4, 0.61)
    assert t == pytest.approx((2.0 / (0.61 * 5e-4)) * np.sqrt(2 / g) * (np.sqrt(4.0) - np.sqrt(1.0)))
    assert drain_level(t, 4.0, 2.0, 5e-4, 0.61) == pytest.approx(1.0, abs=1e-9)
