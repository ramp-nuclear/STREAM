"""The post-run bulk-saturation checker in stream.analysis.thresholds.

The transient guard (``stop_at_saturation``) stops a run at the crossing. This
post-mortem checker covers the steady case and guard-off / already-computed
results: it finds cells whose bulk coolant is at/past saturation in a saved State,
StateTimeseries, or raw solve vector/trajectory (e.g. a caught error's ``e.y``),
and can raise the same :class:`SaturationReachedError`.
"""

import numpy as np
import pytest

from stream.aggregator import Aggregator
from stream.analysis.thresholds import (
    channel_saturation_crossings,
    first_saturation_crossing,
    raise_on_saturation,
)
from stream.calculations import ChannelAndContacts, SaturationReachedError
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water

pipe = EffectivePipe.rectangular(0.6, 0.06, 0.003, 0.003)
Z = np.linspace(0.0, 0.6, 5)  # 4 cells; Tsat(~1 bar) ≈ 100 C


def _agr_vec(T_cool):
    C = ChannelAndContacts(Z, light_water, pipe)
    agr = Aggregator.from_decoupled(C, funcs={C: dict(mdot=0.05, T_left=80.0, T_right=80.0, Tin=80.0, p_abs=1e5)})
    vec = agr.load(
        {C.name: dict(T_cool=np.full(4, T_cool), pressure=0.0, h_left=np.full(4, 1e4), h_right=np.full(4, 1e4))}
    )
    return C, agr, vec


def test_no_crossing_when_subcooled():
    C, agr, vec = _agr_vec(80.0)
    state = agr.save(vec)
    assert channel_saturation_crossings(state, agr) == []
    raise_on_saturation(state, agr)  # must not raise


def test_flags_and_raises_on_a_past_saturation_state():
    C, agr, vec = _agr_vec(130.0)
    state = agr.save(vec)
    crossings = channel_saturation_crossings(state, agr)
    assert len(crossings) == 1 and crossings[0].channel == C.name
    assert crossings[0].cells  # some cells past Tsat
    with pytest.raises(SaturationReachedError) as exc:
        raise_on_saturation(state, agr)
    assert C.name in str(exc.value) and "saturation" in str(exc.value).lower()


def test_accepts_a_raw_vector_last_iterate_with_a_note():
    """A failed steady's last iterate (raw vector, e.g. AlgRuntimeError.y) is
    attributed to saturation, with the caller's note preserved on the message."""
    C, agr, vec = _agr_vec(130.0)
    with pytest.raises(SaturationReachedError) as exc:
        raise_on_saturation(vec, agr, note="no converged steady solution — ")
    assert str(exc.value).startswith("no converged steady solution — ")


def test_finds_earliest_crossing_in_a_trajectory():
    C, agr, sub = _agr_vec(80.0)
    _, _, hot = _agr_vec(130.0)
    traj = np.vstack([sub, sub, hot])
    times = [0.0, 1.0, 2.0]
    found = first_saturation_crossing(traj, agr, times=times)
    assert found is not None
    t, crossings = found
    assert t == 2.0 and crossings
    with pytest.raises(SaturationReachedError):
        raise_on_saturation(traj, agr, times=times)
