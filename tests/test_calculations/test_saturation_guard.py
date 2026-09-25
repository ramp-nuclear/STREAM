"""The opt-in bulk-saturation guard on ChannelAndContacts.

STREAM's channel model (single-phase forced convection + subcooled boiling) is
valid up to *bulk* saturation; past it the flow is two-phase, which STREAM does
not model, and the h = q/ΔT term develops a finite-time pole. With
``stop_at_saturation=True`` the channel exposes a per-cell ``Tsat(p) − T_cool``
event margin so a transient stops cleanly at the crossing and raises
:class:`SaturationReachedError` in domain terms — per cell, no wall side (bulk is
one value per cell). Default OFF, so existing systems are untouched.
"""

import numpy as np
import pytest

from stream.aggregator import Aggregator
from stream.calculations import ChannelAndContacts, SaturationReachedError
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water

pipe = EffectivePipe.rectangular(0.6, 0.06, 0.003, 0.003)  # length, edge1, edge2, heated_edge
Z = np.linspace(0.0, 0.6, 5)  # 4 cells
# Tsat(light water, ~1 bar) ≈ 100 C, so T_cool=80 is subcooled and 110 is past it.


def _agr(stop_at_saturation, T_cool, **func_overrides):
    C = ChannelAndContacts(Z, light_water, pipe, stop_at_saturation=stop_at_saturation)
    funcs = dict(mdot=0.05, T_left=80.0, T_right=80.0, Tin=80.0, p_abs=1e5)
    funcs.update(func_overrides)
    agr = Aggregator.from_decoupled(C, funcs={C: funcs})
    y = agr.load(
        {C.name: dict(T_cool=np.full(C.n, T_cool), pressure=0.0, h_left=np.full(C.n, 1e4), h_right=np.full(C.n, 1e4))}
    )
    return C, agr, y


def test_guard_is_off_by_default_and_exposes_no_margin_even_past_saturation():
    """Default OFF: no event margin, and _handle_event is a no-op even when the
    bulk is already past Tsat — existing systems keep their exact solve path."""
    C, agr, y = _agr(False, T_cool=110.0)
    assert C.stop_at_saturation is False
    assert agr._event_margins(y, 0.0).size == 0
    assert agr._handle_event(y, 0.0) is True


def test_guard_margin_is_per_cell_and_positive_while_subcooled():
    C, agr, y = _agr(True, T_cool=80.0)
    margins = agr._event_margins(y, 0.0)
    assert margins.size == C.n  # one per cell (bulk-based), not per wall
    assert np.all(margins > 0.0)
    assert agr._handle_event(y, 0.0) is True


def test_guard_raises_at_bulk_saturation_with_domain_context():
    C, agr, y = _agr(True, T_cool=110.0)  # past Tsat
    assert np.any(agr._event_margins(y, 0.0) <= 0.0)
    with pytest.raises(SaturationReachedError) as exc:
        agr._handle_event(y, 3.5)
    e = exc.value
    assert e.channel == C.name
    assert e.cells and all(0 <= c < C.n for c in e.cells)
    assert min(e.T_bulk) >= min(e.Tsat) - 1e-6  # bulk reported at/above Tsat
    assert "saturation" in str(e).lower()
    assert C.name in str(e)


def test_guarded_transient_stops_at_crossing_with_valid_trajectory():
    """End-to-end: walls ramp up, the bulk heats past Tsat, and the DAE solve stops
    at the crossing with SaturationReachedError carrying the pre-crossing trajectory
    (via the generic on-event trajectory attachment)."""
    ramp = lambda t: np.full(4, 80.0 + 20.0 * t)  # noqa: E731
    C = ChannelAndContacts(Z, light_water, pipe, stop_at_saturation=True)
    agr = Aggregator.from_decoupled(
        C, funcs={C: dict(mdot=0.02, T_left=ramp, T_right=ramp, Tin=80.0, p_abs=1e5)}
    )
    steady = agr.solve_steady(
        {C.name: dict(T_cool=np.full(4, 80.0), pressure=1e5, h_left=np.full(4, 3e4), h_right=np.full(4, 3e4))}
    )
    with pytest.raises(SaturationReachedError) as exc:
        agr.solve(steady, np.linspace(0.0, 10.0, 41), eq_type="DAE")
    e = exc.value
    assert hasattr(e, "t") and hasattr(e, "y")  # trajectory attached by the solver
    assert e.y.shape[0] == e.t.shape[0] >= 1
    # every returned bulk temperature is at or below saturation (valid regime only)
    tcool = e.y[:, agr.sections[C].start : agr.sections[C].start + C.n]
    Tsat = float(np.atleast_1d(light_water.sat_temperature(1e5))[0])
    assert tcool.max() <= Tsat + 1.0  # crossing point may just touch Tsat
