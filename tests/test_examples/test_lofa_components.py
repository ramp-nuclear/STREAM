import numpy as np
import pytest

from stream.examples.lofa.components import LowFlowTrip, horizontal_pressure
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water


def test_trip_latches_once_and_ramps_rods():
    trip = LowFlowTrip(threshold=200.0, delay=0.5, worth=0.065, duration=1.0)
    y = np.zeros(1)
    assert trip.event_margin(y, ref_mdot=250.0, t=1.0)[0] == pytest.approx(50.0)
    trip.change_state(y, ref_mdot=250.0, t=1.0)
    assert not trip.tripped
    trip.change_state(y, ref_mdot=199.0, t=12.0)
    assert trip.tripped and trip.t_trip == 12.0
    trip.change_state(y, ref_mdot=100.0, t=13.0)
    assert trip.t_trip == 12.0
    assert trip.rho(12.4) == 0.0
    assert trip.rho(13.0) == pytest.approx(-0.065 / 2, rel=1e-9)
    assert trip.rho(14.0) == pytest.approx(-0.065)
    assert trip.rho(100.0) == pytest.approx(-0.065)


def test_trip_margins_track_the_two_events():
    trip = LowFlowTrip(threshold=200.0, delay=0.5, worth=0.065, duration=1.0)
    y = np.zeros(1)
    assert trip.event_margin(y, ref_mdot=300.0, t=0.0).shape == (2,)
    assert trip.event_margin(y, ref_mdot=300.0, t=0.0)[1] > 0
    trip.change_state(y, ref_mdot=150.0, t=10.0)
    m = trip.event_margin(y, ref_mdot=150.0, t=10.2)
    assert m[0] > 0
    assert m[1] == pytest.approx(0.3)
    trip.change_state(y, ref_mdot=150.0, t=11.0)
    m = trip.event_margin(y, ref_mdot=150.0, t=11.0)
    assert m[1] > 0


def test_rod_start_margin_crosses_zero_until_consumed():
    trip = LowFlowTrip(threshold=200.0, delay=0.5, worth=0.065, duration=1.0)
    y = np.zeros(1)
    trip.change_state(y, ref_mdot=150.0, t=10.0)
    assert trip.event_margin(y, ref_mdot=150.0, t=10.2)[1] == pytest.approx(0.3)
    assert trip.event_margin(y, ref_mdot=150.0, t=10.6)[1] == pytest.approx(-0.1)
    trip.change_state(y, ref_mdot=150.0, t=10.6)
    assert trip.rods_moving
    assert trip.event_margin(y, ref_mdot=150.0, t=10.6)[1] == 1.0
    trip.change_state(y, ref_mdot=150.0, t=11.0)
    assert trip.rods_moving
    assert trip.event_margin(y, ref_mdot=150.0, t=11.0)[1] > 0
    trip.arm()
    assert not trip.rods_moving
    assert trip.event_margin(y, ref_mdot=300.0, t=11.0)[1] == 1.0


def test_trip_never_fires_above_threshold():
    trip = LowFlowTrip(threshold=100.0, delay=0.5, worth=0.065, duration=1.0)
    y = np.zeros(1)
    for t in (0.0, 10.0, 1000.0):
        trip.change_state(y, ref_mdot=150.0, t=t)
    assert not trip.tripped and trip.rho(1e6) == 0.0


def test_horizontal_pressure_has_no_gravity():
    pipe = EffectivePipe.rectangular(length=5.0, edge1=2.0, edge2=2.0, heated_edge=2.0)
    f = horizontal_pressure()
    dp = f(T=np.full(5, 40.0), Tw=np.full(5, 40.0), fluid=light_water, mdot=0.0, pipe=pipe, dz=np.full(5, 1.0))
    assert np.allclose(dp, 0.0)
