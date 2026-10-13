import numpy as np
import pytest

from stream.examples.lofa import scenarios
from stream.examples.lofa.params import HEATED, NC_SIGNS, Params


def test_record_empty_marks_times_unknown():
    rec = scenarios.Record.empty("failed", error="x")
    assert rec.status == "failed" and rec.error == "x"
    assert rec.t_trip is None and rec.t_open is None and rec.t_rev_hot is None
    assert np.isnan(rec.peak_max)


def test_stuck_flapper_forces_the_stuck_parameter():
    assert scenarios._effective(Params(), "stuck").flapper_stuck is True
    assert scenarios._effective(Params(), "both").flapper_stuck is False


@pytest.mark.slow
def test_scenario_functions_override_the_scenario_field():
    _, rec = scenarios.one_pump(Params(scenario="both"), t_end=120.0, n=121)
    assert rec.status == "completed", rec.error
    assert rec.t_open is None
    assert rec.t_valve_B is not None and rec.t_valve_A is None


@pytest.mark.slow
def test_short_both_pumps_transient_trips_and_opens():
    sol, rec = scenarios.both_pumps(Params(), t_end=120.0, n=121)
    assert rec.status == "completed", rec.error
    assert rec.t_trip is not None and 10.0 < rec.t_trip < 20.0
    assert rec.t_rods == pytest.approx(rec.t_trip + 0.5)
    assert rec.t_open is not None and rec.t_trip < rec.t_open < 120.0
    assert 0 < rec.final_leg_A < 0.2 * rec.mdot_total / 2
    assert 0 < rec.final_leg_B < 0.2 * rec.mdot_total / 2
    assert sol.time[-1] == pytest.approx(120.0)


@pytest.mark.slow
def test_both_pumps_hour_holds_every_condition():
    _, rec = scenarios.both_pumps(Params())
    assert rec.status == "completed", rec.error
    assert rec.t_last >= 3600.0 * (1 - 1e-9)
    assert rec.t_trip is not None and rec.t_open is not None
    assert rec.t_valve_A is not None and rec.t_valve_B is not None
    assert all(getattr(rec, f"t_rev_{k}") is not None for k in HEATED)
    assert rec.reversal_in_buoyancy_order
    assert rec.final_pool * NC_SIGNS["pool"] > 0 and rec.final_flapper * NC_SIGNS["flapper"] > 0
    assert rec.margin > 0
    assert abs(rec.final_leg_A) < 0.01 * rec.mdot_total / 2 and abs(rec.final_leg_B) < 0.01 * rec.mdot_total / 2
    assert 11.0 < rec.t_trip < 14.0 and 35.0 < rec.t_open < 45.0


@pytest.mark.slow
def test_one_pump_half_hour_keeps_the_flapper_shut():
    _, rec = scenarios.one_pump(Params())
    assert rec.status == "completed", rec.error
    assert rec.t_last >= 1800.0 * (1 - 1e-9)
    assert rec.t_valve_B is not None and rec.t_valve_A is None
    assert rec.t_open is None
    assert abs(rec.final_leg_B) < 0.01 * rec.mdot_total / 2
    assert abs(rec.final_leg_A) > 0.5 * rec.mdot_total


@pytest.mark.slow
def test_stuck_flapper_stops_at_saturation_with_the_named_error():
    sol, rec = scenarios.stuck_flapper(Params())
    assert rec.status == "saturation"
    assert "SaturationReachedError" in rec.error
    assert rec.t_last < 3600.0
    assert sol.time[-1] == pytest.approx(rec.t_last)
