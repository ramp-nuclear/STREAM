import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, lists

from stream.aggregator import Aggregator
from stream.calculations import PointKinetics
from stream.calculations.point_kinetics import (
    OneWayToSCRAM,
    PointKineticsWInput,
    ReactivityController,
    temperature_reactivity,
)
from stream.composition import Calculation_factory
from stream.utilities import identity, just

from .conftest import are_close, medium_floats, pos_floats

U235_lambdak = np.array([55.72, 22.72, 6.22, 2.3, 0.618, 0.23])
mock_calc = Calculation_factory(just(1.0), [False], {})(name="mock")


def mock_point_kinetics():
    return PointKinetics(
        generation_time=1,
        delayed_neutron_fractions=np.array([0.25]),
        delayed_groups_decay_rates=np.array([2]),
        temp_worth={mock_calc: np.array([10])},
        ref_temp={mock_calc: 0},
    )


@pytest.mark.implementation
def test_pkc():
    mock_pk = mock_point_kinetics()
    mock_pk.controls.input_reactivity = just(1.0)
    mock_pk.calculate([0, 0], source=1, T={mock_calc: 0}, t=0.0)
    assert np.allclose(mock_pk._A, ((1 - 0.25, 2), (0.25, -2)))
    assert np.isclose(mock_pk._s[0], 1)
    assert np.isclose(mock_pk.reactivity({mock_calc: np.array([2])}, 15), -2 * 10 + 15)
    assert mock_pk.indices("ck") == slice(1, 2)


@pytest.mark.slow
@settings(deadline=None)
@given(
    nums := floats(allow_infinity=False, allow_nan=False, max_value=1e7, min_value=1e-1),
    lists(elements=nums, min_size=6, max_size=6),
)
def test_precursor_death(p0, ck):
    """
    Having only precursors in a critical system with beta=0 should yield an
    exponentially dependent power (like capacitor charging)
    """
    lambdak = U235_lambdak
    time = np.linspace(0, 8.0, 100)
    pkm = dict(
        generation_time=1,
        delayed_groups_decay_rates=lambdak,
        delayed_neutron_fractions=np.zeros(len(ck)),
    )
    pk = PointKinetics(**pkm)
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T=0, t=0)})

    calculation = agr.solve(y0=np.array([p0] + ck), time=time)
    analytical = p0 - np.array(ck) @ np.expm1(-np.outer(lambdak, time))
    are_close(calculation[:, 0], analytical, rtol=1.0e-3, atol=1.0e-6)


@given(medium_floats, medium_floats, medium_floats, medium_floats)
def test_pk_save_follows_known_pattern_for_mock(p, ck, inp, T):
    mock_pk = mock_point_kinetics()
    mock_pk.controls.input_reactivity = just(inp)
    save = mock_pk.save([p, ck], T={mock_calc: T}, t=0)
    r = inp - mock_pk.temp_worth[mock_calc] * T
    known = dict(
        power=p,
        ck=[ck],
        reactivity=r,
        dPdt=np.dot(mock_pk.lambdak, [ck]) + (r - mock_pk.dollar) * p / mock_pk.Lambda,
    )
    for key, value in known.items():
        are_close(save[key], value, rtol=1e-5, atol=1e-8)


def test_feedbackless_point_kinetics_calculate_without_T():
    """Without temperature feedback (default temp_worth and ref_temp) no temperatures
    are routed to PointKinetics, and calculate() works without T."""
    pk = PointKinetics(
        generation_time=1e-4,
        delayed_neutron_fractions=np.array([0.0065]),
        delayed_groups_decay_rates=np.array([0.08]),
    )
    out = pk.calculate(np.array([1.0, 1.0]), t=0.0)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_pk_load(p, ck):
    mock_pk = mock_point_kinetics()
    load = mock_pk.load(dict(power=p, ck=[ck]))
    assert np.allclose(load, [p, ck])


@pytest.mark.parametrize(
    ("w", "result"),
    [({1: np.ones(5), 2: np.ones(5)}, 0), ({1: np.ones(5), 2: np.zeros(5)}, -5)],
)
def test_reactivity_for_linear_temperature_in_relation_to_reference(w, result):
    T = {1: np.arange(5), 2: np.ones(5)}
    T0 = {1: np.ones(5), 2: np.arange(5)}
    # noinspection PyTypeChecker
    assert np.isclose(temperature_reactivity(T, T0, w), result)


def test_pk_with_decay():
    lambdak = U235_lambdak
    time = np.linspace(0, 8.0, 100)
    p0 = 10
    ck = [1, 2, 3, 4, 5, 6]
    pkm = dict(
        generation_time=1,
        delayed_groups_decay_rates=lambdak,
        delayed_neutron_fractions=np.zeros(6),
        temp_worth={},
        ref_temp={},
    )
    pk = PointKineticsWInput(**pkm)

    def power_input(t):
        return t

    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T=0, power_input=power_input, t=identity)})

    calculation = agr.solve(y0=np.array([p0] + ck + [p0]), time=time)
    analytical = p0 + np.array(ck) @ (-np.expm1(-np.outer(lambdak, time)))

    pk_power = calculation[:, 0]
    total_power = calculation[:, -1]

    assert np.allclose(pk_power, analytical, rtol=1.0e-3, atol=1.0e-6)
    assert np.allclose(total_power, analytical + power_input(time), rtol=1.0e-3, atol=1.0e-6)


@given(pos_floats)
def test_pk_change_state_sets_SCRAM_time(t):
    mock_pk = mock_point_kinetics()
    assert mock_pk.controls.state == OneWayToSCRAM.NORMAL
    mock_pk.controls.state_machine = just(OneWayToSCRAM.SCRAM)
    mock_pk.change_state([0, 0], T=mock_pk.T0, t=t)
    assert mock_pk.controls.state == OneWayToSCRAM.SCRAM
    assert mock_pk.controls.t_state == t


@given(pos_floats)
def test_pk_should_continue_stops_at_SCRAM_time(t):
    mock_pk = mock_point_kinetics()
    mock_pk.controls.state = OneWayToSCRAM.SCRAM
    mock_pk.controls.t_state = t
    mock_pk.controls.abort_states = {OneWayToSCRAM.SCRAM}
    assert not mock_pk.should_continue([0, 0], T=mock_pk.T0, t=t)


# Keepin six-group U-235 thermal data as tabulated in Lamarsh, Introduction to Nuclear Reactor Theory, Table 7-1.
_lambdak = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
_betak = np.array([0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027])
_Lam = 2e-5


def _scram_pk(P0=1e6, limit_factor=1.2):
    """A PointKinetics with a +20 pcm ramp that drives power up to a SCRAM trip
    and a strong rod-insertion ramp afterwards. Abort on SCRAM."""
    limit = limit_factor * P0

    def machine(state, t, power, dPdt, **kw):
        return OneWayToSCRAM.SCRAM if state == OneWayToSCRAM.NORMAL and power > limit else state

    def rho_in(state, t_state, t, **_):
        if state == OneWayToSCRAM.SCRAM:
            return -0.05 * (t - t_state)
        return 20e-5 if t > 1.0 else 0.0

    ctrl = ReactivityController(input_reactivity=rho_in, state_machine=machine, abort_states={OneWayToSCRAM.SCRAM})
    pk = PointKinetics(
        generation_time=_Lam, delayed_neutron_fractions=_betak, delayed_groups_decay_rates=_lambdak, controls=ctrl
    )
    ck0 = _betak * P0 / (_lambdak * _Lam)
    return pk, ctrl, np.concatenate([[P0], ck0])


def test_scram_abort_stops_dae_solve_at_trip_time():
    """The abort predicate in should_continue must see the plain time, so a SCRAM
    transition stops the DAE solve at the trip time instead of running to t_end."""
    pk, ctrl, y0 = _scram_pk()
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T={}, t=identity)})
    time = np.linspace(0, 60, 601)
    sol = agr.solve(y0=y0.copy(), time=time, eq_type="DAE")

    assert ctrl.state == OneWayToSCRAM.SCRAM
    assert sol.time[-1] < time[-1]
    assert sol.time[-1] == pytest.approx(ctrl.t_state)


