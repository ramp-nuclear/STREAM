import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, lists

from stream.aggregator import Aggregator
from stream.calculations import PointKinetics
from stream.calculations.point_kinetics import (
    PointKineticsWInput, temperature_reactivity)
from stream.composition import Calculation_factory
from stream.utilities import just
from .conftest import are_close, medium_floats, pos_floats


U235_lambdak = np.array([55.72, 22.72, 6.22, 2.3, 0.618, 0.23])
mock_calc = Calculation_factory(just(1.0), [False], {})(name='mock')


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
    mock_pk.input_reactivity = just(1.0)
    mock_pk.calculate([0, 0], source={mock_pk: 1}, T={mock_calc: 0})
    assert np.allclose(mock_pk._A, ((1 - 0.25, 2), (0.25, -2)))
    assert np.isclose(mock_pk._s[0], 1)
    assert np.isclose(mock_pk.reactivity({mock_calc: np.array([2])}, 15), -2 * 10 + 15)
    assert mock_pk.indices("ck") == slice(1, 2)


@pytest.mark.slow
@settings(deadline=None)
@given(nums := floats(allow_infinity=False, allow_nan=False,
                      max_value=1e7, min_value=1e-1),
       lists(elements=nums, min_size=6, max_size=6))
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
        input_reactivity_func=just(0),
    )
    pk = PointKinetics(**pkm)
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T=0)})

    calculation = agr.solve(y0=np.array([p0] + ck), time=time)
    analytical = p0 - np.array(ck) @ np.expm1(-np.outer(lambdak, time))
    are_close(calculation[:, 0], analytical, rtol=1.0e-3, atol=1.0e-6)


@given(medium_floats, medium_floats, medium_floats, medium_floats)
def test_pk_save_follows_known_pattern_for_mock(p, ck, inp, T):
    mock_pk = mock_point_kinetics()
    mock_pk.input_reactivity = just(inp)
    save = mock_pk.save([p, ck], T={mock_calc: T})
    r = inp - mock_pk.temp_worth[mock_calc] * T
    assert save == dict(
        power=p, ck=[ck], reactivity=r,
        dPdt=np.dot(mock_pk.lambdak, [ck]) + (r - mock_pk.dollar) * p / mock_pk.Lambda
    )


@given(floats(allow_nan=False), floats(allow_nan=False))
def test_pk_load(p, ck):
    mock_pk = mock_point_kinetics()
    load = mock_pk.load(dict(power=p, ck=[ck]))
    assert np.allclose(load, [p, ck])


@pytest.mark.parametrize(("w", "result"), [
    ({1: np.ones(5), 2: np.ones(5)}, 0),
    ({1: np.ones(5), 2: np.zeros(5)}, -5)
])
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
        input_reactivity_func=just(0),
    )
    pk = PointKineticsWInput(**pkm)

    def power_input(t): return t

    agr = Aggregator.from_decoupled(
        pk, funcs={pk: dict(T=0, power_input=power_input)}
    )

    calculation = agr.solve(y0=np.array([p0] + ck + [p0]), time=time)
    analytical = p0 + np.array(ck) @ (-np.expm1(-np.outer(lambdak, time)))

    pk_power = calculation[:, 0]
    total_power = calculation[:, -1]

    assert np.allclose(pk_power, analytical, rtol=1.0e-3, atol=1.0e-6)
    assert np.allclose(
        total_power, analytical + power_input(time), rtol=1.0e-3, atol=1.0e-6
    )


@given(pos_floats)
def test_pk_change_state_sets_SCRAM_time(t):
    mock_pk = mock_point_kinetics()
    assert mock_pk.SCRAM_time is None
    mock_pk.SCRAM_condition = just(True)
    mock_pk.change_state([0, 0], T=mock_pk.T0, t={mock_pk: t})
    assert mock_pk.SCRAM_time == t


@given(pos_floats)
def test_pk_should_continue_stops_at_SCRAM_time(t):
    mock_pk = mock_point_kinetics()
    mock_pk.SCRAM_time = t
    assert not mock_pk.should_continue([0, 0], T=mock_pk.T0, t={mock_pk: t})
