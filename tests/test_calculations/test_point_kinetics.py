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
    SCRAM_at_power,
    scram_at_power_margin,
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


def test_temperature_reactivity_accepts_scalar_weight_with_multicell_temperature():
    """E10: a scalar temp_worth (as the PerC scalar type hint invites) combined
    with a multi-cell T array made np.dot(scalar, vector).item() raise
    ValueError. A scalar weight must now act per-cell (summed over cells)."""
    rho = temperature_reactivity({"ch": np.array([300.0, 310.0, 320.0])}, {"ch": 290.0}, {"ch": 2e-5})
    assert rho == pytest.approx(-2e-5 * (10 + 20 + 30))


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

    ctrl = ReactivityController(
        input_reactivity=rho_in,
        state_machine=machine,
        abort_states={OneWayToSCRAM.SCRAM},
        trip_margin=lambda state, t, power, dPdt: 1.0 if state == OneWayToSCRAM.SCRAM else limit - power,
    )
    pk = PointKinetics(
        generation_time=_Lam, delayed_neutron_fractions=_betak, delayed_groups_decay_rates=_lambdak, controls=ctrl
    )
    ck0 = _betak * P0 / (_lambdak * _Lam)
    return pk, ctrl, np.concatenate([[P0], ck0])


def test_scram_abort_actually_stops_dae_solve_at_trip_time():
    """E5: should_continue lacked @unpacked, so the aggregator handed it a
    {pk: t} dict; the abort predicate ``t == t_state`` was never True and the
    DAE solve ran to t_end despite the SCRAM transition. It must now stop at the
    trip time."""
    pk, ctrl, y0 = _scram_pk()
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T={}, t=identity)})
    time = np.linspace(0, 60, 601)
    sol = agr.solve(y0=y0.copy(), time=time, eq_type="DAE")

    assert ctrl.state == OneWayToSCRAM.SCRAM  # the transition itself always worked
    assert sol.time[-1] < time[-1]  # the abort actually stopped the run early
    assert sol.time[-1] == pytest.approx(ctrl.t_state)  # ... exactly at the trip time


def test_winput_feeds_true_power_derivative_to_state_machine_and_save():
    """E6: PointKineticsWInput.indices('power') = m+1 is the algebraic total-power
    residual (~0 at any accepted state), so change_state fed the state machine
    dPdt~0 and save recorded dPdt~0. The true power derivative is row 0 of
    calculate()."""
    seen = {}

    def spy(state, t, power, dPdt, **kw):
        seen.update(power=power, dPdt=dPdt)
        return state

    ctrl = ReactivityController(state_machine=spy, input_reactivity=just(100e-5))
    pk = PointKineticsWInput(
        generation_time=_Lam,
        delayed_neutron_fractions=_betak,
        delayed_groups_decay_rates=_lambdak,
        temp_worth={},
        ref_temp={},
        controls=ctrl,
    )
    P0 = 1e6
    ck0 = _betak * P0 / (_lambdak * _Lam)
    y = np.concatenate([[P0], ck0, [P0 + 5e4]])  # [pk_power, ck..., total_power]

    true_dpdt = pk.calculate(y, T={}, t=1.0, power_input=5e4)[0]
    assert abs(true_dpdt) > 1e6  # genuinely large with +100 pcm, not the ~0 residual

    pk.change_state(y, T={}, t=1.0, power_input=5e4)
    assert seen["power"] == pytest.approx(P0 + 5e4)  # total power stays correct
    assert seen["dPdt"] == pytest.approx(true_dpdt)  # ... and dPdt is the real derivative

    saved = pk.save(y, T={}, t=1.0, power_input=5e4)
    assert saved["dPdt"] == pytest.approx(true_dpdt)


def test_save_dpdt_uses_historical_reactivity_not_final_controller_state():
    """E7: save() records reactivity via worth_history(t) (the state active at
    time t) but computed dPdt via calculate() -> worth(t), which uses the
    controller's FINAL state. After a SCRAM the post-processed dPdt for every
    earlier output time was evaluated with post-transition reactivity and a
    negative elapsed time (sign-flipped), disagreeing with the saved reactivity."""

    def ramp(state, t_state, t, **_):
        return -0.05 * (t - t_state) if state == OneWayToSCRAM.SCRAM else 0.0

    ctrl = ReactivityController(
        input_reactivity=ramp,
        state_machine=lambda s, t, p, d, **k: OneWayToSCRAM.SCRAM if t >= 5.0 else s,
    )
    pk = PointKinetics(
        generation_time=_Lam, delayed_neutron_fractions=_betak, delayed_groups_decay_rates=_lambdak, controls=ctrl
    )
    P0 = 1e6
    ck0 = _betak * P0 / (_lambdak * _Lam)  # steady precursors -> dPdt = 0 at rho = 0
    y = np.concatenate([[P0], ck0])

    ctrl.change_state(5.0, P0, 0.0)  # controller now sits in its final SCRAM state (t_state = 5)
    assert ctrl.state == OneWayToSCRAM.SCRAM

    saved = pk.save(y, T={}, t=2.0)  # post-process an output time BEFORE the transition
    # worth_history(2) = 0 (still NORMAL), so both entries must be consistent with rho = 0.
    assert saved["reactivity"] == pytest.approx(0.0)
    assert saved["dPdt"] == pytest.approx(0.0, abs=1e3)  # pre-fix: 7.5e9 from worth(2) = +0.15


def test_scram_at_power_is_usable_as_a_state_machine():
    """E9: SCRAM_at_power had the wrong arity for the StateMachine protocol
    (TypeError when ReactivityController.change_state calls it with
    (state, t, power, dPdt)) and returned a bool instead of a state. It must now
    curry the limit and behave as a real state machine returning an Enum."""
    machine = SCRAM_at_power(1.2e6)  # curried on power_limit
    ctrl = ReactivityController(
        state_machine=machine,
        abort_states={OneWayToSCRAM.SCRAM},
        trip_margin=scram_at_power_margin(1.2e6),
    )

    # Below the limit: stays NORMAL (and returns a state, not a bool).
    returned = ctrl.change_state(t=1.0, power=1.0e6, dPdt=0.0)
    assert returned == OneWayToSCRAM.NORMAL
    assert not isinstance(returned, bool)
    assert ctrl.should_continue(1.0)

    # Above the limit: latches to SCRAM and aborts.
    ctrl.change_state(t=2.0, power=1.5e6, dPdt=1e7)
    assert ctrl.state == OneWayToSCRAM.SCRAM
    assert not ctrl.should_continue(2.0)
    assert ctrl.t_state == 2.0

    # Companion margin crosses zero from positive exactly at the limit.
    margin = scram_at_power_margin(1.2e6)
    assert margin(OneWayToSCRAM.NORMAL, 1.0, 1.0e6, 0.0) > 0
    assert margin(OneWayToSCRAM.NORMAL, 2.0, 1.5e6, 1e7) < 0


def _power_trip_pk():
    """An all-differential PointKinetics with a +50 pcm ramp that drives power to a
    SCRAM trip at 1.2 MW and a strong rod insertion afterwards (no abort, so the
    run continues and the shutdown is observable)."""
    limit = 1.2e6
    ctrl = ReactivityController(
        input_reactivity=lambda state, t_state, t, **_: (
            -0.05 * (t - t_state) if state == OneWayToSCRAM.SCRAM else (50e-5 if t > 1.0 else 0.0)
        ),
        state_machine=SCRAM_at_power(limit),
        trip_margin=scram_at_power_margin(limit),
    )
    pk = PointKinetics(
        generation_time=_Lam,
        delayed_neutron_fractions=_betak,
        delayed_groups_decay_rates=_lambdak,
        controls=ctrl,
    )
    y0 = np.concatenate([[1e6], _betak * 1e6 / (_lambdak * _Lam)])
    return pk, ctrl, y0


@pytest.mark.parametrize("mode", ["DAE", "ODE", None])
def test_scram_fires_in_ode_dae_and_auto_selected_modes(mode):
    """E4: the ODE branch wired no events to solve_ivp, so an all-differential PK
    system (which auto-selects ODE) never invoked its SCRAM state machine and ran
    to full power. The trip must now fire in DAE, explicit ODE, and the
    auto-selected (None -> ODE) path, and its negative reactivity must enter F so
    the power actually falls."""
    pk, ctrl, y0 = _power_trip_pk()
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T={}, t=identity)})
    sol = agr.solve(y0=y0, time=np.linspace(0, 40, 401), eq_type=mode)

    assert ctrl.state == OneWayToSCRAM.SCRAM  # tripped (was NORMAL forever in broken ODE)
    assert sol.data[:, 0].max() > 1.15e6  # climbed to near the 1.2 MW trip
    assert sol.data[-1, 0] < 1e5  # SCRAM reactivity entered F: power fell far below the trip


def test_marginless_controller_scram_reactivity_feeds_back_in_dae():
    """A controller with a state machine but NO trip_margin (margin-less) must
    still feed its SCRAM reactivity into F during a DAE solve. The old boolean
    rootfn ran change_state every step; the poll fallback must restart on the
    F-changing transition rather than returning the unprotected trajectory."""

    def machine(state, t, power, dPdt, **k):
        return OneWayToSCRAM.SCRAM if power >= 1.2e6 else state

    def rho_in(state, t_state, t, **_):
        return -0.05 * (t - t_state) if state == OneWayToSCRAM.SCRAM else (50e-5 if t > 1.0 else 0.0)

    ctrl = ReactivityController(input_reactivity=rho_in, state_machine=machine)  # no margin, no abort
    pk = PointKinetics(
        generation_time=_Lam, delayed_neutron_fractions=_betak, delayed_groups_decay_rates=_lambdak, controls=ctrl
    )
    y0 = np.concatenate([[1e6], _betak * 1e6 / (_lambdak * _Lam)])
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T={}, t=identity)})
    sol = agr.solve(y0=y0, time=np.linspace(0, 40, 401), eq_type="DAE")

    assert ctrl.state == OneWayToSCRAM.SCRAM
    assert sol.data[:, 0].max() > 1.15e6  # climbed to the trip
    assert sol.data[-1, 0] < 1e5  # rods inserted -> power fell: F-feedback preserved (was ~1.6 MW)


def test_event_condition_already_satisfied_at_start_fires():
    """An event whose margin is already <= 0 at t0 (power already over the limit)
    must still fire; a direction/sign-change crossing alone would miss it."""
    limit = 1.2e6
    ctrl = ReactivityController(
        state_machine=SCRAM_at_power(limit),
        trip_margin=scram_at_power_margin(limit),
        abort_states={OneWayToSCRAM.SCRAM},
    )
    pk = PointKinetics(
        generation_time=_Lam, delayed_neutron_fractions=_betak, delayed_groups_decay_rates=_lambdak, controls=ctrl
    )
    y0 = np.concatenate([[2e6], _betak * 2e6 / (_lambdak * _Lam)])  # power 2e6 already over the limit
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T={}, t=identity)})
    sol = agr.solve(y0=y0, time=np.linspace(0, 10, 101), eq_type="DAE")

    assert ctrl.state == OneWayToSCRAM.SCRAM  # tripped at the very start
    assert sol.time[-1] == pytest.approx(0.0)  # aborted immediately, not run to t_end
