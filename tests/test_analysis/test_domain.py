"""The fluid-domain checker family in stream.analysis.thresholds.

``domain_report`` is the post-hoc surface that makes an out-of-domain fluid state
attributable: it turns a silent absurd h or a silent negative-pressure NaN into
named violations (variable/cell/calc/value/range). ``raise_on_domain`` mirrors
``raise_on_saturation``; both add a non-converged-iterate caveat on a raw-array
input. The h≈0 guard in ``twall_limit`` and the Bergles ONB margin returns NaN
(not a q/h blow-up) with one warning naming the stagnant cells, and is unchanged
at healthy h.
"""

import warnings

import numpy as np
import pytest

from stream.aggregator import Aggregator, Solution
from stream.analysis.thresholds import (
    Bergles_Rohsenow_T_ONB,
    DomainValidityError,
    DomainViolation,
    domain_report,
    raise_on_domain,
    raise_on_saturation,
    twall_limit,
)
from stream.calculations import ChannelAndContacts
from stream.calculations.channel import ChannelVar, Direction, SaturationReachedError
from stream.errors import StreamError
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water

pipe = EffectivePipe.rectangular(0.6, 0.06, 0.003, 0.003)
Z = np.linspace(0.0, 0.6, 5)  # 4 cells; Tsat(~1 bar) ≈ 100 C


def _agr_vec(T_cool, pressure=1e5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Tin_minus construction warning
        C = ChannelAndContacts(Z, light_water, pipe)
        agr = Aggregator.from_decoupled(
            C, funcs={C: dict(mdot=0.05, T_left=80.0, T_right=80.0, Tin=80.0, p_abs=pressure)}
        )
    vec = agr.load(
        {C.name: dict(T_cool=np.full(4, T_cool), pressure=0.0, h_left=np.full(4, 1e4), h_right=np.full(4, 1e4))}
    )
    return C, agr, vec


# --- domain_report ------------------------------------------------------------


def test_domain_report_empty_on_healthy_state():
    C, agr, vec = _agr_vec(80.0)
    assert domain_report(agr.save(vec), agr) == []
    raise_on_domain(agr.save(vec), agr)  # no-op, must not raise


def test_domain_report_finds_planted_T_violation():
    """A far-out-of-range bulk temperature is named by variable/cell/calc/value/range."""
    C, agr, vec = _agr_vec(-287.0)
    report = domain_report(agr.save(vec), agr)
    tcool = [v for v in report if v.variable == "T_cool"]
    assert len(tcool) == 4  # one per cell
    v0 = tcool[0]
    assert isinstance(v0, DomainViolation)
    assert v0.calc_name == C.name
    assert v0.cell == 0
    assert v0.value == -287.0
    assert v0.bound == (0.1, 350.0)
    assert v0.t is None


def test_domain_report_finds_planted_negative_pressure():
    C, agr, _ = _agr_vec(80.0)
    state = {
        C.name: {
            ChannelVar.tbulk: np.full(4, 80.0),
            ChannelVar.static_pressure: np.full(4, -2e5),
            ChannelVar.tin: np.array([80.0]),
        }
    }
    report = domain_report(state, agr)
    press = [v for v in report if "pressure" in v.variable]
    assert press and all(v.value == -2e5 and v.bound == 0.0 for v in press)


def test_domain_report_ignores_signed_pressure_drop():
    """The signed pressure-*drop* is legitimately negative and must not be flagged."""
    C, agr, _ = _agr_vec(80.0)
    state = {
        C.name: {
            ChannelVar.tbulk: np.full(4, 80.0),
            ChannelVar.static_pressure: np.full(4, 1e5),  # healthy absolute pressure
            ChannelVar.pressure_drop: np.full(4, -500.0),  # a real, negative drop
        }
    }
    assert domain_report(state, agr) == []


def test_domain_report_accepts_a_raw_vector():
    C, agr, vec = _agr_vec(-287.0)
    report = domain_report(vec, agr)  # raw solve vector, not a saved State
    assert any(v.variable == "T_cool" and v.value == -287.0 for v in report)


def test_domain_report_handles_a_timeseries_earliest_first():
    C, agr, sub = _agr_vec(80.0)
    _, _, hot = _agr_vec(400.0)  # above the 350 C upper bound
    traj = np.vstack([sub, sub, hot])
    report = domain_report(traj, agr, times=[0.0, 1.0, 2.0])
    assert report and all(v.t == 2.0 for v in report)  # only the hot frame violates
    assert any(v.variable == "T_cool" and v.value == 400.0 for v in report)


def test_domain_report_silent_on_fluid_without_validity(monkeypatch):
    """A node whose fluid declares no validity range is skipped entirely."""
    from dataclasses import replace

    C, agr, vec = _agr_vec(-287.0)
    saved = agr.save(vec)  # save before swapping the fluid (save reads fluid too)
    monkeypatch.setattr(C, "fluid", replace(light_water, validity=None))
    assert domain_report(saved, agr) == []


# --- raise_on_domain ----------------------------------------------------------


def test_raise_on_domain_raises_catchable_via_StreamError():
    C, agr, vec = _agr_vec(-287.0)
    with pytest.raises(DomainValidityError) as exc:
        raise_on_domain(agr.save(vec), agr)
    # Domain-terms message: value, cell, calc, and the violated range.
    msg = str(exc.value)
    assert "T_cool" in msg and "-287" in msg and C.name in msg
    assert "[0.1, 350]" in msg and "outside" in msg
    # StreamError-family, so the one-stop catch works.
    with pytest.raises(StreamError):
        raise_on_domain(agr.save(vec), agr)


def test_raise_on_domain_note_is_prepended():
    C, agr, vec = _agr_vec(-287.0)
    with pytest.raises(DomainValidityError) as exc:
        raise_on_domain(agr.save(vec), agr, note="no converged steady solution — ")
    assert str(exc.value).startswith("no converged steady solution — ")


# --- non-converged-iterate sentence ------------------------------------

_SENTENCE = "input may be a non-converged iterate — verify against a converged solution"


def test_iterate_sentence_present_for_raw_vector_domain():
    C, agr, vec = _agr_vec(-287.0)
    with pytest.raises(DomainValidityError) as exc:
        raise_on_domain(vec, agr)  # raw ndarray
    assert str(exc.value).endswith(_SENTENCE)


def test_iterate_sentence_absent_for_state_domain():
    C, agr, vec = _agr_vec(-287.0)
    with pytest.raises(DomainValidityError) as exc:
        raise_on_domain(agr.save(vec), agr)  # a saved State
    assert _SENTENCE not in str(exc.value)


def test_iterate_sentence_absent_for_solution_domain():
    """A Solution carries authoritative status, so no iterate caveat is added."""
    C, agr, vec = _agr_vec(-287.0)
    sol = Solution(np.array([0.0]), vec[None, :])
    with pytest.raises(DomainValidityError) as exc:
        raise_on_domain(sol, agr)
    assert _SENTENCE not in str(exc.value)


def test_iterate_sentence_present_for_raw_vector_saturation():
    C, agr, vec = _agr_vec(130.0)
    with pytest.raises(SaturationReachedError) as exc:
        raise_on_saturation(vec, agr)  # raw ndarray
    assert str(exc.value).endswith(_SENTENCE)


def test_iterate_sentence_absent_for_state_saturation():
    C, agr, vec = _agr_vec(130.0)
    with pytest.raises(SaturationReachedError) as exc:
        raise_on_saturation(agr.save(vec), agr)  # a saved State
    assert _SENTENCE not in str(exc.value)


def test_iterate_sentence_absent_for_solution_saturation():
    """A Solution carries authoritative status, so no iterate caveat is added."""
    C, agr, vec = _agr_vec(130.0)
    sol = Solution(np.array([0.0]), vec[None, :])
    with pytest.raises(SaturationReachedError) as exc:
        raise_on_saturation(sol, agr)
    assert _SENTENCE not in str(exc.value)


# --- h≈0 analysis guard ----------------------------------------------


def _wall_state(h, tbulk=(110.0, 111.0), q=2.5e5):
    return {
        ChannelVar.tbulk: np.array(tbulk),
        ChannelVar.get("heatflux", Direction.left): np.full(len(tbulk), q),
        ChannelVar.get("heatflux", Direction.right): np.full(len(tbulk), q),
        ChannelVar.get("h", Direction.left): np.asarray(h, dtype=float),
        ChannelVar.get("h", Direction.right): np.asarray(h, dtype=float),
    }


def test_twall_limit_nan_and_warning_at_stagnant_h():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tw = np.asarray(twall_limit(state=_wall_state(np.array([0.0, 0.0]))))
    assert np.isnan(tw).all()  # NaN, not a q/h blow-up
    assert w, "a stagnant cell must warn"
    msg = str(w[0].message)
    assert "cell" in msg and "[0, 1]" in msg and "h" in msg


def test_twall_limit_exact_at_healthy_h():
    h = np.array([1e4, 2e4])
    st = _wall_state(h)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tw = np.asarray(twall_limit(state=st))
    tbulk = st[ChannelVar.tbulk]
    q = st[ChannelVar.get("heatflux", Direction.left)]
    expected = np.maximum(tbulk + q / h, tbulk + q / h)
    assert np.array_equal(tw, expected)
    assert not w  # healthy harvest never warns


def test_twall_limit_exact_at_partially_stagnant_h():
    """Only the stagnant cell is NaN; the healthy cell keeps its exact value."""
    h = np.array([1e4, 0.0])
    st = _wall_state(h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tw = np.asarray(twall_limit(state=st))
    tbulk = st[ChannelVar.tbulk]
    q = st[ChannelVar.get("heatflux", Direction.left)]
    assert tw[0] == tbulk[0] + q[0] / h[0]  # healthy cell untouched
    assert np.isnan(tw[1])


def test_onb_margin_nan_and_warning_at_stagnant_h():
    st = {
        ChannelVar.static_pressure: np.full(2, 1e5),
        ChannelVar.tbulk: np.array([110.0, 111.0]),
        ChannelVar.get("h", Direction.left): np.array([0.0, 0.0]),
        ChannelVar.get("heatflux", Direction.left): np.array([2.5e5, 2.5e5]),
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        margin = np.asarray(Bergles_Rohsenow_T_ONB(state=st, direction=Direction.left))
    assert np.isnan(margin).all()
    assert w and "cell" in str(w[0].message)


def test_onb_margin_no_warning_and_finite_at_healthy_h():
    st = {
        ChannelVar.static_pressure: np.full(2, 1e5),
        ChannelVar.tbulk: np.array([110.0, 111.0]),
        ChannelVar.get("h", Direction.left): np.array([3e4, 3e4]),
        ChannelVar.get("heatflux", Direction.left): np.array([2.5e5, 2.5e5]),
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        margin = np.asarray(Bergles_Rohsenow_T_ONB(state=st, direction=Direction.left))
    assert np.isfinite(margin).all()
    assert not w
