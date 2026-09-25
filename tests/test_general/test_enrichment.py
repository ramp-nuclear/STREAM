"""Failure-enrichment tests.

:meth:`Aggregator._enrich_failure` decorates an in-flight solver failure with
domain-term notes — non-finite locations, the top-3 scaled worst residuals, and a
saturation crossing plus the ``stop_at_saturation`` hint. It is best-effort: each
probe is guarded, so one failing probe never suppresses the error nor the other
notes, and it touches notes only, never message/args.
"""

import numpy as np
import pytest

from stream.aggregator import Aggregator
from stream.composition import Calculation_factory
from stream.solvers import AlgRuntimeError


def _notes(err):
    return "\n".join(getattr(err, "__notes__", []))


def _vec_scalar_agr():
    """One node named 'CC': a 3-cell vector 'T' (rows 0..2) and a scalar 'p' (row 3)."""
    A = Calculation_factory(lambda y: np.asarray(y, dtype=float), [False] * 4, {"T": slice(0, 3), "p": 3})("CC")
    return Aggregator.from_decoupled(A)


def _no_root_agr(const=1.0):
    """A steady system with no real root: F(y) = y^2 + const."""
    Calc = Calculation_factory(lambda y, c=const: np.array([y[0] ** 2 + c]), [False], {"y": 0})
    return Aggregator.from_decoupled(Calc("noroot"))


# --- _enrich_failure unit behaviour ------------------------------------------


def test_enrich_nonfinite_note_names_location():
    """A NaN-bearing 1-D iterate yields a non-finite note with the exact Location text."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("boom")
    err.y = np.array([1.0, np.nan, 3.0, 5.0])  # NaN at cell 1 of 'T'
    agr._enrich_failure(err)
    notes = _notes(err)
    assert "non-finite values" in notes
    assert "T=nan at cell 1 of 'CC' (source: y)" in notes


def test_enrich_returns_silently_without_y():
    """An error carrying no ``.y`` gets no notes and never raises."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("boom")  # no .y attribute
    agr._enrich_failure(err)  # must not raise
    assert not getattr(err, "__notes__", [])


def test_enrich_is_best_effort(monkeypatch):
    """A probe that raises must not suppress the error, and the other notes
    (here, the worst-residual note) must still attach."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("boom")
    err.y = np.array([1.0, 2.0, 3.0, 5.0])  # all finite -> worst_residuals still applies

    def boom(*a, **k):
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(agr, "locate_nonfinite", boom)
    agr._enrich_failure(err)  # must return silently despite the failing probe
    assert "worst scaled residuals" in _notes(err)


def test_enrich_message_and_args_untouched():
    """Enrichment adds notes only; the message and args are never edited."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("original message")
    err.y = np.array([1.0, np.nan, 3.0, 5.0])
    agr._enrich_failure(err)
    assert str(err) == "original message"
    assert err.args == ("original message",)
    assert getattr(err, "__notes__", [])  # but notes were added


def test_enrich_handles_2d_trajectory_shape():
    """A 2-D trajectory uses its LAST row as the failure state."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("boom")
    err.y = np.array([[1.0, 2.0, 3.0, 5.0], [1.0, 2.0, np.nan, 5.0]])  # NaN in last row, cell 2
    err.t = np.array([0.0, 1.0])
    agr._enrich_failure(err)
    assert "T=nan at cell 2 of 'CC' (source: y)" in _notes(err)


def test_enrich_handles_1d_ic_payload_shape():
    """A 1-D IC-recovery payload is used as-is."""
    agr = _vec_scalar_agr()
    err = AlgRuntimeError("boom")
    err.y = np.array([1.0, 2.0, 3.0, np.nan])  # NaN at the scalar 'p' (row 3, cell None)
    agr._enrich_failure(err)
    notes = _notes(err)
    assert "p=nan of 'CC' (source: y)" in notes  # scalar -> no "at cell N"


# --- cascade failure (solve_steady) enrichment -------------------------------


def test_cascade_error_carries_worst_residual_note():
    """An unsolvable algebraic system exhausts the globalize cascade; the single
    aggregate AlgRuntimeError is enriched with a 'worst scaled residuals' note."""
    agr = _no_root_agr(1.0)
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(np.array([3.0]))  # globalize='auto' -> full cascade fails
    assert "worst scaled residuals" in _notes(exc.value)


# --- transient saturation death ----------------------


@pytest.mark.slow
def test_transient_saturation_death_enriched():
    """A guard-off channel driven past Tsat dies with a TransientRuntimeError whose
    notes name the channel, a crossing time, and 'stop_at_saturation'."""
    from stream.calculations import ChannelAndContacts
    from stream.pipe_geometry import EffectivePipe
    from stream.solvers import TransientRuntimeError
    from stream.substances import light_water

    pipe = EffectivePipe.rectangular(0.6, 0.06, 0.003, 0.003)
    Z = np.linspace(0.0, 0.6, 5)  # 4 cells; Tsat(~1 bar) ~ 99.6 C
    ramp = lambda t: np.full(4, 80.0 + 20.0 * t)  # noqa: E731
    C = ChannelAndContacts(Z, light_water, pipe, stop_at_saturation=False)  # GUARD OFF
    agr = Aggregator.from_decoupled(C, funcs={C: dict(mdot=0.02, T_left=ramp, T_right=ramp, Tin=80.0, p_abs=1e5)})
    steady = agr.solve_steady(
        {C.name: dict(T_cool=np.full(4, 80.0), pressure=1e5, h_left=np.full(4, 3e4), h_right=np.full(4, 3e4))}
    )
    with pytest.raises(TransientRuntimeError) as exc:
        agr.solve(steady, np.linspace(0.0, 10.0, 41), eq_type="DAE")
    notes = _notes(exc.value)
    assert "CC" in notes  # the channel name
    assert "crossed Tsat" in notes  # a crossing time / cells
    assert "stop_at_saturation" in notes  # the discoverability hint
    # The message itself is untouched: the untranslated backend line survives.
    assert "stop_at_saturation" not in exc.value.message
