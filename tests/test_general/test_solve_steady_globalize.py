"""Tests for the globalized ``solve_steady`` and ``scaled_atol``.

The fixture is the compact FlowGraph loop from ``test_scales.py`` (two junctions,
a pump and a resistor closed by a Kirchhoff node). It is small and *exactly*
solvable, so ``hybr`` succeeds — which is what the bit-stability and skip-scipy
assertions need. Cases that must exercise the fallback rungs (scipy failure,
``scaled_newton`` failure, the PTC rung) force those failures with monkeypatches,
because a natural tiny system on which scipy fails but the globalized backend
succeeds is a LOFA-scale hard problem, not a unit-test fixture.
"""

import numpy as np
import pytest

from stream.aggregator import aggregator as aggmod
from stream.calculations import Junction, Pump, Resistor
from stream.composition import FlowGraph, flow_edge
from stream.jacobians import ALG_jacobian
from stream.scales import DEFAULT_SCALES, scale_vector
from stream.solvers import AlgRuntimeError, scaled_newton


@pytest.fixture
def system():
    """A minimal, exactly-solvable aggregator (Kirchhoff node + ideal comps)."""
    a, b = Junction("A"), Junction("B")
    pump = Pump(pressure=5.0, name="pump")
    res = Resistor(1.0, name="res")
    fg = FlowGraph(
        flow_edge((a, b), pump),
        flow_edge((b, a), res),
        abs_pressure_comps=[pump, res],
        reference_node=(a, 2e5),
    )
    return fg.aggregator


def _residual(agr, y):
    return float(np.linalg.norm(agr.compute(y, 0.0)))


def test_auto_preserves_scipy_success_bitwise(system):
    """'auto' (default) takes the scipy path and returns hybr's bits exactly."""
    guess = np.ones(len(system))
    auto = system.solve_steady(guess)
    baseline = system.solve_steady(guess, globalize=False)
    assert np.array_equal(auto, baseline)


def test_auto_rescues_scipy_failure(system, monkeypatch):
    """When scipy raises, 'auto' falls back to scaled_newton and finds a root."""
    guess = np.ones(len(system))

    def raising_algebraic(*args, **kwargs):
        raise AlgRuntimeError("forced scipy failure")

    monkeypatch.setattr(aggmod, "algebraic", raising_algebraic)
    root = system.solve_steady(guess)  # default 'auto'
    assert _residual(system, root) < 1e-6


def test_globalize_false_propagates_scipy_failure(system, monkeypatch):
    """globalize=False reproduces today's behavior: the AlgRuntimeError escapes."""
    guess = np.ones(len(system))

    def raising_algebraic(*args, **kwargs):
        raise AlgRuntimeError("forced scipy failure")

    monkeypatch.setattr(aggmod, "algebraic", raising_algebraic)
    with pytest.raises(AlgRuntimeError):
        system.solve_steady(guess, globalize=False)


def test_globalize_true_skips_scipy(system, monkeypatch):
    """globalize=True never touches scipy yet still converges."""
    guess = np.ones(len(system))

    def poisoned_algebraic(*args, **kwargs):
        raise RuntimeError("scipy path must not run when globalize=True")

    monkeypatch.setattr(aggmod, "algebraic", poisoned_algebraic)
    root = system.solve_steady(guess, globalize=True)
    assert _residual(system, root) < 1e-6


def test_caller_jac_respected(system, monkeypatch):
    """A caller-supplied jac is used by the fallback; no ALG_jacobian is built."""
    agr = system
    guess = np.ones(len(agr))
    real_jac = ALG_jacobian(agr)
    jac_calls = {"n": 0}

    def spy_jac(y, t=0):
        jac_calls["n"] += 1
        return real_jac(y, t)

    sv_calls = {"n": 0}
    orig_scale_vector = aggmod.scale_vector

    def spy_scale_vector(*args, **kwargs):
        sv_calls["n"] += 1
        return orig_scale_vector(*args, **kwargs)

    monkeypatch.setattr(aggmod, "scale_vector", spy_scale_vector)
    root = agr.solve_steady(guess, globalize=True, jac=spy_jac)
    assert jac_calls["n"] > 0  # the fallback used the caller's jac
    assert sv_calls["n"] == 0  # so scale_vector / ALG_jacobian were never consulted
    assert _residual(agr, root) < 1e-6


def test_already_converged_rescue(system):
    """An already-converged input is accepted immediately (jac never evaluated)."""
    agr = system
    exact = agr.solve_steady(np.ones(len(agr)), globalize=True)
    jac_calls = {"n": 0}
    base_jac = ALG_jacobian(agr)

    def spy_jac(y, t=0):
        jac_calls["n"] += 1
        return base_jac(y, t)

    root = agr.solve_steady(exact, globalize=True, jac=spy_jac)
    assert jac_calls["n"] == 0  # returned before the first Newton step
    assert _residual(agr, root) < 1e-6
    np.testing.assert_array_equal(root, exact)


def test_ptc_rung(system, monkeypatch):
    """scaled_newton failure routes through pseudo_transient when fallback_ptc=True.

    The fixture is exactly solvable, so pseudo_transient returns at its entry
    guard (``||F|| < tol``); the wiring under test is that it *is* invoked and its
    result is polished into a root, and that fallback_ptc=False bypasses it.
    """
    agr = system
    exact = agr.solve_steady(np.ones(len(agr)), globalize=True)

    newton_calls = {"n": 0}

    def flaky_newton(F, y0, jac, **kw):
        newton_calls["n"] += 1
        if newton_calls["n"] == 1:  # fail the first (pre-PTC) attempt only
            err = AlgRuntimeError("forced scaled_newton failure")
            err.y = np.asarray(y0, float)
            raise err
        return scaled_newton(F, y0, jac, **kw)

    monkeypatch.setattr(aggmod, "scaled_newton", flaky_newton)

    ptc_calls = {"n": 0}
    real_ptc = aggmod.pseudo_transient

    def spy_ptc(*args, **kwargs):
        ptc_calls["n"] += 1
        return real_ptc(*args, **kwargs)

    monkeypatch.setattr(aggmod, "pseudo_transient", spy_ptc)

    root = agr.solve_steady(exact, globalize=True, fallback_ptc=True)
    assert ptc_calls["n"] == 1  # the PTC rung ran
    assert _residual(agr, root) < 1e-6

    newton_calls["n"] = 0
    ptc_calls["n"] = 0
    with pytest.raises(AlgRuntimeError):
        agr.solve_steady(exact, globalize=True, fallback_ptc=False)
    assert ptc_calls["n"] == 0  # no PTC when fallback_ptc=False


def test_scaled_atol_forms(system):
    """scaled_atol returns rel * typ for the None/dict/array scales forms."""
    agr = system
    n = len(agr)

    # None -> DEFAULT_SCALES registry.
    np.testing.assert_array_equal(agr.scaled_atol(), 1e-6 * scale_vector(agr))
    np.testing.assert_array_equal(agr.scaled_atol(rel=1e-3), 1e-3 * scale_vector(agr))

    # dict -> custom registry.
    registry = dict(DEFAULT_SCALES, Tin=7.0)
    np.testing.assert_array_equal(
        agr.scaled_atol(rel=1e-4, scales=registry),
        1e-4 * scale_vector(agr, registry=registry),
    )

    # array -> used directly.
    typ = np.arange(1.0, n + 1.0)
    np.testing.assert_array_equal(agr.scaled_atol(rel=2.0, scales=typ), 2.0 * typ)


def test_scaled_atol_array_length_validated(system):
    """A wrong-length scales array is rejected."""
    agr = system
    with pytest.raises(ValueError):
        agr.scaled_atol(scales=np.ones(len(agr) + 1))


def test_new_keywords_never_reach_scipy(system):
    """The keyword-only extras must not be forwarded to scipy.optimize.root."""
    guess = np.ones(len(system))
    # Would raise TypeError from scipy if these leaked into **options.
    root = system.solve_steady(guess, globalize=False, scales=None, fallback_ptc=True)
    assert _residual(system, root) < 1e-6


# --- cascade aggregation and widening ---
from stream.aggregator import Aggregator
from stream.composition import Calculation_factory


def _no_root_agr(const):
    """A steady system with no real root: F(y) = y^2 + const."""
    Calc = Calculation_factory(
        calculate=lambda y, c=const: np.array([y[0] ** 2 + c]),
        mass_vector=[False],
        variables=dict(y=0),
    )
    return Aggregator.from_decoupled(Calc())


def _singular_agr():
    """A structurally singular (rank-deficient) steady system: F = [a-b, a-b]."""
    Calc = Calculation_factory(
        calculate=lambda y: np.array([y[0] - y[1], y[0] - y[1]]),
        mass_vector=[False, False],
        variables=dict(a=0, b=1),
    )
    return Aggregator.from_decoupled(Calc())


def test_cascade_aggregates_all_rungs_in_one_error():
    """On a no-root system all rungs of the globalize cascade fail; the user must get
    ONE aggregate AlgRuntimeError naming every rung that ran (not just the last), with
    a structured .rungs record, __cause__ chained to the last failure, and the
    guess-vs-physics hint. F=y^2+1 exhausts scipy -> scaled_newton -> PTC (PTC cannot
    reach the basin, so no polish runs): three rungs."""
    agr = _no_root_agr(1.0)
    guess = np.array([3.0])
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(guess)  # globalize='auto'
    e = exc.value
    msg = str(e)
    assert "globalize cascade failed" in msg
    for name in ("scipy hybr", "scaled_newton", "pseudo_transient"):
        assert name in msg
    assert hasattr(e, "rungs")
    names = [r[0] for r in e.rungs]
    assert names == ["scipy hybr", "scaled_newton", "pseudo_transient"]
    assert all(np.array_equal(r[1], guess) for r in e.rungs)  # all three start from the guess
    assert getattr(e, "y", None) is not None  # last rung's iterate
    assert e.__cause__ is not None
    assert "try:" in msg  # hint_block text present
    # the hybr record's outcome carries scipy's actual reason, not just the
    # "...failed with the following message:" preamble line
    assert not e.rungs[0][2].endswith(":")


def test_cascade_records_four_rungs_including_polish():
    """When PTC reaches the (loose) basin but the tight polish then fails, all four
    rungs are recorded in order — the PTC record reads 'reached the basin' and the
    polish starts from PTC's iterate (not the guess). F=y^2+1e-4 exercises this."""
    agr = _no_root_agr(1e-4)
    guess = np.array([3.0])
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(guess)
    e = exc.value
    names = [r[0] for r in e.rungs]
    assert names == ["scipy hybr", "scaled_newton", "pseudo_transient", "scaled_newton polish"]
    for name in names:
        assert name in str(e)
    ptc = e.rungs[2]
    assert ptc[2] == "reached the basin"
    assert np.array_equal(ptc[1], guess)  # PTC still starts from the guess
    polish = e.rungs[3]
    assert not np.array_equal(polish[1], guess)  # polish starts from PTC's iterate
    assert all(np.isfinite(r[3]) for r in e.rungs)  # every recorded ||F|| is finite


def test_cascade_fallback_ptc_false_two_rungs():
    """fallback_ptc=False cuts the cascade at two rungs (scipy -> scaled_newton);
    the aggregate must record exactly those two and never invoke PTC."""
    agr = _no_root_agr(1.0)
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(np.array([3.0]), fallback_ptc=False)
    e = exc.value
    assert [r[0] for r in e.rungs] == ["scipy hybr", "scaled_newton"]
    assert "all 2 rungs" in str(e)


def test_globalize_false_single_rung_stays_raw():
    """globalize=False keeps the un-globalized behavior: the single scipy rung's
    AlgRuntimeError propagates raw, with no aggregation and no .rungs."""
    agr = _no_root_agr(1.0)
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(np.array([3.0]), globalize=False)
    e = exc.value
    assert not hasattr(e, "rungs")
    assert "globalize cascade failed" not in str(e)


def test_singular_cascade_reaches_pseudo_transient_rung(monkeypatch):
    """A singular scaled_newton rung converts its bare numpy LinAlgError to an
    AlgRuntimeError, so the cascade proceeds to the PTC rung — proven by
    'pseudo_transient' appearing in the aggregate .rungs. scipy is forced to fail (it
    otherwise reports false success on a singular system), so 'auto' drops into the
    globalized backend."""
    agr = _singular_agr()

    def raising_algebraic(*args, **kwargs):
        raise AlgRuntimeError("forced scipy failure")

    monkeypatch.setattr(aggmod, "algebraic", raising_algebraic)
    with pytest.raises(AlgRuntimeError) as exc:
        agr.solve_steady(np.array([3.0, 0.0]))  # globalize='auto'
    e = exc.value
    names = [r[0] for r in e.rungs]
    assert "pseudo_transient" in names  # the cascade reached the PTC rung
    assert "scaled_newton" in names
