"""Tests for the globalized ``solve_steady``, ``scaled_atol`` and ``scaled_step``.

The fixture is a compact FlowGraph loop (two junctions, a pump and a resistor
closed by a Kirchhoff node). It is small and exactly solvable, so ``hybr``
succeeds. Cases that must exercise the fallback (scipy failure) or its own
failure (``scaled_newton`` failure) force those failures with monkeypatches.
"""

import numpy as np
import pytest

from stream.aggregator import aggregator as aggmod
from stream.calculations import Junction, Pump, Resistor
from stream.composition import FlowGraph, flow_edge
from stream.jacobians import ALG_jacobian, scaled_step
from stream.scales import DEFAULT_SCALES, scale_vector
from stream.solvers import AlgRuntimeError


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
    root = system.solve_steady(guess)
    assert _residual(system, root) < 1e-6


def test_globalize_false_propagates_scipy_failure(system, monkeypatch):
    """globalize=False has no fallback: the AlgRuntimeError escapes."""
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
    assert jac_calls["n"] > 0
    assert sv_calls["n"] == 0
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
    assert jac_calls["n"] == 0
    assert _residual(agr, root) < 1e-6
    np.testing.assert_array_equal(root, exact)


def test_scaled_newton_failure_propagates(system, monkeypatch):
    """When the scaled_newton fallback also fails, its AlgRuntimeError escapes."""
    agr = system
    exact = agr.solve_steady(np.ones(len(agr)), globalize=True)

    def failing_newton(F, y0, jac, **kw):
        err = AlgRuntimeError("forced scaled_newton failure")
        err.y = np.asarray(y0, float)
        raise err

    monkeypatch.setattr(aggmod, "scaled_newton", failing_newton)
    with pytest.raises(AlgRuntimeError, match="forced scaled_newton failure"):
        agr.solve_steady(exact, globalize=True)


def test_scaled_step_uses_the_larger_of_value_and_nominal_scale():
    """The step is sqrt(eps) * max(|y|, typ), whether or not ydot is passed."""
    typ = np.array([1.0, 1e5, 1e-3])
    y = np.array([10.0, 0.0, -2.0])
    step = scaled_step(typ)
    expected = np.sqrt(np.finfo(float).eps) * np.array([10.0, 1e5, 2.0])
    np.testing.assert_array_equal(step(y), expected)
    np.testing.assert_array_equal(step(y, np.zeros(3)), expected)


def test_scaled_atol_forms(system):
    """scaled_atol returns rel * typ for the None/dict/array scales forms."""
    agr = system
    n = len(agr)

    np.testing.assert_array_equal(agr.scaled_atol(), 1e-6 * scale_vector(agr))
    np.testing.assert_array_equal(agr.scaled_atol(rel=1e-3), 1e-3 * scale_vector(agr))

    registry = dict(DEFAULT_SCALES, Tin=7.0)
    np.testing.assert_array_equal(
        agr.scaled_atol(rel=1e-4, scales=registry),
        1e-4 * scale_vector(agr, registry=registry),
    )

    typ = np.arange(1.0, n + 1.0)
    np.testing.assert_array_equal(agr.scaled_atol(rel=2.0, scales=typ), 2.0 * typ)


def test_scaled_atol_array_length_validated(system):
    """A wrong-length scales array is rejected."""
    agr = system
    with pytest.raises(ValueError):
        agr.scaled_atol(scales=np.ones(len(agr) + 1))


def test_new_keywords_never_reach_scipy(system):
    """The keyword-only extras must not be forwarded to scipy.optimize.root,
    which would raise TypeError on them."""
    guess = np.ones(len(system))
    root = system.solve_steady(guess, globalize=False, scales=None)
    assert _residual(system, root) < 1e-6
