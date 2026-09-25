"""Benchmark-scale conditioning on the real LOFA benchmark system.

The synthetic-matrix properties of ``stream.solvers._equilibrate`` (single
pass, floors, reconstruction) are covered in ``test_scaled_newton.py``. This
module asserts the empirical claim on the benchmark: the raw ALG Jacobian at the
benchmark operating point (83.6 kW) is ill-conditioned by scaling alone (cond ~1e12,
dominated by the raw inertia coefficients), and one row+column equilibration pass
drops it below 1e8 and leaves no numerical null space (the benchmark grounds
pressure through its reference node).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from stream.jacobians import ALG_jacobian
from stream.solvers import _equilibrate

_CASE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "lofa" / "case.py"


def _load_case():
    spec = importlib.util.spec_from_file_location("lofa_benchmark_case", _CASE_PATH)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


@pytest.fixture(scope="module")
def benchmark_jacobians():
    case = _load_case()
    agr, K, refs = case.build(power=case.POWER_REALISTIC)
    jac = ALG_jacobian(agr)
    vec_ballpark = agr.load(case.ballpark_guess(agr, K, refs))
    vec_expert = agr.load(case.expert_guess(refs, case.POWER_REALISTIC))
    return {
        "ballpark": jac(vec_ballpark, 0.0).copy(),
        "expert": jac(vec_expert, 0.0).copy(),
    }


@pytest.mark.parametrize("guess", ["ballpark", "expert"])
def test_equilibration_drops_benchmark_cond_below_1e8(benchmark_jacobians, guess):
    J = benchmark_jacobians[guess]
    sv_raw = np.linalg.svd(J, compute_uv=False)
    cond_raw = sv_raw[0] / sv_raw[-1]
    assert cond_raw > 1e10, f"raw cond(J) at {guess} guess is {cond_raw:.3e}"

    J_eq, rs, cs = _equilibrate(J)
    sv_eq = np.linalg.svd(J_eq, compute_uv=False)
    cond_eq = sv_eq[0] / sv_eq[-1]
    assert cond_eq < 1e8, f"equilibrated cond(J) at {guess} guess is {cond_eq:.3e}"


@pytest.mark.parametrize("guess", ["ballpark", "expert"])
def test_equilibrated_benchmark_jacobian_has_no_null_space(benchmark_jacobians, guess):
    J_eq, rs, cs = _equilibrate(benchmark_jacobians[guess])
    sv = np.linalg.svd(J_eq, compute_uv=False)
    assert int(np.sum(sv < 1e-8 * sv[0])) == 0

    # positive scales mean the pass is sign-preserving by construction
    assert np.all(rs > 0) and np.all(cs > 0)
