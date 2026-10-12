import pytest

from stream.examples.lofa import checks, scenarios
from stream.examples.lofa.params import Params
from stream.examples.lofa.system import operating_point


@pytest.fixture(scope="module")
def short_run():
    p = Params()
    sol, rec = scenarios.both_pumps(p, t_end=300.0, n=1501)
    agr, K, refs, y = operating_point(p)
    return p, agr, K, refs, sol


@pytest.mark.slow
def test_junction_mass_balance_closes_to_round_off(short_run):
    p, agr, K, refs, sol = short_run
    worst, per = checks.junction_mass_balance(agr, K, refs, sol)
    assert worst < 1e-8
    assert set(per) == {str(j) for j in K.g.nodes}


@pytest.mark.slow
def test_energy_budget_closes_within_one_percent(short_run):
    p, agr, K, refs, sol = short_run
    e = checks.energy_budget(agr, K, refs, sol, p)
    assert e["fed_J"] > 0
    assert abs(e["unaccounted_J"]) < 0.01 * e["fed_J"]


@pytest.mark.slow
def test_energy_budget_accepts_a_partial_solution(short_run):
    p, agr, K, refs, sol = short_run
    partial = scenarios.Solution(sol.time[:40], sol.data[:40])
    e = checks.energy_budget(agr, K, refs, partial, p)
    assert abs(e["unaccounted_J"]) < 0.01 * e["fed_J"]


@pytest.mark.slow
def test_loop_pressure_budget_closes_around_the_pool_loop(short_run):
    p, agr, K, refs, sol = short_run
    names = ["HotChannel", "Flapper", "Pool"]
    budget, buoyancy = checks.loop_pressure_budget(agr, K, refs, sol, names)
    assert set(budget) == set(names)
    assert buoyancy > 0
    assert abs(sum(budget.values())) < 0.05 * buoyancy


@pytest.mark.slow
def test_loop_pressure_budget_closes_around_the_pump_loop(short_run):
    p, agr, K, refs, sol = short_run
    names = [
        "HotChannel",
        "FrictionPrimary", "BendPrimary0", "BendPrimary1", "GravityPrimary",
        "DecayTank",
        "FrictionBasement", "BendBasement0", "GravityBasement",
        "PumpA", "PumpCurveA", "FlywheelA", "ValveA",
        "HX", "HXLoss", "FrictionRiser", "BendRiser0", "BendRiser1", "GravityRiser",
        "Pool",
    ]
    budget, buoyancy = checks.loop_pressure_budget(agr, K, refs, sol, names)
    assert set(budget) == set(names)
    assert abs(sum(budget.values())) < 0.05 * buoyancy
