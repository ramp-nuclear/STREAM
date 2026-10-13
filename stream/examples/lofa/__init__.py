"""A pool-reactor loss-of-flow case: a plate core in forced downward flow, a flapper to the
pool below the core, a decay tank, two pump legs with flywheels and check valves, a heat
exchanger, and the pool as the natural-circulation heat sink. Point kinetics with
temperature feedback and a low-flow trip drive the power."""

from stream.examples.lofa.params import CORE, HEATED, Params, nominal_flows, power_share
from stream.examples.lofa.system import build, draw, guess, operating_point, steady, steady_checks
from stream.examples.lofa.scenarios import Record, both_pumps, one_pump, stuck_flapper
from stream.examples.lofa.checks import energy_budget, junction_mass_balance, loop_pressure_budget

__all__ = [
    "CORE",
    "HEATED",
    "Params",
    "Record",
    "both_pumps",
    "build",
    "draw",
    "energy_budget",
    "guess",
    "junction_mass_balance",
    "loop_pressure_budget",
    "nominal_flows",
    "one_pump",
    "operating_point",
    "power_share",
    "steady",
    "steady_checks",
    "stuck_flapper",
]
