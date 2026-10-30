# Pool-reactor loss of flow

A loss-of-flow case for a generic materials-testing pool reactor: a plate core in downward forced
flow, a decay tank, two pump legs with flywheels and check valves, a heat exchanger, and a flapper
below the core that opens when the pumps coast down, so that the core goes over to natural
circulation with the pool as the heat sink. Point kinetics with temperature feedback and a low-flow
trip drive the power. Every heated plate channel runs the regime-dependent wall heat-transfer law, which
composes laminar, turbulent and natural convection down to zero flow; `Params(htc="dittus")` gives
the Dittus-Boelter correlation for comparison. The case ships with the library as
`stream.examples.lofa`; this directory runs it as a benchmark: a ladder of rungs with pass
conditions, a stress sweep over its parameters, and a physics audit of a solved transient against
conservation laws.

## Commands

From the repository root, with the library installed (`pip install -e .`):

```bash
python benchmarks/lofa_pool/run.py --all              # every rung, appends a dated table to RESULTS.md
python benchmarks/lofa_pool/sweeps.py --axis all --workers 10   # the stress sweep, one CSV per axis in results/
python benchmarks/lofa_pool/metrics.py --params '{"power_scale": 1.2}' --out row.json   # one scenario as a JSON row
```

`run.py --rung R` runs one rung in the current process. `physics_checks.py --case {case,top_flapper}`
audits a solved hour of the shipped loop, or of the second loop in `top_flapper.py` where the pump
return discharges into the pool surface and the flapper sits above the core.

The sweep's `guesses` axis solves the default case from the ballpark guess, the expert guess and
five `perturbed` guesses, each the expert guess with every entry scaled by a random factor; every
guess is expected to pass. Elsewhere in the sweep, the `tolerance` axis's atol 1e-6 row is known to
fail in the steady solve's line search, and the three `EXPECTED_SATURATION` rows of `sweeps.py`
reach saturation by design. The `htc` axis runs the default case with the regime-dependent law and
with Dittus-Boelter; both rows complete, and the Dittus-Boelter row's hot-plate meat peaks about
16 °C higher. The `cells` axis's 40-cell row records one NumPy warning, an invalid value in the
laminar heat-transfer law's entry length, so it completes but counts as warned.

## Rungs

Measured on the library at commit `8fe2d91`, the first commit with the regime-dependent wall law.

| Rung | Title | Status | Wall (s) |
|---|---|---|---|
| 1 | steady, ballpark guess | PASS | 8 |
| 3 | both pumps trip, 1 h | PASS | 28 |
| 4 | one pump trips, 30 min | PASS | 20 |
| 5 | stuck flapper | PASS | 22 |
| 6 | event-time invariance | PASS | 104 |

Rung 2 solves the same steady state as rung 1, starting from `stream.composition.guess_steady_state`
instead of the ballpark guess; its pass conditions are those of rung 1.

Rungs 1, 3, 4 and 5 also run as slow tests of `stream.examples.lofa`.
Rung 6 and the sweep are benchmark-only.
