# Pool-reactor loss of flow

A loss-of-flow case for a generic materials-testing pool reactor: a plate core in downward
forced flow, a decay tank, two pump legs with flywheels and check valves, a heat exchanger, and a
flapper below the core that opens when the pumps coast down, so that the core goes over to
natural circulation with the pool as the heat sink. Point kinetics with temperature feedback and a
low-flow trip drive the power. The case ships with the library as `stream.examples.lofa`; this
directory runs it as a benchmark: a ladder of rungs with pass conditions, a stress sweep over its
parameters, and a physics audit of a solved transient against conservation laws.

## Commands

From the repository root, with the repository on `PYTHONPATH`:

```bash
python benchmarks/lofa_pool/run.py --all              # every rung, appends a dated table to RESULTS.md
python benchmarks/lofa_pool/sweeps.py --axis all --workers 10   # the stress sweep, one CSV per axis in results/
python benchmarks/lofa_pool/metrics.py --params '{"power_scale": 1.2}' --out row.json   # one scenario as a JSON row
```

`run.py --rung R` runs one rung in the current process. `physics_checks.py --case {case,top_flapper}`
audits a solved hour of the shipped loop, or of the second loop in `top_flapper.py` where the pump
return discharges into the pool surface and the flapper sits above the core.

## Rungs

Measured on the library at commit `1802387`.

| Rung | Title | Status | Wall (s) |
|---|---|---|---|
| 1 | steady, ballpark guess | PASS | 6 |
| 3 | both pumps trip, 1 h | PASS | 23 |
| 4 | one pump trips, 30 min | PASS | 16 |
| 5 | stuck flapper | PASS | 20 |
| 6 | event-time invariance | PASS | 91 |

Rung 2 is reserved for the steady state from the library's guess toolkit and is absent until that
library feature ships.

Rungs 1, 3, 4 and 5 also run as slow tests of `stream.examples.lofa`.
Rung 6 and the sweep are benchmark-only.
