# LOFA benchmark

A loss-of-flow accident (LOFA) in an MTR-like research-reactor loop, used as an end-to-end
solver-robustness benchmark. The loop has a pump, flywheel and heat exchanger on the return
leg, a heated `ChannelAndContacts` + `Fuel` plate cooled downward, and a flapper bypass leg.
Pump trip → flywheel coastdown → the flapper opens near zero flow → buoyancy reverses the
channel into natural circulation.

On STREAM v1.1.2 this scenario only ran with three hand workarounds: the flapper was opened
manually at a pre-computed time instead of through the event machinery, the whole run used a
loose `atol=1e-1`, and the plate carried 1 kW instead of the intended 83.6 kW. The benchmark
is therefore a graded ladder. Stages B and C keep the workarounds and guard against
regressions; every other stage removes one workaround.

## Stages

| Stage | What it removes | On v1.1.2 |
|---|---|---|
| A | Expert guess → a uniform "ballpark" guess, at 83.6 kW | **CRASH** — scipy `hybr` stalls ("no improvement") |
| B | *(nothing — expert guess, steady solve at 1 kW)* | **PASS** — regression guard |
| C | *(nothing — choreographed transient at `atol=1e-1`)* | **PASS** — regression guard |
| D | Manual pre-open → the natural event path; the opening time must not depend on the output grid | **FAIL** — `t_open` latches to output-grid points (0.082 s spread across three grids) |
| E | Loose tolerance → `atol=rtol=1e-6` | **TIMEOUT** — wedged for more than 20 minutes |
| F | 1 kW → 83.6 kW (expert guess + choreographed transient) | **CRASH** — the transient dies at t ≈ 15.9 s in the coastdown |
| G | One channel → the general multichannel LOFA below | **FAIL** — steady state solves, the coastdown dies at t ≈ 89.6 s, before the flapper opens (~101 s) |

At 1 kW the loop is nearly isothermal (ΔT ≈ 0.5 °C), so any non-smoothness in the
correlations is microscopic. Stages A, D, E and F discriminate only because they raise the
power, tighten the tolerance, or demand invariance.

## Stage G — the general multichannel LOFA

Four parallel channels between shared plena, each isolating one physics axis:

- **Hot** (2 mm gap, 84 kW) and **Warm** (2 mm, 33.6 kW) share geometry, so the hot channel
  must reverse first; the ordering is asserted.
- **Wide** (3 mm, 50 kW) varies geometry and heat together (a different Re/Gr regime); its
  reversal time is recorded, not asserted.
- **Bypass** (4 mm, unheated plain `Channel`) is a pressure-balance passenger and, with the
  flapper leg, the natural-circulation downcomer.

At pump trip every plate switches to a decay-heat curve, P(t) = P₀·0.066·(t+0.1)^−0.2. The
state vector has 555 variables. The staggered reversals force recirculation between channels
through the shared junctions, which exercises junction mixing, events and regime switches at
once. Sub-stages are recorded separately: wiring and gravity-loop closure, the expert steady
state at 167.6 kW, the scram transient, and the staggered-reversal checks.

The target physics is reachable on v1.1.2 when the flapper is pre-opened early: the run then
completes 2500 s, with the hot channel reversing at t ≈ 1000 s, warm ≈ 1500 s, wide ≈ 2000 s,
and the bypass staying a downcomer. With the physically timed opening, v1.1.2 dies during the
closed-flapper coastdown, so stage G measures the solver, not the physics.

## Running

```bash
conda run -n stream-env python benchmarks/lofa/run.py --all        # full ladder
conda run -n stream-env python benchmarks/lofa/run.py --stage D    # one stage
```

Each stage runs in its own subprocess with a timeout, because some baseline failures loop
forever. A stage writes its record to `results/<stage>.json`; `--all` also appends a dated
table, keyed to `git describe`, to `RESULTS.md`. Both are local outputs and are not tracked.

The ladder is not a pytest suite on purpose: its stages are expected to fail on the baseline.
Each defect it exposes is pinned by a unit regression test under `tests/`.
