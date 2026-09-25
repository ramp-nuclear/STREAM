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

| Stage | What it removes | On v1.1.2 | Now |
|---|---|---|---|
| A | Expert guess → a uniform "ballpark" guess, at 83.6 kW | **CRASH** — scipy `hybr` stalls ("no improvement") | **PASS** |
| B | *(nothing — expert guess, steady solve at 1 kW)* | **PASS** — regression guard | **PASS**, same result |
| C | *(nothing — choreographed transient at `atol=1e-1`)* | **PASS** — regression guard | **PASS**, same result |
| D | Manual pre-open → the natural event path; the opening time must not depend on the output grid | **FAIL** — `t_open` latches to output-grid points (0.082 s spread across three grids) | **PASS** — 4.9e-5 s spread at `atol=rtol=1e-6` (gate: 0.02 s) |
| E | Loose tolerance → `atol=rtol=1e-6` | **TIMEOUT** — wedged for more than 20 minutes | **PASS** in about 2 minutes |
| F | 1 kW → 83.6 kW with a ramped scram, `regime_dependent` friction and the natural event path | **CRASH** — the transient dies at t ≈ 15.9 s in the coastdown | **PASS** — flapper opens at 25.4 s, reversal at 34 s, peak coolant 112.8 °C (7.5 °C below saturation) |
| G | One channel → the general multichannel LOFA below | **FAIL** — the coastdown dies at t ≈ 89.6 s, before the flapper opens | **PASS** — reaches 2500 s; reversals at 816 / 1176 / 1508 s (hot / warm / wide); peak 107.1 °C (13.2 °C margin) |

At 1 kW the loop is nearly isothermal (ΔT ≈ 0.5 °C), so any non-smoothness in the
correlations is microscopic. Stages A, D, E and F discriminate only because they raise the
power, tighten the tolerance, or demand invariance.

Stage D runs its three output grids at both loose and tight tolerance. Only the tight spread
gates: the loose spread (0.028 s) is ordinary `rtol=1e-3` divergence between the runs, not
event error, and is reported for information.

## Stage G — the general multichannel LOFA

Four parallel channels between shared plena, each isolating one physics axis:

- **Hot** (2 mm gap) and **Warm** (2 mm) share geometry but not power, so the hot channel
  must reverse first; the ordering is asserted.
- **Wide** (3 mm) varies geometry and heat together (a different Re/Gr regime); its reversal
  time is recorded, not asserted.
- **Bypass** (4 mm, unheated plain `Channel`) is a pressure-balance passenger and, with the
  flapper leg, the natural-circulation downcomer, which must keep flowing downward.

Full power is 84 / 33.6 / 50 kW. At full power the hot channel crosses saturation on the
reversal spike, which is outside what the single-phase models cover, so the stage runs at 70 %
(`GEN_POWER_SCALE`) with the `regime_dependent` friction law. At t = 3 s a 0.5 s ramp takes the
pump head to zero and blends every plate onto a decay-heat curve,
P(t) = P₀·0.066·(t+0.1)^−0.2. The state vector has 555 variables.

The steady state is solved from the ballpark guess through `solve_steady`'s automatic
fallback, and the transient uses the natural event path: nothing is choreographed. The stage
passes when the run reaches 2500 s, the three heated channels have all reversed, the hot
channel reversed before the warm one, the bypass is still a downcomer, and the hot coolant
never reached saturation. The staggered reversals force recirculation between channels through
the shared junctions, which exercises junction mixing, events and regime switches at once.

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
