# LOFA benchmark

A loss-of-flow accident (LOFA) in an MTR-like research-reactor loop, used as an end-to-end
solver-robustness benchmark. The loop has a pump, flywheel and heat exchanger on the return
leg, a heated `ChannelAndContacts` + `Fuel` plate cooled downward, and a flapper bypass leg.
The pump trips, the flywheel coasts down, the flapper opens near zero flow, and buoyancy
reverses the channel into natural circulation.

On STREAM v1.1.2 this scenario only ran with three hand workarounds: the flapper was opened
by hand at a precomputed time instead of through the event machinery, the whole run used a
loose `atol=1e-1`, and the plate carried 1 kW instead of the intended 83.6 kW. The benchmark
is therefore a graded ladder, run in order. Stages B and C keep the workarounds and guard against
regressions; every other stage removes one workaround or widens the system.

## Stages

| Stage | What it removes | v1.1.2 | v1.1.2 with the solver and component fixes |
|---|---|---|---|
| `A_ballpark_guess` | The expert guess: a uniform "ballpark" guess, at 83.6 kW | **CRASH**: scipy `hybr` gives up at t = 0, "not making good progress" | **CRASH**, same |
| `B_expert_steady` | *(nothing: expert guess, steady solve at 1 kW)* | **PASS**: 0.543 kg/s, outlet 40.44 °C | **PASS**, same to 1e-13 |
| `C_loose_transient` | *(nothing: choreographed transient at `atol=1e-1`)* | **PASS**: opens at 23.6 s, reversed by 400 s, peak wall 72.8 °C | **PASS**, peak wall 72.9 °C |
| `D_event_timing` | The manual pre-open: the natural event path, whose opening time must not depend on the output grid | **FAIL**: `t_open` latches at 23.812 / 23.836 / 23.754 s on three grids, a 0.082 s spread against a 0.02 s gate | **FAIL**, same latched times |
| `E_tight_tolerance` | The loose tolerance: `atol=rtol=1e-6` | **TIMEOUT**: killed after 20 minutes | **TIMEOUT** |
| `F_full_power` | The 1 kW: 83.6 kW with the expert guess and the choreographed transient | steady state passes; **CRASH** at t = 15.9 s in the coastdown (repeated convergence test failures) | **CRASH**, same time |
| `G_multichannel` | The single channel: the general multichannel LOFA below | steady state passes (hot 0.564, warm 0.552, wide 1.09, bypass 1.75 kg/s); **FAIL**: the closed-flapper coastdown dies at t = 89.6 s, before the estimated opening at 101 s | **FAIL**, same time |

Measured on 2026-10-04 with `run.py --all`. The first column is `dev` at tag `v1.1.2` (`407fb4a`). The
second is the same tree with the branches `fix/silent-solver-failures` (`1331dd1`) and
`fix/component-correlation-correctness` (`e0a85eb`) merged in. Every stage ends the same way and
at the same time in both runs; the steady states agree to 1e-13 and the transient peaks move by
less than 0.15 °C, inside `atol=1e-1`. Each run takes about 20 minutes, almost all of it stage E's
timeout.

At 1 kW the loop is nearly isothermal (ΔT ≈ 0.5 °C), so any non-smoothness in the
correlations is microscopic. Stages A, D, E and F discriminate only because they raise the
power, tighten the tolerance, or demand invariance.

## Stage G, the general multichannel LOFA

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

## Running

```bash
conda run -n stream-env python benchmarks/lofa/run.py --all        # full ladder
conda run -n stream-env python benchmarks/lofa/run.py --stage D_event_timing    # one stage
conda run -n stream-env python benchmarks/lofa/run.py --report     # rebuild the table from results/
```

Each stage runs in its own subprocess with a timeout, because some failures loop forever. A
stage writes its record to `results/<stage>.json`; `--all` also appends a dated table, keyed to
`git describe`, to `RESULTS.md`. Both are local outputs and are not tracked.

The ladder is not a pytest suite on purpose: its stages are expected to fail until the solver
copes with them. Each defect it exposes is pinned by a unit regression test under `tests/`.
