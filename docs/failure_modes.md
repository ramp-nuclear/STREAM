# Failure modes: reading a STREAM solve that died

This is the long-form companion to the one-line hints STREAM prints at its raise
sites. It is organised by the *journey* a failure takes you on — a steady solve
that will not converge, a transient that dies mid-run, a natural-circulation loop
that misbehaves — and it ends with a post-mortem toolkit that ties the machinery
together. Everything below refers to APIs that exist today; the point is to save
you the archaeology.

Two facts frame all of it:

- **STREAM raises on failure, it never returns a broken answer.** A returned
  `Solution` is either `COMPLETED` or legitimately `STOPPED` (`Solution.status`,
  `Solution.completed`, `Solution.t_stop`). There is no `FAILED` status — a
  failure is an exception, and the exception carries the state it died on.
- **Every STREAM exception derives from `stream.errors.StreamError`.** So
  `except StreamError` is the one-stop catch; the specific types
  (`TransientRuntimeError`, `AlgRuntimeError`, `SaturationReachedError`,
  `DomainValidityError`, `GravityMismatchError`, `StreamConstructionError`, …)
  still catch under their old bases (`RuntimeError`/`ValueError`) too.

---

## 1. Steady-solve failures

### The globalize cascade

`Aggregator.solve_steady(guess, *, globalize='auto', scales=None, fallback_ptc=True)`
runs a *cascade* of solvers, each a rung. Under `globalize='auto'` the rungs are:

1. **scipy hybr** — `scipy.optimize.root`, started from your `guess`. On success
   it returns exactly that solver's result.
2. **scaled_newton** — an equilibrated, Armijo-damped Newton, started from `guess`.
   The damping is the load-bearing cure for far/ballpark guesses where hybr stalls.
3. **pseudo_transient** — implicit-Euler pseudo-transient continuation, started
   from `guess` (only if `fallback_ptc=True`).
4. **scaled_newton polish** — a final Newton started *from pseudo_transient's
   iterate*, not from `guess`.

`globalize=True` skips straight to rung 2; `globalize=False` runs rung 1 only —
a raw `AlgRuntimeError` from hybr with no fallback.

If **every** rung fails, `solve_steady` raises a single `AlgRuntimeError` that
lists each rung in order — its name, start point, first-line outcome and final
`‖F‖` — and carries:

- `err.rungs` — a tuple of `(rung_name, start_vector, outcome, ‖F‖)` records;
- `err.y` — the last rung's iterate (the best state reached);
- notes attached by the aggregator's failure enrichment (see section 2, note families).

### Guess or physics? The honest discriminator

There is **no automatic discriminator** — STREAM will not tell you whether a stuck
cascade means "bad guess" or "broken model", because no honest test exists. The
recipe is manual:

- **Re-solve from several different guesses** (e.g. an expert guess, a uniform
  guess, a perturbed guess).
- If the failure is **the same** each time *and* `agr.worst_residuals(err.y)`
  points at the same rows with a stable magnitude, the problem is almost certainly
  **physics or topology** — an unclosed loop, a missing supplier, an
  under-determined system.
- If the failure is **guess-dependent** (converges from some starts, not others),
  you are fighting a **basin** problem — start closer, or lean on the cascade
  (`globalize=True`).

`agr.worst_residuals(y, t=0.0, *, n=5, scales=None)` returns the `n` least-converged
rows as `(Location, scaled_residual)` pairs, ranked by `|F/typ|` — **scale-aware on
purpose**. Ranking the raw residual is unit-dominated (a pressure residual in Pa
dwarfs a temperature residual in K by units alone); dividing by the nominal
magnitudes `typ` gives the solver's own view of which equation is worst.

### `err.y` post-mortems

`err.y` is the failure iterate. Bridge it into readable domain terms:

```python
try:
    y = agr.solve_steady(guess)
except AlgRuntimeError as e:
    for loc, res in agr.worst_residuals(e.y, n=5):
        print(loc.variable, "of", loc.name, "cell", loc.cell, "->", res)
    state = agr.state_from(e.y)          # 1-D iterate -> State
```

`agr.locate_nonfinite(e.y)` names every NaN/inf entry of the iterate; `agr.locate(row)`
is the raw row → `Location(calculation, name, variable, cell)` inverse of `var_index`.

---

## 2. Transient deaths

### The IDA flag table

When the DAE (SUNDIALS IDA) backend fails, the `TransientRuntimeError` message is
`IDA <symbol>(<flag>): <backend message> — <STREAM meaning>`, and it carries
`err.flag` (the numeric SUNDIALS code) and `err.symbol` (the `IDA_*` name) — both
are how the SUNDIALS troubleshooting docs are indexed. The table below is
reproduced verbatim from `stream.solvers._IDA_STATUS`:

| flag | symbol | STREAM meaning |
|---|---|---|
| −1 | `IDA_TOO_MUCH_WORK` | mxsteps internal steps taken before an output time — the step collapsed as the system stiffened (e.g. approaching bulk Tsat / an SCB switch) |
| −2 | `IDA_TOO_MUCH_ACC` | the requested tolerance is unreachable at the system scale (atol/rtol too tight) |
| −3 | `IDA_ERR_FAIL` | repeated local error-test failures drove the step down to hmin (a stiff transient) |
| −4 | `IDA_CONV_FAIL` | the modified-Newton corrector could not converge and the step fell to hmin — the canonical LOFA death past ONB |
| −5 | `IDA_LINIT_FAIL` | — |
| −6 | `IDA_LSETUP_FAIL` | the linear solver's setup (Jacobian factorization) failed unrecoverably (a near-singular reversal Jacobian) |
| −7 | `IDA_LSOLVE_FAIL` | the linear solver's solve stage failed unrecoverably |
| −8 | `IDA_RES_FAIL` | the residual function (compute) raised or returned NaN inside IDA (a property-domain excursion) |
| −9 | `IDA_REP_RES_ERR` | the residual was repeatedly non-finite near a domain edge |
| −10 | `IDA_RTFUNC_FAIL` | a user event_margin/rootfn failed (but this usually surfaces as SystemError, not this flag) |
| −11 | `IDA_CONSTR_FAIL` | the inequality constraints option could not be met |
| −12 | `IDA_FIRST_RES_FAIL` | compute(y0, t0) was non-finite — a bad guess that is already unphysical |
| −13 | `IDA_LINESEARCH_FAIL` | — |
| −14 | `IDA_NO_RECOVERY` | the consistent-IC solve (IDACalcIC) could not recover — an IC-stage failure |
| −15 | `IDA_NLS_INIT_FAIL` | — |
| −16 | `IDA_NLS_SETUP_FAIL` | — |
| −17 | `IDA_NLS_FAIL` | — |
| −20 | `IDA_MEM_NULL` | — |
| −21 | `IDA_MEM_FAIL` | — |
| −22 | `IDA_ILL_INPUT` | malformed options / mass / algebraic_vars_idx mismatch |
| −23 | `IDA_NO_MALLOC` | — |
| −24 | `IDA_BAD_EWT` | a zero in the error-weight vector (a scale or atol of 0) |
| −25 | `IDA_BAD_K` | — |
| −26 | `IDA_BAD_T` | — |
| −27 | `IDA_BAD_DKY` | — |
| −28 | `IDA_VECTOROP_ERR` | — |
| −99 | `IDA_UNRECOGNIZED_ERROR` | — |

The `—` rows carry a symbol but no STREAM-specific note; they are rare and mostly
mean malformed input or an internal SUNDIALS state. The everyday one is `−4`
(`IDA_CONV_FAIL`) — the canonical stiff death, usually past onset of nucleate
boiling. The ODE (`solve_ivp`) and steady (`scipy.optimize.root`) backends have the
smaller `_IVP_STATUS` / `_HYBR_STATUS` tables in the same module.

### The saturation journey

STREAM's channel model is single-phase forced convection plus subcooled boiling —
valid up to *bulk* saturation. Past it the flow is two-phase (not modelled) and the
wall `h = q/ΔT` term develops a finite-time pole, so IDA dies with a cryptic `−4`
unless you stop first. A `ChannelAndContacts` built with `stop_at_saturation=True`
raises `SaturationReachedError(channel, cells, T_bulk, Tsat)` at the boundary — in
domain terms, with the valid pre-crossing trajectory attached to `err.t`/`err.y`.

To find saturation without letting the solve run into it:

- `analysis.thresholds.first_saturation_crossing(result, agg, *, times=None)` →
  `(time_or_None, crossings)` or `None`, where each `SaturationCrossing` names the
  channel and its at/above-Tsat cells. `result` may be a `State`, a
  `StateTimeseries`, a `Solution`, or a raw vector/trajectory (e.g. a caught
  `e.y`).
- `analysis.thresholds.raise_on_saturation(result, agg, *, times=None, note="")`
  raises `SaturationReachedError` for the worst/earliest crossing, or no-ops. Handy
  after a steady solve or on a caught failure's state.

### Trajectory on the exception

A transient death carries what it reached. `err.y` is the trajectory (2-D) plus
the failing partial segment; `err.t` its times. Bridge it:

```python
try:
    sol = agr.solve(y0, time, eq_type="DAE")
except TransientRuntimeError as e:
    history = agr.state_from((e.t, e.y))     # 2-D (t, y) -> StateTimeseries
    last = history[max(history)]             # the failing row, as a State
```

`agr.state_from(obj)` is the documented bridge: a 1-D vector → `State`; a `(t, y2d)`
pair → `StateTimeseries` (identical to `agr.save(Solution(t, y))`); a `Solution` →
`StateTimeseries`. On an **IC-stage** DAE failure there is no trajectory, so `err.y`
is instead the single failing state IDA recorded (a note on the exception says so);
`agr.state_from(e.y)` still works.

### Enrichment notes: the four families

On any transient/steady failure the aggregator best-effort **adds notes** (never
edits the message, never masks the error) via `Aggregator._enrich_failure`. Read
them with `e.__notes__` or just let them print in the traceback. Four families:

1. **Non-finite locations** — every NaN/inf in the failure state (and its
   residual, when computable), named: `T_wall=nan at cell 3 of 'CC' (source: y)`.
2. **Worst scaled residuals** — the top-3 by `|F/typ|` (the same scale-aware view
   as `worst_residuals`).
3. **Saturation crossing** — if any channel crossed Tsat, which channel and cells,
   plus the `stop_at_saturation` / `first_saturation_crossing` hint.
4. **Fluid-domain violations** — a `T*` variable outside its fluid's declared
   validity range, or a negative absolute pressure (from `domain_report`, section 3).

---

## 3. Natural-circulation loops

### The two-HX gravity sandwich

A buoyancy-driven loop is built from `Gravity` legs whose density depends on the
advected temperature. Under forced flow each leg's inlet temperature (`Tin`) is the
upstream one; when the flow **reverses** (the whole point of natural circulation),
physics wants the *downstream* temperature, supplied as `Tin_minus`. The robust
construction sandwiches each gravity leg between two `HeatExchanger`s, so both the
forward and reversed density temperatures are **fixed-temperature boundaries**, not
state-dependent junction outlets. This sandwich is **load-bearing** — and a **silent
trap**: get it wrong and buoyancy quietly uses the hot outlet temperature under
reversal, reversing the loop's drive with no error.

Mind the sign too: `Gravity(disposition=…)` takes **positive = downward** (a
descending leg gives a positive hydrostatic `dp_out`); a climbing leg needs a
negative `disposition`. Getting the sign wrong silently reverses buoyancy.

### What `check_gravity_mismatch` does — and does not — see

`FlowGraph.check_gravity_mismatch(temperature=10.0, strategy=None, tol=1e-5, head=1.0)`
(a convenience method on the flow graph; the underlying free function
`stream.composition.subsystems.check_gravity_mismatch(kirchhoff, ...)` takes the
`Kirchhoff` explicitly) does two things:

- **A static, zero-flow consistency check.** It evaluates every `Δp` at `ṁ = 0` and
  raises `GravityMismatchError` if any loop's pressures do not sum to zero —
  usually a gravity-height bookkeeping error. Because it is evaluated at zero flow
  it is **direction-blind**.
- **The topology pass (warn-only).** When a natural-convection-intent pump is
  present (one imposing *neither* a nonzero head *nor* a nonzero flow), it
  classifies each gravity leg's *reversed*-flow
  temperature supplier. If that supplier is anything other than a fixed-temperature
  boundary (a `HeatExchanger`), it **warns** that under reversed/NC flow the leg's
  buoyancy will use the hot outlet temperature, and suggests the two-HX sandwich.

What it does **not** see: it is direction-blind at nonzero flow, and its topology
pass has a known limitation — a reverse-flow temperature wired from a *hot junction*
(rather than being absent) passes the classifier silently. Treat the warning as
necessary, not sufficient: for a loop that will reverse, verify each gravity leg's
reversed-flow temperature source yourself.

---

## 4. The `[ERROR]` lines on stderr

When IDA dies you may see raw `[ERROR]` lines printed to stderr, e.g.
`[ERROR] IDA ... At t = ... convergence test failed repeatedly`. These come from
the **SUNDIALS C layer**, which writes to stderr from C directly — Python cannot
intercept them. They are **not** a second, separate
failure: they duplicate what the translated `TransientRuntimeError` already carries
(`err.flag`, `err.symbol`, the message, the `_IDA_STATUS` meaning). Read the
Python exception; treat the `[ERROR]` lines as redundant C-side echo.

---

## 5. The post-mortem toolkit (worked recipe)

Put it together. Catch the failure, read what it already tells you, then drill in:

```python
from stream.analysis.debugging import debug_derivatives
from stream.analysis.thresholds import raise_on_saturation, domain_report
from stream.analysis.report import report

try:
    sol = agr.solve(y0, time, eq_type="DAE")
except StreamError as e:
    # 1. Read the notes the aggregator already attached (the four families).
    print("\n".join(getattr(e, "__notes__", [])))

    # 2. Locate the trouble in domain terms.
    print(agr.locate_nonfinite(e.y))           # named NaN/inf entries
    print(agr.worst_residuals(e.y, n=5))       # scale-aware worst equations

    # 3. Bridge the failure state into readable form.
    state = agr.state_from(e.y)                 # or state_from((e.t, e.y)) for a trajectory

    # 4. Inspect residuals with the solver's own (scaled) view.
    print(debug_derivatives(agr, e.y, scales="default"))

    # 5. Ask the domain checkers why it is invalid.
    raise_on_saturation(e.y, agr)              # SaturationReachedError, in domain terms
    print(domain_report(e.y, agr))             # out-of-validity / negative-pressure cells

    # 6. Sanity-check the wiring itself.
    report(agr, "terminal")                    # Unset / Set Externally / Missing columns
```

Notes on the tools:

- `debug_derivatives(agr, guess, scales=None)` accepts a raw vector directly (no
  hand-built `State` needed) and, with `scales="default"` (or a dict / `typ`
  array), returns residuals divided by `typ` — the solver's view. The default
  (`scales=None`) is the *raw*, unit-dominated view; prefer the scaled one when
  comparing across variables.
- `domain_report` / `raise_on_saturation` on a **raw array** append a caveat that
  the input may be a non-converged iterate — a checker verdict on a mid-iteration
  state is not a verdict on a solution. Pass a `Solution`/`State` when you have one.
- `report(agr)` prints a readable table whether or not you are in a notebook: with
  no live IPython frontend it falls back to the terminal renderer automatically
  (`report(agr, "terminal")` forces it). The `Missing` column is the one to watch —
  a non-empty `Missing` means the simulation is expected to crash.
