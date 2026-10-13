"""The three loss-of-flow scenarios and the record each one leaves behind."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import pairwise

import numpy as np
import pandas as pd

from stream.aggregator import Aggregator, Solution
from stream.calculations.channel import SaturationReachedError
from stream.examples.lofa.params import CORE, HEATED, Params, nominal_flows
from stream.examples.lofa.system import operating_point
from stream.jacobians import DAE_jacobian
from stream.substances import light_water

REVERSAL_THRESHOLD = -1e-3
VALVE_SHUT_FRACTION = 0.01
SCENARIOS = ("both", "one", "stuck")
DEFAULT_PARAMS = Params()


@dataclass
class Record:
    """Outcome of one scenario run: ``status`` is ``completed``, ``saturation`` or ``failed``.

    A valve's shut time ``t_valve_A`` or ``t_valve_B`` is the first output sample at which that
    leg's flow is below 1 % of the nominal leg flow, so a leg still coasting forward counts as shut.
    ``margin`` is the saturation temperature at the reference pressure at the core top minus the
    sampled peak coolant temperature; the saturation stop uses the local pressure, so it can fire at
    a slightly higher local temperature than ``margin`` suggests. Reversal and valve times are
    quantised to the output grid, and the sampled peak moves with the grid (about 0.01 °C between
    the 901- and 3601-point grids). ``t_half`` is the first output time at which the riser flow is
    below half its initial value, counted from t=0 and so including the ``t_pumps`` before the
    pumps trip; ``t_half_hand`` is the flywheel estimate of the time from the pump trip, inertia
    times nominal leg flow over the solved pump head, in seconds.

    Attributes
    ----------
    status: str
        ``completed``, ``saturation`` or ``failed``.
    error: str
        The error that ended a run early, empty for a completed run.
    residual_entry: float
        Norm of the thermal-hydraulic residual at the steady-state guess, in the mixed units of the residual.
    residual_steady: float
        Norm of the full system's residual at the operating point, in the mixed units of the residual.
    head: float
        Solved head of pump A at the operating point, Pa.
    mdot_total: float
        Riser flow at the operating point, the core total, kg/s.
    mdot_hot: float
        Per-plate flow of the hot group at the operating point, kg/s.
    mdot_normal: float
        Per-plate flow of the normal group at the operating point, kg/s.
    mdot_bypass: float
        Per-plate flow of the bypass group at the operating point, kg/s.
    T_out_hot: float
        Coolant temperature in the hot channel's outlet cell at the operating point, °C.
    T_out_error_max: float
        Largest deviation of a heated group's outlet temperature from its energy balance, K.
    split_error_max: float
        Largest relative deviation of a group's flow share from the turbulent gap law.
    t_trip: float | None
        Time the low-flow trip latched, s.
    t_rods: float | None
        Time the rods start to insert, the trip time plus the trip delay, s.
    t_open: float | None
        Time the flapper opened, s.
    t_valve_A: float | None
        Shut time of leg A's check valve, s.
    t_valve_B: float | None
        Shut time of leg B's check valve, s.
    t_rev_hot: float | None
        Flow reversal time of the hot group, s.
    t_rev_normal: float | None
        Flow reversal time of the normal group, s.
    t_rev_low: float | None
        Flow reversal time of the low group, s.
    reversal_in_buoyancy_order: bool
        Whether the heated groups reversed in decreasing order of power factor over nominal flow.
    peak_hot: float
        Sampled peak coolant temperature of the hot channel, °C.
    peak_max: float
        Sampled peak coolant temperature over the heated channels, °C.
    margin: float
        Saturation temperature at the reference pressure minus ``peak_max``, K.
    final_hot: float
        Per-plate flow of the hot group at the last output time, kg/s.
    final_bypass: float
        Per-plate flow of the bypass group at the last output time, kg/s.
    final_pool: float
        Flow through the pool at the last output time, kg/s.
    final_leg_A: float
        Flow through pump leg A at the last output time, kg/s.
    final_leg_B: float
        Flow through pump leg B at the last output time, kg/s.
    final_flapper: float
        Flow through the flapper at the last output time, kg/s.
    pool_rise: float
        Rise of the mean pool coolant temperature over the run, K.
    t_last: float
        Last output time, s.
    t_half: float | None
        First output time the riser flow is below half its initial value, s.
    t_half_hand: float
        Flywheel estimate of the riser flow's half time from the pump trip, s.
    """

    status: str
    error: str
    residual_entry: float
    residual_steady: float
    head: float
    mdot_total: float
    mdot_hot: float
    mdot_normal: float
    mdot_bypass: float
    T_out_hot: float
    T_out_error_max: float
    split_error_max: float
    t_trip: float | None
    t_rods: float | None
    t_open: float | None
    t_valve_A: float | None
    t_valve_B: float | None
    t_rev_hot: float | None
    t_rev_normal: float | None
    t_rev_low: float | None
    reversal_in_buoyancy_order: bool
    peak_hot: float
    peak_max: float
    margin: float
    final_hot: float
    final_bypass: float
    final_pool: float
    final_leg_A: float
    final_leg_B: float
    final_flapper: float
    pool_rise: float
    t_last: float
    t_half: float | None
    t_half_hand: float

    @classmethod
    def empty(cls, status: str, error: str = "") -> Record:
        """A record with every time unknown (``None``), every flag false and every number NaN."""
        values = {}
        for f in fields(cls):
            if f.name.startswith("t_") and f.name not in ("t_last", "t_half_hand"):
                values[f.name] = None
            elif f.type == "bool":
                values[f.name] = False
            else:
                values[f.name] = np.nan
        return cls(**values | {"status": status, "error": error})


def flows(agr, K, refs, sol: Solution) -> pd.DataFrame:
    """Mass flow of every core group, the pool, the flapper, both pump legs and the riser at each output time."""
    comps = {k: c for k, c in refs["channels"].items()}
    comps |= {"pool": refs["pool"], "flapper": refs["flapper"], "A": refs["pumps"]["A"], "B": refs["pumps"]["B"], "riser": refs["hx"]}
    data = {k: agr.at_times(sol, K, K.component_edge(c)) for k, c in comps.items()}
    return pd.DataFrame({"t": np.asarray(sol.time)} | data)


def _first_time_below(df: pd.DataFrame, column: str, threshold: float) -> float | None:
    idx = np.flatnonzero(df[column].to_numpy() < threshold)
    return float(df["t"].iloc[idx[0]]) if len(idx) else None


def reversal_times(df: pd.DataFrame) -> dict[str, float | None]:
    """First output time each heated group's flow drops below ``REVERSAL_THRESHOLD``, ``None`` if it never does."""
    return {k: _first_time_below(df, k, REVERSAL_THRESHOLD) for k in HEATED}


def buoyancy_ratios(p: Params) -> dict[str, float]:
    """Power factor over nominal per-plate flow of each heated group."""
    nominal = nominal_flows(p)
    return {k: float(CORE[k].factor / nominal[k]) for k in HEATED}


def _in_buoyancy_order(rev: dict, p: Params) -> bool:
    ratios = buoyancy_ratios(p)
    ordered = sorted(HEATED, key=lambda k: -ratios[k])
    times = [rev[k] for k in ordered]
    if any(t is None for t in times):
        return False
    return all(a <= b for a, b in pairwise(times))


def _transient_metrics(agr, K, refs, sol: Solution, p: Params) -> dict:
    df = flows(agr, K, refs, sol)
    rev = reversal_times(df)
    peaks = {k: float(np.max(agr.at_times(sol, refs["channels"][k], "T_cool"))) for k in HEATED}
    trip, flapper = refs["trip"], refs["flapper"]
    leg_nominal = float(df["riser"].iloc[0]) / 2
    pool = refs["pool"]
    pool_rise = np.nan
    if "T_cool" in pool.variables:
        pool_T = agr.at_times(sol, pool, "T_cool")
        pool_rise = float(np.mean(pool_T[-1]) - np.mean(pool_T[0]))
    out = {
        "t_trip": float(trip.t_trip) if np.isfinite(trip.t_trip) else None,
        "t_rods": float(trip.t_rods) if np.isfinite(trip.t_trip) else None,
        "t_open": float(flapper.t_open) if np.isfinite(flapper.t_open) else None,
        "t_valve_A": _first_time_below(df, "A", VALVE_SHUT_FRACTION * leg_nominal),
        "t_valve_B": _first_time_below(df, "B", VALVE_SHUT_FRACTION * leg_nominal),
        "reversal_in_buoyancy_order": _in_buoyancy_order(rev, p),
        "peak_hot": peaks["hot"],
        "peak_max": max(peaks.values()),
        "margin": float(light_water.sat_temperature(refs["p_ref"])) - max(peaks.values()),
        "final_hot": float(df["hot"].iloc[-1]),
        "final_bypass": float(df["bypass"].iloc[-1]),
        "final_pool": float(df["pool"].iloc[-1]),
        "final_leg_A": float(df["A"].iloc[-1]),
        "final_leg_B": float(df["B"].iloc[-1]),
        "final_flapper": float(df["flapper"].iloc[-1]),
        "pool_rise": pool_rise,
        "t_last": float(sol.time[-1]),
        "t_half": _first_time_below(df, "riser", 0.5 * float(df["riser"].iloc[0])),
    }
    out |= {f"t_rev_{k}": rev[k] for k in HEATED}
    return out


def _partial_solution(e: BaseException) -> Solution | None:
    t, y = getattr(e, "t", None), getattr(e, "y", None)
    if t is None or y is None or np.ndim(y) != 2 or len(np.atleast_1d(t)) == 0:
        return None
    return Solution(np.asarray(t), np.asarray(y))


def smooth_down(t: float, t0: float, w: float) -> float:
    """Cosine ramp from 1 at ``t0`` to 0 at ``t0 + w``, flat outside."""
    return 1.0 if t <= t0 else (0.0 if t >= t0 + w else 0.5 * (1 + np.cos(np.pi * (t - t0) / w)))


def _ramped_head(head: float, t0: float, width: float):
    return lambda t: head * smooth_down(t, t0, width)


def start_transient(agr: Aggregator, refs: dict, y: np.ndarray, p: Params, scenario: str) -> None:
    """Switch both pumps from imposed flow to imposed head at their solved heads, and ramp
    the tripping ones to zero over ``[t_pumps, t_pumps + ramp_width]``."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if scenario == "stuck" and not p.flapper_stuck:
        raise ValueError("the stuck-flapper scenario needs Params(flapper_stuck=True)")
    st = agr.save(y)
    tripping = {"both": "AB", "stuck": "AB", "one": "B"}[scenario]
    for leg, pump in refs["pumps"].items():
        head = float(np.asarray(st[pump.name]["pressure"]))
        if leg in tripping:
            agr.funcs.setdefault(pump, {})["pressure"] = _ramped_head(head, p.t_pumps, p.ramp_width)
        else:
            agr.funcs.setdefault(pump, {})["pressure"] = head


def transient(agr: Aggregator, y: np.ndarray, *, t_end: float, n: int, atol_rel: float, rtol: float) -> Solution:
    """Integrate from ``y`` over ``[0, t_end]`` at ``n`` output times. Algebraic variables are
    left out of the local error test: the Kirchhoff ``mdot2`` rows are the time derivative of
    flows an algebraic law pins (index 2), and holding them to the error test stalls the step
    at the flapper opening at tight tolerance."""
    return agr.solve(y, time=np.linspace(0.0, t_end, n), jacfn=DAE_jacobian(agr),
                     atol=agr.scaled_atol(atol_rel), rtol=rtol, max_steps=1_000_000,
                     exclude_algvar_from_error=True)


def _effective(p: Params, scenario: str) -> Params:
    """``p`` as the scenario runs it: the stuck-flapper scenario forces ``flapper_stuck``."""
    return replace(p, scenario=scenario, flapper_stuck=True) if scenario == "stuck" else replace(p, scenario=scenario)


def _steady_fields(rec: Record, agr: Aggregator, K, refs: dict, y: np.ndarray) -> None:
    chk = refs["th_checks"]
    st = agr.save(y)
    rec.residual_entry = refs["th_residual_entry"]
    rec.residual_steady = float(np.linalg.norm(agr.compute(y, 0.0)))
    rec.head = chk["heads"]["A"]
    rec.mdot_total = float(np.asarray(st[K.name][K.component_edge(refs["hx"])]))
    inertia = refs["params"].inertia * refs["params"].inertia_scale
    rec.t_half_hand = inertia * (rec.mdot_total / 2) / rec.head
    rec.mdot_hot, rec.mdot_normal, rec.mdot_bypass = chk["mdot"]["hot"], chk["mdot"]["normal"], chk["mdot"]["bypass"]
    rec.T_out_hot = float(np.asarray(st[refs["channels"]["hot"].name]["T_cool"])[-1])
    rec.T_out_error_max, rec.split_error_max = chk["T_out_error_max"], chk["split_error_max"]


def _run(p: Params, scenario: str, *, t_end: float, n: int, atol_rel: float, rtol: float, guess_kind: str,
         seed: int, band: float) -> tuple[Solution, Record]:
    """Operating point, then the scenario's transient. Reaching saturation ends the run with status
    ``saturation`` and the trajectory up to the stop; any other exception raises; a run the solver
    ends early without raising returns status ``failed``."""
    p = _effective(p, scenario)
    agr, K, refs, y = operating_point(p, guess_kind=guess_kind, seed=seed, band=band)
    rec = Record.empty("failed")
    _steady_fields(rec, agr, K, refs, y)
    start_transient(agr, refs, y, p, scenario)
    try:
        sol = transient(agr, y, t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol)
    except SaturationReachedError as e:
        sol = _partial_solution(e)
        if sol is None:
            raise
        rec.status, rec.error = "saturation", f"{type(e).__name__}: {e}"
    else:
        rec.status = "completed" if sol.time[-1] >= t_end * (1 - 1e-9) else "failed"
        if rec.status == "failed":
            rec.error = f"Solution ended at t={sol.time[-1]} before t_end={t_end}"
    for k, v in _transient_metrics(agr, K, refs, sol, p).items():
        setattr(rec, k, v)
    return sol, rec


def both_pumps(p: Params = DEFAULT_PARAMS, *, t_end: float = 3600.0, n: int = 1801, atol_rel: float = 1e-4, rtol: float = 1e-4,
               guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3) -> tuple[Solution, Record]:
    """Both pumps trip at ``p.t_pumps``: coastdown, low-flow trip, flapper opening, flow
    reversal into natural circulation through the pool."""
    return _run(p, "both", t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol, guess_kind=guess_kind, seed=seed, band=band)


def one_pump(p: Params = DEFAULT_PARAMS, *, t_end: float = 1800.0, n: int = 901, atol_rel: float = 1e-4, rtol: float = 1e-4,
             guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3) -> tuple[Solution, Record]:
    """Pump B trips, pump A keeps running: its check valve shuts and the flapper stays closed."""
    return _run(p, "one", t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol, guess_kind=guess_kind, seed=seed, band=band)


def stuck_flapper(p: Params = DEFAULT_PARAMS, *, t_end: float = 3600.0, n: int = 1801, atol_rel: float = 1e-4, rtol: float = 1e-4,
                  guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3) -> tuple[Solution, Record]:
    """Both pumps trip with the flapper stuck shut: no natural-circulation path, the hot
    channel reaches saturation and the run stops there."""
    return _run(p, "stuck", t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol, guess_kind=guess_kind, seed=seed, band=band)
