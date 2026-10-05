"""Parametrised loss-of-flow case: four parallel plate channels between shared plena.

The hot, warm and wide channels are heated by their own fuel plates; the bypass is an
unheated passenger. A flapper leg opens on low pump flow, and the pump leg carries a
flywheel and a heat exchanger. At ``t_scram`` the pump head ramps to zero and reactor
power blends to decay heat. ``Params()`` reproduces the benchmark's multichannel case.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass, fields
from functools import partial

import numpy as np
import pandas as pd

from stream.aggregator import Aggregator, Solution
from stream.calculations import (
    Fuel,
    Gravity,
    HeatExchanger,
    Inertia,
    Junction,
    Kirchhoff,
    KirchhoffWDerivatives,
    Pump,
    Resistor,
    Solid,
)
from stream.calculations.channel import Channel, ChannelAndContacts, SaturationReachedError
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.composition import (
    FlowGraph,
    check_gravity_mismatch,
    flow_edge,
    guess_steady_state,
    seed_steady_state,
    symmetric_plate,
    uniform_x_power_shape,
    x_boundaries,
)
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.jacobians import ALG_jacobian, DAE_jacobian
from stream.physical_models.heat_transfer_coefficient import spl_htc, wall_heat_transfer_coeff
from stream.physical_models.pressure_drop import pressure_diff
from stream.physical_models.pressure_drop.friction import friction_factor, rectangular_laminar_correction
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.utilities import identity

TIN_DEFAULT = 40.0
DP0 = 35000.0
MDOT0 = 0.5
LENGTH, WIDTH = 0.7, 0.070
CLAD_N, FUEL_N = 2, 8
CLAD_W, MEAT_W = 0.0005, 0.0006
CLADDING = Solid(density=2700, specific_heat=900, conductivity=167)
FUEL_MEAT = Solid(density=3500, specific_heat=750, conductivity=40)
CHANNELS = ("hot", "warm", "wide", "bypass")
HEATED = ("hot", "warm", "wide")
GUESS_KINDS = ("toolkit", "seed", "ballpark", "expert", "perturbed")
REVERSAL_THRESHOLD = -1e-3


@dataclass(frozen=True)
class Params:
    """Operating point and modelling choices; ``powers_full`` are the hot, warm and wide
    plate powers (W) at ``power_scale=1``, ``gaps`` the hot, warm, wide and bypass gaps (m)."""

    power_scale: float = 0.70
    powers_full: tuple[float, float, float] = (84e3, 33.6e3, 50e3)
    gaps: tuple[float, float, float, float] = (0.002, 0.002, 0.003, 0.004)
    pump_head: float = DP0
    inertia: float = 3e5
    hx_outlet: float = TIN_DEFAULT
    p_ref: float = 2e5
    flapper_fraction: float = 0.3
    flapper_area: float = 1e-3
    friction: str = "regime"
    htc: str = "dittus"
    t_scram: float = 3.0
    ramp_width: float = 0.5
    z_n: int = 10
    stop_at_saturation: bool = True

    @property
    def powers(self) -> dict[str, float]:
        return dict(zip(HEATED, (self.power_scale * p for p in self.powers_full)))

    @property
    def gap(self) -> dict[str, float]:
        return dict(zip(CHANNELS, self.gaps))


def TSAT(p_ref: float) -> float:
    """Saturation temperature (°C) of light water at ``p_ref`` (Pa)."""
    return float(light_water.sat_temperature(p_ref))


def _pipe(gap: float) -> EffectivePipe:
    return EffectivePipe.rectangular(length=LENGTH, edge1=WIDTH, edge2=gap, heated_edge=WIDTH)


def _fuel(z_n: int, name: str) -> Fuel:
    shape = (z_n, FUEL_N + 2 * CLAD_N)
    meat = np.zeros(shape, dtype=bool)
    meat[:, CLAD_N:-CLAD_N] = True
    materials = np.empty(shape, dtype=object)
    materials[meat] = FUEL_MEAT
    materials[~meat] = CLADDING
    return Fuel(
        z_boundaries=np.linspace(0.0, LENGTH, z_n + 1),
        x_boundaries=x_boundaries(CLAD_N, FUEL_N, CLAD_W, MEAT_W),
        material=Solid.from_array(materials),
        meat_indices=meat,
        power_shape=uniform_x_power_shape(z_n, FUEL_N, CLAD_N, CLAD_W, MEAT_W, WIDTH),
        y_length=WIDTH,
        name=name,
    )


def _nominal_split(p: Params) -> dict[str, float]:
    w = {k: _pipe(g).area * np.sqrt(_pipe(g).hydraulic_diameter) for k, g in p.gap.items()}
    scale = MDOT0 * np.sqrt(p.pump_head / DP0) / w["hot"]
    return {k: scale * wk for k, wk in w.items()}


def _htc(p: Params, gap: float):
    if p.htc == "dittus":
        return wall_heat_transfer_coeff
    if p.htc == "regime":
        return partial(wall_heat_transfer_coeff,
                       h_spl=spl_htc("regime_dependent", re_bounds=(2000.0, 5000.0),
                                     aspect_ratio=gap / WIDTH, Lh=LENGTH))
    raise ValueError(f"htc must be 'dittus' or 'regime', got {p.htc!r}")


def _apply_friction(channel: Channel, p: Params, gap: float) -> None:
    if p.friction == "blasius":
        return
    if p.friction != "regime":
        raise ValueError(f"friction must be 'regime' or 'blasius', got {p.friction!r}")
    k_R = float(rectangular_laminar_correction(gap / WIDTH))
    f = friction_factor("regime_dependent", re_bounds=(2000.0, 5000.0), k_R=k_R)
    channel.pressure = partial(pressure_diff, fluid=channel.fluid, pipe=channel.pipe, dz=channel.dz, f=f)


def build(p: Params) -> tuple[Aggregator, Kirchhoff, dict]:
    """Build the system at ``p``; returns the aggregator, its Kirchhoff and component handles."""
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=p.pump_head, name="Pump")
    flywheel = Inertia(inertia=p.inertia, name="Flywheel")
    hx = HeatExchanger(outlet=p.hx_outlet, name="HX")
    gravity_ret = Gravity(fluid=light_water, disposition=-LENGTH, name="GravityReturn")
    gravity_flap = Gravity(fluid=light_water, disposition=LENGTH, name="GravityFlapper")
    resist_flap = Resistor(resistance=10.0, name="ResistFlap")
    split = _nominal_split(p)
    mdot_nominal = sum(split.values())
    flapper = Flapper(
        open_at_current=p.flapper_fraction * mdot_nominal, f=5.0, fluid=light_water,
        area=p.flapper_area, open_rate=1.0, stop_on_open=False,
        relaxation=continuously_differentiable_relaxation, name="Flapper",
    )
    z = np.linspace(0.0, LENGTH, p.z_n + 1)
    channels = {
        k: ChannelAndContacts(z_boundaries=z, fluid=light_water, pipe=_pipe(p.gap[k]),
                              h_wall_func=_htc(p, p.gap[k]), name=f"{k.capitalize()}Channel",
                              stop_at_saturation=p.stop_at_saturation)
        for k in HEATED
    }
    channels["bypass"] = Channel(z_boundaries=z, fluid=light_water, pipe=_pipe(p.gap["bypass"]), name="Bypass")
    for k, ch in channels.items():
        _apply_friction(ch, p, p.gap[k])
    fuels = {k: _fuel(p.z_n, f"{k.capitalize()}Fuel") for k in HEATED}
    fg = FlowGraph(
        *(flow_edge((j_top, j_bot), channels[k]) for k in CHANNELS),
        flow_edge((j_top, j_bot), flapper, gravity_flap, resist_flap),
        flow_edge((j_bot, j_top), pump, flywheel, hx, gravity_ret, ref_mdot_for=(flapper,)),
        inertial_comps=[flywheel, *channels.values()],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channels[k] for k in HEATED],
        funcs={flapper: {"t": identity}},
        reference_node=(j_top, p.p_ref),
    )
    check_gravity_mismatch(fg.kirchhoff)
    agr = fg.aggregator
    for k in HEATED:
        agr = agr + symmetric_plate(channels[k], fuels[k], funcs={fuels[k]: {"power": p.powers[k]}}).to_aggregator()
    refs = {"fg": fg, "pump": pump, "flywheel": flywheel, "hx": hx, "flapper": flapper, "channels": channels,
                "fuels": fuels, "powers": p.powers, "mdot_nominal": mdot_nominal, "split": split, "params": p}
    return agr, fg.kirchhoff, refs


def _ballpark(agr: Aggregator, K: Kirchhoff, p: Params) -> State:
    non_k = [n for n in agr.graph if not hasattr(n, "variables_by_type")]
    vec = np.zeros(len(K))
    vec[K.variables_by_type["mdot"]] = MDOT0
    vec[K.variables_by_type["abs_pressure"]] = p.p_ref
    return State.merge(
        State.uniform(non_k, p.hx_outlet, "T_cool", "T", "T_wall_left", "T_wall_right", "Tin"),
        State.uniform(non_k, 1e3, "h_left", "h_right"),
        State.uniform(non_k, 0.0, "pressure"),
        {K.name: K.save(vec)},
    )


def _expert(refs: dict, p: Params) -> State:
    channels, fuels, split = refs["channels"], refs["fuels"], refs["split"]
    hydraulic = refs["fg"].guess_steady_state(
        mdots={channels[k]: split[k] for k in CHANNELS}
              | {refs["pump"]: refs["mdot_nominal"], refs["flapper"]: 0.0},
        temperature=p.hx_outlet,
    )
    thermals = [
        symmetric_plate_steady_state(channels[k], fuels[k], mdot=split[k], p_abs=p.p_ref,
                                     power=p.powers[k], Tin=p.hx_outlet)
        for k in HEATED
    ]
    return State.merge(hydraulic, *thermals)


def guess(agr: Aggregator, K: Kirchhoff, refs: dict, kind: str, seed: int = 0, band: float = 0.3) -> np.ndarray:
    """A steady-state guess vector of one of ``GUESS_KINDS``; ``perturbed`` scales each entry
    of the toolkit guess by a uniform factor in ``[1 - band, 1 + band]`` drawn from ``seed``."""
    p = refs["params"]
    flows = {refs["pump"]: refs["mdot_nominal"]}
    if kind == "toolkit":
        return agr.load(guess_steady_state(agr, K, flows=flows))
    if kind == "seed":
        return agr.load(seed_steady_state(agr, K, flows=flows))
    if kind == "ballpark":
        return agr.load(_ballpark(agr, K, p))
    if kind == "expert":
        return agr.load(_expert(refs, p))
    if kind == "perturbed":
        y = agr.load(guess_steady_state(agr, K, flows=flows))
        rng = np.random.default_rng(seed)
        return y * (1 + rng.uniform(-band, band, size=y.shape))
    raise ValueError(f"guess kind must be one of {GUESS_KINDS}, got {kind!r}")


def steady(agr: Aggregator, y0: np.ndarray, **options) -> np.ndarray:
    """Steady state from ``y0`` with the analytic algebraic Jacobian."""
    return agr.solve_steady(y0, jac=ALG_jacobian(agr), **options)


def _decay_power(p0: float):
    return lambda t: p0 * 0.066 * (t + 0.1) ** -0.2


def _smooth_down(t: float, t0: float, w: float) -> float:
    return 1.0 if t <= t0 else (0.0 if t >= t0 + w else 0.5 * (1 + np.cos(np.pi * (t - t0) / w)))


def scram(agr: Aggregator, refs: dict, p: Params) -> None:
    """Ramp the pump head to zero and blend each plate's power to decay heat over
    ``[t_scram, t_scram + ramp_width]``."""
    t0, w = p.t_scram, p.ramp_width
    agr.funcs.setdefault(refs["pump"], {})["pressure"] = lambda t: p.pump_head * _smooth_down(t, t0, w)

    def blended(p0: float):
        dpf = _decay_power(p0)
        return lambda t: p0 if t <= t0 else _smooth_down(t, t0, w) * p0 + (1 - _smooth_down(t, t0, w)) * dpf(t - t0)

    for k in HEATED:
        agr.funcs[refs["fuels"][k]]["power"] = blended(p.powers[k])


def transient(agr: Aggregator, refs: dict, y_steady: np.ndarray, *, t_end: float = 2500.0, n: int = 1251,
              atol_rel: float = 1e-4, rtol: float = 1e-4) -> Solution:
    """Integrate from ``y_steady`` over ``n`` output times on ``[0, t_end]``."""
    return agr.solve(y_steady, time=np.linspace(0.0, t_end, n), jacfn=DAE_jacobian(agr),
                     atol=agr.scaled_atol(atol_rel), rtol=rtol, max_steps=1_000_000)


@dataclass
class Record:
    """Outcome of one :func:`run`: ``status`` is ``completed``, ``saturation`` or ``failed``
    (or a caller-defined label for rows made with :meth:`empty`)."""

    status: str
    error: str
    n_warnings: int
    warnings: str
    residual_entry: float
    residual_steady: float
    mdot_hot: float
    mdot_warm: float
    mdot_wide: float
    mdot_bypass: float
    mdot_pump: float
    T_out_hot: float
    t_open: float | None
    t_rev_hot: float | None
    t_rev_warm: float | None
    t_rev_wide: float | None
    peak_hot: float
    peak_warm: float
    peak_wide: float
    margin: float
    final_hot: float
    final_warm: float
    final_wide: float
    final_bypass: float
    t_last: float
    wall_s: float
    steady_dist: float

    @classmethod
    def empty(cls, status: str, error: str = "") -> Record:
        values = {f.name: None if f.name.startswith(("t_open", "t_rev")) else np.nan for f in fields(cls)}
        return cls(**values | {"status": status, "error": error, "n_warnings": 0, "warnings": ""})


def flows(agr: Aggregator, K: Kirchhoff, refs: dict, sol: Solution) -> pd.DataFrame:
    """Mass flows (kg/s, positive downward in the channels) at the solution times."""
    comps = refs["channels"] | {"flapper": refs["flapper"], "pump": refs["pump"]}
    data = {k: agr.at_times(sol, K, K.component_edge(c)) for k, c in comps.items()}
    return pd.DataFrame({"t": np.asarray(sol.time)} | data)


def temperatures(agr: Aggregator, refs: dict, sol: Solution, channel: str):
    """``(t, T_cool, T_wall_left)`` for one channel; ``T_wall_left`` is None for the bypass."""
    T_cool = agr.at_times(sol, refs["channels"][channel], "T_cool")
    fuel = refs["fuels"].get(channel)
    T_wall = agr.at_times(sol, fuel, "T_wall_left") if fuel is not None else None
    return np.asarray(sol.time), T_cool, T_wall


def peak_series(agr: Aggregator, refs: dict, sol: Solution) -> pd.DataFrame:
    """Hottest coolant cell (°C) of each heated channel at the solution times."""
    data = {k: np.max(agr.at_times(sol, refs["channels"][k], "T_cool"), axis=1) for k in HEATED}
    return pd.DataFrame({"t": np.asarray(sol.time)} | data)


def reversal_times(flows_df: pd.DataFrame) -> dict[str, float | None]:
    """First time each channel's flow falls below ``REVERSAL_THRESHOLD``, else None."""
    out = {}
    for k in CHANNELS:
        neg = np.flatnonzero(flows_df[k].to_numpy() < REVERSAL_THRESHOLD)
        out[k] = float(flows_df["t"].iloc[neg[0]]) if len(neg) else None
    return out


def _steady_metrics(agr: Aggregator, K: Kirchhoff, refs: dict, y: np.ndarray) -> dict:
    st = agr.save(y)
    comps = refs["channels"] | {"pump": refs["pump"]}
    out = {f"mdot_{k}": float(st[K.name][K.component_edge(c)]) for k, c in comps.items()}
    out["T_out_hot"] = float(np.asarray(st[refs["channels"]["hot"].name]["T_cool"])[-1])
    out["residual_steady"] = float(np.linalg.norm(agr.compute(y, 0.0)))
    return out


def _transient_metrics(agr: Aggregator, K: Kirchhoff, refs: dict, sol: Solution, p: Params) -> dict:
    fl = flows(agr, K, refs, sol)
    rev = reversal_times(fl)
    peaks = peak_series(agr, refs, sol)
    t_open = refs["flapper"].t_open
    out = {"t_open": float(t_open) if np.isfinite(t_open) else None, "t_last": float(sol.time[-1])}
    out |= {f"t_rev_{k}": rev[k] for k in HEATED}
    out |= {f"peak_{k}": float(peaks[k].max()) for k in HEATED}
    out |= {f"final_{k}": float(fl[k].iloc[-1]) for k in CHANNELS}
    out["margin"] = TSAT(p.p_ref) - max(out[f"peak_{k}"] for k in HEATED)
    return out


def _partial_solution(e: BaseException) -> Solution | None:
    t, y = getattr(e, "t", None), getattr(e, "y", None)
    if t is None or y is None or np.ndim(y) != 2 or len(np.atleast_1d(t)) == 0:
        return None
    return Solution(np.asarray(t), np.asarray(y))


def run(p: Params, *, guess_kind: str = "toolkit", seed: int = 0, band: float = 0.3, atol_rel: float = 1e-4,
        rtol: float = 1e-4, t_end: float = 2500.0, n: int = 1251, reference: np.ndarray | None = None) -> Record:
    """Build, solve the steady state, scram and integrate; never raises on a solver failure."""
    rec = Record.empty("failed")
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            agr, K, refs = build(p)
            y0 = guess(agr, K, refs, guess_kind, seed=seed, band=band)
            rec.residual_entry = float(np.linalg.norm(agr.compute(y0, 0.0)))
            y = steady(agr, y0)
            for k, v in _steady_metrics(agr, K, refs, y).items():
                setattr(rec, k, v)
            if reference is not None:
                rec.steady_dist = float(np.max(np.abs(y - reference)))
            scram(agr, refs, p)
            sol = transient(agr, refs, y, t_end=t_end, n=n, atol_rel=atol_rel, rtol=rtol)
            for k, v in _transient_metrics(agr, K, refs, sol, p).items():
                setattr(rec, k, v)
            if rec.t_last < t_end * (1 - 1e-9):
                rec.error = f"Solution ended at t={rec.t_last} before t_end={t_end} ({sol.status.name})"
            else:
                rec.status = "completed"
        except Exception as e:
            rec.status = "saturation" if isinstance(e, SaturationReachedError) else "failed"
            rec.error = f"{type(e).__name__}: {e}"
            partial_sol = _partial_solution(e)
            if partial_sol is not None:
                try:
                    for k, v in _transient_metrics(agr, K, refs, partial_sol, p).items():
                        setattr(rec, k, v)
                except Exception:
                    rec.t_last = float(np.asarray(partial_sol.time)[-1])
    rec.wall_s = time.perf_counter() - start
    rec.n_warnings = len(caught)
    rec.warnings = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught[:3])
    return rec


def _flatten_params(p: Params) -> dict:
    row = {}
    for name, value in asdict(p).items():
        if name == "powers_full":
            row |= {f"p_power_full_{k}": v for k, v in zip(HEATED, value)}
        elif name == "gaps":
            row |= {f"p_gap_{k}": v for k, v in zip(CHANNELS, value)}
        else:
            row[f"p_{name}"] = value
    return row


def record_to_row(p: Params, rec: Record, **extra) -> dict:
    """One flat table row: ``Params`` fields prefixed ``p_``, the record's fields, then ``extra``."""
    return _flatten_params(p) | asdict(rec) | extra


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def params_from_json(text: str) -> Params:
    """``Params`` from a JSON object of its fields; lists become tuples."""
    raw = json.loads(text)
    return Params(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one loss-of-flow case and write its row as JSON.")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--guess", default="toolkit", choices=GUESS_KINDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--band", type=float, default=0.3)
    parser.add_argument("--atol-rel", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=2500.0)
    parser.add_argument("--n", type=int, default=1251)
    parser.add_argument("--axis", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    p = params_from_json(args.params)
    rec = run(p, guess_kind=args.guess, seed=args.seed, band=args.band, atol_rel=args.atol_rel,
              rtol=args.rtol, t_end=args.t_end, n=args.n)
    row = record_to_row(p, rec, axis=args.axis, label=args.label, guess=args.guess, seed=args.seed,
                        band=args.band, atol_rel=args.atol_rel, rtol=args.rtol, t_end=args.t_end, n=args.n)
    with open(args.out, "w") as f:
        json.dump({k: _json_safe(v) for k, v in row.items()}, f)


if __name__ == "__main__":
    main()
