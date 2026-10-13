"""Hand checks of a solved loss-of-flow transient against conservation laws and against estimates
that do not go through the solver: mass at every junction, energy over the whole run and in the
natural-circulation state, the pressure budget around the natural-circulation loop, the mixing
at the pool-top junction, the decay-heat law, and a tolerance comparison of the reversal times.

    python benchmarks/lofa_pool/physics_checks.py --case case --out checks-main.json --figures figs/
    python benchmarks/lofa_pool/physics_checks.py --case top_flapper --out checks-top.json --figures figs/

``--case case`` runs the shipped loop of :mod:`stream.examples.lofa`, ``--case top_flapper`` the
loop of the sibling ``top_flapper.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stream.examples.lofa.checks import energy_budget, junction_mass_balance, loop_pressure_budget, series
from stream.examples.lofa.params import CORE, HEATED, Params, power_share
from stream.examples.lofa.scenarios import flows, start_transient, transient
from stream.examples.lofa.scenarios import reversal_times as flow_reversal_times
from stream.examples.lofa.system import decay_heat, operating_point
from stream.substances import light_water

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lofa_bench_top_flapper", HERE / "top_flapper.py")
top_flapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(top_flapper)

LOOPS = {
    "case": SimpleNamespace(operating_point=operating_point, start_transient=start_transient, transient=transient),
    "top_flapper": top_flapper,
}
LOOP_NAMES = {
    "case": ["HotChannel", "Flapper", "Pool"],
    "top_flapper": ["HotChannel", "PipePlenumRiser", "BendPlenumRiser0", "Flapper", "PoolLower"],
}


def _cp(T):
    return np.asarray(light_water.specific_heat(T), dtype=float)


def channel_energy(agr, K, refs, sol):
    """Per heated group: the coolant enthalpy rise through the plates, direction-aware, against
    the power fed to those plates, at every output time; W."""
    t = np.asarray(sol.time)
    power = series(agr, sol, refs["pk"], "power")[:, 0]
    T_top = series(agr, sol, agr["J_pool_top"], "Tin")[:, 0]
    T_plenum = series(agr, sol, agr["J_plenum"], "Tin")[:, 0]
    out = {}
    for k in HEATED:
        ch = refs["channels"][k]
        n = CORE[k].plates
        m = series(agr, sol, K, K.component_edge(ch))[:, 0]
        T = series(agr, sol, ch, "T_cool")
        T_end = np.where(m >= 0, T[:, -1], T[:, 0])
        T_start = np.where(m >= 0, T_top, T_plenum)
        removed = n * np.abs(m) * _cp(0.5 * (T_start + T_end)) * (T_end - T_start)
        fed = n * power_share(k) * power
        out[k] = {"removed_W": removed, "fed_W": fed, "mdot": m, "T_start": T_start, "T_end": T_end}
    return t, power, out


def sink_energy_top_flapper(agr, K, refs, sol, p):
    """For the top-flapper loop: energy carried into the lower pool section by reversed flow (the
    relabelled sink) over the run; J."""
    t = np.asarray(sol.time)
    m = series(agr, sol, K, K.component_edge(refs["pool"]))[:, 0]
    T_top = series(agr, sol, agr["J_pool_top"], "Tin")[:, 0]
    carried = np.trapezoid(np.maximum(-m, 0.0) * _cp(T_top) * (T_top - p.hx_outlet), t)
    return float(carried)


def mixing_check(agr, K, refs, sol, idx=-1):
    """The pool-top junction temperature against the flow-weighted mean of the streams entering
    it at output index ``idx``."""
    st = agr.save(sol.data[idx])
    num = den = 0.0
    for k, ch in refs["channels"].items():
        m = float(st[K.name][K.component_edge(ch)])
        if m < 0:
            T_out = float(np.asarray(st[ch.name]["T_cool"])[0])
            num += CORE[k].plates * (-m) * T_out
            den += CORE[k].plates * (-m)
    pool_m = float(st[K.name][K.component_edge(refs["pool"])])
    hand = num / den if den else float("nan")
    return {"junction": float(np.asarray(st["J_pool_top"]["Tin"])), "hand_mix": hand, "inflow_kg_s": den, "pool_flow": pool_m}


def reversal_times(agr, K, refs, sol):
    df = flows(agr, K, refs, sol)
    return flow_reversal_times(df), df


def run(module, p, atol_rel, t_end, n):
    agr, K, refs, y = module.operating_point(p)
    module.start_transient(agr, refs, y, p, p.scenario)
    sol = module.transient(agr, y, t_end=t_end, n=n, atol_rel=atol_rel, rtol=atol_rel)
    return agr, K, refs, sol


def tight_tolerance_comparison(module, p, args, df):
    """Re-run the case at ``args.tight`` up to ``args.tight_t_end`` and compare its reversal times and
    per-plate flows with the loose-tolerance flows ``df``. Returns the report entries and the tight flows.

    The comparison is optional: a failure of the tight run, from its steady state to its transient, is
    recorded under ``tight_error`` instead of raised, and the tight flows are then None, so the checks
    of the loose-tolerance run are still written."""
    try:
        agr2, K2, refs2, sol2 = run(module, p, args.tight, args.tight_t_end, round(args.tight_t_end / args.tight_dt) + 1)
        rev2, df2 = reversal_times(agr2, K2, refs2, sol2)
    except Exception as e:  # noqa: BLE001
        return {"tight_error": f"{type(e).__name__}: {e}"[:600]}, None
    diffs = {}
    for k in CORE:
        a = np.interp(df2["t"], df["t"], df[k])
        diffs[k] = {"max_abs_diff": float(np.max(np.abs(a - df2[k]))), "scale": float(np.max(np.abs(df2[k])))}
    events = {"t_trip": float(refs2["trip"].t_trip), "t_open": float(refs2["flapper"].t_open)}
    return {"reversals_tight": rev2, "events_tight": events, "flow_trace_diff_tolerances": diffs}, df2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case", choices=tuple(LOOPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="")
    ap.add_argument("--t-end", type=float, default=3600.0)
    ap.add_argument("--n", type=int, default=1801)
    ap.add_argument("--tight", type=float, default=1e-5)
    ap.add_argument("--tight-t-end", type=float, default=900.0)
    ap.add_argument("--tight-dt", type=float, default=2.0)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*shadow edge-routed.*")
    module = LOOPS[args.case]
    p = Params()
    agr, K, refs, sol = run(module, p, 1e-4, args.t_end, args.n)
    t = np.asarray(sol.time)
    report = {"case": args.case, "t_end": float(t[-1])}

    worst, per = junction_mass_balance(agr, K, refs, sol)
    report["kcl_worst_kg_s"] = worst
    report["kcl_per_junction"] = per

    t, power, ce = channel_energy(agr, K, refs, sol)
    late = t >= 0.8 * t[-1]
    report["channel_energy_late"] = {k: {"removed_W": float(np.mean(v["removed_W"][late])), "fed_W": float(np.mean(v["fed_W"][late])),
                                        "mdot": float(np.mean(v["mdot"][late])), "T_start": float(np.mean(v["T_start"][late])),
                                        "T_end": float(np.mean(v["T_end"][late]))}
                                     for k, v in ce.items()}
    report["channel_energy_steady"] = {k: {"removed_W": float(v["removed_W"][0]), "fed_W": float(v["fed_W"][0])} for k, v in ce.items()}
    report["power_W"] = {"t0": float(power[0]), "end": float(power[-1]),
                         "decay_law_end": float(decay_heat(refs["power"])(t[-1] - refs["trip"].t_trip))}

    report["global_energy"] = energy_budget(agr, K, refs, sol, p)
    if args.case == "top_flapper":
        report["sink_carried_J"] = sink_energy_top_flapper(agr, K, refs, sol, p)

    budget, buoyancy = loop_pressure_budget(agr, K, refs, sol, LOOP_NAMES[args.case])
    report["nc_pressure_budget_Pa"] = budget
    report["hot_plate_buoyancy_head_Pa"] = buoyancy
    report["mixing_pool_top"] = mixing_check(agr, K, refs, sol)

    rev, df = reversal_times(agr, K, refs, sol)
    report["reversals_1e-4"] = rev
    report["events_1e-4"] = {"t_trip": float(refs["trip"].t_trip), "t_open": float(refs["flapper"].t_open)}

    report["tight_tolerance"] = args.tight
    tight, df2 = tight_tolerance_comparison(module, p, args, df)
    report.update(tight)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)

    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.figures, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        for k, v in ce.items():
            ax.plot(t, v["removed_W"] / 1e3, label=f"{k} removed")
            ax.plot(t, v["fed_W"] / 1e3, ls="--", label=f"{k} fed")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title("per-group power fed (dashed) and carried away by the coolant (solid), kW")
        ax.legend(fontsize=7, ncol=2)
        ax = axes[0, 1]
        for k in CORE:
            ax.plot(df["t"], df[k], label=f"{k} 1e-4")
            if df2 is not None:
                ax.plot(df2["t"], df2[k], ls=":", label=f"{k} {args.tight:g}")
        ax.set_xlim(0, args.tight_t_end)
        ax.set_ylim(-0.1, 0.3)
        ax.set_title("per-plate flows at two tolerances, kg/s")
        ax.legend(fontsize=7, ncol=2)
        ax = axes[1, 0]
        T_top = series(agr, sol, agr["J_pool_top"], "Tin")[:, 0]
        T_pl = series(agr, sol, agr["J_plenum"], "Tin")[:, 0]
        ax.plot(t, T_top, label="J_pool_top")
        ax.plot(t, T_pl, label="J_plenum")
        for k in HEATED:
            ax.plot(t, ce[k]["T_end"], ls="--", label=f"{k} exit")
        ax.set_title("junction and channel exit temperatures, C")
        ax.legend(fontsize=7)
        ax = axes[1, 1]
        ax.plot(t, power / 1e3, label="total power")
        tot = sum(v["removed_W"] for v in ce.values())
        ax.plot(t, tot / 1e3, label="sum carried away")
        ax.set_yscale("log")
        ax.set_title("core power and coolant removal, kW")
        ax.legend(fontsize=7)
        for ax in axes.ravel():
            ax.set_xlabel("time (s)")
            ax.grid(True, lw=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures, f"checks-{args.case}-{args.tight:g}.png"), dpi=110)
    print(json.dumps({k: v for k, v in report.items() if k not in ("kcl_per_junction",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
