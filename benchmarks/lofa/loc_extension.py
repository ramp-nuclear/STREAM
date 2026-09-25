"""
Loss-of-coolant extension of the multichannel LOFA benchmark.

The general four-channel loop with its fixed top boundary replaced by a pool: the
channels still hang between shared plena, but the plena are now fed by a free-surface
Tank through hydrostatic heads, the return line carries its own friction, and a break
sits on the lower plenum — the pump's suction. One continuous run carries the system
through

    forced steady -> scram + pump trip -> flapper opens -> natural circulation
    -> break opens -> drain -> uncovery

on the same Way-Wigner-like decay curve the ladder's capstone uses. The script exits 0
only if every check holds, and prints its record as one JSON line.

Usage:
  conda run -n stream-env python benchmarks/lofa/loc_extension.py
"""
import importlib
import json
import os
import sys
import time as _time
import warnings

import numpy as np

from stream.calculations import (
    Environment,
    Friction,
    Gravity,
    HeatExchanger,
    Inertia,
    Junction,
    KirchhoffWDerivatives,
    Orifice,
    Pump,
    Resistor,
    Tank,
)
from stream.calculations.channel import Channel, ChannelAndContacts
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.composition import (
    FlowGraph,
    break_to_ambient,
    check_gravity_mismatch,
    flow_edge,
    pool,
    symmetric_plate,
)
from stream.composition.subsystems import loc_steady_state, symmetric_plate_steady_state
from stream.jacobians import ALG_jacobian, DAE_jacobian
from stream.physical_models.pressure_drop.discharge import discharge_cd
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.utilities import identity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
case = importlib.import_module("case")

POOL_AREA = 2.0
POOL_LEVEL0 = 4.0
POOL_UNCOVERY = 1.0
COVER_GAS = case.P_REF

RETURN_LINE = EffectivePipe.circular(15.0, 0.05)
RETURN_F = 0.02
BREAK_AREA = 5e-4
BREAK_CD = discharge_cd("sharp")

T_SCRAM = 3.0
T_BREAK = 1500.0
T_SETTLED = 200.0
T_END = 3400.0
RTOL = 1e-4
WALL_BUDGET = 600.0

HEATED = ("hot", "warm", "wide")


def build():
    """The LOFA loop drawing from a pool, with a suction break and a return line."""
    tank = Tank(light_water, POOL_AREA, POOL_LEVEL0, z_uncovery=POOL_UNCOVERY,
                surface_pressure=COVER_GAS, fixed_temperature=case.TIN, name="Pool")
    ambient = Environment(name="Ambient")
    j_top, j_bot, j_return = Junction(name="J_top"), Junction(name="J_bot"), Junction(name="J_return")

    pump = Pump(pressure=case.DP0_CHANNEL, name="Pump")
    flywheel = Inertia(inertia=case.GEN_INERTIA_L, name="Flywheel")
    hx = HeatExchanger(outlet=case.TIN, name="HX")
    gravity_ret = Gravity(fluid=light_water, disposition=-case.CHANNEL_LENGTH, name="GravityReturn")
    gravity_flap = Gravity(fluid=light_water, disposition=case.CHANNEL_LENGTH, name="GravityFlapper")
    resist_flap = Resistor(resistance=10.0, name="ResistFlap")
    return_line = Friction(RETURN_F, light_water, RETURN_LINE.length,
                           RETURN_LINE.hydraulic_diameter, RETURN_LINE.area, name="ReturnLine")
    flapper = Flapper(
        open_at_current=case.GEN_FLAPPER_THRESHOLD, f=case.FLAPPER_F, fluid=light_water,
        area=case.FLAPPER_AREA, open_rate=1.0, stop_on_open=False,
        relaxation=continuously_differentiable_relaxation, name="Flapper",
    )
    breach = Orifice(light_water, BREAK_AREA, BREAK_CD, t_break=T_BREAK, open_rate=1.0, name="Breach")

    z = np.linspace(0.0, case.CHANNEL_LENGTH, case.CHANNEL_Z_N + 1)
    channels = {
        name: ChannelAndContacts(z_boundaries=z, fluid=light_water, pipe=case._gap_pipe(case.GEN_GAPS[name]),
                                 name=f"{name.capitalize()}Channel")
        for name in HEATED
    }
    channels["bypass"] = Channel(z_boundaries=z, fluid=light_water,
                                 pipe=case._gap_pipe(case.GEN_GAPS["bypass"]), name="Bypass")
    for name, channel in channels.items():
        case._apply_regime_friction(channel, case.GEN_GAPS[name])
    fuels = {name: case._fuel(case.CHANNEL_Z_N, name=f"{name.capitalize()}Fuel") for name in HEATED}

    fg = FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_return: 0.0}),
        flow_edge((j_top, j_bot), channels["hot"]),
        flow_edge((j_top, j_bot), channels["warm"]),
        flow_edge((j_top, j_bot), channels["wide"]),
        flow_edge((j_top, j_bot), channels["bypass"]),
        flow_edge((j_top, j_bot), flapper, gravity_flap, resist_flap),
        flow_edge((j_bot, j_return), return_line, pump, flywheel, hx, gravity_ret, ref_mdot_for=(flapper,)),
        break_to_ambient(j_bot, breach, ambient),
        surface_nodes={tank: None, ambient: None},
        inertial_comps=[flywheel, *channels.values()],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channels[name] for name in HEATED],
        funcs={flapper: dict(t=identity), breach: dict(t=identity)},
    )
    check_gravity_mismatch(fg.kirchhoff)

    agr = fg.aggregator
    for name in HEATED:
        agr = agr + symmetric_plate(channels[name], fuels[name],
                                    funcs={fuels[name]: dict(power=case.GEN_POWERS[name])}).to_aggregator()

    refs = dict(tank=tank, pump=pump, flapper=flapper, breach=breach, channels=channels, fuels=fuels)
    return agr, fg.kirchhoff, refs


def guess(agr, k, refs):
    """Per-channel expert guess for the intact loop: pool full, break sealed."""
    channels, total = refs["channels"], case.GEN_MDOT_TOTAL
    hydraulic = loc_steady_state(
        k,
        {channels[name]: case.GEN_MDOTS[name] for name in channels}
        | {agr["head_Pool_J_top"]: total, agr["head_J_return_Pool"]: total,
           refs["pump"]: total, refs["flapper"]: 0.0},
        case.TIN,
    )
    thermals = [
        symmetric_plate_steady_state(channels[name], refs["fuels"][name], mdot=case.GEN_MDOTS[name],
                                     p_abs=case.P_REF, power=case.GEN_POWERS[name], Tin=case.TIN)
        for name in HEATED
    ]
    return State.merge(hydraulic, *thermals)


def grid():
    """Output times: dense through the coastdown, through the opening, and over the drain."""
    return np.unique(np.concatenate((
        np.linspace(0.0, 200.0, 201),
        np.linspace(200.0, T_BREAK, 401),
        np.linspace(T_BREAK, T_BREAK + 40.0, 81),
        np.linspace(T_BREAK + 40.0, T_END, 1000),
    )))


def tolerances(agr, tank):
    atol = np.full(len(agr), 1e-1)
    atol[agr.var_index(tank, "level")] = 1e-8
    return atol


def run():
    t0 = _time.time()
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        agr, k, refs = build()
    unrouted = sorted(str(w.message).split(" has a ")[0] for w in raised if "Tin_minus" in str(w.message))
    tank, breach, flapper = refs["tank"], refs["breach"], refs["flapper"]
    channels, fuels = refs["channels"], refs["fuels"]

    vec = agr.solve_steady(guess(agr, k, refs), jac=ALG_jacobian(agr))
    intact = agr.save(vec)
    steady = dict(
        residual_norm=float(np.linalg.norm(agr.compute(vec, 0.0))),
        mdots={name: float(intact[k.name][k.component_edge(c)]) for name, c in channels.items()},
        T_outlet_hot=float(np.asarray(intact[channels["hot"].name]["T_cool"])[-1]),
        level=float(intact[tank.name]["level"]),
    )

    case.wire_ramp_scram(agr, refs["pump"], {fuels[name]: case.GEN_POWERS[name] for name in HEATED},
                         t_scram=T_SCRAM)
    tank.unpin()
    agr.refresh_mass()
    sol = agr.solve(vec, time=grid(), jacfn=DAE_jacobian(agr), atol=tolerances(agr, tank),
                    rtol=RTOL, max_steps=1000000)

    t = np.asarray(sol.time)
    level = np.asarray(agr.at_times(sol, tank, "level")).squeeze()
    m_break = np.asarray(agr.at_times(sol, k, k.component_edge(breach))).squeeze()
    m_channel = {name: np.asarray(agr.at_times(sol, k, k.component_edge(c))).squeeze()
                 for name, c in channels.items()}
    peak = {name: float(np.max(agr.at_times(sol, channels[name], "T_cool"))) for name in HEATED}
    peak_wall = {name: float(np.max(agr.at_times(sol, fuels[name], "T_wall_left"))) for name in HEATED}
    t_sat = float(light_water.sat_temperature(COVER_GAS))

    t_rev = {}
    for name, m in m_channel.items():
        reversed_at = np.flatnonzero(m < -1e-3)
        t_rev[name] = float(t[reversed_at[0]]) if len(reversed_at) else None
    t_open = float(flapper.t_open) if np.isfinite(flapper.t_open) else None
    t_uncovery = float(sol.t_stop) if sol.t_stop is not None else None

    discharged = float(np.trapezoid(m_break, t))
    lost = float(light_water.density(case.TIN)) * POOL_AREA * (POOL_LEVEL0 - float(level[-1]))
    closure = abs(discharged - lost) / lost if lost else np.inf

    settled = (t >= T_BREAK - T_SETTLED) & (t <= T_BREAK)
    circulating = all(bool(np.all(m_channel[name][settled] < 0.0)) for name in HEATED)
    reversed_before_break = all(t_rev[name] is not None and t_rev[name] < T_BREAK for name in HEATED)
    stopped_by = {name for event in sol.events for name in event.stopped}
    wall = round(_time.time() - t0, 1)

    checks = dict(
        bidirectional_advection_routed=not unrouted,
        event_ordering=bool(t_open is not None and t_uncovery is not None and t_open < T_BREAK < t_uncovery),
        natural_convection_before_break=bool(circulating and reversed_before_break),
        staggered_reversal=bool(t_rev["hot"] is not None and t_rev["warm"] is not None
                                and t_rev["hot"] < t_rev["warm"]),
        mass_balance=bool(closure < 1e-2),
        terminal_uncovery=bool(t_uncovery is not None and tank.name in stopped_by),
        level_monotone_after_break=bool(np.all(np.diff(level[t >= T_BREAK]) <= 1e-9)),
        sub_saturation=bool(max(peak.values()) < t_sat),
        runtime=bool(wall < WALL_BUDGET),
    )
    metrics = dict(
        unrouted_reverse_advection=unrouted,
        steady=steady,
        t_flapper_open=t_open,
        t_break=T_BREAK,
        t_reversal=t_rev,
        t_uncovery=t_uncovery,
        level_end=float(level[-1]),
        break_peak=float(np.max(m_break)),
        discharged_kg=discharged,
        inventory_lost_kg=lost,
        mass_balance_closure=closure,
        mdot_final={name: float(m[-1]) for name, m in m_channel.items()},
        peak_T_cool=peak,
        peak_T_wall=peak_wall,
        T_sat=t_sat,
        margin=t_sat - max(peak.values()),
    )
    detail = (
        f"t_open={t_open:.2f}s, t_break={T_BREAK:.0f}s, t_rev hot/warm/wide="
        f"{t_rev['hot']}/{t_rev['warm']}/{t_rev['wide']}, t_uncovery={t_uncovery:.1f}s; "
        f"drained {discharged:.0f} kg, closure {closure:.2e}; "
        f"peak={max(peak.values()):.2f}C margin={metrics['margin']:+.2f}C (sat {t_sat:.1f})"
    )
    return dict(stage="LOC", status="PASS" if all(checks.values()) else "FAIL", wall_s=wall,
                checks=checks, detail=detail, metrics=metrics)


if __name__ == "__main__":
    record = run()
    print(json.dumps(record))
    sys.exit(0 if record["status"] == "PASS" else 1)
