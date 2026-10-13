"""A second pool-reactor loss-of-flow loop: the pump return discharges into the pool surface,
the pool stands above the core in two gravity sections, and the flapper joins the pool just
above the channel tops to the plenum riser at the same height. Natural circulation runs
core up, pool up, flapper, plenum riser down; the driving head is the core buoyancy alone.

The two pool sections are one-cell channels carrying the pool's water. The lower section's
reversed-flow inlet temperature is fixed at the heat-exchanger outlet, so the core outflow
entering it under natural circulation is relabelled cold pool water and the pool is an
infinite sink.

Core, kinetics, trip, decay heat, pumps and scenarios are those of :mod:`stream.examples.lofa`."""
from __future__ import annotations

from functools import partial

import numpy as np

from stream.aggregator import Aggregator
from stream.calculations import Bend, HeatExchanger, Inertia, Junction, KirchhoffWDerivatives, Pump
from stream.calculations.channel import Channel
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.calculations.kirchhoff import Kirchhoff
from stream.composition import FlowGraph, check_gravity_mismatch, flow_edge, symmetric_plate
from stream.composition.constructors import ResistorFromKnownPoint
from stream.composition.subsystems import point_kinetics_steady_state
from stream.examples.lofa import scenarios, system
from stream.examples.lofa.components import CheckValve, LowFlowTrip
from stream.examples.lofa.params import (
    CORE,
    HEATED,
    LENGTH,
    PER_PLATE_REFERENCE,
    PIPE1_DROP,
    Params,
    nominal_flows,
    plate_pipe,
    power_share,
)
from stream.examples.lofa.system import GUESS_KINDS, make_channel, make_fuel
from stream.physical_models.pressure_drop import pressure_diff
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.utilities import identity

TAP_ABOVE_CORE = 0.5
NC_SIGNS = {"pool": -1.0, "flapper": 1.0}
CHANNEL_PIPES = ("PlenumRiser", "Upper")
PIPE_CELLS = 4
G = 9.80665

steady = system.steady
start_transient = scenarios.start_transient
transient = scenarios.transient
decay_heat = system.decay_heat


def depths(p: Params) -> dict[str, float]:
    """Metres below the pool surface of the flapper tap, the channel tops, the plenum, the
    decay tank and the pump manifold."""
    tops = p.core_depth
    plenum = tops + LENGTH
    return {"tap": tops - TAP_ABOVE_CORE, "tops": tops, "plenum": plenum, "tank": plenum + PIPE1_DROP, "pumps": plenum + p.basement_drop}


def _pipe_channel(p: Params, tag: str, length: float, drop: float) -> Channel:
    """An unheated channel standing in for a pipe leg: ``PIPE_CELLS`` cells of the pipe's bore
    over ``length``, climbing or falling by ``drop`` (positive downward), so that the column's
    temperature, and with it its weight, follows the water that actually fills it."""
    pipe = EffectivePipe.circular(length=length, diameter=p.pipe_diameter)
    return Channel(z_boundaries=np.linspace(0.0, length, PIPE_CELLS + 1), fluid=light_water, pipe=pipe, name=f"Pipe{tag}",
                   pressure_func=partial(pressure_diff, g=G * drop / length))


def _pipe_leg(p: Params, tag: str, length: float, drop: float, bends: int) -> list:
    """The parts of a pipe leg: a channel where ``CHANNEL_PIPES`` names the leg, else the lumped
    friction, bends and gravity of the shipped loop's pipe legs."""
    if tag not in CHANNEL_PIPES:
        return system._pipe_parts(p, tag, length, drop, bends)
    area = np.pi * p.pipe_diameter**2 / 4
    bend_parts = [Bend(fluid=light_water, hydraulic_diameter=p.pipe_diameter, area=area, bend_radius=1.5 * p.pipe_diameter,
                       bend_angle=np.pi / 2, name=f"Bend{tag}{i}") for i in range(bends)]
    return [_pipe_channel(p, tag, length, drop), *bend_parts]


def build(p: Params, kinetics: bool = False, ref_state: State | None = None) -> tuple[Aggregator, Kirchhoff, dict]:
    """Build the loop at ``p``; same contract as :func:`stream.examples.lofa.build`."""
    if kinetics and ref_state is None:
        raise ValueError("kinetics needs the solved thermal-hydraulic state as ref_state")
    nominal = nominal_flows(p)
    total = nominal["total"]
    P0 = p.power * p.power_scale
    p_ref = p.p_surface + float(light_water.density(p.hx_outlet)) * 9.80665 * p.core_depth
    z = depths(p)

    j_surface, j_tap, j_top = Junction(name="J_surface"), Junction(name="J_pool_tap"), Junction(name="J_pool_top")
    j_plenum, j_pipe_tap = Junction(name="J_plenum"), Junction(name="J_pipe_tap")
    j_tank_in, j_tank_out = Junction(name="J_tank_in"), Junction(name="J_tank_out")
    j_split, j_merge = Junction(name="J_split"), Junction(name="J_merge")

    channels = {k: make_channel(f"{k.capitalize()}Channel", p.gap_hot if k == "hot" else g.gap, k in HEATED,
                                p.friction, p.htc, p.stop_at_saturation) for k, g in CORE.items()}
    fuels = {k: make_fuel(f"{k.capitalize()}Fuel", share=power_share(k)) for k in HEATED}
    pool_volume = p.pool_volume * p.pool_area_scale
    pool_upper = system._square_channel("PoolUpper", pool_volume * z["tap"] / p.core_depth, z["tap"], 1, horizontal=False)
    pool_lower = system._square_channel("PoolLower", pool_volume * TAP_ABOVE_CORE / p.core_depth, TAP_ABOVE_CORE, 1, horizontal=False)
    tank = system._square_channel("DecayTank", p.tank_volume * p.tank_volume_scale, 5.0, 5, horizontal=True)

    flapper = Flapper(open_at_current=(-1e9 if p.flapper_stuck else p.flapper_fraction * total), f=p.flapper_f,
                      fluid=light_water, area=p.flapper_area, open_rate=1.0, stop_on_open=False,
                      relaxation=continuously_differentiable_relaxation, name="Flapper")
    trip = LowFlowTrip(threshold=p.trip_threshold * total, delay=p.trip_delay, worth=system._dollars(p),
                       duration=p.rod_duration, name="Trip")
    pumps = {leg: Pump(mdot0=total / 2, name=f"Pump{leg}") for leg in "AB"}
    flywheels = {leg: Inertia(inertia=p.inertia * p.inertia_scale, name=f"Flywheel{leg}") for leg in "AB"}
    valves = {leg: CheckValve(k_forward=p.valve_k, reverse_ratio=p.valve_reverse_ratio, name=f"Valve{leg}") for leg in "AB"}
    pump_curves = {leg: ResistorFromKnownPoint(dp=-p.pump_droop * p.pump_head_estimate, mdot=total / 2, behavior="parabolic",
                                               fluid=light_water, Tin=p.hx_outlet, name=f"PumpCurve{leg}") for leg in "AB"}
    hx = HeatExchanger(outlet=p.hx_outlet, name="HX")
    hx_loss = ResistorFromKnownPoint(dp=-p.hx_dp, mdot=total, behavior="parabolic", fluid=light_water,
                                     Tin=p.hx_outlet, name="HXLoss")

    riser_climb = z["plenum"] - z["tap"]
    upper_drop = z["tank"] - z["tap"]
    basement_drop = z["pumps"] - z["tank"]
    riser_parts = _pipe_leg(p, "PlenumRiser", riser_climb, -riser_climb, 1)
    upper_parts = _pipe_leg(p, "Upper", z["tap"] + z["tank"], upper_drop, 2)
    fg = FlowGraph(
        *(flow_edge((j_top, j_plenum), channels[k], signify=float(CORE[k].plates)) for k in CORE),
        flow_edge((j_tap, j_top), pool_lower),
        flow_edge((j_surface, j_tap), pool_upper),
        flow_edge((j_tap, j_pipe_tap), flapper),
        flow_edge((j_plenum, j_pipe_tap), *riser_parts),
        flow_edge((j_pipe_tap, j_tank_in), *upper_parts),
        flow_edge((j_tank_in, j_tank_out), tank),
        flow_edge((j_tank_out, j_split), *system._pipe_parts(p, "Basement", basement_drop, basement_drop, 1)),
        *(flow_edge((j_split, j_merge), pumps[leg], pump_curves[leg], flywheels[leg], valves[leg]) for leg in "AB"),
        flow_edge((j_merge, j_surface), hx, hx_loss, *system._pipe_parts(p, "Return", z["pumps"], -z["pumps"], 2),
                  ref_mdot_for=(flapper, trip)),
        inertial_comps=[*flywheels.values(), *channels.values(), pool_upper, pool_lower, tank,
                        *[c for c in riser_parts + upper_parts if isinstance(c, Channel)]],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channels[k] for k in HEATED],
        funcs={flapper: {"t": identity}, trip: {"t": identity}, pool_lower: {"Tin_minus": p.hx_outlet}},
        reference_node=(j_top, p_ref),
    )
    check_gravity_mismatch(fg.kirchhoff)

    refs = {"fg": fg, "channels": channels, "fuels": fuels, "pool": pool_lower, "pool_upper": pool_upper, "tank": tank, "pumps": pumps,
            "flywheels": flywheels, "valves": valves, "pump_curves": pump_curves, "flapper": flapper, "trip": trip, "hx": hx, "pk": None,
            "params": p, "nominal": nominal, "power": P0, "p_ref": p_ref, "depths": z}
    agr = fg.aggregator
    if not kinetics:
        for k in HEATED:
            agr = agr + symmetric_plate(channels[k], fuels[k], funcs={fuels[k]: {"power": P0}}).to_aggregator()
        return agr, fg.kirchhoff, refs
    agr = agr + system._kinetics_graph(p, channels, fuels, trip, ref_state, P0).to_aggregator()
    refs["pk"] = agr["PK"]
    return agr, fg.kirchhoff, refs


def _ballpark(agr: Aggregator, K: Kirchhoff, refs: dict) -> State:
    """Uniform temperatures, zero pressure drops, order-of-magnitude heat-transfer
    coefficients, and nominal flows by leg type: one plate's flow on core edges, the core
    total elsewhere, zero through the flapper."""
    p, nominal = refs["params"], refs["nominal"]
    non_k = [n for n in agr.graph if not hasattr(n, "variables_by_type")]
    vec = np.zeros(len(K))
    vec[K.variables_by_type["mdot"]] = nominal["total"]
    vec[K.variables_by_type["abs_pressure"]] = refs["p_ref"]
    k_state = K.save(vec)
    for ch in refs["channels"].values():
        k_state[K.component_edge(ch)] = PER_PLATE_REFERENCE
    k_state[K.component_edge(refs["flapper"])] = 0.0
    for leg in "AB":
        k_state[K.component_edge(refs["pumps"][leg])] = nominal["total"] / 2
    return State.merge(
        State.uniform(non_k, p.hx_outlet, "T_cool", "T", "T_wall_left", "T_wall_right", "Tin"),
        State.uniform(non_k, 1e3, "h_left", "h_right"),
        State.uniform(non_k, 0.0, "pressure", "flag"),
        {K.name: k_state},
    )


def guess(agr: Aggregator, K: Kirchhoff, refs: dict, kind: str, seed: int = 0, band: float = 0.3) -> np.ndarray:
    """A steady-state guess vector of one of ``GUESS_KINDS``, as :func:`stream.examples.lofa.guess`."""
    if kind == "ballpark":
        return agr.load(_ballpark(agr, K, refs))
    if kind not in GUESS_KINDS:
        raise ValueError(f"guess kind must be one of {GUESS_KINDS}, got {kind!r}")
    return system.guess(agr, K, refs, kind, seed=seed, band=band)


def steady_checks(agr: Aggregator, K: Kirchhoff, refs: dict, y: np.ndarray) -> dict:
    """The hand checks of :func:`stream.examples.lofa.steady_checks`; here ``pool_flow`` is the lower pool
    section's flow, positive downward, and ``inlet_matches_pool`` compares the core inlet
    with the heat-exchanger outlet."""
    st = agr.save(y)
    p = refs["params"]
    cp = float(light_water.specific_heat(p.hx_outlet + 10.0))
    mdot = {k: float(st[K.name][K.component_edge(c)]) for k, c in refs["channels"].items()}
    T_in = float(np.asarray(st["J_pool_top"]["Tin"]))
    T_out_err = []
    for k in HEATED:
        T_out = float(np.asarray(st[refs["channels"][k].name]["T_cool"])[-1])
        T_out_err.append(abs(T_out - (T_in + refs["power"] * power_share(k) / (mdot[k] * cp))))
    pipes = {k: plate_pipe(p.gap_hot if k == "hot" else g.gap) for k, g in CORE.items()}
    law = {k: pipe.area * pipe.hydraulic_diameter ** (5 / 7) for k, pipe in pipes.items()}
    ratios = [mdot[k] / mdot["normal"] / (law[k] / law["normal"]) for k in CORE]
    wall = [float(np.asarray(st[refs["fuels"][k].name]["T_wall_left"])[-1]) >
            float(np.asarray(st[refs["channels"][k].name]["T_cool"])[-1]) for k in HEATED]
    return {
        "T_out_error_max": max(T_out_err),
        "split_error_max": max(abs(r - 1.0) for r in ratios),
        "pool_flow": float(st[K.name][K.component_edge(refs["pool"])]),
        "inlet_matches_pool": abs(T_in - p.hx_outlet) < 0.05,
        "wall_above_coolant": all(wall),
        "heads": {leg: float(np.asarray(st[refs["pumps"][leg].name]["pressure"])) for leg in "AB"},
        "mdot": mdot,
        "T_in": T_in,
    }


def operating_point(p: Params, guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3):
    """As :func:`stream.examples.lofa.operating_point`, on this loop."""
    agr_th, K_th, refs_th = build(p)
    y_th = steady(agr_th, guess(agr_th, K_th, refs_th, guess_kind, seed=seed, band=band))
    st_th = agr_th.save(y_th)
    agr, K, refs = build(p, kinetics=True, ref_state=st_th)
    refs["th_checks"] = steady_checks(agr_th, K_th, refs_th, y_th)
    refs["th_residual_entry"] = float(np.linalg.norm(agr_th.compute(guess(agr_th, K_th, refs_th, guess_kind, seed=seed, band=band), 0.0)))
    P0 = refs["power"]
    kin = point_kinetics_steady_state(refs["pk"], P0, power_input=decay_heat(P0)(0.0))
    y = agr.load(State.merge(st_th, kin))
    return agr, K, refs, y
