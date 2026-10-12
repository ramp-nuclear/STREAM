"""Building the loss-of-flow loop, guessing and solving its operating point, and drawing it."""
from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from stream.aggregator import Aggregator, CalculationGraph
from stream.calculations import (
    Bend,
    Friction,
    Fuel,
    Gravity,
    HeatExchanger,
    Inertia,
    Junction,
    KirchhoffWDerivatives,
    Pump,
    Solid,
)
from stream.calculations.channel import Channel, ChannelAndContacts
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.calculations.kirchhoff import Kirchhoff
from stream.calculations.point_kinetics import PointKineticsWInput, ReactivityController
from stream.composition import (
    FlowGraph,
    check_gravity_mismatch,
    flow_edge,
    symmetric_plate,
    uniform_x_power_shape,
    x_boundaries,
)
from stream.composition.constructors import ResistorFromKnownPoint
from stream.composition.subsystems import point_kinetics_steady_state, symmetric_plate_steady_state
from stream.examples.lofa.components import CheckValve, LowFlowTrip, horizontal_pressure
from stream.examples.lofa.params import (
    BETA,
    CLAD_N,
    CLAD_W,
    CLADDING,
    CORE,
    FUEL_MEAT,
    FUEL_N,
    GENERATION_TIME,
    HEATED,
    LAMBDA,
    LENGTH,
    MEAT_W,
    PER_PLATE_REFERENCE,
    PIPE1_DROP,
    WIDTH,
    Z_N,
    Params,
    nominal_flows,
    plate_count_weighted_sum,
    plate_pipe,
    power_share,
)
from stream.jacobians import ALG_jacobian
from stream.physical_models.pressure_drop import pressure_diff
from stream.physical_models.pressure_drop.friction import friction_factor, rectangular_laminar_correction
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.units import g as local_gravity
from stream.utilities import identity


def make_fuel(name: str, share: float = 1.0) -> Fuel:
    """One plate; ``share`` scales the power shape so the plate takes that fraction of the
    ``power`` it is fed."""
    shape = (Z_N, FUEL_N + 2 * CLAD_N)
    meat = np.zeros(shape, dtype=bool)
    meat[:, CLAD_N:-CLAD_N] = True
    materials = np.empty(shape, dtype=object)
    materials[meat] = FUEL_MEAT
    materials[~meat] = CLADDING
    return Fuel(
        z_boundaries=np.linspace(0.0, LENGTH, Z_N + 1),
        x_boundaries=x_boundaries(CLAD_N, FUEL_N, CLAD_W, MEAT_W),
        material=Solid.from_array(materials),
        meat_indices=meat,
        power_shape=share * uniform_x_power_shape(Z_N, FUEL_N, CLAD_N, CLAD_W, MEAT_W, WIDTH),
        y_length=WIDTH,
        name=name,
    )


def _regime_friction(gap: float):
    k_R = float(rectangular_laminar_correction(gap / WIDTH))
    return friction_factor("regime_dependent", re_bounds=(2000.0, 5000.0), k_R=k_R)


def make_channel(name: str, gap: float, heated: bool, friction: str, stop_at_saturation: bool):
    """One plate channel of coolant ``gap`` in m, with heat transfer to both walls when ``heated``;
    ``friction`` is ``"regime"`` (laminar-turbulent blend) or ``"blasius"``."""
    if friction not in ("regime", "blasius"):
        raise ValueError(f"friction must be 'regime' or 'blasius', got {friction!r}")
    z = np.linspace(0.0, LENGTH, Z_N + 1)
    pipe = plate_pipe(gap)
    if heated:
        ch = ChannelAndContacts(z_boundaries=z, fluid=light_water, pipe=pipe, name=name,
                                stop_at_saturation=stop_at_saturation)
    else:
        ch = Channel(z_boundaries=z, fluid=light_water, pipe=pipe, name=name)
    if friction == "regime":
        ch.pressure = partial(pressure_diff, fluid=ch.fluid, pipe=ch.pipe, dz=ch.dz, f=_regime_friction(gap))
    return ch


def _square_channel(name: str, volume: float, length: float, cells: int, horizontal: bool, upward: bool = False) -> Channel:
    """A square-section unheated column; ``upward`` orients its positive flow against gravity."""
    edge = np.sqrt(volume / length)
    pipe = EffectivePipe.rectangular(length=length, edge1=edge, edge2=edge, heated_edge=edge)
    kwargs = {}
    if horizontal:
        kwargs = dict(pressure_func=horizontal_pressure())
    elif upward:
        kwargs = dict(pressure_func=partial(pressure_diff, g=-local_gravity))
    return Channel(z_boundaries=np.linspace(0.0, length, cells + 1), fluid=light_water, pipe=pipe, name=name, **kwargs)


def _pipe_parts(p: Params, tag: str, length: float, drop: float, bends: int) -> list:
    area = np.pi * p.pipe_diameter**2 / 4
    parts = [Friction(f=p.pipe_friction, fluid=light_water, length=length, hydraulic_diameter=p.pipe_diameter,
                      area=area, name=f"Friction{tag}")]
    parts += [Bend(fluid=light_water, hydraulic_diameter=p.pipe_diameter, area=area, bend_radius=1.5 * p.pipe_diameter,
                   bend_angle=np.pi / 2, name=f"Bend{tag}{i}") for i in range(bends)]
    parts.append(Gravity(fluid=light_water, disposition=drop, name=f"Gravity{tag}"))
    return parts


def _dollars(p: Params) -> float:
    return p.rod_worth_dollars * float(np.sum(BETA))


def build(p: Params, kinetics: bool = False, ref_state: State | None = None) -> tuple[Aggregator, Kirchhoff, dict]:
    """Build the loop at ``p``. With ``kinetics=False`` every plate takes its share of the
    nominal power as a constant; with ``kinetics=True`` a point-kinetics node feeds the
    plates and takes temperature feedback with reference temperatures read from
    ``ref_state`` (a solved thermal-hydraulic state)."""
    if kinetics and ref_state is None:
        raise ValueError("kinetics needs the solved thermal-hydraulic state as ref_state")
    nominal = nominal_flows(p)
    total = nominal["total"]
    P0 = p.power * p.power_scale
    p_ref = p.p_surface + float(light_water.density(p.hx_outlet)) * 9.80665 * p.core_depth

    j_top, j_plenum, j_bot = Junction(name="J_pool_top"), Junction(name="J_plenum"), Junction(name="J_pool_bot")
    j_tank_in, j_tank_out = Junction(name="J_tank_in"), Junction(name="J_tank_out")
    j_split, j_merge = Junction(name="J_split"), Junction(name="J_merge")

    channels = {k: make_channel(f"{k.capitalize()}Channel", p.gap_hot if k == "hot" else g.gap, k in HEATED,
                                p.friction, p.stop_at_saturation) for k, g in CORE.items()}
    fuels = {k: make_fuel(f"{k.capitalize()}Fuel", share=power_share(k)) for k in HEATED}
    pool = _square_channel("Pool", p.pool_volume * p.pool_area_scale, LENGTH, 10, horizontal=False, upward=True)
    tank = _square_channel("DecayTank", p.tank_volume * p.tank_volume_scale, 5.0, 5, horizontal=True)

    flapper = Flapper(open_at_current=(-1e9 if p.flapper_stuck else p.flapper_fraction * total), f=p.flapper_f,
                      fluid=light_water, area=p.flapper_area, open_rate=1.0, stop_on_open=False,
                      relaxation=continuously_differentiable_relaxation, name="Flapper")
    trip = LowFlowTrip(threshold=p.trip_threshold * total, delay=p.trip_delay, worth=_dollars(p),
                       duration=p.rod_duration, name="Trip")
    pumps = {leg: Pump(mdot0=total / 2, name=f"Pump{leg}") for leg in "AB"}
    flywheels = {leg: Inertia(inertia=p.inertia * p.inertia_scale, name=f"Flywheel{leg}") for leg in "AB"}
    valves = {leg: CheckValve(k_forward=p.valve_k, reverse_ratio=p.valve_reverse_ratio, name=f"Valve{leg}") for leg in "AB"}
    pump_curves = {leg: ResistorFromKnownPoint(dp=-p.pump_droop * p.pump_head_estimate, mdot=total / 2, behavior="parabolic",
                                               fluid=light_water, Tin=p.hx_outlet, name=f"PumpCurve{leg}") for leg in "AB"}
    hx = HeatExchanger(outlet=p.hx_outlet, name="HX")
    hx_loss = ResistorFromKnownPoint(dp=-p.hx_dp, mdot=total, behavior="parabolic", fluid=light_water,
                                     Tin=p.hx_outlet, name="HXLoss")

    fg = FlowGraph(
        *(flow_edge((j_top, j_plenum), channels[k], signify=float(CORE[k].plates)) for k in CORE),
        flow_edge((j_bot, j_top), pool),
        flow_edge((j_bot, j_plenum), flapper),
        flow_edge((j_plenum, j_tank_in), *_pipe_parts(p, "Primary", PIPE1_DROP, PIPE1_DROP, 2)),
        flow_edge((j_tank_in, j_tank_out), tank),
        flow_edge((j_tank_out, j_split), *_pipe_parts(p, "Basement", p.basement_drop - PIPE1_DROP, p.basement_drop - PIPE1_DROP, 1)),
        *(flow_edge((j_split, j_merge), pumps[leg], pump_curves[leg], flywheels[leg], valves[leg]) for leg in "AB"),
        flow_edge((j_merge, j_bot), hx, hx_loss, *_pipe_parts(p, "Riser", p.basement_drop, -p.basement_drop, 2),
                  ref_mdot_for=(flapper, trip)),
        inertial_comps=[*flywheels.values(), *channels.values(), pool, tank],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channels[k] for k in HEATED],
        funcs={flapper: dict(t=identity), trip: dict(t=identity)},
        reference_node=(j_top, p_ref),
    )
    check_gravity_mismatch(fg.kirchhoff)

    refs = dict(fg=fg, channels=channels, fuels=fuels, pool=pool, tank=tank, pumps=pumps, flywheels=flywheels,
                valves=valves, pump_curves=pump_curves, flapper=flapper, trip=trip, hx=hx, pk=None, params=p, nominal=nominal,
                power=P0, p_ref=p_ref)
    agr = fg.aggregator
    if not kinetics:
        for k in HEATED:
            agr = agr + symmetric_plate(channels[k], fuels[k], funcs={fuels[k]: dict(power=P0)}).to_aggregator()
        return agr, fg.kirchhoff, refs
    agr = agr + _kinetics_graph(p, channels, fuels, trip, ref_state, P0).to_aggregator()
    refs["pk"] = agr["PK"]
    return agr, fg.kirchhoff, refs


GUESS_KINDS = ("ballpark", "expert", "perturbed")


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
    k_state[K.component_edge(refs["pool"])] = nominal["total"]
    k_state[K.component_edge(refs["flapper"])] = 0.0
    for leg in "AB":
        k_state[K.component_edge(refs["pumps"][leg])] = nominal["total"] / 2
    return State.merge(
        State.uniform(non_k, p.hx_outlet, "T_cool", "T", "T_wall_left", "T_wall_right", "Tin"),
        State.uniform(non_k, 1e3, "h_left", "h_right"),
        State.uniform(non_k, 0.0, "pressure", "flag"),
        {K.name: k_state},
    )


def _expert(refs: dict) -> State:
    p, nominal = refs["params"], refs["nominal"]
    channels, fuels = refs["channels"], refs["fuels"]
    mdots = {channels[k]: nominal[k] for k in CORE}
    mdots |= {refs["pool"]: nominal["total"], refs["flapper"]: 0.0, refs["tank"]: nominal["total"]}
    mdots |= {refs["pumps"][leg]: nominal["total"] / 2 for leg in "AB"}
    for comp in refs["fg"].kirchhoff.g.edges(data="comps"):
        for c in comp[2]:
            mdots.setdefault(c, nominal["total"])
    hydraulic = refs["fg"].guess_steady_state(mdots=mdots, temperature=p.hx_outlet)
    thermals = [symmetric_plate_steady_state(channels[k], fuels[k], mdot=nominal[k], p_abs=refs["p_ref"],
                                             power=refs["power"], Tin=p.hx_outlet) for k in HEATED]
    return State.merge(hydraulic, *thermals, {refs["trip"].name: dict(flag=0.0)})


def guess(agr: Aggregator, K: Kirchhoff, refs: dict, kind: str, seed: int = 0, band: float = 0.3) -> np.ndarray:
    """A steady-state guess vector of one of ``GUESS_KINDS``: ``ballpark`` is uniform
    temperatures and nominal flows by leg type, ``expert`` is the composed hydraulic and
    plate guesses, ``perturbed`` scales each entry of the expert guess by a uniform factor in
    ``[1 - band, 1 + band]`` drawn from ``seed``."""
    if kind == "ballpark":
        return agr.load(_ballpark(agr, K, refs))
    if kind == "expert":
        return agr.load(_expert(refs))
    if kind == "perturbed":
        y = agr.load(_expert(refs))
        rng = np.random.default_rng(seed)
        return y * (1 + rng.uniform(-band, band, size=y.shape))
    raise ValueError(f"guess kind must be one of {GUESS_KINDS}, got {kind!r}")


def steady(agr: Aggregator, y0: np.ndarray, **options) -> np.ndarray:
    """Steady state from ``y0`` with the analytic algebraic Jacobian."""
    return agr.solve_steady(y0, jac=ALG_jacobian(agr), **options)


def steady_checks(agr: Aggregator, K: Kirchhoff, refs: dict, y: np.ndarray) -> dict:
    """Hand checks of a solved steady state against the energy balance and the turbulent gap
    law: at a common pressure drop the per-plate flow goes as ``A * Dh**(5/7)``."""
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
    pool_flow = float(st[K.name][K.component_edge(refs["pool"])])
    T_pool_top_cell = float(np.asarray(st["Pool"]["T_cool"])[0])
    return dict(
        T_out_error_max=max(T_out_err),
        split_error_max=max(abs(r - 1.0) for r in ratios),
        pool_flow=pool_flow,
        inlet_matches_pool=abs(T_in - T_pool_top_cell) < 0.05,
        wall_above_coolant=all(wall),
        heads={leg: float(np.asarray(st[refs["pumps"][leg].name]["pressure"])) for leg in "AB"},
        mdot=mdot,
        T_in=T_in,
    )


def decay_heat(P0: float):
    """Fission-product decay heat in watts after infinite operation at ``P0``, as a function of
    the time since shutdown in seconds; the library's decay-heat standards ship without data,
    so this is the classic power-law fit with a one-second offset so that it is finite and
    about seven percent of ``P0`` at shutdown."""
    return lambda tau: 0.066 * P0 * (tau + 1.0) ** -0.2


def _meat_temperatures(fuel: Fuel, state: State) -> np.ndarray:
    T = np.asarray(state[fuel.name]["T"]).reshape(-1)
    return T[np.asarray(fuel.meat, dtype=bool).reshape(-1)]


def _kinetics_graph(p: Params, channels: dict, fuels: dict, trip: LowFlowTrip, ref_state: State, P0: float) -> CalculationGraph:
    weight_sum = plate_count_weighted_sum()
    temp_worth, ref_temp = {}, {}
    for k in HEATED:
        fraction = p.feedback_scale * CORE[k].plates * CORE[k].factor / weight_sum
        temp_worth[channels[k]] = np.full(Z_N, p.coolant_worth * fraction / Z_N)
        temp_worth[fuels[k]] = np.full(Z_N * FUEL_N, p.fuel_worth * fraction / (Z_N * FUEL_N))
        ref_temp[channels[k]] = np.asarray(ref_state[channels[k].name]["T_cool"], dtype=float).copy()
        ref_temp[fuels[k]] = _meat_temperatures(fuels[k], ref_state)
    controls = ReactivityController(input_reactivity=lambda state, t_state, t: trip.rho(t))
    pk = PointKineticsWInput(generation_time=GENERATION_TIME, delayed_neutron_fractions=BETA,
                             delayed_groups_decay_rates=LAMBDA, temp_worth=temp_worth, ref_temp=ref_temp,
                             controls=controls, name="PK")
    dh = decay_heat(P0)
    graph = CalculationGraph.from_decoupled(
        pk, funcs={pk: dict(t=identity, power_input=lambda t: dh(max(t - trip.t_trip, 0.0)))})
    for k in HEATED:
        graph = graph + symmetric_plate(channels[k], fuels[k])
        graph.graph.add_edge(channels[k], pk, variables=("T",))
        graph.graph.add_edge(fuels[k], pk, variables=("T",))
        graph.graph.add_edge(pk, fuels[k], variables=("power",))
    return graph


def operating_point(p: Params, guess_kind: str = "ballpark", seed: int = 0, band: float = 0.3):
    """Thermal-hydraulics solved first at the nominal power, then the kinetics attached with
    its feedback references at the solved temperatures. Returns ``(agr, K, refs, y)`` for the
    full system, ``y`` consistent to solver tolerance."""
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


_LAYOUT = {
    "J_pool_top": (-1.5, 1.5), "J_plenum": (1.5, 1.5), "J_pool_bot": (-1.5, -0.5),
    "J_tank_in": (1.5, -1.5), "J_tank_out": (-0.5, -1.5), "J_split": (-3.5, -1.5), "J_merge": (-3.5, 0.0),
}


def draw(K: Kirchhoff, ax=None):
    """Draw the flow graph of a built loop: junctions at their elevation-like positions, one
    labelled curve per edge (parallel edges fan out). Returns the axes drawn on."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    g = K.g
    pos = {node: _LAYOUT[str(node)] for node in g.nodes}
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=900, node_color="white", edgecolors="black")
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node), ha="center", va="center", fontsize=8)
    for i, (u, v, key, comps) in enumerate(g.edges(keys=True, data="comps")):
        parallel = g.number_of_edges(u, v)
        rad = 0.0 if parallel == 1 else -0.35 + 0.7 * key / (parallel - 1)
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], ax=ax, connectionstyle=f"arc3,rad={rad}",
                               arrows=True, arrowstyle="-|>", node_size=900)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        xm, ym = xm + rad * dy * 0.5, ym - rad * dx * 0.5
        ax.text(xm, ym, comps[0].name, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_axis_off()
    return ax
