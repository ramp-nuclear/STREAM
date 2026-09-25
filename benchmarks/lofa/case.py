"""
LOFA benchmark case — a self-contained MTR-like research-reactor loop.

Topology:
    J_top ──[Channel (ChannelAndContacts, heated, DOWN)]── J_bot
    J_top ──[Flapper + GravityFlapper (bypass leg)]─────── J_bot
    J_bot ──[Pump | Flywheel | HX | GravityReturn]───────── J_top

Forced convection drives flow DOWN the heated channel. After pump trip the flow
coasts down on the flywheel; near-zero flow the flapper opens; buoyancy from the
heated channel then reverses the flow UPWARD into natural circulation.

This file only *builds* systems and guesses. Stage logic lives in run.py.
"""
from functools import partial

import numpy as np

from stream.calculations import (
    Fuel, Gravity, HeatExchanger, Inertia, Junction, KirchhoffWDerivatives, Pump, Solid,
)
from stream.calculations.channel import ChannelAndContacts
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.composition import (
    FlowGraph, flow_edge, symmetric_plate, check_gravity_mismatch, x_boundaries,
    uniform_x_power_shape,
)
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.physical_models.pressure_drop import pressure_diff
from stream.physical_models.pressure_drop.friction import (
    friction_factor, rectangular_laminar_correction,
)
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.utilities import identity

# ── Nominal parameters ────────────────────────────────────────────────────────
TIN = 40.0                 # °C cold-leg inlet
MDOT0 = 0.5                # kg/s nominal forced flow
POWER_DEMO = 1000.0        # W  — the power the legacy model could actually solve at
POWER_REALISTIC = 83.6e3   # W  — the originally intended power (ΔT ≈ 40 °C at MDOT0)
DP0_CHANNEL = 35000.0      # Pa pump head at steady state
INERTIA_L = 7e5            # Pa·s²/kg flywheel coefficient
FLAPPER_THRESHOLD = 0.15   # kg/s
FLAPPER_F = 5.0
FLAPPER_AREA = 1e-3        # m²
P_REF = 2e5                # Pa absolute reference at J_top
CHANNEL_Z_N = 10

# Estimated flapper opening (hyperbolic coastdown with quadratic friction)
_K_QUAD = DP0_CHANNEL / MDOT0**2
T_OPEN_ESTIMATE = (INERTIA_L / (_K_QUAD * MDOT0)) * (MDOT0 / FLAPPER_THRESHOLD - 1)

# ── Geometry / materials (MTR-like plate) ─────────────────────────────────────
CHANNEL_LENGTH = 0.7       # m
CHANNEL_WIDTH = 0.070      # m
CHANNEL_GAP = 0.002        # m
CLAD_N, FUEL_N = 2, 8
CLAD_W, MEAT_W = 0.0005, 0.0006

CLADDING = Solid(density=2700, specific_heat=900, conductivity=167)
FUEL_MEAT = Solid(density=3500, specific_heat=750, conductivity=40)


def _pipe() -> EffectivePipe:
    return EffectivePipe.rectangular(
        length=CHANNEL_LENGTH, edge1=CHANNEL_WIDTH, edge2=CHANNEL_GAP,
        heated_edge=CHANNEL_WIDTH,
    )


def _fuel(z_n: int, name: str = "Fuel") -> Fuel:
    shape = (z_n, FUEL_N + 2 * CLAD_N)
    meat = np.zeros(shape, dtype=bool)
    meat[:, CLAD_N:-CLAD_N] = True
    materials = np.empty(shape, dtype=object)
    materials[meat] = FUEL_MEAT
    materials[~meat] = CLADDING
    return Fuel(
        z_boundaries=np.linspace(0.0, CHANNEL_LENGTH, z_n + 1),
        x_boundaries=x_boundaries(CLAD_N, FUEL_N, CLAD_W, MEAT_W),
        material=Solid.from_array(materials),
        meat_indices=meat,
        power_shape=uniform_x_power_shape(z_n, FUEL_N, CLAD_N, CLAD_W, MEAT_W, CHANNEL_WIDTH),
        y_length=CHANNEL_WIDTH,
        name=name,
    )


def _apply_regime_friction(channel, gap: float) -> None:
    """Swap ``channel``'s pressure-drop closure for the non-optimistic
    ``regime_dependent`` friction law (laminar / transition / turbulent) with
    the rectangular-duct laminar correction ``k_R`` for this gap. This is the
    physically-correct law (Blasius alone is optimistic in the laminar regime a
    coasting-down channel enters)."""
    k_R = float(rectangular_laminar_correction(gap / CHANNEL_WIDTH))
    f = friction_factor("regime_dependent", re_bounds=(2000., 5000.), k_R=k_R)
    channel.pressure = partial(pressure_diff, fluid=channel.fluid,
                               pipe=channel.pipe, dz=channel.dz, f=f)


def build(power: float = POWER_DEMO, stop_on_open: bool = False,
          regime_friction: bool = False):
    """Build the LOFA system.

    stop_on_open=False matches the legacy choreographed run (manual pre-open);
    stop_on_open=True is the natural event path (solver detects the opening).

    regime_friction=False keeps the channel's default Blasius law so the
    bit-stability guard stages (B/C/E) are untouched; True applies the
    non-optimistic ``regime_dependent`` law (used by stage F).

    Returns (agr, K, refs) where refs holds the component handles.
    """
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=DP0_CHANNEL, name="Pump")
    flywheel = Inertia(inertia=INERTIA_L, name="Flywheel")
    hx = HeatExchanger(outlet=TIN, name="HX")
    gravity_ret = Gravity(fluid=light_water, disposition=-CHANNEL_LENGTH, name="GravityReturn")
    gravity_flap = Gravity(fluid=light_water, disposition=CHANNEL_LENGTH, name="GravityFlapper")
    flapper = Flapper(
        open_at_current=FLAPPER_THRESHOLD, f=FLAPPER_F, fluid=light_water,
        area=FLAPPER_AREA, open_rate=1.0, stop_on_open=stop_on_open,
        relaxation=continuously_differentiable_relaxation, name="Flapper",
    )
    channel = ChannelAndContacts(
        z_boundaries=np.linspace(0.0, CHANNEL_LENGTH, CHANNEL_Z_N + 1),
        fluid=light_water, pipe=_pipe(), name="Channel",
    )
    if regime_friction:
        _apply_regime_friction(channel, CHANNEL_GAP)
    fuel = _fuel(CHANNEL_Z_N)

    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_top, j_bot), flapper, gravity_flap),
        flow_edge((j_bot, j_top), pump, flywheel, hx, gravity_ret, ref_mdot_for=(flapper,)),
        inertial_comps=[flywheel, channel],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channel],
        funcs={flapper: dict(t=identity)},
        reference_node=(j_top, P_REF),
    )
    check_gravity_mismatch(fg.kirchhoff)

    thermal = symmetric_plate(channel, fuel, funcs={fuel: dict(power=power)}).to_aggregator()
    agr = fg.aggregator + thermal

    refs = dict(fg=fg, pump=pump, flapper=flapper, channel=channel, fuel=fuel,
                power=power)
    return agr, fg.kirchhoff, refs


def expert_guess(refs, power: float) -> State:
    """The legacy 'good guess' recipe: hydraulic helper + thermal pre-solve."""
    fg, channel, fuel = refs["fg"], refs["channel"], refs["fuel"]
    hydraulic = fg.guess_steady_state(
        mdots={channel: MDOT0, refs["pump"]: MDOT0, refs["flapper"]: 0.0},
        temperature=TIN,
    )
    thermal = symmetric_plate_steady_state(
        channel, fuel, mdot=MDOT0, p_abs=P_REF, power=power, Tin=TIN,
    )
    return State.merge(hydraulic, thermal)


def ballpark_guess(agr, K, refs) -> State:
    """A sane, no-expert-knowledge guess: uniform temperatures, nominal flows,
    zero pressure drops, order-of-magnitude HTC. The steady solve must
    converge from it without expert help (stage A)."""
    channel, fuel, flapper, pump = refs["channel"], refs["fuel"], refs["flapper"], refs["pump"]
    n = CHANNEL_Z_N
    k_state = {}
    for comp, mdot in ((channel, MDOT0), (pump, MDOT0), (flapper, 0.0)):
        k_state[K.component_edge(comp)] = mdot
    # every remaining Kirchhoff variable (other edges, mdot2, p_abs) → 0 / P_REF
    vec = np.zeros(len(K))
    vec[K.variables_by_type["abs_pressure"]] = P_REF
    k_state = K.save(vec) | k_state

    guess = {
        K.name: k_state,
        channel.name: dict(
            T_cool=np.full(n, TIN), pressure=0.0,
            h_left=np.full(n, 1e3), h_right=np.full(n, 1e3),
        ),
        fuel.name: dict(
            T=np.full(fuel.shape[0] * fuel.shape[1], TIN),
            T_wall_left=np.full(n, TIN), T_wall_right=np.full(n, TIN),
        ),
    }
    for comp in ("Pump", "Flywheel", "HX", "GravityReturn", "GravityFlapper", "Flapper"):
        guess[comp] = dict(Tin=TIN, pressure=0.0)
    for j in ("J_top", "J_bot"):
        guess[j] = dict(Tin=TIN)
    return State(guess)


# ══════════════════════════════════════════════════════════════════════════════
# General multichannel LOFA system (benchmark stage G — the capstone case)
#
# Four parallel channels between the same plena, each isolating one physics axis:
#   HotChannel  2.0 mm gap, 84.0 kW — reverses FIRST (assertable vs Warm)
#   WarmChannel 2.0 mm gap, 33.6 kW — same geometry, less power → reverses later
#   WideChannel 3.0 mm gap, 50.0 kW — different geometry AND heat (regime variety)
#   Bypass      4.0 mm gap, unheated plain Channel — pressure-balance passenger
# plus the flapper NC leg and the pump/flywheel/HX return leg.
#
# Scram at pump trip: power follows a Way-Wigner-like decay 0.066*(t+0.1)^-0.2.
#
# Wiring decisions:
#   * all channel edges J_top -> J_bot (positive = down); identical 0.7 m span
#     (same plena!), so check_gravity_mismatch closes all 5 loops at zero flow;
#   * every Channel is in inertial_comps (required by KirchhoffWDerivatives);
#   * only ChannelAndContacts instances need abs_pressure_comps;
#   * each heated channel gets its OWN Fuel via symmetric_plate; the per-plate
#     thermal aggregators share no nodes/edges/funcs keys with each other or
#     with the flow aggregator, so CalculationGraph merging is clobber-safe.
# ══════════════════════════════════════════════════════════════════════════════

GEN_GAPS = dict(hot=0.002, warm=0.002, wide=0.003, bypass=0.004)
GEN_POWERS_FULL = dict(hot=84.0e3, warm=33.6e3, wide=50.0e3)   # W at full power
# De-rate: at full power the hot channel crosses saturation on
# the reversal spike; 0.70 keeps the trajectory peak sub-saturation with margin.
GEN_POWER_SCALE = 0.70
GEN_POWERS = {k: GEN_POWER_SCALE * v for k, v in GEN_POWERS_FULL.items()}
GEN_INERTIA_L = 3e5          # Pa·s²/kg — recalibrated for the lower parallel resistance


def _gap_pipe(gap: float) -> EffectivePipe:
    return EffectivePipe.rectangular(
        length=CHANNEL_LENGTH, edge1=CHANNEL_WIDTH, edge2=gap,
        heated_edge=CHANNEL_WIDTH,
    )


def _gen_mdot_split():
    """Turbulent-ish split guess: mdot_i ∝ A_i*sqrt(Dh_i), anchored to the
    validated single 2mm-gap channel drawing 0.5 kg/s at DP0_CHANNEL."""
    w = {k: (_p := _gap_pipe(g)).area * np.sqrt(_p.hydraulic_diameter)
         for k, g in GEN_GAPS.items()}
    w_ref = w["hot"]
    mdots = {k: MDOT0 * wk / w_ref for k, wk in w.items()}
    return mdots, sum(mdots.values())


GEN_MDOTS, GEN_MDOT_TOTAL = _gen_mdot_split()
GEN_FLAPPER_THRESHOLD = 0.3 * GEN_MDOT_TOTAL
_GEN_K_QUAD = DP0_CHANNEL / GEN_MDOT_TOTAL**2
GEN_T_OPEN_ESTIMATE = (GEN_INERTIA_L / (_GEN_K_QUAD * GEN_MDOT_TOTAL)) \
    * (GEN_MDOT_TOTAL / GEN_FLAPPER_THRESHOLD - 1)


def decay_power(p0: float):
    """Scram decay-heat curve (Way-Wigner-like): ~10.5% at t=0+, ~2.6% at 100 s."""
    return lambda t: p0 * 0.066 * (t + 0.1) ** -0.2


def smooth_down(t: float, t0: float, w: float) -> float:
    """Cosine half-ramp: 1 for ``t <= t0``, 0 for ``t >= t0 + w``, a C1 cosine
    descent between. Ramps the pump head to zero and blends reactor power to
    decay heat at scram so the trip's IC stays consistent (‖F(0)‖ ≈ 0)."""
    return 1.0 if t <= t0 else (0.0 if t >= t0 + w else 0.5 * (1 + np.cos(np.pi * (t - t0) / w)))


def wire_ramp_scram(agr, pump, fuels_and_powers, t_scram: float = 3.0, width: float = 0.5):
    """Wire the ramp-scram boundary conditions used by stages F and G.

    Over ``[t_scram, t_scram + width]`` the pump head coasts from ``DP0_CHANNEL``
    to zero and each fuel's power blends from its pre-scram value ``p0`` to its
    decay-heat curve. ``fuels_and_powers`` maps each Fuel to its ``p0`` (one
    entry for the single-channel case, one per plate for the general case).

    The double-lambda binds
    ``p0``/``dpf`` per fuel — without it every closure would capture the last."""
    agr.funcs.setdefault(pump, {})["pressure"] = \
        lambda t: DP0_CHANNEL * smooth_down(t, t_scram, width)
    for fuel, p0 in dict(fuels_and_powers).items():
        dpf = decay_power(p0)
        agr.funcs[fuel]["power"] = (
            lambda p0, dpf: (lambda t: p0 if t <= t_scram
                             else smooth_down(t, t_scram, width) * p0
                             + (1 - smooth_down(t, t_scram, width)) * dpf(t - t_scram))
        )(p0, dpf)


def build_general(stop_on_open: bool = False, regime_friction: bool = True):
    """Build the general multichannel LOFA system at the de-rated stage-G power.

    regime_friction=True (the default) applies the physically-correct
    ``regime_dependent`` friction law per channel — this stage carries no
    bit-stability guard, so it uses the non-optimistic law. Each channel's
    ``k_R`` comes from its own gap: ``rectangular_laminar_correction(gap/width)``.

    Returns (agr, K, refs); refs["channels"]/["fuels"] are dicts keyed
    hot/warm/wide (+ bypass in channels)."""
    from stream.calculations.channel import Channel

    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=DP0_CHANNEL, name="Pump")
    flywheel = Inertia(inertia=GEN_INERTIA_L, name="Flywheel")
    hx = HeatExchanger(outlet=TIN, name="HX")
    gravity_ret = Gravity(fluid=light_water, disposition=-CHANNEL_LENGTH, name="GravityReturn")
    gravity_flap = Gravity(fluid=light_water, disposition=CHANNEL_LENGTH, name="GravityFlapper")
    from stream.calculations import Resistor
    resist_flap = Resistor(resistance=10.0, name="ResistFlap")
    flapper = Flapper(
        open_at_current=GEN_FLAPPER_THRESHOLD, f=FLAPPER_F, fluid=light_water,
        area=FLAPPER_AREA, open_rate=1.0, stop_on_open=stop_on_open,
        relaxation=continuously_differentiable_relaxation, name="Flapper",
    )

    z = np.linspace(0.0, CHANNEL_LENGTH, CHANNEL_Z_N + 1)
    channels = {
        k: ChannelAndContacts(z_boundaries=z, fluid=light_water,
                              pipe=_gap_pipe(GEN_GAPS[k]), name=f"{k.capitalize()}Channel")
        for k in ("hot", "warm", "wide")
    }
    channels["bypass"] = Channel(z_boundaries=z, fluid=light_water,
                                 pipe=_gap_pipe(GEN_GAPS["bypass"]), name="Bypass")
    if regime_friction:
        for k, ch in channels.items():
            _apply_regime_friction(ch, GEN_GAPS[k])
    fuels = {k: _fuel(CHANNEL_Z_N, name=f"{k.capitalize()}Fuel") for k in ("hot", "warm", "wide")}

    fg = FlowGraph(
        flow_edge((j_top, j_bot), channels["hot"]),
        flow_edge((j_top, j_bot), channels["warm"]),
        flow_edge((j_top, j_bot), channels["wide"]),
        flow_edge((j_top, j_bot), channels["bypass"]),
        flow_edge((j_top, j_bot), flapper, gravity_flap, resist_flap),
        flow_edge((j_bot, j_top), pump, flywheel, hx, gravity_ret, ref_mdot_for=(flapper,)),
        inertial_comps=[flywheel, *channels.values()],
        k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channels["hot"], channels["warm"], channels["wide"]],
        funcs={flapper: dict(t=identity)},
        reference_node=(j_top, P_REF),
    )
    check_gravity_mismatch(fg.kirchhoff)

    agr = fg.aggregator
    for k in ("hot", "warm", "wide"):
        agr = agr + symmetric_plate(channels[k], fuels[k],
                                    funcs={fuels[k]: dict(power=GEN_POWERS[k])}).to_aggregator()

    refs = dict(fg=fg, pump=pump, flapper=flapper, channels=channels, fuels=fuels)
    return agr, fg.kirchhoff, refs


def expert_guess_general(refs) -> State:
    """Per-channel expert guesses: hydraulic split + one thermal pre-solve per plate."""
    fg, channels, fuels = refs["fg"], refs["channels"], refs["fuels"]
    hydraulic = fg.guess_steady_state(
        mdots={channels[k]: GEN_MDOTS[k] for k in channels}
              | {refs["pump"]: GEN_MDOT_TOTAL, refs["flapper"]: 0.0},
        temperature=TIN,
    )
    thermals = [
        symmetric_plate_steady_state(channels[k], fuels[k], mdot=GEN_MDOTS[k],
                                     p_abs=P_REF, power=GEN_POWERS[k], Tin=TIN)
        for k in ("hot", "warm", "wide")
    ]
    return State.merge(hydraulic, *thermals)


def ballpark_guess_general(agr, K) -> State:
    """A sane, no-expert-knowledge guess for the general multichannel system:
    uniform temperatures, nominal flows, zero pressure drops, order-of-magnitude
    HTC. Stage G's steady solve starts from it."""
    non_k = [n for n in agr.graph if not hasattr(n, "variables_by_type")]
    vec = np.zeros(len(K))
    vec[K.variables_by_type["mdot"]] = MDOT0             # nominal flow on every edge
    vec[K.variables_by_type["abs_pressure"]] = P_REF
    return State.merge(
        State.uniform(non_k, TIN, "T_cool", "T", "T_wall_left", "T_wall_right", "Tin"),
        State.uniform(non_k, 1e3, "h_left", "h_right"),  # crude HTC (only channels have these)
        State.uniform(non_k, 0.0, "pressure"),           # component dp -> 0
        {K.name: K.save(vec)},                           # dict keyed by the opaque edge names
    )
