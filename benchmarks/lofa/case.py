"""
LOFA benchmark case: a self-contained MTR-like research-reactor loop.

Topology:
    J_top ──[Channel (ChannelAndContacts, heated, DOWN)]── J_bot
    J_top ──[Flapper + GravityFlapper (bypass leg)]─────── J_bot
    J_bot ──[Pump | Flywheel | HX | GravityReturn]───────── J_top

Forced convection drives flow DOWN the heated channel. After pump trip the flow
coasts down on the flywheel; near-zero flow the flapper opens; buoyancy from the
heated channel then reverses the flow UPWARD into natural circulation.

This file only *builds* systems and guesses. Stage logic lives in run.py.
"""
import numpy as np

from stream.calculations import (
    Fuel,
    Gravity,
    HeatExchanger,
    Inertia,
    Junction,
    KirchhoffWDerivatives,
    Pump,
    Resistor,
    Solid,
)
from stream.calculations.channel import Channel, ChannelAndContacts
from stream.calculations.flapper import Flapper, continuously_differentiable_relaxation
from stream.composition import (
    FlowGraph,
    check_gravity_mismatch,
    flow_edge,
    symmetric_plate,
    uniform_x_power_shape,
    x_boundaries,
)
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.pipe_geometry import EffectivePipe
from stream.state import State
from stream.substances import light_water
from stream.utilities import identity

TIN = 40.0                 # °C cold-leg inlet
MDOT0 = 0.5                # kg/s nominal forced flow
POWER_DEMO = 1000.0        # W, leaves the loop nearly isothermal
POWER_REALISTIC = 83.6e3   # W, the intended plate power (ΔT ≈ 40 °C at MDOT0)
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


def build(power: float = POWER_DEMO, stop_on_open: bool = False):
    """Build the single-channel LOFA system.

    With ``stop_on_open=False`` the caller opens the flapper at a time of its
    choosing; with ``stop_on_open=True`` the solver detects the opening and
    stops there.

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
    """Expert guess: the hydraulic helper plus a thermal pre-solve of the plate."""
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
    """A guess without expert knowledge: uniform temperatures, nominal flows,
    zero pressure drops and an order-of-magnitude HTC."""
    channel, fuel, flapper, pump = refs["channel"], refs["fuel"], refs["flapper"], refs["pump"]
    n = CHANNEL_Z_N
    k_state = {}
    for comp, mdot in ((channel, MDOT0), (pump, MDOT0), (flapper, 0.0)):
        k_state[K.component_edge(comp)] = mdot
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


GEN_GAPS = dict(hot=0.002, warm=0.002, wide=0.003, bypass=0.004)
GEN_POWERS = dict(hot=84.0e3, warm=33.6e3, wide=50.0e3)   # W at full power
GEN_INERTIA_L = 3e5          # Pa·s²/kg, sized for the lower parallel resistance


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


def build_general(stop_on_open: bool = False):
    """Build the general multichannel LOFA system at full power.

    Four parallel channels run between the same plena, positive flow downward:
    HotChannel (2.0 mm gap, 84.0 kW), WarmChannel (2.0 mm, 33.6 kW),
    WideChannel (3.0 mm, 50.0 kW) and an unheated Bypass (4.0 mm, a plain
    ``Channel``). A flapper leg with a resistor and the pump, flywheel and
    heat-exchanger return leg close the loop. Each heated channel has its own
    ``Fuel`` plate.

    Returns (agr, K, refs); refs["channels"] and refs["fuels"] are dicts keyed
    hot/warm/wide, with bypass in channels as well.
    """
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=DP0_CHANNEL, name="Pump")
    flywheel = Inertia(inertia=GEN_INERTIA_L, name="Flywheel")
    hx = HeatExchanger(outlet=TIN, name="HX")
    gravity_ret = Gravity(fluid=light_water, disposition=-CHANNEL_LENGTH, name="GravityReturn")
    gravity_flap = Gravity(fluid=light_water, disposition=CHANNEL_LENGTH, name="GravityFlapper")
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


def scram(agr, refs):
    """Pump trip + reactor scram: kill pump head, switch power to decay heat."""
    refs["pump"].p = 0.0
    for k, fuel in refs["fuels"].items():
        agr.funcs[fuel]["power"] = decay_power(GEN_POWERS[k])


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


def gen_t_open_from(m0: float) -> float:
    """Flapper-opening estimate from the solved steady total flow (the static
    GEN_T_OPEN_ESTIMATE uses the split guess and lands about 35 % early)."""
    k = DP0_CHANNEL / m0**2
    return (GEN_INERTIA_L / (k * m0)) * (m0 / GEN_FLAPPER_THRESHOLD - 1)
