"""The pump=0 buoyancy-driven natural-convection loop solves to steady state from a
realistic guess for a constant, a regime-dependent, and a pure-Elenbaas (natural)
wall HTC, and lands on the same NC state regardless of the (sane) initial guess.
The loop mass flow is set by the friction law alone, so all three HTC kinds must
agree on mdot; the HTC choice only moves the wall temperatures.

The Gravity leg runs in both flow directions at pump=0, so it is sandwiched between
two HX (``hx1 -> grav -> hx2``) to source the cold HX temperature regardless of flow
sign. A single HX would give the wrong buoyancy in reversed (NC) flow and no NC
fixed point.
"""
from functools import partial

import numpy as np
import pytest

from stream.calculations import (
    Fuel, Gravity, HeatExchanger, Junction, KirchhoffWDerivatives, Pump, Resistor, Solid,
)
from stream.calculations.channel import ChannelAndContacts
from stream.composition import (
    FlowGraph, flow_edge, symmetric_plate, check_gravity_mismatch,
    x_boundaries, uniform_x_power_shape,
)
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.pipe_geometry import EffectivePipe
from stream.physical_models.heat_transfer_coefficient import wall_heat_transfer_coeff, spl_htc
from stream.state import State
from stream.substances import light_water

TIN, P_REF, POWER = 40.0, 2e5, 1000.0
L, W, GAP, N = 0.7, 0.070, 0.002, 10
CLAD_N, FUEL_N = 2, 8
CLAD_W, MEAT_W = 0.0005, 0.0006
CLADDING = Solid(density=2700, specific_heat=900, conductivity=167)
FUEL_MEAT = Solid(density=3500, specific_heat=750, conductivity=40)

NC_MDOT = -0.010841  # kg/s, upflow NC for this MTR-plate geometry @ 1000 W, 2 bar


def _fuel() -> Fuel:
    shape = (N, FUEL_N + 2 * CLAD_N)
    meat = np.zeros(shape, dtype=bool)
    meat[:, CLAD_N:-CLAD_N] = True
    materials = np.empty(shape, dtype=object)
    materials[meat] = FUEL_MEAT
    materials[~meat] = CLADDING
    return Fuel(
        z_boundaries=np.linspace(0.0, L, N + 1),
        x_boundaries=x_boundaries(CLAD_N, FUEL_N, CLAD_W, MEAT_W),
        material=Solid.from_array(materials),
        meat_indices=meat,
        power_shape=uniform_x_power_shape(N, FUEL_N, CLAD_N, CLAD_W, MEAT_W, W),
        y_length=W, name="Fuel",
    )


def _htc(kind: str):
    if kind == "constant":
        return partial(wall_heat_transfer_coeff, h_spl=spl_htc("laminar_constant_nu"))
    if kind == "natural":
        return partial(wall_heat_transfer_coeff, h_spl=spl_htc("natural", Lh=L))
    return partial(wall_heat_transfer_coeff,
                   h_spl=spl_htc("regime_dependent", re_bounds=(2000.0, 5000.0),
                                 aspect_ratio=GAP / W, Lh=L))


def _build(dp_pump: float, kind: str):
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=dp_pump, name="Pump")
    hx1, hx2 = HeatExchanger(outlet=TIN, name="HX1"), HeatExchanger(outlet=TIN, name="HX2")
    grav = Gravity(fluid=light_water, disposition=-L, name="Grav")
    channel = ChannelAndContacts(z_boundaries=np.linspace(0.0, L, N + 1),
                                 fluid=light_water, pipe=EffectivePipe.rectangular(
                                     length=L, edge1=W, edge2=GAP, heated_edge=W),
                                 h_wall_func=_htc(kind), name="Channel")
    fuel = _fuel()
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), pump, hx1, grav, hx2),  # gravity sandwiched by HX
        inertial_comps=[channel], k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channel], reference_node=(j_top, P_REF),
    )
    check_gravity_mismatch(fg.kirchhoff)
    agr = fg.aggregator + symmetric_plate(channel, fuel, funcs={fuel: dict(power=POWER)}).to_aggregator()
    return agr, fg, channel, fuel, pump


def _guess(fg, channel, fuel, pump, mdot_g: float) -> State:
    return State.merge(
        fg.guess_steady_state(mdots={channel: mdot_g, pump: mdot_g}, temperature=TIN),
        symmetric_plate_steady_state(channel, fuel, mdot=mdot_g, p_abs=P_REF, power=POWER, Tin=TIN),
    )


def _mdot(agr, fg, y) -> float:
    K = fg.kirchhoff
    return float(np.asarray(y)[agr.sections[K]][K.variables_by_type["mdot"]][0])


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["constant", "regime_dependent", "natural"])
def test_natural_convection_converges(kind):
    """pump=0 buoyancy-driven NC converges to the expected upflow state."""
    agr, fg, channel, fuel, pump = _build(0.0, kind)
    y = agr.solve_steady(_guess(fg, channel, fuel, pump, -0.01))
    assert np.linalg.norm(agr.compute(y)) < 1e-6
    assert _mdot(agr, fg, y) == pytest.approx(NC_MDOT, abs=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["constant", "regime_dependent", "natural"])
def test_natural_convection_guess_insensitive(kind):
    """Different sane guesses land on the same NC state (no flakiness)."""
    agr, fg, channel, fuel, pump = _build(0.0, kind)
    mdots = [_mdot(agr, fg, agr.solve_steady(_guess(fg, channel, fuel, pump, g)))
             for g in (-0.005, -0.02, -0.05)]
    assert max(mdots) - min(mdots) < 1e-6
    assert mdots[0] == pytest.approx(NC_MDOT, abs=1e-4)


@pytest.mark.slow
def test_throttled_loop_converges_cold():
    """A heavily throttled NC loop (through-flow at stagnation scale) converges from a
    cold guess. The old Gr/Re^2 interpolation left this region with a folded wall
    characteristic (three/zero roots), so no cold solve could land; the monotone
    composition leaves a unique root."""
    power = 20.0
    j_top, j_bot = Junction(name="J_top"), Junction(name="J_bot")
    pump = Pump(pressure=0.0, name="Pump")
    throttle = Resistor(resistance=1e6, name="Throttle")
    hx1, hx2 = HeatExchanger(outlet=TIN, name="HX1"), HeatExchanger(outlet=TIN, name="HX2")
    grav = Gravity(fluid=light_water, disposition=-L, name="Grav")
    channel = ChannelAndContacts(z_boundaries=np.linspace(0.0, L, N + 1),
                                 fluid=light_water, pipe=EffectivePipe.rectangular(
                                     length=L, edge1=W, edge2=GAP, heated_edge=W),
                                 h_wall_func=_htc("regime_dependent"), name="Channel")
    fuel = _fuel()
    fg = FlowGraph(
        flow_edge((j_top, j_bot), channel),
        flow_edge((j_bot, j_top), pump, throttle, hx1, grav, hx2),
        inertial_comps=[channel], k_constructor=KirchhoffWDerivatives,
        abs_pressure_comps=[channel], reference_node=(j_top, P_REF),
    )
    agr = fg.aggregator + symmetric_plate(channel, fuel, funcs={fuel: dict(power=power)}).to_aggregator()
    guess = State.merge(
        fg.guess_steady_state(mdots={channel: -3e-4, pump: -3e-4}, temperature=TIN),
        symmetric_plate_steady_state(channel, fuel, mdot=-3e-4, p_abs=P_REF, power=power, Tin=TIN),
    )
    y = agr.solve_steady(guess)
    assert np.linalg.norm(agr.compute(y)) < 1e-6
    mdot = _mdot(agr, fg, y)
    assert -1e-3 < mdot < -5e-5
