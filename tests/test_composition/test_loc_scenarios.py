"""What a system does while it loses its inventory: drain curves, reversal, conservation."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from stream.calculations import (
    Environment,
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
from stream.calculations.flapper import Flapper
from stream.calculations.kirchhoff import to_str
from stream.composition import FlowGraph, break_to_ambient, check_gravity_mismatch, flow_edge, pool
from stream.composition.subsystems import loc_steady_state
from stream.jacobians import DAE_jacobian
from stream.physical_models.pressure_drop.discharge import discharge_cd, drain_level, drain_time
from stream.substances import light_water
from stream.units import g
from stream.utilities import identity

T0 = 30.0
A_TANK, L0, Z_UNCOVERY = 2.0, 4.0, 1.0
A_HOLE, CD = 5e-4, discharge_cd("sharp")
RHO = float(light_water.density(T0))
DP_PUMP, R_CORE, R_SUCTION = 5e3, 5e3, 2e3
STIFF = dict(rtol=1e-6, atol=1e-3, max_steps=100000)


def _series(agr, sol, node, var):
    return np.asarray(agr.at_times(sol, node, var)).squeeze()


def _flow(agr, sol, k, component):
    return _series(agr, sol, k, k.component_edge(component))


def _drained(agr, tank, vec, time, **options):
    tank.unpin()
    agr.refresh_mass()
    return agr.solve(vec, time, **options)


def _opening_grid(t_end, points=400):
    """A grid fine enough over the first seconds to integrate an opening ramp accurately."""
    return np.concatenate((np.linspace(0.0, 4.0, 21), np.linspace(4.0, t_end, points)[1:]))


def test_a_tank_drains_on_the_torricelli_curve():
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    mid = Junction(name="mid")
    hole = Orifice(light_water, A_HOLE, CD, dp_eps=1e-3, name="break")
    fg = FlowGraph(
        *pool(tank, outflows={mid: 0.0}),
        break_to_ambient(mid, hole, env),
        surface_nodes={tank: None, env: None},
        funcs={hole: dict(t=identity)},
    )
    agr, k = fg.aggregator, fg.kirchhoff
    vec = agr.solve_steady(loc_steady_state(k, {agr["head_pool_mid"]: 0.0}, T0))

    hole.open(0.0)
    sol = _drained(agr, tank, vec, np.linspace(0.0, 5000.0, 501))

    level = _series(agr, sol, tank, "level")
    assert sol.t_stop == pytest.approx(drain_time(L0, Z_UNCOVERY, A_TANK, A_HOLE, CD), abs=0.5)
    assert np.abs(level - drain_level(sol.time, L0, A_TANK, A_HOLE, CD)).max() < 1e-3
    assert level[-1] == pytest.approx(Z_UNCOVERY, abs=1e-3)


def _coasting_loop(*, break_area=5e-3, t_break=30.0):
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    j_top, j_suction = Junction(name="j_top"), Junction(name="j_suction")
    j_bot, j_return = Junction(name="j_bot"), Junction(name="j_return")

    m0, suction_line = 1.0, Resistor(R_SUCTION, name="suction_line")
    dp0 = (R_CORE + R_SUCTION) * m0
    pump = Pump(pressure=dp0, name="pump")
    flywheel = Inertia(inertia=2e4, name="flywheel")
    hx = HeatExchanger(T0, name="hx")
    down = Gravity(light_water, 3.0, name="down_leg")
    core = Resistor(R_CORE, name="core")
    up = Gravity(light_water, -3.0, name="core_riser")
    flapper = Flapper(open_at_current=0.15, f=5.0, fluid=light_water, area=1e-3, open_rate=1.0, name="flapper")
    bypass = Gravity(light_water, -3.0, name="bypass_riser")
    breach = Orifice(light_water, break_area, CD, t_break=t_break, open_rate=1.0, name="breach")

    fg = FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_return: 0.0}),
        flow_edge((j_top, j_suction), suction_line),
        flow_edge((j_suction, j_bot), pump, flywheel, hx, down, ref_mdot_for=(flapper, tank)),
        flow_edge((j_bot, j_return), core, up),
        flow_edge((j_bot, j_suction), flapper, bypass),
        break_to_ambient(j_suction, breach, env),
        surface_nodes={tank: None, env: None},
        inertial_comps=[flywheel],
        k_constructor=KirchhoffWDerivatives,
        funcs={flapper: dict(t=identity), breach: dict(t=identity)},
    )
    check_gravity_mismatch(fg.kirchhoff)
    agr = fg.aggregator
    mdots = {
        agr["head_pool_j_top"]: m0, agr["head_j_return_pool"]: m0, suction_line: m0,
        pump: m0, flywheel: m0, hx: m0, down: m0, core: m0, up: m0,
        flapper: 0.0, bypass: 0.0,
    }
    return fg, dict(tank=tank, pump=pump, core=core, flapper=flapper, breach=breach,
                    mdots=mdots, m0=m0, dp0=dp0)


def test_a_coasting_loop_reverses_and_drains_through_a_suction_break():
    """Pump trip, flapper opening, and a suction break on one inertial free-surface graph."""
    t_break = 30.0
    fg, r = _coasting_loop(t_break=t_break)
    agr, k = fg.aggregator, fg.kirchhoff
    tank, flapper = r["tank"], r["flapper"]
    vec = agr.solve_steady(loc_steady_state(k, r["mdots"], T0))
    assert vec[agr.var_index(k, k.component_edge(r["core"]))] == pytest.approx(r["m0"], rel=1e-6)

    dp0 = r["dp0"]
    agr.funcs[r["pump"]] = dict(pressure=lambda t: dp0 * np.exp(-t / 3.0))
    sol = _drained(
        agr, tank, vec, np.linspace(0.0, 600.0, 301), jacfn=DAE_jacobian(agr), **STIFF
    )

    level = _series(agr, sol, tank, "level")
    m_core = _flow(agr, sol, k, r["core"])
    after_break = sol.time >= t_break

    assert not sol.completed
    assert m_core.min() < 0.0
    assert np.isfinite(flapper.t_open)
    assert np.all(np.diff(level[after_break]) <= 0.0)
    assert sol.t_stop is not None
    assert tank.name in {name for event in sol.events for name in event.stopped}


def _bottom_break_loop(*, break_area=A_HOLE, t_break=0.0, abs_pressure=False):
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    j_top, j_bot, j_return = Junction(name="j_top"), Junction(name="j_bot"), Junction(name="j_return")
    pump = Pump(pressure=DP_PUMP, name="pump")
    core = Resistor(R_CORE, name="core")
    breach = Orifice(light_water, break_area, CD, dp_eps=1e-3, t_break=t_break, open_rate=1.0, name="breach")
    fg = FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_return: 0.0}),
        flow_edge((j_top, j_bot), core),
        flow_edge((j_bot, j_return), pump),
        break_to_ambient(j_bot, breach, env),
        surface_nodes={tank: None, env: None},
        abs_pressure_comps=[core] if abs_pressure else None,
        funcs={breach: dict(t=identity)},
    )
    agr = fg.aggregator
    m0 = DP_PUMP / R_CORE
    mdots = {agr["head_pool_j_top"]: m0, agr["head_j_return_pool"]: m0, pump: m0, core: m0}
    return fg, dict(tank=tank, core=core, pump=pump, breach=breach, mdots=mdots, m0=m0)


def _run_bottom_break(**kwargs):
    fg, r = _bottom_break_loop(**kwargs)
    agr, k = fg.aggregator, fg.kirchhoff
    vec = agr.solve_steady(loc_steady_state(k, r["mdots"], T0))
    sol = _drained(agr, r["tank"], vec, _opening_grid(4000.0) + kwargs.get("t_break", 0.0))
    return fg, r, sol


def test_a_bottom_break_leaves_the_circulation_untouched():
    fg, r, sol = _run_bottom_break()
    agr, k = fg.aggregator, fg.kirchhoff
    m_core = _flow(agr, sol, k, r["core"])
    m_break = _flow(agr, sol, k, r["breach"])
    level = _series(agr, sol, r["tank"], "level")
    driving = RHO * g * level[-1] - DP_PUMP

    assert np.allclose(m_core, r["m0"], rtol=1e-4)
    assert np.all(np.diff(m_break[sol.time > 5.0]) < 0.0)
    assert m_break[-1] == pytest.approx(CD * A_HOLE * np.sqrt(2.0 * RHO * driving), rel=1e-3)
    assert level[-1] == pytest.approx(Z_UNCOVERY, abs=1e-3)
    assert r["tank"].name in {name for event in sol.events for name in event.stopped}


def test_the_break_carries_exactly_the_inventory_the_pool_loses():
    fg, r, sol = _run_bottom_break()
    agr, k = fg.aggregator, fg.kirchhoff
    m_break = _flow(agr, sol, k, r["breach"])
    level = _series(agr, sol, r["tank"], "level")

    discharged = np.trapezoid(m_break, sol.time)
    assert discharged == pytest.approx(RHO * A_TANK * (L0 - level[-1]), rel=1e-3)


def _suction_break_loop(*, break_area, t_break=0.0):
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    j_top, j_suction, j_return = Junction(name="j_top"), Junction(name="j_suction"), Junction(name="j_return")
    m0, suction_line = 1.0, Resistor(R_SUCTION, name="suction_line")
    dp0 = (R_CORE + R_SUCTION) * m0
    pump = Pump(pressure=dp0, name="pump")
    core = Resistor(R_CORE, name="core")
    breach = Orifice(light_water, break_area, CD, dp_eps=1e-3, t_break=t_break, open_rate=1.0, name="breach")
    fg = FlowGraph(
        *pool(tank, outflows={j_top: 0.0}, inflows={j_return: 0.0}),
        flow_edge((j_top, j_suction), suction_line),
        flow_edge((j_suction, j_return), pump, core),
        break_to_ambient(j_suction, breach, env),
        surface_nodes={tank: None, env: None},
        funcs={breach: dict(t=identity)},
    )
    agr = fg.aggregator
    mdots = {
        agr["head_pool_j_top"]: m0, agr["head_j_return_pool"]: m0,
        suction_line: m0, pump: m0, core: m0,
    }
    return fg, dict(tank=tank, pump=pump, core=core, breach=breach, mdots=mdots, m0=m0, dp0=dp0)


def test_a_suction_break_reverses_the_core_and_empties_faster_than_a_bottom_break():
    area, t_break = 5e-4, 20.0
    fg, r = _suction_break_loop(break_area=area, t_break=t_break)
    agr, k = fg.aggregator, fg.kirchhoff
    dp0 = r["dp0"]
    vec = agr.solve_steady(loc_steady_state(k, r["mdots"], T0))
    agr.funcs[r["pump"]] = dict(pressure=lambda t: dp0 * np.exp(-t / 3.0))
    sol = _drained(agr, r["tank"], vec, np.linspace(0.0, 4000.0, 401))

    m_core = _flow(agr, sol, k, r["core"])
    reversed_at = sol.time[np.argmax(m_core < 0.0)]

    assert m_core[0] == pytest.approx(r["m0"], rel=1e-6)
    assert m_core.min() < 0.0
    assert reversed_at > t_break

    _, _, bottom = _run_bottom_break(break_area=area, t_break=t_break)
    assert sol.t_stop < bottom.t_stop


def test_holes_stop_flowing_as_the_level_passes_their_elevation():
    z_high, z_low, area = 2.5, 0.5, 5e-3
    tank = Tank(
        light_water, A_TANK, L0, z_uncovery=0.2, fixed_temperature=T0,
        marks={"upper break": z_high}, name="pool",
    )
    env = Environment(name="ambient")
    j_high, j_low = Junction(name="j_high"), Junction(name="j_low")
    high = Orifice(light_water, area, CD, dp_eps=100.0, closes_below=(tank, z_high), name="upper_break")
    low = Orifice(light_water, area, CD, dp_eps=100.0, name="lower_break")
    fg = FlowGraph(
        *pool(tank, outflows={j_high: z_high, j_low: z_low}),
        break_to_ambient(j_high, high, env),
        break_to_ambient(j_low, low, env),
        surface_nodes={tank: None, env: None},
        funcs={high: dict(t=identity), low: dict(t=identity)},
    )
    agr, k = fg.aggregator, fg.kirchhoff
    vec = agr.solve_steady(
        loc_steady_state(k, {agr["head_pool_j_high"]: 0.0, agr["head_pool_j_low"]: 0.0}, T0)
    )

    high.open(0.0)
    low.open(0.0)
    sol = _drained(agr, tank, vec, np.linspace(0.0, 700.0, 301))

    level = _series(agr, sol, tank, "level")
    m_high = _flow(agr, sol, k, high)
    m_low = _flow(agr, sol, k, low)
    assert np.isfinite(high.t_close)
    assert m_high[sol.time < high.t_close].max() > 0.1 * m_low.max()
    assert np.interp(high.t_close, sol.time, level) == pytest.approx(z_high, abs=1e-3)
    assert m_high[-1] == pytest.approx(0.0, abs=1e-9)
    assert m_low[-1] < 1e-4 * m_low.max()
    assert level[-1] == pytest.approx(z_low, abs=1e-3)


def test_a_severed_pipe_discharges_from_both_ends_on_one_graph():
    t_cut = 20.0
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    j_hot, j_cold, j_return = Junction(name="j_hot"), Junction(name="j_cold"), Junction(name="j_return")
    pipe_run = Orifice(
        light_water, 5e-2, discharge_cd("pipe_stub"), t_break=-1.0, close_rate=1.0, name="pipe_run"
    )
    pump = Pump(pressure=DP_PUMP, name="pump")
    core = Resistor(R_CORE, name="core")
    stub_hot = Orifice(light_water, 1e-3, discharge_cd("pipe_stub"), open_rate=1.0, name="hot_stub")
    stub_cold = Orifice(light_water, 1e-3, discharge_cd("pipe_stub"), open_rate=1.0, name="cold_stub")
    fg = FlowGraph(
        *pool(tank, outflows={j_hot: 0.0}, inflows={j_return: 0.0}),
        flow_edge((j_hot, j_cold), pipe_run),
        flow_edge((j_cold, j_return), pump, core),
        break_to_ambient(j_hot, stub_hot, env),
        break_to_ambient(j_cold, stub_cold, env),
        surface_nodes={tank: None, env: None},
        funcs={c: dict(t=identity) for c in (pipe_run, stub_hot, stub_cold)},
    )
    agr, k = fg.aggregator, fg.kirchhoff
    equations = len(agr)
    m0 = DP_PUMP / R_CORE
    vec = agr.solve_steady(
        loc_steady_state(
            k,
            {agr["head_pool_j_hot"]: m0, agr["head_j_return_pool"]: m0, pipe_run: m0, pump: m0, core: m0},
            T0,
        )
    )

    intact = _drained(agr, tank, vec, np.linspace(0.0, t_cut, 21))
    assert len(agr) == equations

    cut_state = intact.data[-1]
    pipe_run.close(t_cut, cut_state[agr.var_index(k, k.component_edge(pipe_run))])
    stub_hot.open(t_cut)
    stub_cold.open(t_cut)
    agr.funcs[pump] = dict(
        pressure=lambda t: DP_PUMP if t <= t_cut else DP_PUMP * np.exp(-(t - t_cut) / 2.0)
    )
    severed = agr.solve(cut_state, np.linspace(t_cut, 1200.0, 301), jacfn=DAE_jacobian(agr), **STIFF)

    m_hot = _flow(agr, severed, k, stub_hot)
    m_cold = _flow(agr, severed, k, stub_cold)
    m_run = _flow(agr, severed, k, pipe_run)
    m_core = _flow(agr, severed, k, core)

    assert len(agr) == equations
    assert m_hot.max() > 0.1 and m_cold.max() > 0.1
    assert m_run[-1] == pytest.approx(0.0, abs=1e-9)
    assert m_core.min() < 0.0


def test_absolute_pressure_follows_the_falling_level():
    fg, r, sol = _run_bottom_break(abs_pressure=True)
    agr, k = fg.aggregator, fg.kirchhoff
    level = _series(agr, sol, r["tank"], "level")
    p_abs = _series(agr, sol, k, to_str(("p_abs", r["core"])))

    assert np.allclose(p_abs - p_abs[0], RHO * g * (level - L0), rtol=1e-3, atol=1.0)


def test_a_heated_pool_follows_its_own_energy_balance():
    m_in, t_hot, t_end = 2.0, 40.0, 3000.0
    tank = Tank(light_water, A_TANK, L0, z_uncovery=0.5, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    j_in, j_out = Junction(name="j_in"), Junction(name="j_out")
    feed = Pump(mdot0=m_in, name="feed")
    heater = HeatExchanger(t_hot, name="heater")
    drain = Orifice(light_water, A_HOLE, CD, dp_eps=1e-3, open_rate=1.0, name="drain")
    fg = FlowGraph(
        flow_edge((env, j_in), feed, heater),
        *pool(tank, outflows={j_out: 0.0}, inflows={j_in: 0.0}),
        break_to_ambient(j_out, drain, env),
        surface_nodes={tank: None, env: None},
        funcs={drain: dict(t=identity)},
    )
    agr, k = fg.aggregator, fg.kirchhoff
    vec = agr.solve_steady(
        loc_steady_state(
            k,
            {feed: m_in, heater: m_in, agr["head_j_in_pool"]: m_in, agr["head_pool_j_out"]: 0.0},
            T0,
        )
    )

    drain.open(0.0)
    tank.fixed_temperature = None
    sol = _drained(agr, tank, vec, _opening_grid(t_end, points=300))

    level = _series(agr, sol, tank, "level")
    temperature = _series(agr, sol, tank, "T")
    m_out = _flow(agr, sol, k, drain)

    def reduced(_, y):
        depth, T = y
        rho = float(light_water.density(T))
        out = CD * A_HOLE * rho * np.sqrt(2.0 * g * max(depth, 0.0))
        return [(m_in - out) / (rho * A_TANK), m_in * (t_hot - T) / (rho * A_TANK * depth)]

    reference = solve_ivp(reduced, (0.0, t_end), [L0, T0], t_eval=sol.time, rtol=1e-10, atol=1e-12)
    assert np.allclose(level, reference.y[0], rtol=1e-3)
    assert np.allclose(temperature, reference.y[1], rtol=1e-3)

    cp = float(light_water.specific_heat(T0))
    energy = RHO * A_TANK * cp * level * temperature
    balance = np.trapezoid(cp * (m_in * t_hot - m_out * temperature), sol.time)
    assert energy[-1] - energy[0] == pytest.approx(balance, rel=1e-2)
