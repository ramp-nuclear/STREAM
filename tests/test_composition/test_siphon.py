"""A pool draining over a crest: what a siphon does that a plain hole cannot, and how a
breaker stops it.

The leg lifts the liquid above the free surface before dropping it below the pool bottom,
so the drive is the outlet elevation rather than the connection elevation, the crest
carries the lowest pressure in the system, and the only thing that arrests the discharge
at a chosen level is a break that latches shut there.
"""

import numpy as np
import pytest

from stream.analysis.thresholds import cavitation_crossings
from stream.calculations import Environment, Junction, Orifice, Tank
from stream.composition import FlowGraph, break_to_ambient, pool, siphon_leg
from stream.composition.subsystems import loc_steady_state
from stream.errors import StreamConstructionError
from stream.physical_models.pressure_drop.discharge import discharge_cd, drain_time
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water
from stream.units import g
from stream.utilities import identity

T0 = 30.0
A_TANK, L0, Z_UNCOVERY = 2.0, 4.0, 0.2
A_HOLE, CD = 5e-4, discharge_cd("sharp")
Z_INTAKE, Z_CREST, Z_OUTLET = 1.5, 5.0, -3.0
Z_BREAKER = 2.0
PIPE = EffectivePipe.circular(length=12.0, diameter=0.05)


def _series(agr, sol, node, var):
    return np.asarray(agr.at_times(sol, node, var)).squeeze()


def _flow(agr, sol, k, component):
    return _series(agr, sol, k, k.component_edge(component))


def _grid(t_end, points=300):
    """Fine over the opening ramp, then spread over the drain."""
    return np.concatenate((np.linspace(0.0, 4.0, 21), np.linspace(4.0, t_end, points)[1:]))


def _torricelli_rate(level, break_area=A_HOLE, temperature=T0):
    """How fast the pool surface falls at ``level``, driven down to the outlet."""
    rho = float(light_water.density(temperature))
    mdot = CD * break_area * np.sqrt(2.0 * rho * rho * g * (level - Z_OUTLET))
    return mdot / (rho * A_TANK)


def _ramp_allowance(level, close_rate, break_area=A_HOLE):
    """How far the level may still fall while a break of that size closes at that rate."""
    return 3.0 * _torricelli_rate(level, break_area) / close_rate


def _siphon(*, temperature=T0, break_area=A_HOLE, z_breaker=None, close_rate=10.0):
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=temperature, name="pool")
    env = Environment(name="ambient")
    breach = Orifice(
        light_water,
        break_area,
        CD,
        dp_eps=1e-3,
        closes_below=None if z_breaker is None else (tank, z_breaker),
        close_rate=close_rate,
        name="breach",
    )
    edges, crest = siphon_leg(
        tank,
        env,
        z_intake=Z_INTAKE,
        z_crest=Z_CREST,
        z_outlet=Z_OUTLET,
        pipe=PIPE,
        fluid=light_water,
        orifice=breach,
        z_breaker=z_breaker,
    )
    fg = FlowGraph(
        *edges,
        surface_nodes={tank: None, env: None},
        abs_pressure_comps=[crest],
        funcs={breach: dict(t=identity)},
    )
    comps = [c for *_, data in edges for c in data["comps"]]
    return fg, dict(tank=tank, breach=breach, crest=crest, mdots={c: 0.0 for c in comps})


def _drain(fg, r, t_end, points=300):
    agr, k = fg.aggregator, fg.kirchhoff
    vec = agr.solve_steady(loc_steady_state(k, r["mdots"], r["tank"].fixed_temperature))
    r["breach"].open(0.0)
    r["tank"].unpin()
    agr.refresh_mass()
    return agr, k, agr.solve(vec, _grid(t_end, points))


@pytest.fixture(scope="module")
def uncovering_siphon():
    fg, r = _siphon()
    return (*_drain(fg, r, 3500.0), r)


@pytest.fixture(scope="module")
def arrested_siphon():
    cache = {}

    def _arrested(close_rate=5.0, break_area=A_HOLE):
        key = (close_rate, break_area)
        if key not in cache:
            fg, r = _siphon(break_area=break_area, z_breaker=Z_BREAKER, close_rate=close_rate)
            cache[key] = (*_drain(fg, r, 1800.0), r)
        return cache[key]

    return _arrested


def test_the_pool_keeps_discharging_after_its_level_passes_the_connection(uncovering_siphon):
    """A hole stops when the surface reaches it; a siphon does not, because what drives it
    is the elevation of the outlet, not of the connection to the pool."""
    agr, k, sol, r = uncovering_siphon
    level = _series(agr, sol, r["tank"], "level")
    m_break = _flow(agr, sol, k, r["breach"])
    below = level < Z_INTAKE

    assert below.any()
    assert m_break[below].min() > 0.5 * m_break.max()
    assert level[-1] == pytest.approx(Z_UNCOVERY, abs=1e-3)
    assert sol.t_stop is not None
    assert r["tank"].name in {name for event in sol.events for name in event.stopped}


def test_a_latching_break_arrests_the_drain_at_its_elevation(arrested_siphon):
    close_rate = 5.0
    agr, k, sol, r = arrested_siphon(close_rate=close_rate)
    level = _series(agr, sol, r["tank"], "level")
    m_break = _flow(agr, sol, k, r["breach"])

    assert np.isfinite(r["breach"].t_close)
    assert sol.completed
    assert level[-1] == pytest.approx(Z_BREAKER, abs=_ramp_allowance(Z_BREAKER, close_rate))
    assert m_break[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_slower_closure_lets_the_level_undershoot_further(arrested_siphon):
    agr_f, _, sol_f, r_f = arrested_siphon(close_rate=5.0)
    agr_s, _, sol_s, r_s = arrested_siphon(close_rate=0.05)
    quick = _series(agr_f, sol_f, r_f["tank"], "level")[-1]
    lazy = _series(agr_s, sol_s, r_s["tank"], "level")[-1]

    assert lazy < quick - 1e-3


def test_the_arrest_level_does_not_follow_the_size_of_the_break(arrested_siphon):
    close_rate, wide = 5.0, 2.0 * A_HOLE
    agr_n, _, sol_n, r_n = arrested_siphon(close_rate=close_rate)
    agr_w, _, sol_w, r_w = arrested_siphon(close_rate=close_rate, break_area=wide)
    narrow = _series(agr_n, sol_n, r_n["tank"], "level")[-1]
    broad = _series(agr_w, sol_w, r_w["tank"], "level")[-1]

    assert sol_w.t_stop is None and sol_n.t_stop is None
    assert abs(broad - narrow) < _ramp_allowance(Z_BREAKER, close_rate, wide)


def test_a_hot_pool_reaches_saturation_at_the_crest():
    fg, r = _siphon(temperature=90.0)
    agr, _, sol = _drain(fg, r, 3500.0)
    crossings = cavitation_crossings(sol, agr)

    assert [c.component for c in crossings] == [r["crest"].name]
    assert 0.0 < crossings[0].t < sol.time[-1]
    assert crossings[0].margin < 0.0


def test_a_cold_pool_crosses_the_same_crest_subcooled():
    fg, r = _siphon(temperature=20.0)
    agr, _, sol = _drain(fg, r, 3500.0)
    assert cavitation_crossings(sol, agr) == []


def test_a_stub_drain_reproduces_the_closed_form_at_a_measured_discharge_coefficient():
    """A tank emptying through a short stub, against the quasi-steady Torricelli integral
    at the coefficient such a stub is measured to have."""
    cd, z_stop = 0.38, 1.0
    tank = Tank(light_water, A_TANK, L0, z_uncovery=z_stop, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    mid = Junction(name="mid")
    stub = Orifice(light_water, A_HOLE, cd, dp_eps=1e-3, name="stub")
    fg = FlowGraph(
        *pool(tank, outflows={mid: 0.0}),
        break_to_ambient(mid, stub, env),
        surface_nodes={tank: None, env: None},
        funcs={stub: dict(t=identity)},
    )
    agr = fg.aggregator
    vec = agr.solve_steady(loc_steady_state(fg.kirchhoff, {agr["head_pool_mid"]: 0.0}, T0))

    stub.open(0.0)
    tank.unpin()
    agr.refresh_mass()
    sol = agr.solve(vec, _grid(5400.0))

    expected = drain_time(L0, z_stop, A_TANK, A_HOLE, cd)
    assert sol.t_stop == pytest.approx(expected, rel=0.01)


def test_a_breaker_elevation_without_a_latching_break_is_refused():
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    breach = Orifice(light_water, A_HOLE, CD, name="breach")
    with pytest.raises(StreamConstructionError, match="closes_below"):
        siphon_leg(
            tank, env, z_intake=Z_INTAKE, z_crest=Z_CREST, z_outlet=Z_OUTLET,
            pipe=PIPE, fluid=light_water, orifice=breach, z_breaker=Z_BREAKER,
        )


def test_a_crest_below_the_leg_ends_is_refused():
    tank = Tank(light_water, A_TANK, L0, z_uncovery=Z_UNCOVERY, fixed_temperature=T0, name="pool")
    env = Environment(name="ambient")
    breach = Orifice(light_water, A_HOLE, CD, name="breach")
    with pytest.raises(StreamConstructionError, match="highest point"):
        siphon_leg(
            tank, env, z_intake=Z_INTAKE, z_crest=Z_INTAKE - 1.0, z_outlet=Z_OUTLET,
            pipe=PIPE, fluid=light_water, orifice=breach,
        )
