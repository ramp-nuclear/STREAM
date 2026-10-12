"""Geometry, core groups, operating point and modelling choices of the loss-of-flow case.

The module constants are the plate geometry (``LENGTH``, ``WIDTH`` in m, cell counts and
widths of cladding and meat, ``Z_N`` axial cells), the cladding and fuel-meat solids, the core
table (``CORE``, ``HEATED``, ``UNHEATED``), the primary pipe drop ``PIPE1_DROP`` in m, the
natural-circulation flow signs ``NC_SIGNS``, the nominal per-plate flow ``PER_PLATE_REFERENCE``
in kg/s, and six-group delayed-neutron data (``BETA``, ``LAMBDA`` in 1/s, ``GENERATION_TIME``
in s)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stream.calculations import Solid
from stream.pipe_geometry import EffectivePipe

LENGTH, WIDTH = 0.7, 0.070
CLAD_N, FUEL_N = 2, 8
CLAD_W, MEAT_W = 0.0005, 0.0006
CLADDING = Solid(density=2700, specific_heat=900, conductivity=167)
FUEL_MEAT = Solid(density=3500, specific_heat=750, conductivity=40)
Z_N = 10


@dataclass(frozen=True)
class Group:
    """One core group: coolant ``gap`` in m, power peaking ``factor`` and number of ``plates``."""

    gap: float
    factor: float
    plates: int


CORE: dict[str, Group] = {
    "hot": Group(0.0022, 1.6, 1),
    "normal": Group(0.0022, 1.0, 300),
    "low": Group(0.0022, 0.5, 90),
    "bypass": Group(0.0100, 0.0, 10),
}
HEATED = tuple(k for k, g in CORE.items() if g.factor > 0)
UNHEATED = tuple(k for k, g in CORE.items() if g.factor == 0)


def plate_count_weighted_sum() -> float:
    """Sum over the core groups of plate count times peaking factor."""
    return sum(g.plates * g.factor for g in CORE.values())


def power_share(name: str) -> float:
    """Fraction of the core power one plate of group ``name`` receives."""
    return CORE[name].factor / plate_count_weighted_sum()


def plate_pipe(gap: float) -> EffectivePipe:
    """Rectangular channel of one plate with coolant ``gap`` in m, heated on its wide side."""
    return EffectivePipe.rectangular(length=LENGTH, edge1=WIDTH, edge2=gap, heated_edge=WIDTH)


@dataclass(frozen=True)
class Params:
    """Operating point and modelling choices. Scales multiply the nominal value named.
    ``pump_head_estimate`` is the head the loop needs at nominal flow without the pump
    curve, Pa; each pump's curve is a quadratic droop that loses ``pump_droop`` times that
    at the nominal leg flow, so the pump's own nominal head is the estimate plus the droop."""

    power: float = 10e6
    power_scale: float = 1.0
    hx_outlet: float = 38.0
    p_surface: float = 101325.0
    core_depth: float = 8.0
    pool_volume: float = 300.0
    pool_area_scale: float = 1.0
    tank_volume: float = 20.0
    tank_volume_scale: float = 1.0
    pipe_diameter: float = 0.4
    pipe_friction: float = 0.02
    basement_drop: float = 7.3
    hx_dp: float = 5e4
    pump_head_estimate: float = 8.15e4
    pump_droop: float = 0.6
    inertia: float = 1.35e4
    inertia_scale: float = 1.0
    valve_k: float = 0.25
    valve_reverse_ratio: float = 1e5
    flapper_fraction: float = 0.3
    flapper_f: float = 5.0
    flapper_area: float = 0.05
    flapper_stuck: bool = False
    friction: str = "regime"
    gap_hot: float = 0.0022
    trip_threshold: float = 0.85
    trip_delay: float = 0.5
    rod_worth_dollars: float = 10.0
    rod_duration: float = 1.0
    feedback_scale: float = 1.0
    coolant_worth: float = -1e-4
    fuel_worth: float = -2e-5
    t_pumps: float = 10.0
    ramp_width: float = 0.5
    scenario: str = "both"
    stop_at_saturation: bool = True


PIPE1_DROP = 4.3
NC_SIGNS = dict(pool=-1.0, flapper=1.0)
PER_PLATE_REFERENCE = 0.5
BETA = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
GENERATION_TIME = 5e-5


def nominal_flows(p: Params) -> dict[str, float]:
    """Per-plate nominal flow of each group (proportional to area times the square root of
    the hydraulic diameter, anchored at ``PER_PLATE_REFERENCE`` for a 2.2 mm gap) and the
    core total under ``"total"``. ``PER_PLATE_REFERENCE`` is the flow through a 2.2 mm plate
    channel at the nominal point, kg/s."""
    gaps = {k: (p.gap_hot if k == "hot" else g.gap) for k, g in CORE.items()}
    weight = lambda gap: plate_pipe(gap).area * np.sqrt(plate_pipe(gap).hydraulic_diameter)
    scale = PER_PLATE_REFERENCE / weight(0.0022)
    flows = {k: scale * weight(gap) for k, gap in gaps.items()}
    flows["total"] = sum(CORE[k].plates * flows[k] for k in CORE)
    return flows
