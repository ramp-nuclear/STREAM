"""Conservation checks of a solved loss-of-flow run: mass at every junction, energy over the run, pressure around the natural-circulation loop."""
from __future__ import annotations

import numpy as np

from stream.examples.lofa.params import CORE
from stream.substances import light_water
from stream.units import g as G


def series(agr, sol, node, var):
    """Values of ``var`` of ``node`` at every output time of ``sol``, one row per time."""
    return np.asarray(agr.at_times(sol, node, var)).reshape(len(sol.time), -1)


def junction_mass_balance(agr, K, refs, sol):
    """Largest weighted mass imbalance at any junction over the run, kg/s, and the largest per junction.
    Edge flows are weighted by their ``signify`` multiplicity; the balance closes when every value is round-off."""
    g = K.g
    worst = 0.0
    per = {}
    for j in g.nodes:
        total = np.zeros(len(sol.time))
        for u, v, key, comps in g.edges(keys=True, data="comps"):
            w = float(g.edges[u, v, key].get("signify", 1.0))
            flow = series(agr, sol, K, K.component_edge(comps[0]))[:, 0]
            if v == j:
                total += w * flow
            if u == j:
                total -= w * flow
        per[str(j)] = float(np.max(np.abs(total)))
        worst = max(worst, per[str(j)])
    return worst, per


def _cp(T):
    return np.asarray(light_water.specific_heat(T), dtype=float)


def _rho(T):
    return np.asarray(light_water.density(T), dtype=float)


def energy_budget(agr, K, refs, sol, p):
    """Energy balance of the whole run, J.

    ``fed_J`` is the heat the core produced: the point-kinetics power integrated over the output times.
    ``hx_removed_J`` is the heat the exchanger took out: its forward flow times the specific heat times
    the drop from the ``J_merge`` temperature to ``p.hx_outlet``, integrated over the run. ``stored_J``
    is the rise in internal energy from the first to the last output time of the water in every
    component with a coolant temperature (its density and specific heat taken at the first output
    time) and of the fuel plates, each core channel and plate counted as many times as its group has
    plates; ``parts_J`` lists it per component. ``unaccounted_J`` is
    ``fed_J - hx_removed_J - stored_J``, the closure residual: positive when heat the core produced
    was neither removed by the exchanger nor stored. The balance closes when it is small against ``fed_J``."""
    t = np.asarray(sol.time)
    power = series(agr, sol, refs["pk"], "power")[:, 0]
    fed = np.trapezoid(power, t)
    hx_flow = series(agr, sol, K, K.component_edge(refs["hx"]))[:, 0]
    hx_in = series(agr, sol, agr["J_merge"], "Tin")[:, 0]
    hx_out = p.hx_outlet
    hx_removed = np.trapezoid(np.maximum(hx_flow, 0.0) * _cp(hx_in) * (hx_in - hx_out), t)
    stored = 0.0
    parts = {}
    plates = {ch: CORE[k].plates for k, ch in refs["channels"].items()}
    water = [(comp.name, comp) for comp in agr.graph if "T_cool" in getattr(comp, "variables", {})]
    for name, comp in water:
        T = series(agr, sol, comp, "T_cool")
        n = plates.get(comp, 1)
        mass = comp.pipe.area * comp.dz * _rho(T[0])
        dU = n * float(np.sum(mass * _cp(T[0]) * (T[-1] - T[0])))
        parts[name] = dU
        stored += dU
    for name, fuel in refs["fuels"].items():
        T = series(agr, sol, fuel, "T")
        n = CORE[name].plates
        shape = (fuel.m, fuel.n)
        rho = np.broadcast_to(np.asarray(fuel.material.density, dtype=float), shape)
        cp = np.broadcast_to(np.asarray(fuel.material.specific_heat, dtype=float), shape)
        cap = rho * cp * np.outer(fuel.dz, fuel.dx) * fuel.y_length
        parts[f"{name}_fuel"] = n * float(np.sum(cap * (T[-1] - T[0]).reshape(shape)))
        stored += parts[f"{name}_fuel"]
    return {"fed_J": float(fed), "hx_removed_J": float(hx_removed), "stored_J": float(stored), "parts_J": parts,
            "unaccounted_J": float(fed - hx_removed - stored)}


def loop_pressure_budget(agr, K, refs, sol, loop_names, idx=-1):
    """Pressure difference of every named component at output index ``idx``, Pa, signed along the loop
    traversed in the order given, so the values sum to the loop residual; and the hot plate's buoyancy head, Pa.
    The names must chain edge to edge around a closed loop in that order, otherwise ``ValueError`` is raised;
    components sharing an edge are listed consecutively and take that edge's sign."""
    edges = {comp.name: (u, v, key) for u, v, key, comps in K.g.edges(keys=True, data="comps") for comp in comps}
    st = agr.save(sol.data[idx])
    budget = {}
    start, node, edge = None, None, None
    for name in loop_names:
        u, v, key = edges.get(name, (None, None, None))
        if u is None:
            raise ValueError(f"{name} is not a component on a flow edge")
        if (u, v, key) != edge:
            if node is None:
                start, sign, node = u, 1.0, v
            elif u == node:
                sign, node = 1.0, v
            elif v == node:
                sign, node = -1.0, u
            else:
                raise ValueError(f"{name} does not chain with the loop at junction {node}")
        edge = (u, v, key)
        budget[name] = sign * float(np.sum(np.asarray(st[name]["pressure"], dtype=float)))
    if node != start:
        raise ValueError(f"{loop_names[-1]} does not return the loop to its start junction {start}")
    ch = refs["channels"]["hot"]
    T = np.asarray(st[ch.name]["T_cool"], dtype=float)
    T_in = float(np.asarray(st["J_plenum"]["Tin"]))
    buoyancy = float(G * np.sum((_rho(T_in) - _rho(T)) * ch.dz))
    return budget, buoyancy
