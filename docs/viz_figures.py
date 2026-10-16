"""Make the figures of the visualization page from a plate-and-channel sweep.

The sweep is 120 steady states of a symmetric fuel plate and its coolant
channel over power, mass flow and axial power shape; the five figures are
written as PNG files to ``docs/source/_static/viz`` or to the directory given
as the only argument. It takes about two minutes and is run by hand, never by
the docs build.
"""

import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stream import EffectivePipe, Solid, light_water
from stream.analysis.thresholds import Saha_Zuber_OSV, Sudo_Kaminaga_CHF, threshold_analysis
from stream.calculations import ChannelAndContacts, Fuel
from stream.calculations.channel import Direction
from stream.composition import symmetric_plate, uniform_x_power_shape, x_boundaries
from stream.composition.subsystems import symmetric_plate_steady_state
from stream.physical_models.heat_transfer_coefficient import spl_htc, wall_heat_transfer_coeff
from stream.utilities import cosine_shape
from stream.viz import Sweep, at_least, at_most, cases, run_cases

TIN, P_ABS = 40.0, 2e5
L, W, GAP, N = 0.7, 0.070, 0.002, 14
CLAD_N, FUEL_N = 2, 6
CLAD_W, MEAT_W = 0.0005, 0.0006
CLADDING = Solid(density=2700, specific_heat=900, conductivity=167)
FUEL_MEAT = Solid(density=3500, specific_heat=750, conductivity=40)
PIPE = EffectivePipe.rectangular(length=L, edge1=W, edge2=GAP, heated_edge=W)

H_WALL = partial(
    wall_heat_transfer_coeff,
    h_spl=spl_htc("regime_dependent", re_bounds=(2000.0, 5000.0), aspect_ratio=GAP / W, Lh=L),
)
SHAPES = {"cosine": cosine_shape, "flat": lambda zb: np.diff(zb) / zb[-1]}
LABELS = {
    "power": ("Power per plate", "kW", 1e-3),
    "mdot": ("Mass flow", "kg/s"),
    "shape": ("Axial power shape", ""),
    "OSVR": ("OSV ratio", ""),
    "CHFR": ("CHF ratio", ""),
    "T_wall, left": ("Clad surface temperature", "°C"),
}
LIMITS = {"OSVR": at_least(1.3), "CHFR": at_least(1.3), "T_wall, left": at_most(110.0)}
POWERS = [10e3, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]
MDOTS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def fuel(kind) -> Fuel:
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
        power_shape=uniform_x_power_shape(N, FUEL_N, CLAD_N, CLAD_W, MEAT_W, W, z_shaper=SHAPES[kind]),
        y_length=W,
        name="Fuel",
    )


def plates():
    built = {}
    for shape in SHAPES:
        c = ChannelAndContacts(
            z_boundaries=np.linspace(0.0, L, N + 1), fluid=light_water, pipe=PIPE, h_wall_func=H_WALL, name="Channel"
        )
        f = fuel(shape)
        agr = symmetric_plate(c, f, {c: {"mdot": 0.2, "p_abs": P_ABS, "Tin": TIN}, f: {"power": 1e4}}).to_aggregator()
        built[shape] = (c, f, agr)
    return built


PLATES = plates()
POST = threshold_analysis(CHF=Sudo_Kaminaga_CHF, OSV=partial(Saha_Zuber_OSV, direction=Direction.left))


def model(power, mdot, shape):
    c, f, agr = PLATES[shape]
    state = symmetric_plate_steady_state(c, f, mdot=mdot, p_abs=P_ABS, power=power, Tin=TIN)
    state = POST(state, agr, c.name)
    ch = state[c.name]
    q = np.maximum(np.asarray(ch["q, left"]), np.asarray(ch["q, right"]))
    tsat = light_water.sat_temperature(np.asarray(ch["static_pressure"]))
    if np.any(np.asarray(ch["T_cool"]) >= tsat):
        raise RuntimeError("bulk coolant reached saturation")
    ch["CHFR"] = np.asarray(ch["CHF"]) / q
    ch["OSVR"] = np.asarray(ch["OSV"]) / q
    return state.to_dataframe()


def sweep():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs, frames = run_cases(model, cases(power=POWERS, mdot=MDOTS, shape=list(SHAPES)))
        return Sweep(cs, frames, PLATES["cosine"][2], labels=LABELS, limits=LIMITS)


def figures(s):
    return {
        "profile": s.profile(["T_cool", "T_wall, left"], where={"power": 40e3, "mdot": 0.2, "shape": "cosine"}),
        "curves": s.reduce("OSVR", "min", versus="power", where={"shape": "cosine"}),
        "panels": s.reduce("T_wall, left", "max", versus="power"),
        "map": s.reduce("OSVR", "min", versus=("power", "mdot"), where={"shape": "cosine"}),
        "required": s.required(["OSVR", "CHFR", "T_wall, left"], solve_for="power", versus="mdot", limiting=True),
    }


OPTIONS = {
    "profile": {},
    "curves": {"color": "mdot", "crossing": "mark"},
    "panels": {"color": "mdot", "col": "shape", "show_limit": "shade"},
    "map": {"show_limit": "shade"},
    "required": {"color": "shape"},
}


def main(out):
    matplotlib.use("Agg")
    out.mkdir(parents=True, exist_ok=True)
    for name, result in figures(sweep()).items():
        fig, _ = result.plot(**OPTIONS[name])
        fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "source" / "_static" / "viz")
