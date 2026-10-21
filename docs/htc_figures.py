"""Make the figures of the wall HTC and friction pages at low and reversed flow.

The three figures are written as PNG files to ``docs/source/_static/htc`` or to
the directory given as the only argument. The channel is a heated rectangular
gap with water at 40 °C. The script runs in seconds and is run by hand, never
by the docs build.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stream import EffectivePipe, light_water
from stream.physical_models.heat_transfer_coefficient import spl_htc
from stream.physical_models.pressure_drop import pressure_diff
from stream.physical_models.pressure_drop.friction import friction_factor, rectangular_laminar_correction

T_COOL, SUPERHEAT = 40.0, 10.0
L, W, N = 0.7, 0.070, 10
GAPS = (0.002, 0.006, 0.010)
RE_BOUNDS = (2000.0, 5000.0)
Q_WALL = 15e3
DPI = 120


def pipe(gap):
    return EffectivePipe.rectangular(length=L, edge1=W, edge2=gap, heated_edge=W)


def mdot_of_re(re, gap):
    p = pipe(gap)
    mu = float(light_water.viscosity(np.array([T_COOL]))[0])
    return re * p.area * mu / p.hydraulic_diameter


def h_wall(gap, re, delta_t, **options):
    p = pipe(gap)
    h_spl = spl_htc("regime_dependent", re_bounds=RE_BOUNDS, aspect_ratio=gap / W, Lh=L, **options)
    t_cool = np.array([T_COOL])
    t_wall = t_cool + delta_t
    return h_spl(
        coolant=light_water.to_properties((t_cool + t_wall) / 2),
        mdot=mdot_of_re(re, gap),
        Dh=p.hydraulic_diameter,
        A=p.area,
        T_cool=t_cool,
        T_wall=t_wall,
        coolant_funcs=light_water,
        depth=p.depth,
        develop_length=np.array([L / 2]),
    )[0]


def h_versus_re(out):
    side = np.logspace(-1, 4, 301)
    re = np.concatenate([-side[::-1], [0.0], side])
    fig, ax = plt.subplots(figsize=(7, 4))
    for sign in (-1, 1):
        ax.axvspan(*(sign * np.array(RE_BOUNDS)), color="0.9", zorder=0)
    for gap in GAPS:
        h = [h_wall(gap, r, SUPERHEAT) for r in re]
        (line,) = ax.plot(re, h, label=f"gap {gap * 1e3:.0f} mm")
        if gap == GAPS[0]:
            h_down = [h_wall(gap, r, SUPERHEAT, downflow="sudo") for r in re]
            ax.plot(re, h_down, "-.", color=line.get_color(), label=f"gap {gap * 1e3:.0f} mm, downflow law")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("signed Re, positive along gravity")
    ax.set_ylabel("h [W/m²K]")
    ax.legend(loc="upper center")
    fig.savefig(out / "h_vs_re.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def wall_balance(out):
    delta_t = np.logspace(-1, 2, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    for re, style in ((100.0, "-"), (20.0, "--")):
        q = [h_wall(GAPS[0], re, dt) * dt for dt in delta_t]
        ax.plot(delta_t, q, style, label=f"Re {re:.0f}")
    ax.axhline(Q_WALL, color="k", ls=":", label=f"{Q_WALL / 1e3:.0f} kW/m²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall superheat ΔT [K]")
    ax.set_ylabel("wall heat flux h·ΔT [W/m²]")
    ax.legend()
    fig.savefig(out / "wall_balance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def friction_through_zero(out):
    gap = GAPS[0]
    p = pipe(gap)
    t = np.full(N, T_COOL)
    k_r = rectangular_laminar_correction(gap / W)
    laws = {
        "regime dependent": (friction_factor("regime_dependent", re_bounds=RE_BOUNDS, k_R=k_r), "-"),
        "Blasius": (friction_factor("Blasius"), "--"),
    }
    side = np.logspace(-5, np.log10(0.5), 200)
    mdot = np.concatenate([-side[::-1], [0.0], side])
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (f, style) in laws.items():
        dp = [pressure_diff(t, t, light_water, m, p, L / N, f=f, mdot2=0.0).sum() for m in mdot]
        ax.plot(mdot, np.array(dp) / 1e3, style, label=label)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("mass flow, positive along gravity [kg/s]")
    ax.set_ylabel("pressure difference [kPa]")
    ax.legend()
    fig.savefig(out / "friction_through_zero.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main(out: Path) -> None:
    matplotlib.use("Agg")
    out.mkdir(parents=True, exist_ok=True)
    h_versus_re(out)
    wall_balance(out)
    friction_through_zero(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "source" / "_static" / "htc")
