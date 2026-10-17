import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import frame, make_case_frame
from networkx import DiGraph

from stream import Aggregator, State, viz
from stream.calculations.channel import ChannelAndContacts
from stream.pipe_geometry import EffectivePipe
from stream.solvers import AlgRuntimeError
from stream.substances import light_water
from stream.viz import Sweep, at, at_least, cases, run_cases

PIPE = EffectivePipe.rectangular(length=0.7, edge1=0.070, edge2=0.002, heated_edge=0.070)
N = 6
Z = np.linspace(0.0, 0.7, N + 1)


def solve(wall, mdot):
    channel = ChannelAndContacts(z_boundaries=Z, fluid=light_water, pipe=PIPE)
    graph = DiGraph()
    graph.add_node(channel)
    agr = Aggregator(graph, funcs={channel: dict(Tin=40.0, Tin_minus=40.0, mdot=mdot, T_left=np.full(N, wall), p_abs=2e5)})
    guess = State({"CC": dict(T_cool=40.0, pressure=0.0, h_left=1e4, h_right=1e4)})
    if wall > 200.0:
        raise AlgRuntimeError("wall too hot for this test")
    state = agr.save(agr.solve_steady(guess))
    state["CC"]["margin"] = np.asarray(state["CC"]["T_wall, left"]) - np.asarray(state["CC"]["T_cool"])
    return agr, state.to_dataframe()


@pytest.fixture(scope="module")
def real_sweep():
    cases = [dict(wall=w, mdot=m) for w in (60.0, 80.0, 100.0, 250.0) for m in (0.2, 0.3)]
    agr_holder = {}

    def model(wall, mdot):
        agr, frame = solve(wall, mdot)
        agr_holder.setdefault("agr", agr)
        return frame

    with pytest.warns(UserWarning, match="wall too hot"):
        frames = run_cases(model, cases)
    with pytest.warns(UserWarning, match="holes"):
        return Sweep(cases, frames, agr_holder["agr"], labels={"wall": ("Wall temperature", "°C"), "mdot": ("Mass flow", "kg/s"), "margin": ("Wall superheat", "°C")})


def test_profile_reduce_and_required_run_on_a_real_channel(real_sweep):
    s = real_sweep
    assert s.holes.keys() == {6, 7}
    fig, ax = s.profile("T_cool", where=dict(wall=80.0, mdot=0.3)).plot()
    assert ax.get_xlabel() == "z [m]" and len(ax.get_lines()) == 1
    plt.close(fig)
    with pytest.warns(UserWarning, match="case 7"):
        fig, ax = s.profile(["T_cool", "T_wall, left"], where=dict(mdot=0.3)).plot()
    assert len(ax.get_lines()) == 6
    plt.close(fig)
    with pytest.warns(UserWarning, match="skipping unsolved"):
        r = s.reduce("margin", "max", versus="wall", where=dict(mdot=0.3))
    assert r.frame.value.is_monotonic_increasing
    with pytest.warns(UserWarning, match="skipping unsolved"):
        fig, ax = s.reduce("T_cool", at(z=0.65), versus="wall").plot(color="mdot")
    assert ax.get_legend().get_title().get_text() == "Mass flow [kg/s]"
    plt.close(fig)
    with pytest.warns(UserWarning, match="skipping unsolved"):
        req = s.required("margin", equals=25.0, reduce="max", solve_for="wall", versus="mdot")
    assert np.isfinite(req.frame.value).all()
    assert (req.frame.value > 60.0).all() and (req.frame.value < 100.0).all()
    fig, ax = req.plot()
    assert ax.get_ylabel() == "Wall temperature [°C]"
    plt.close(fig)


def test_sweep_with_a_choice_a_limit_a_hole_and_a_map(agr):
    def model(power, mdot, wall_law):
        if power == 6e6 and mdot == 0.3 and wall_law == "regime":
            raise AlgRuntimeError("no convergence")
        return make_case_frame(power, mdot * (1.1 if wall_law == "regime" else 1.0))

    cs = cases(power=[2e6, 4e6, 6e6, 8e6], mdot=[0.2, 0.3], wall_law=["Dittus", "regime"])
    with pytest.warns(UserWarning):
        frames = run_cases(model, cs)
    with pytest.warns(UserWarning) as record:
        sweep = Sweep(cs, frames, agr, limits={"CHFR": at_least(1.5)}, fill_holes=True)
    assert any("filled unsolved" in str(w.message) for w in record)
    assert "wall_law [Dittus, regime]" in repr(sweep)
    with pytest.warns(UserWarning, match="rests on interpolated case 11"):
        r = sweep.required("CHFR", solve_for="power", versus="mdot")
    assert r.dims == {"wall_law": ["Dittus", "regime"]} and r.safe_side == "below"
    fig, axes = r.plot(col="wall_law", show_limit="shade")
    assert axes.shape == (1, 2)
    plt.close(fig)
    fig, axes = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(col="wall_law", show_limit="shade")
    assert axes.shape == (1, 2)
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where=dict(wall_law="regime")).plot(color="mdot")
    assert any(ln.get_linestyle() == "--" for ln in ax.get_lines())
    plt.close(fig)


@pytest.mark.filterwarnings("error")
def test_the_module_example_runs(agr):
    def model(power, mdot, shape):
        r = power / 20e3 * 0.2 / mdot
        q = r * (np.array([0.7, 1.0, 1.2, 0.9]) if shape == "cosine" else np.ones(4))
        quantities = {
            "T_cool": 35.0 + 3.0 * np.arange(4) + 8.0 * q,
            "T_wall, left": 50.0 + 60.0 / 1.7 * q,
            "CHFR": 2.2 * 1.3 / q,
            "OSVR": 1.9 * 1.3 / q,
        }
        return frame([("CC", name, 0, j, v[j]) for name, v in quantities.items() for j in range(4)])

    lines = viz.__doc__.split("::\n\n", 1)[1].splitlines()
    code = []
    for line in lines:
        if line and not line.startswith("    "):
            break
        code.append(line)
    exec(textwrap.dedent("\n".join(code)), {**{n: getattr(viz, n) for n in viz.__all__}, "model": model, "agr": agr})
