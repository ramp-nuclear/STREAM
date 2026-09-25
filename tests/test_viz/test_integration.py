import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from networkx import DiGraph

from stream import Aggregator, State
from stream.calculations.channel import ChannelAndContacts
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water
from stream.viz import Sweep, at, run_cases

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
        raise RuntimeError("wall too hot for this test")
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
        cases, frames = run_cases(model, cases)
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
