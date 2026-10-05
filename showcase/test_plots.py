import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from showcase import plots


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _flows():
    t = np.linspace(0, 100, 21)
    return pd.DataFrame({"t": t} | {k: np.cos(t / (20 + i)) for i, k in enumerate(plots.LEGS)})


def test_flows_full_and_zoom():
    with plots.style():
        assert isinstance(plots.flows(_flows()), Figure)
        assert isinstance(plots.flows(_flows(), zoom=(20, 60), legs=["hot", "warm"]), Figure)


def test_peaks():
    t = np.linspace(0, 10, 5)
    df = pd.DataFrame({"t": t, "hot": 60 + t, "warm": 50 + t, "wide": 45 + t})
    with plots.style():
        assert isinstance(plots.peaks(df, 120.3), Figure)


def test_profile_on_given_axes():
    z = np.linspace(0, 0.7, 10)
    with plots.style():
        fig, ax = plt.subplots()
        assert plots.profile(z, 40 + 30 * z, 60 + 30 * z, ax=ax, title="t = 0 s") is fig


def test_schedule():
    with plots.style():
        assert isinstance(plots.schedule({"Scram": 3.0, "Flapper opens": 78.0, "Hot reverses": 814.0}), Figure)


def test_sweep_line_and_status_strip():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "peak_hot": [100, 105, 110, 118], "peak_warm": [80, 84, 88, 92],
                       "status": ["completed", "saturation", "failed", "timeout"]})
    with plots.style():
        assert isinstance(plots.sweep_line(df, "x", ["peak_hot", "peak_warm"]), Figure)
        assert isinstance(plots.status_strip(df, "x"), Figure)
