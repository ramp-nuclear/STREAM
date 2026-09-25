import numpy as np
import pandas as pd
import pytest

from stream.viz.sweep import Sweep


def frame(rows, sys=None, stat=None):
    """Build a State-like frame from (calculation, variable, i, j, value) rows."""
    df = pd.DataFrame(rows, columns=["calculation", "variable", "i", "j", "value"]).astype(
        dict(calculation="category", variable="category", i="uint16", j="uint16", value="float64")
    )
    if sys is not None:
        df = df.assign(sys=np.asarray(sys, dtype=float), stat=np.asarray(stat, dtype=float))
    return df


def channel_rows(T, chfr, power):
    """A four-cell channel named CC with a scalar power and a plate named plate."""
    rows = [("CC", "T_cool", 0, j, T[j]) for j in range(4)]
    rows += [("CC", "CHFR", 0, j, chfr[j]) for j in range(4)]
    rows += [("CC", "power", 0, 0, power)]
    rows += [("plate", "T", i, j, 50.0 + 10 * i + j) for i in range(2) for j in range(3)]
    return rows


class FakeChannel:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dz = np.abs(np.diff(self.bounds))
        self.centers = 0.5 * (self.bounds[1:] + self.bounds[:-1])


class FakePlate:
    def __init__(self):
        self.z_bounds = np.array([0.0, 0.5, 1.0])
        self.x_bounds = np.array([0.0, 1e-3, 2e-3, 3e-3])
        self.z_centers = np.array([0.25, 0.75])
        self.x_centers = np.array([0.5e-3, 1.5e-3, 2.5e-3])
        self.dz = np.diff(self.z_bounds)
        self.dx = np.diff(self.x_bounds)
        self.meat = np.array([[0, 1, 0], [0, 1, 0]])


class FakeAgr(dict):
    """Only ``agr[name]`` is needed by the module."""


@pytest.fixture
def agr():
    return FakeAgr(CC=FakeChannel([0.0, 0.1, 0.2, 0.3, 0.4]), plate=FakePlate())


def make_case_frame(power, mdot, banded=False):
    T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
    chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
    rows = channel_rows(T, chfr, power)
    if not banded:
        return frame(rows)
    n = len(rows)
    return frame(rows, sys=np.full(n, 0.1), stat=np.full(n, 0.05))


@pytest.fixture
def grid_cases():
    return [dict(power=p, mdot=m) for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3)]


LABELS = {"mdot": ("Mass flow", "kg/s")}


@pytest.fixture
def sweep(grid_cases, agr):
    frames = [make_case_frame(**c) for c in grid_cases]
    return Sweep(grid_cases, frames, agr, labels=LABELS)


@pytest.fixture
def banded_sweep(grid_cases, agr):
    frames = [make_case_frame(**c, banded=True) for c in grid_cases]
    return Sweep(grid_cases, frames, agr, labels=LABELS)


@pytest.fixture
def holed_sweep(grid_cases, agr):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[7] = None
    with pytest.warns(UserWarning, match="holes"):
        return Sweep(grid_cases, frames, agr, labels=LABELS)
