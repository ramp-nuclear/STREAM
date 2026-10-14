"""The container: cases, frames and the aggregator, stacked and checked."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from pandas import DataFrame

from .labels import STREAM_LABELS, Labels

STATE_COLUMNS = ("calculation", "variable", "i", "j", "value")
BAND_COLUMNS = ("sys", "stat")
RESERVED_COLUMNS = ("case",) + STATE_COLUMNS + BAND_COLUMNS


@dataclass(frozen=True)
class Coords:
    """Cell centres, edges and lengths of a calculation's mesh.

    ``names`` is ``("z",)`` for a channel and ``("z", "x")`` for a plate;
    the other tuples follow the same order. ``mask`` marks the plate's meat.
    """

    names: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    bounds: tuple[np.ndarray | None, ...]
    weights: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None


class FrameList(list):
    """A list of result frames that remembers why some slots are ``None``."""

    failures: dict[int, str]

    def __init__(self, frames: Iterable = ()):
        super().__init__(frames)
        self.failures = {}


def _case_text(case: Mapping[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in case.items())


def run_cases(
    model: Callable[..., DataFrame],
    cases: list[dict[str, float]],
    catch: tuple[type[BaseException], ...] = (Exception,),
) -> tuple[list[dict[str, float]], FrameList]:
    """Call ``model(**case)`` for every case and collect the frames.

    A call raising one of ``catch`` leaves ``None`` in its slot; the
    exception text is kept in the returned list's ``failures`` and one
    warning lists every failed case. Other exceptions propagate.
    """
    frames = FrameList()
    for index, case in enumerate(cases):
        try:
            frames.append(model(**case))
        except catch as error:
            frames.append(None)
            frames.failures[index] = str(error)
    if frames.failures:
        lines = [f"case {k} ({_case_text(cases[k])}): {msg}" for k, msg in frames.failures.items()]
        warnings.warn(f"{len(lines)} of {len(cases)} cases failed:\n  " + "\n  ".join(lines), stacklevel=2)
    return list(cases), frames


class Sweep:
    """Cases, result frames and the aggregator they came from.

    Parameters
    ----------
    cases:
        One dict per case, all with the same keys, float values only.
    frames:
        One ``State.to_dataframe()`` result per case, or ``None`` for a
        case whose solve failed.
    agr:
        The aggregator, static across the sweep, used for coordinates.
    labels:
        Display names and units, merged over the module defaults.
    """

    def __init__(self, cases, frames, agr, labels=None):
        self.cases = _check_cases(cases)
        self.parameters: tuple[str, ...] = tuple(self.cases[0]) if self.cases else ()
        if len(self.cases) != len(frames):
            raise ValueError(f"{len(self.cases)} cases but {len(frames)} frames; the two lists must align")
        self.frames = list(frames)
        self.agr = agr
        self.labels = labels
        failures = getattr(frames, "failures", {})
        self.holes: dict[int, str | None] = {k: failures.get(k) for k, f in enumerate(self.frames) if f is None}
        if self.holes:
            lines = [f"case {k} ({_case_text(self.cases[k])})" + (f": {m}" if m else "") for k, m in self.holes.items()]
            warnings.warn("holes in the sweep:\n  " + "\n  ".join(lines), stacklevel=2)
        checked = {k: _check_frame(k, f) for k, f in enumerate(self.frames) if f is not None}
        self.banded = any(_has_band(f) for f in checked.values())
        self.bad_rows: dict[int, list[int]] = {}
        self.table = self._stack(checked)
        self._coords_cache: dict[str, object] = {}

    @classmethod
    def single(cls, frame: DataFrame, agr, labels=None) -> Sweep:
        """A sweep of one case and no parameters."""
        return cls([{}], [frame], agr, labels)

    @property
    def labels(self) -> Labels:
        return self._labels

    @labels.setter
    def labels(self, value) -> None:
        self._labels = STREAM_LABELS.merged(value)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def case_of(self, index: int) -> dict[str, float]:
        """The parameter values of case ``index``."""
        return dict(self.cases[index])

    def levels(self, parameter: str) -> np.ndarray:
        """The sorted distinct values of ``parameter`` over all cases."""
        if parameter not in self.parameters:
            raise ValueError(f"{parameter!r} is not a sweep parameter; the parameters are {', '.join(self.parameters)}")
        return np.unique([c[parameter] for c in self.cases])

    def _stack(self, checked: dict[int, DataFrame]) -> DataFrame:
        pieces = []
        zero_band_lines = []
        non_finite_lines = []
        for k, f in checked.items():
            f = f.copy()
            if self.banded:
                if not _has_band(f):
                    zero_band_lines.append(
                        f"case {k} ({_case_text(self.cases[k])}) carries no nonzero uncertainty; its band is zero"
                    )
                    f = f.assign(sys=0.0, stat=0.0)
                f["sys"] = f["sys"].astype(float)
                f["stat"] = f["stat"].astype(float)
                bad = ~np.isfinite(f[["value", "sys", "stat"]]).all(axis=1)
            else:
                f = f.drop(columns=[c for c in BAND_COLUMNS if c in f.columns])
                bad = ~np.isfinite(f["value"])
            if bad.any():
                self.bad_rows[k] = list(f.index[bad])
                non_finite_lines.append(
                    f"case {k} ({_case_text(self.cases[k])}) has non-finite values in row "
                    f"{', '.join(map(str, self.bad_rows[k]))}; treated as missing"
                )
                f = f[~bad]
            front = {"case": k, **self.cases[k]}
            pieces.append(f.assign(**front)[list(front) + [c for c in f.columns if c not in front]])
        if zero_band_lines:
            warnings.warn("cases with no nonzero uncertainty get a zero band:\n  " + "\n  ".join(zero_band_lines), stacklevel=3)
        if non_finite_lines:
            warnings.warn("non-finite values treated as missing:\n  " + "\n  ".join(non_finite_lines), stacklevel=3)
        if not pieces:
            columns = ["case", *self.parameters, *STATE_COLUMNS] + (list(BAND_COLUMNS) if self.banded else [])
            return DataFrame(columns=columns)
        table = pd.concat(pieces, ignore_index=True)
        for column in ("calculation", "variable"):
            table[column] = table[column].astype("category")
        return table

    def is_hole(self, index: int) -> bool:
        return index in self.holes

    def resolve(self, where: Mapping[str, float] | None, nearest: bool = False) -> dict[str, float]:
        """``where`` with each value snapped to the grid, or raised on."""
        out = {}
        for name, wanted in (where or {}).items():
            levels = self.levels(name)
            hits = np.isclose(levels, wanted, rtol=1e-9, atol=0.0)
            if hits.any():
                out[name] = float(levels[hits][0])
                continue
            if not nearest:
                raise ValueError(
                    f"{name}={wanted:g} is not on the grid; the values of {name} are "
                    f"{', '.join(f'{v:g}' for v in levels)} (pass nearest=True to snap)"
                )
            used = float(levels[np.argmin(np.abs(levels - wanted))])
            warnings.warn(f"{name}={wanted:g} is not on the grid; using {name}={used:g}", stacklevel=3)
            out[name] = used
        return out

    def select(self, where: Mapping[str, float] | None = None, nearest: bool = False) -> np.ndarray:
        """Indices of the cases matching ``where``; every case when ``where`` is empty."""
        wanted = self.resolve(where, nearest)
        keep = [
            k
            for k, case in enumerate(self.cases)
            if all(np.isclose(case[n], v, rtol=1e-9, atol=0.0) for n, v in wanted.items())
        ]
        return np.asarray(keep, dtype=int)

    def rank(self, quantity: str, calculation: str) -> int:
        """0, 1 or 2 from the index layout of the quantity's rows.

        A profile stored in a single cell is indistinguishable from a scalar and reads as rank 0.
        """
        rows = self.table[(self.table.calculation == calculation) & (self.table.variable == quantity)]
        if rows.empty:
            raise ValueError(f"{quantity!r} of {calculation!r} is not in the sweep")
        if rows.i.max() > 0:
            return 2
        return 1 if rows.j.max() > 0 else 0

    def names(self, quantity=None, of=None) -> tuple[list[str], list[str]]:
        """Resolve ``quantity`` and ``of`` to lists, defaulting when unambiguous."""
        calcs = sorted(self.table.calculation.unique().tolist())
        quantities = [quantity] if isinstance(quantity, str) else list(quantity or [])
        if of is None:
            owners = sorted(
                {c for c in calcs for q in quantities if not self.table[(self.table.calculation == c) & (self.table.variable == q)].empty}
            )
            if len(owners) != 1:
                raise ValueError(
                    f"{', '.join(map(repr, quantities))} belong(s) to {', '.join(owners) or 'no calculation'}; pass of= to choose"
                )
            ofs = owners
        else:
            ofs = [of] if isinstance(of, str) else list(of)
        for c in ofs:
            if c not in calcs:
                raise ValueError(f"{c!r} is not a calculation in the sweep; the calculations are {', '.join(calcs)}")
            have = sorted(self.table.variable[self.table.calculation == c].unique().tolist())
            for q in quantities:
                if q not in have:
                    raise ValueError(f"{q!r} is not a variable of {c}; it has {', '.join(have)}")
        return quantities, ofs

    def coords(self, calculation: str) -> Coords | None:
        """The mesh of ``calculation`` from the aggregator, or ``None`` when it has none."""
        if calculation not in self._coords_cache:
            try:
                calc = self.agr[calculation]
            except KeyError as error:
                known = ", ".join(_calculation_names(self.agr)) or "no calculations"
                raise ValueError(
                    f"the frames hold {calculation!r} but the aggregator does not; it holds {known}"
                ) from error
            self._coords_cache[calculation] = _coords_of(calc)
        return self._coords_cache[calculation]

    def profile(self, quantity, of=None, where=None, nearest=False, sigma=1.0, uncertainty=True):
        """A rank-1 quantity against its coordinate for every selected case."""
        from .results import profile

        return profile(self, quantity, of, where, nearest, sigma, uncertainty)

    def field(self, quantity, of=None, where=None, nearest=False, sigma=1.0, uncertainty=True):
        """A rank-2 quantity of exactly one case, ready to map."""
        from .results import field

        return field(self, quantity, of, where, nearest, sigma, uncertainty)

    def reduce(self, quantity, reducer, versus=None, of=None, where=None, axis=None, nearest=False, sigma=1.0, uncertainty=True):
        """Collapse a quantity per case and plot it against a parameter, or along one field axis."""
        from .results import reduce

        return reduce(self, quantity, reducer, versus, of, where, axis, nearest, sigma, uncertainty)

    def required(self, quantity, equals, solve_for, versus, reduce=None, of=None, at=None, where=None,
                 method="linear", nearest=False, sigma=1.0, uncertainty=True):
        """The value of ``solve_for`` at which the reduced quantity equals ``equals``, against ``versus``."""
        from .inverse import required

        return required(self, quantity, equals, solve_for, versus, reducer=reduce, of=of, at=at, where=where,
                        method=method, nearest=nearest, sigma=sigma, uncertainty=uncertainty)


def _calculation_names(agr) -> list[str]:
    graph = getattr(agr, "graph", None)
    try:
        return sorted(str(getattr(node, "name", node)) for node in (agr if graph is None else graph))
    except TypeError:
        return []


def _coords_of(calc) -> Coords | None:
    if hasattr(calc, "z_centers") and hasattr(calc, "x_centers"):
        zc, xc = np.asarray(calc.z_centers, float), np.asarray(calc.x_centers, float)
        zb = np.asarray(calc.z_bounds, float) if hasattr(calc, "z_bounds") else None
        xb = np.asarray(calc.x_bounds, float) if hasattr(calc, "x_bounds") else None
        zw = np.abs(np.diff(zb)) if zb is not None else np.ones_like(zc)
        xw = np.abs(np.diff(xb)) if xb is not None else np.ones_like(xc)
        return Coords(
            names=("z", "x"),
            centers=(zc, xc),
            bounds=(zb, xb),
            weights=(zw, xw),
            mask=np.asarray(calc.meat) if hasattr(calc, "meat") else None,
        )
    if hasattr(calc, "centers"):
        centers = np.asarray(calc.centers, float)
        bounds = np.asarray(calc.bounds, float) if hasattr(calc, "bounds") else None
        weights = np.abs(np.diff(bounds)) if bounds is not None else np.ones_like(centers)
        return Coords(names=("z",), centers=(centers,), bounds=(bounds,), weights=(weights,))
    return None


def _check_cases(cases) -> list[dict[str, float]]:
    out = []
    keys = None
    for k, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TypeError(f"case {k} must be a dict of parameter values, not {type(case).__name__}")
        if keys is None:
            keys = tuple(case)
            reserved = [name for name in keys if name in RESERVED_COLUMNS]
            if reserved:
                raise ValueError(
                    f"parameter name(s) {', '.join(map(repr, reserved))} clash with the sweep table's own columns; "
                    f"the reserved names are {', '.join(RESERVED_COLUMNS)}"
                )
        elif tuple(case) != keys and set(case) != set(keys):
            raise ValueError(f"case {k} has parameters {', '.join(case)} but case 0 has {', '.join(keys)}")
        clean = {}
        for name in keys:
            v = case[name]
            if isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(f"case {k}: parameter {name!r} must be a number, not {v!r}")
            clean[name] = float(v)
        out.append(clean)
    return out


def _check_frame(k: int, f: DataFrame) -> DataFrame:
    if "time" in f.columns:
        raise ValueError(f"case {k}: the frame has a 'time' column; only steady-state frames are supported")
    missing = [c for c in STATE_COLUMNS if c not in f.columns]
    if missing:
        raise ValueError(f"case {k}: the frame lacks column(s) {', '.join(repr(c) for c in missing)}")
    for c in BAND_COLUMNS:
        if c in f.columns and (f[c] < 0).any():
            rows = ", ".join(str(r) for r in f.index[f[c] < 0])
            raise ValueError(f"case {k}: negative {c} in row {rows}; uncertainties are non-negative")
    return f


def _has_band(f: DataFrame) -> bool:
    return all(c in f.columns for c in BAND_COLUMNS) and bool((f["sys"] != 0).any() or (f["stat"] != 0).any())
