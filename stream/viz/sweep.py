"""The container: cases, frames and the aggregator, stacked and checked."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from numbers import Real
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas import DataFrame

from .labels import STREAM_LABELS, Labels
from .limits import Limit

__all__ = ["Sweep", "cases", "run_cases"]


STATE_COLUMNS = ("calculation", "variable", "i", "j", "value")
BAND_COLUMNS = ("sys", "stat")
RESERVED_COLUMNS = ("case",) + STATE_COLUMNS + BAND_COLUMNS + ("filled", "safe", "binder")
RANKS = ("scalar", "profile", "field")


@dataclass(frozen=True)
class _Coords:
    """Cell centres, edges and lengths of a calculation's mesh.

    ``names`` is ``("z",)`` for a channel and ``("z", "x")`` for a plate;
    the other tuples follow the same order. ``mask`` marks the plate's meat.
    """

    names: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    bounds: tuple[np.ndarray | None, ...]
    weights: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None


class _FrameList(list):
    """A list of result frames that remembers why some slots are ``None``."""

    failures: dict[int, str]

    def __init__(self, frames: Iterable = ()):
        super().__init__(frames)
        self.failures = {}


def _level_text(v: float | str) -> str:
    return v if isinstance(v, str) else format(v, "g")


def _case_text(case: Mapping[str, float | str]) -> str:
    return ", ".join(f"{k}={v if isinstance(v, str) else format(v, 'g')}" for k, v in case.items())


def cases(**levels) -> list[dict]:
    """The product of the given levels, one dict per case.

    Each keyword is a parameter and its levels; a single number or string is a
    parameter with one level. The last parameter varies fastest::

        cases(power=[2e6, 4e6], mdot=[0.2, 0.3], Tin=300.0)

    gives four cases, ``power=2e6, mdot=0.2`` first. Pass the list to
    :func:`run_cases` and the result to :class:`Sweep`.
    """
    if not levels:
        raise ValueError("cases() needs at least one parameter")
    axes = []
    for name, values in levels.items():
        values = [] if values is None else [values] if isinstance(values, (str, Real)) else list(values)
        if not values:
            raise ValueError(f"{name} has no levels")
        axes.append(values)
    return [dict(zip(levels, combo)) for combo in product(*axes)]


def run_cases(
    model: Callable[..., DataFrame],
    cases: list[dict[str, float | str]],
    catch: tuple[type[BaseException], ...] = (Exception,),
) -> tuple[list[dict[str, float | str]], _FrameList]:
    """Call ``model(**case)`` for every case and collect the frames.

    A call raising one of ``catch`` leaves ``None`` in its slot; the
    exception text is kept in the returned list's ``failures`` and one
    warning lists every failed case. Other exceptions propagate.
    """
    frames = _FrameList()
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

    Build one from your own loop or from :func:`cases` and :func:`run_cases`,
    then ask it a question, :meth:`profile`, :meth:`field`, :meth:`reduce`
    (two parameters on ``versus`` make a map) or :meth:`required`, and call
    ``.plot()`` on the answer or read its ``frame``. What is true of the whole
    sweep is set here: labels, limits and whether holes are filled.

    Ask ``print(sweep)`` for the parameters and their levels, the quantities of
    each calculation, the holes and the limits. The sweep keeps its data
    readable: ``cases``, ``frames``, the aggregator ``agr``, the stacked
    ``table``, ``holes``, ``filled``, ``limits`` and ``labels``.

    Parameters
    ----------
    cases:
        One dict per case, all with the same keys. A value is a number, or a
        string for a parameter that is a choice (a wall law, a correlation);
        each parameter is one or the other in every case. A choice can colour,
        style or panel a figure, never be an axis.
    frames:
        One ``State.to_dataframe()`` result per case, or ``None`` for a case
        whose solve failed (a hole).
    agr:
        The aggregator, static across the sweep, used for coordinates.
    labels:
        Display names, units and scales, merged over the module defaults.
    limits:
        The limits the design must respect, one per quantity::

            limits={"CHFR": at_least(1.3), "T_clad": at_most(900.0)}

        A declared limit is drawn on every figure that shows the quantity,
        gives ``required`` its target and ``limiting=True`` its meaning.
        Quantities that share axes and one limit get one line between them.
    fill_holes:
        ``True`` fills a hole when, along a numeric parameter and with every
        other parameter equal, the grid levels immediately below and above it
        are solved cases; the fill interpolates linearly between the two (the
        mean over the parameters that bracket it). A hole that no numeric
        parameter brackets this way (at the edge of the sweep, or next to
        another hole, along every one of them) stays a hole. Filled rows carry
        ``filled=True`` in every frame, draw with an open marker (a field
        ends its subtitle with ``, interpolated``) and are listed in
        ``filled``. Off by default because an interpolated case is a guess.
    """

    def __init__(self, cases, frames, agr, labels=None, limits=None, fill_holes=False):
        self.cases, self._categorical = _check_cases(cases)
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
        self._bad_rows: dict[int, list[int]] = {}
        self.table = self._stack(checked)
        self._fill_holes = bool(fill_holes)
        self.filled: dict[int, dict[str, tuple[int, int]]] = {}
        if self._fill_holes:
            self.table = self._fill(self.table)
        self.limits = MappingProxyType(_check_limits(limits, self.table))
        self._coords_cache: dict[str, object] = {}

    def __repr__(self) -> str:
        n_holes, n_filled = len(self.holes), len(self.filled)
        head = f"Sweep: {self.n_cases} case{'s' if self.n_cases != 1 else ''}, {n_holes} hole{'s' if n_holes != 1 else ''}"
        head += f" ({n_filled} filled)" if n_holes and self._fill_holes else ""
        head += ", uncertainty bands" if self.banded else ", no uncertainty"
        lines = [head]
        if self.parameters:
            params = ", ".join(f"{p} [{', '.join(_level_text(v) for v in self.levels(p))}]" for p in self.parameters)
            lines.append(f"  parameters: {params}")
        for calc in sorted(self.table.calculation.unique().tolist()) if len(self.table) else []:
            names = sorted(self.table.variable[self.table.calculation == calc].unique().tolist())
            lines.append(f"  {calc}: " + ", ".join(f"{q} ({RANKS[self._rank(q, calc)]})" for q in names))
        if self.limits:
            lines.append("  limits: " + ", ".join(f"{q} {lim.text()}" for q, lim in self.limits.items()))
        for k, message in self.holes.items():
            text = "  holes: " if k == next(iter(self.holes)) else "         "
            text += f"case {k} ({_case_text(self.cases[k])})" + (f": {message}" if message else "")
            if k in self.filled:
                text += ", filled along " + ", ".join(self.filled[k])
            lines.append(text)
        return "\n".join(lines)

    @classmethod
    def single(cls, frame: DataFrame, agr, labels=None, limits=None) -> Sweep:
        """A sweep of one case and no parameters."""
        return cls([{}], [frame], agr, labels, limits)

    @property
    def labels(self) -> Labels:
        return self._labels

    @labels.setter
    def labels(self, value) -> None:
        self._labels = STREAM_LABELS.merged(value)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def case_of(self, index: int) -> dict[str, float | str]:
        """The parameter values of case ``index``."""
        return dict(self.cases[index])

    def levels(self, parameter: str) -> np.ndarray:
        """The distinct values of ``parameter``: sorted when numeric, in order of first appearance when a string."""
        if parameter not in self.parameters:
            raise ValueError(f"{parameter!r} is not a sweep parameter; the parameters are {', '.join(self.parameters)}")
        values = [c[parameter] for c in self.cases]
        if parameter in self._categorical:
            return np.asarray(list(dict.fromkeys(values)), dtype=object)
        return np.unique(values)

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
                self._bad_rows[k] = list(f.index[bad])
                non_finite_lines.append(
                    f"case {k} ({_case_text(self.cases[k])}) has non-finite values in row "
                    f"{', '.join(map(str, self._bad_rows[k]))}; treated as missing"
                )
                f = f[~bad]
            front = {"case": k, **self.cases[k]}
            f = f.assign(**front, filled=False)
            pieces.append(f[list(front) + [c for c in f.columns if c not in front and c != "filled"] + ["filled"]])
        if zero_band_lines:
            warnings.warn("cases with no nonzero uncertainty get a zero band:\n  " + "\n  ".join(zero_band_lines), stacklevel=3)
        if non_finite_lines:
            warnings.warn("non-finite values treated as missing:\n  " + "\n  ".join(non_finite_lines), stacklevel=3)
        if not pieces:
            columns = ["case", *self.parameters, *STATE_COLUMNS] + (list(BAND_COLUMNS) if self.banded else []) + ["filled"]
            return DataFrame(columns=columns)
        table = pd.concat(pieces, ignore_index=True)
        for column in ("calculation", "variable"):
            table[column] = table[column].astype("category")
        return table

    def _fill(self, table: DataFrame) -> DataFrame:
        numeric = [p for p in self.parameters if p not in self._categorical]
        keys = ["calculation", "variable", "i", "j"]
        columns = ["value"] + (list(BAND_COLUMNS) if self.banded else [])
        pieces, lines = [], []
        for k in self.holes:
            case = self.cases[k]
            parts, brackets = [], {}
            for p in numeric:
                same = [
                    m for m in range(self.n_cases)
                    if all(self._equal(q, self.cases[m][q], case[q]) for q in self.parameters if q != p)
                ]
                below = [m for m in same if self.cases[m][p] < case[p]]
                above = [m for m in same if self.cases[m][p] > case[p]]
                if not below or not above:
                    continue
                lo = max(below, key=lambda m: self.cases[m][p])
                hi = min(above, key=lambda m: self.cases[m][p])
                if lo in self.holes or hi in self.holes:
                    continue
                a = table[table.case == lo].set_index(keys).sort_index()
                b = table[table.case == hi].set_index(keys).sort_index()
                if not a.index.equals(b.index):
                    warnings.warn(
                        f"case {k}: cases {lo} and {hi} do not share one frame layout along {p}; not used", stacklevel=3
                    )
                    continue
                t = (case[p] - self.cases[lo][p]) / (self.cases[hi][p] - self.cases[lo][p])
                parts.append(a[columns] + t * (b[columns] - a[columns]))
                brackets[p] = (lo, hi)
            if not parts:
                continue
            mean = sum(parts[1:], parts[0]) / len(parts)
            rows = mean.reset_index().assign(case=k, **case, filled=True)[list(table.columns)]
            pieces.append(rows)
            self.filled[k] = brackets
            lines.append(
                f"case {k} ({_case_text(case)}): "
                + "; ".join(f"along {p} between case {lo} and case {hi}" for p, (lo, hi) in brackets.items())
            )
        if lines:
            warnings.warn("filled unsolved case(s) by interpolation:\n  " + "\n  ".join(lines), stacklevel=3)
        if not pieces:
            return table
        out = pd.concat([table, *pieces], ignore_index=True)
        for column in ("calculation", "variable"):
            out[column] = out[column].astype("category")
        return out

    def _equal(self, name: str, a, b) -> bool:
        return a == b if name in self._categorical else bool(np.isclose(a, b, rtol=1e-9, atol=0.0))

    def _is_hole(self, index: int) -> bool:
        return index in self.holes

    def _resolve(self, where: Mapping[str, float | str] | None, nearest: bool = False) -> dict[str, float | str]:
        """``where`` with each value snapped to the grid, or raised on."""
        out = {}
        for name, wanted in (where or {}).items():
            levels = self.levels(name)
            if name in self._categorical:
                if wanted not in levels:
                    raise ValueError(f"{wanted!r} is not a level of {name}; the levels are {', '.join(levels)}")
                out[name] = wanted
                continue
            if isinstance(wanted, str):
                raise ValueError(f"{name} is a number; got {wanted!r}")
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

    def _select(self, where: Mapping[str, float | str] | None = None, nearest: bool = False) -> np.ndarray:
        """Indices of the cases matching ``where``; every case when ``where`` is empty."""
        wanted = self._resolve(where, nearest)
        keep = [k for k, case in enumerate(self.cases) if self._matches(case, wanted)]
        return np.asarray(keep, dtype=int)

    def _matches(self, case: Mapping, wanted: Mapping) -> bool:
        return all(self._equal(n, case[n], v) for n, v in wanted.items())

    def _rank(self, quantity: str, calculation: str) -> int:
        """0, 1 or 2 from the index layout of the quantity's rows.

        A profile stored in a single cell is indistinguishable from a scalar and reads as rank 0.
        """
        rows = self.table[(self.table.calculation == calculation) & (self.table.variable == quantity)]
        if rows.empty:
            raise ValueError(f"{quantity!r} of {calculation!r} is not in the sweep")
        if rows.i.max() > 0:
            return 2
        return 1 if rows.j.max() > 0 else 0

    def _names(self, quantity=None, of=None) -> tuple[list[str], list[str]]:
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

    def _coords(self, calculation: str) -> _Coords | None:
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
        """A profile quantity against its mesh coordinate, one line per selected case.

        Asked by ``where``, ``of``, ``nearest``, ``sigma`` and ``uncertainty``; drawn by
        :meth:`~stream.viz.Result.plot`.

        ::

            sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot()
            sweep.profile("T_cool", where=dict(mdot=0.3)).plot(color="power")
            sweep.profile(["CHFR", "OSVR"], where=dict(power=4e6, mdot=0.3)).plot(col="quantity")

        Parameters
        ----------
        quantity:
            One name or a list of names, all of one unit, each a profile (one
            value per cell along one axis).
        of:
            The calculation(s) that own the quantity, when ambiguous.
        where:
            Parameters fixed at a grid level, as a dict; a choice parameter takes
            one of its strings (``where=dict(power=4e6, wall_law="regime")``).
        nearest:
            Snap ``where`` values to the nearest grid level instead of raising.
        sigma:
            The multiple of ``stat`` in the envelopes; ``sys`` is added as it is.
        uncertainty:
            ``False`` drops the envelopes; ``lower`` and ``upper`` equal ``value``.
        """
        from ._results import profile

        return profile(self, quantity, of, where, nearest, sigma, uncertainty)

    def field(self, quantity, of=None, where=None, nearest=False, sigma=1.0, uncertainty=True):
        """A field quantity of exactly one case, drawn as a colour mesh over its two axes.

        Asked by ``where``, ``of``, ``nearest``, ``sigma`` and ``uncertainty``; drawn by
        :meth:`~stream.viz.Result.plot`.

        ::

            sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3)).plot()
            sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3)).plot(bound="upper", show_limit="shade")

        Parameters
        ----------
        quantity:
            One name, a field (one value per cell over two axes).
        of:
            The calculation(s) that own the quantity, when ambiguous.
        where:
            Parameters fixed at a grid level, as a dict, enough to pick one case; a choice parameter takes
            one of its strings (``where=dict(power=4e6, wall_law="regime")``).
        nearest:
            Snap ``where`` values to the nearest grid level instead of raising.
        sigma:
            The multiple of ``stat`` in the envelopes; ``sys`` is added as it is.
        uncertainty:
            ``False`` drops the envelopes; ``lower`` and ``upper`` equal ``value``.
        """
        from ._results import field

        return field(self, quantity, of, where, nearest, sigma, uncertainty)

    def reduce(self, quantity, reducer, versus=None, of=None, where=None, axis=None, nearest=False, sigma=1.0,
               uncertainty=True, limiting=False):
        """Collapse a quantity per case and plot it against a parameter, against two parameters as a map, or along one field axis.

        Asked by the reducer, ``versus``, ``of``, ``where``, ``axis``, ``nearest``, ``sigma``,
        ``uncertainty`` and ``limiting``; drawn by :meth:`~stream.viz.Result.plot`.

        ::

            sweep.reduce("CHFR", "min", versus="power")
            sweep.reduce("CHFR", "min", versus="power").plot(color="mdot")
            sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot()
            sweep.reduce("T", "max", axis="x")
            sweep.reduce("CHFR", "min", versus="power", of=channels, limiting=True)

        Parameters
        ----------
        quantity:
            One name or a list of names, all of one unit.
        reducer:
            How the cells of each case are collapsed:

            * ``"min"``: the smallest cell value;
            * ``"max"``: the largest cell value;
            * ``"mean"``: the mean weighted by cell size;
            * ``"where_min"``: the coordinate of the smallest value;
            * ``"where_max"``: the coordinate of the largest value;
            * ``at(z=...)``, ``at(x=...)`` or ``at(cell=...)``: the value of one cell;
            * a callable ``f(values, coords)``: whatever it returns.

            A scalar quantity is taken as it is. A ``where_*`` reducer gives a
            location and a callable an unknown value, so their results carry no
            limit: ``show_limit="line"`` draws nothing, and ``show_limit="shade"`` or
            ``limiting=True`` raises.
        versus:
            A numeric parameter for the x axis, or a pair of two numeric
            parameters (a tuple or a list) for a map (filled contours, the limit
            as a contour). When
            the map's reducer is the limit's own (``min`` for ``at_least``,
            ``max`` for ``at_most``), its limit contour is the curve
            ``required(solve_for=versus[0], versus=versus[1])`` draws.
        of:
            The calculation(s) that own the quantity, when ambiguous.
        where:
            Parameters fixed at a grid level.
        axis:
            Collapse a field along one axis and keep a profile along the other,
            instead of ``versus``:

            * ``"z"``: collapse along z, a profile along x;
            * ``"x"``: collapse along x, a profile along z.
        nearest:
            Snap ``where`` values to the nearest grid level instead of raising.
        sigma:
            The multiple of ``stat`` in the envelopes; ``sys`` is added as it is.
        uncertainty:
            ``False`` drops the envelopes; ``lower`` and ``upper`` equal ``value``.
        limiting:
            ``True`` keeps, per case, only the calculation (and quantity) that
            binds: the smallest reduced value for a limit that is ``at_least``, the
            largest for ``at_most``. The frame names it in ``calculation`` and
            ``quantity``. Needs every quantity's limit declared, all with one
            side and one value: values are compared as they are, so across
            quantities it is meaningful only when they share a limit value; use
            ``required(..., limiting=True)`` for quantities with different limits.
        """
        from ._results import reduce

        return reduce(self, quantity, reducer, versus, of, where, axis, nearest, sigma, uncertainty, limiting)

    def required(self, quantity, equals=None, solve_for=None, versus=None, reduce=None, of=None, at=None, where=None,
                 method="linear", nearest=False, sigma=1.0, uncertainty=True, limiting=False):
        """The value of ``solve_for`` at which the reduced quantity meets its limit, against ``versus``.

        Asked by ``equals``, ``solve_for``, ``versus``, ``reduce``, ``of``, ``at``, ``where``,
        ``nearest``, ``sigma``, ``uncertainty`` and ``limiting``; drawn by
        :meth:`~stream.viz.Result.plot`.

        Each quantity is reduced over the mesh per case, the curve along
        ``solve_for`` is walked at every level of ``versus`` (and of every other
        free parameter), and the crossing is interpolated::

            sweep.required("CHFR", solve_for="power", versus="mdot")
            sweep.required("CHFR", equals=1.5, solve_for="power", versus="mdot")
            sweep.required(["CHFR", "OSVR"], solve_for="power", versus="mdot")
            sweep.required(["CHFR", "OSVR"], equals={"CHFR": 1.3, "OSVR": 1.1}, solve_for="power", versus="mdot")
            sweep.required(["CHFR", "OSVR"], of=channels, solve_for="power", versus="mdot", limiting=True)

        Parameters
        ----------
        quantity:
            One name or a list of names. Each is walked on its own, so their
            units may differ.
        equals:
            The target. ``None`` (the default) uses each quantity's declared limit;
            a number applies to every quantity; a quantity name walks the
            difference to that quantity (``equals="q_CHF"``); a dict gives each
            quantity its own.
        solve_for:
            Required. The parameter to walk along. Numeric.
        versus:
            Required. The parameter on the x axis. Numeric.
        reduce:
            The mesh reducer, any value ``reduce()`` takes. ``None`` uses ``min``
            for a limit that is ``at_least`` and ``max`` for one that is
            ``at_most`` (``min`` without a declared limit); one reducer applies to
            every quantity; a dict gives each its own.
        of:
            The calculation(s) that own the quantity, when ambiguous.
        at:
            Parameters fixed at a value, on or off the grid; off the grid, the
            walk is interpolated between the two neighbouring levels.
        where:
            Parameters fixed at a grid level.
        method:
            How the crossing is interpolated between grid points:

            * ``"linear"``: straight lines between them (the default);
            * ``"pchip"``: a monotone cubic through them.
        nearest:
            Snap ``where`` values to the nearest grid level instead of raising.
        sigma:
            The multiple of ``stat`` in the envelopes; ``sys`` is added as it is.
        uncertainty:
            ``False`` drops the envelopes; ``lower`` and ``upper`` equal ``value``.
        limiting:
            ``True`` keeps, at each level of ``versus`` (and of every other free
            parameter), only the quantity and the calculation that bind, the
            smallest required value when the safe side lies below it and the
            largest when above. A quantity safe everywhere at that level does not
            bind there; one safe nowhere makes the level infeasible (its value
            is NaN). The frame names the binder in ``quantity`` and
            ``calculation``, and the plot colours the markers by it. Needs every
            quantity's limit declared, and the quantities that cross at one level
            safe on the same side of their crossings.
            What is compared is the required value of ``solve_for``, so
            quantities of different units and limits combine.

        Returns
        -------
        Result
            A curve whose ``frame`` has one row per crossing with ``lower``,
            ``value``, ``upper`` (the walked envelopes), ``filled`` (whether the
            crossing rests on an interpolated case) and ``safe``, where the
            quantity is safe:

            * ``"below"``: below the crossing, in ``solve_for``;
            * ``"above"``: above it;
            * ``"everywhere"``: the walk never reaches the target and every
              solved point is safe;
            * ``"nowhere"``: the walk never reaches the target and every solved
              point is unsafe;
            * ``None``: no limit is declared for the quantity.
        """
        from ._inverse import required

        return required(self, quantity, equals, solve_for, versus, reducer=reduce, of=of, at=at, where=where,
                        method=method, nearest=nearest, sigma=sigma, uncertainty=uncertainty, limiting=limiting)


def _calculation_names(agr) -> list[str]:
    graph = getattr(agr, "graph", None)
    try:
        return sorted(str(getattr(node, "name", node)) for node in (agr if graph is None else graph))
    except TypeError:
        return []


def _coords_of(calc) -> _Coords | None:
    if hasattr(calc, "z_centers") and hasattr(calc, "x_centers"):
        zc, xc = np.asarray(calc.z_centers, float), np.asarray(calc.x_centers, float)
        zb = np.asarray(calc.z_bounds, float) if hasattr(calc, "z_bounds") else None
        xb = np.asarray(calc.x_bounds, float) if hasattr(calc, "x_bounds") else None
        zw = np.abs(np.diff(zb)) if zb is not None else np.ones_like(zc)
        xw = np.abs(np.diff(xb)) if xb is not None else np.ones_like(xc)
        return _Coords(
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
        return _Coords(names=("z",), centers=(centers,), bounds=(bounds,), weights=(weights,))
    return None


def _check_limits(limits, table: DataFrame) -> dict[str, Limit]:
    if limits is not None and not isinstance(limits, Mapping):
        raise TypeError(
            f"limits= must be a mapping from quantity to a limit from at_least() or at_most(), not {limits!r}"
        )
    out = {}
    variables = sorted(table["variable"].unique().tolist()) if len(table) else []
    for name, limit in (limits or {}).items():
        if not isinstance(limit, Limit):
            raise TypeError(f"limits[{name!r}] must come from at_least() or at_most(), not {limit!r}")
        if name not in variables:
            raise ValueError(
                f"{name!r} in limits= is not a variable of any calculation; the variables are {', '.join(variables)}"
            )
        out[name] = limit
    return out


def _check_cases(cases) -> tuple[list[dict], frozenset[str]]:
    out = []
    keys = None
    kinds: dict[str, tuple[str, int]] = {}
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
            if isinstance(v, str):
                kind = "a string"
            elif isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(f"case {k}: parameter {name!r} must be a number or a string, not {v!r}")
            else:
                kind, v = "a number", float(v)
            seen = kinds.setdefault(name, (kind, k))
            if seen[0] != kind:
                raise TypeError(
                    f"parameter {name!r} is {kind} in case {k} and {seen[0]} in case {seen[1]}; "
                    "a parameter is one or the other"
                )
            clean[name] = v
        out.append(clean)
    categorical = frozenset(name for name, (kind, _) in kinds.items() if kind == "a string")
    return out, categorical


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
