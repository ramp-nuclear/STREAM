"""Answers to the questions: tidy frames with the metadata a figure needs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd
from pandas import DataFrame

from .envelope import ENVELOPES, INNER, mesh_shape, reduce_rows, reducer_name, to_array, with_envelopes
from .labels import Labels
from .sweep import Coords, Sweep, _case_text

COLUMNS = ENVELOPES + INNER


def _unit_phrase(unit: str) -> str:
    return f"in {unit}" if unit else "dimensionless"


def check_units(labels: Labels, quantities: list[str]) -> None:
    """Raise when two quantities have different known units or different scales."""
    known = sorted((q, labels.unit(q), labels.scale(q)) for q in quantities if labels.unit(q) is not None)
    for q, u, s in known[1:]:
        q0, u0, s0 = known[0]
        if u != u0:
            raise ValueError(
                f"{q0!r} is {_unit_phrase(u0)} and {q!r} is {_unit_phrase(u)}; one axis cannot carry both"
            )
        if s != s0:
            raise ValueError(
                f"{q0!r} is scaled by {s0:g} and {q!r} by {s:g}; one axis cannot carry both"
            )


def y_axis_text(labels: Labels, quantities: list[str], prefix: str = "") -> str:
    """``prefix Display1, Display2 [unit]`` for quantities sharing a unit."""
    displays = ", ".join(labels.display(q) for q in quantities)
    units = [labels.unit(q) for q in quantities if labels.unit(q)]
    head = f"{prefix} " if prefix else ""
    return f"{head}{displays} [{units[0]}]" if units else f"{head}{displays}"


def y_axis(result, labels: Labels) -> tuple[str, str]:
    """The y axis text of ``result`` under ``labels``, and the name whose scale its values carry."""
    names = list(result.ylabel_source)
    if result.kind == "field":
        return labels.axis("z"), "z"
    drawn = set(result.frame["quantity"]) if "quantity" in result.frame.columns else set()
    if result.kind == "curve" and result.reducer is None and len(names) == 1 and names[0] not in drawn:
        return labels.axis(names[0]), names[0]
    if result.reducer is not None and result.reducer.startswith("where_"):
        coordinate = result.coordinate or "z"
        unit = f" [{labels.unit(coordinate)}]" if labels.unit(coordinate) else ""
        return f"{result.reducer} {', '.join(labels.display(q) for q in names)}{unit}", coordinate
    return y_axis_text(labels, names, result.reducer or ""), (names[0] if names else "")


@dataclass
class Result:
    """A tidy frame plus what a figure needs to know about it.

    ``frame`` holds one row per drawn point with the dimension columns,
    the x column, and ``lower``, ``value``, ``upper``. ``dims`` lists the
    dimensions that vary and their levels; ``fixed`` the parameters that do
    not; ``level_space`` every parameter's full list of levels in the sweep,
    so a level keeps its colour across figures. ``coordinate`` names the mesh
    axis a ``where_*`` reducer read its values off. ``plot`` draws it.
    """

    kind: str
    frame: DataFrame
    x: str
    dims: dict[str, list]
    numeric: frozenset[str]
    fixed: dict[str, float]
    calculation: str | None
    labels: Labels
    banded: bool
    reducer: str | None = None
    ylabel_source: tuple[str, ...] = ()
    sigma: float = 1.0
    walk_reducer: str | None = None
    level_space: dict[str, list] = dc_field(default_factory=dict)
    coordinate: str | None = None
    target: float | str | None = None
    walk_quantity: str | None = None

    @property
    def ylabel(self) -> str:
        """The y axis text under this result's own labels."""
        return y_axis(self, self.labels)[0]

    @property
    def subtitle(self) -> str:
        """The line above the axes under this result's own labels."""
        return self.subtitle_with(self.labels)

    def subtitle_with(self, labels: Labels) -> str:
        """The line above the axes under ``labels``: the target, the fixed parameters, the calculation."""
        parts = [self._target_text(labels), labels.describe(self.fixed)]
        if self.calculation is not None:
            parts.append(labels.display(self.calculation))
        return ", ".join(p for p in parts if p)

    def _target_text(self, labels: Labels) -> str:
        if self.target is None or self.walk_quantity is None:
            return ""
        head = " ".join(p for p in (self.walk_reducer, labels.display(self.walk_quantity)) if p)
        if isinstance(self.target, str):
            return f"{head} = {labels.display(self.target)}"
        unit = labels[self.walk_quantity].unit
        return f"{head} = {labels.value(self.walk_quantity, self.target)}" + (f" {unit}" if unit else "")

    def plot(self, ax=None, color=None, style=None, marker=None, col=None, row=None, bound="nominal",
             band="total", errorbars=False, style_sheet=None, labels=None):
        """Draw this result; see :func:`stream.viz.draw.plot`."""
        from .draw import plot

        return plot(self, ax, color, style, marker, col, row, bound, band, errorbars, style_sheet, labels)


@dataclass
class Field(Result):
    """A rank-2 quantity of one case with its mesh."""

    quantity: str = ""
    z: np.ndarray = dc_field(default_factory=lambda: np.zeros(0))
    xs: np.ndarray = dc_field(default_factory=lambda: np.zeros(0))
    z_bounds: np.ndarray | None = None
    x_bounds: np.ndarray | None = None
    arrays: dict[str, np.ndarray] = dc_field(default_factory=dict)
    mask: np.ndarray | None = None

    def values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``(lower, value, upper)`` mesh arrays; ``arrays`` holds all five."""
        return self.arrays["lower"], self.arrays["value"], self.arrays["upper"]


def _selection(sweep: Sweep, where, nearest):
    resolved = sweep.resolve(where, nearest)
    indices = sweep.select(resolved)
    holes = [k for k in indices if sweep.is_hole(k)]
    if holes:
        lines = [
            f"case {k} ({_case_text(sweep.case_of(k))})" + (f": {sweep.holes[k]}" if sweep.holes[k] else "")
            for k in holes
        ]
        warnings.warn("skipping unsolved case(s):\n  " + "\n  ".join(lines), stacklevel=4)
    return resolved, [k for k in indices if not sweep.is_hole(k)]


def _dims(sweep: Sweep, quantities, ofs, resolved, indices, exclude=()) -> tuple[dict, dict, str | None]:
    dims: dict[str, list] = {}
    if len(quantities) > 1:
        dims["quantity"] = list(quantities)
    if len(ofs) > 1:
        dims["calculation"] = list(ofs)
    fixed = dict(resolved)
    for p in sweep.parameters:
        if p in resolved or p in exclude:
            continue
        levels = sorted({sweep.cases[k][p] for k in indices})
        if len(levels) > 1:
            dims[p] = levels
        elif levels:
            fixed[p] = levels[0]
    return dims, fixed, ofs[0] if len(ofs) == 1 else None


def _numeric(sweep: Sweep, dims) -> frozenset[str]:
    return frozenset(d for d in dims if d in sweep.parameters)


def _level_space(sweep: Sweep) -> dict[str, list]:
    return {p: [float(v) for v in sweep.levels(p)] for p in sweep.parameters}


def _rows(sweep: Sweep, table: DataFrame, k: int, calc: str, q: str) -> DataFrame:
    return table[(table.case == k) & (table.calculation == calc) & (table.variable == q)]


def _mesh_clash(q: str, calc: str, reach: str, size: str) -> ValueError:
    return ValueError(
        f"{q!r} of {calc} reaches {reach} but the mesh of {calc} has {size}; the frame and the aggregator disagree"
    )


def _nothing_selected(resolved, quantities) -> ValueError:
    where_text = f"where=({_case_text(resolved)})" if resolved else "the whole sweep"
    return ValueError(f"no solved case matches {where_text} for {', '.join(map(repr, quantities))}")


def _x_of(coords: Coords | None, rows: DataFrame, q: str, calc: str) -> tuple[str, np.ndarray]:
    cells = rows.sort_values("j").j.to_numpy(dtype=int)
    if coords is None:
        return "cell", cells.astype(float)
    centers = np.asarray(coords.centers[0], float)
    if cells.size and cells.max() >= len(centers):
        raise _mesh_clash(q, calc, f"cell {cells.max()}", f"{len(centers)} cells")
    return coords.names[0], centers[cells]


def _dense(rows: DataFrame, column: str) -> np.ndarray:
    return rows.sort_values("j")[column].to_numpy(dtype=float)


def _x_clash(xnames: dict[str, str]) -> ValueError:
    return ValueError(
        "the calculations do not share one x axis: "
        + ", ".join(f"{c} along {n}" for c, n in xnames.items())
        + "; draw them in separate figures"
    )


def profile(sweep: Sweep, quantity, of=None, where=None, nearest=False, sigma=1.0, uncertainty=True) -> Result:
    """A rank-1 quantity against its coordinate for every selected case."""
    quantities, ofs = sweep.names(quantity, of)
    check_units(sweep.labels, quantities)
    resolved, indices = _selection(sweep, where, nearest)
    table = with_envelopes(sweep.table, sweep.banded, sigma, uncertainty)
    pieces, xnames = [], {}
    for calc in ofs:
        coords = sweep.coords(calc)
        for q in quantities:
            rank = sweep.rank(q, calc)
            if rank == 0:
                raise ValueError(f"{q!r} of {calc} is a scalar; use reduce() to plot it against a parameter")
            if rank == 2:
                raise ValueError(f"{q!r} of {calc} is a field; use field() for a map or reduce() with axis= for a profile")
            for k in indices:
                rows = _rows(sweep, table, k, calc, q)
                xname, x = _x_of(coords, rows, q, calc)
                xnames[calc] = xname
                pieces.append(
                    DataFrame(
                        {
                            "quantity": q,
                            "calculation": calc,
                            **{p: sweep.cases[k][p] for p in sweep.parameters},
                            "case": k,
                            xname: x,
                            **{c: _dense(rows, c) for c in COLUMNS},
                        }
                    )
                )
    if not pieces:
        raise _nothing_selected(resolved, quantities)
    if len(set(xnames.values())) > 1:
        raise _x_clash(xnames)
    dims, fixed, single = _dims(sweep, quantities, ofs, resolved, indices)
    return Result(
        kind="profile",
        frame=pd.concat(pieces, ignore_index=True),
        x=xnames[ofs[0]],
        dims=dims,
        numeric=_numeric(sweep, dims),
        fixed=fixed,
        calculation=single,
        labels=sweep.labels,
        banded=sweep.banded and uncertainty,
        ylabel_source=tuple(quantities),
        sigma=sigma,
        level_space=_level_space(sweep),
    )


def field(sweep: Sweep, quantity, of=None, where=None, nearest=False, sigma=1.0, uncertainty=True) -> Field:
    """A rank-2 quantity of exactly one case with its mesh."""
    quantities, ofs = sweep.names(quantity, of)
    if len(quantities) != 1 or len(ofs) != 1:
        raise ValueError(
            "field() takes one quantity of one calculation; got "
            f"{', '.join(map(repr, quantities)) or 'no quantity'} of {', '.join(map(repr, ofs)) or 'no calculation'}"
        )
    q, calc = quantities[0], ofs[0]
    resolved, indices = _selection(sweep, where, nearest)
    if len(indices) != 1:
        raise ValueError(
            f"field() draws one case and the selection holds {len(indices)} cases of {q!r}; "
            "add keys to where= to pick one"
        )
    if sweep.rank(q, calc) != 2:
        raise ValueError(f"{q!r} of {calc} is not a field; use profile() or reduce()")
    k = indices[0]
    table = with_envelopes(sweep.table, sweep.banded, sigma, uncertainty)
    rows = _rows(sweep, table, k, calc, q)
    coords = sweep.coords(calc)
    m, n = mesh_shape(rows, 2, coords)
    if not rows.empty:
        if int(rows.i.max()) >= m:
            raise _mesh_clash(q, calc, f"row {int(rows.i.max())}", f"{m} rows")
        if int(rows.j.max()) >= n:
            raise _mesh_clash(q, calc, f"column {int(rows.j.max())}", f"{n} columns")
    arrays = {c: to_array(rows, 2, c, coords) for c in COLUMNS}
    if coords is not None and len(coords.names) == 2:
        z, xs, zb, xb, mask = coords.centers[0], coords.centers[1], coords.bounds[0], coords.bounds[1], coords.mask
    else:
        z, xs, zb, xb, mask = np.arange(m, dtype=float), np.arange(n, dtype=float), None, None, None
    zz, xx = np.meshgrid(z, xs, indexing="ij")
    frame = DataFrame(
        {
            "quantity": q,
            "calculation": calc,
            **{p: sweep.cases[k][p] for p in sweep.parameters},
            "case": k,
            "z": zz.ravel(),
            "x": xx.ravel(),
            **{c: arrays[c].ravel() for c in COLUMNS},
        }
    )
    dims, fixed, single = _dims(sweep, [q], [calc], resolved, indices)
    return Field(
        kind="field",
        frame=frame,
        x="x",
        dims=dims,
        numeric=_numeric(sweep, dims),
        fixed=fixed,
        calculation=single,
        labels=sweep.labels,
        banded=sweep.banded and uncertainty,
        ylabel_source=(q,),
        sigma=sigma,
        level_space=_level_space(sweep),
        quantity=q,
        z=np.asarray(z, float),
        xs=np.asarray(xs, float),
        z_bounds=zb,
        x_bounds=xb,
        arrays=arrays,
        mask=mask,
    )


def reduce(
    sweep: Sweep,
    quantity,
    reducer,
    versus=None,
    of=None,
    where=None,
    axis=None,
    nearest=False,
    sigma=1.0,
    uncertainty=True,
) -> Result:
    """A reducer applied per case, against ``versus``; or a profile when ``axis`` is given."""
    quantities, ofs = sweep.names(quantity, of)
    check_units(sweep.labels, quantities)
    if axis is not None and versus is not None:
        raise ValueError("reduce() takes versus= or axis=, not both: axis= leaves a profile, versus= a curve")
    if axis is None and versus is None:
        raise ValueError("reduce() needs versus= (a parameter for the x axis) or axis= (a field direction to collapse)")
    if versus is not None:
        if versus not in sweep.parameters:
            raise ValueError(f"{versus!r} is not a sweep parameter; the parameters are {', '.join(sweep.parameters)}")
        if where and versus in where:
            raise ValueError(f"{versus} is the x axis and cannot also be fixed in where=")
    resolved, indices = _selection(sweep, where, nearest)
    table = with_envelopes(sweep.table, sweep.banded, sigma, uncertainty)
    name = reducer_name(reducer)
    pieces, xname, coordinate = [], versus or "cell", None
    for calc in ofs:
        coords = sweep.coords(calc)
        for q in quantities:
            rank = sweep.rank(q, calc)
            for k in indices:
                rows = _rows(sweep, table, k, calc, q)
                base = {"quantity": q, "calculation": calc, **{p: sweep.cases[k][p] for p in sweep.parameters}, "case": k}
                if axis is not None:
                    if rank != 2:
                        raise ValueError(f"axis= applies to a field and {q!r} of {calc} has rank {rank}")
                    out = reduce_rows(rows, rank, coords, reducer, axis)
                    keep = 1 - coords.names.index(axis) if coords is not None else (1 if axis == "z" else 0)
                    xname = coords.names[keep] if coords is not None else "cell"
                    x = coords.centers[keep] if coords is not None else np.arange(len(out["value"]), dtype=float)
                    coordinate = axis
                    pieces.append(DataFrame({**base, xname: x, **out}))
                else:
                    if rank:
                        out = reduce_rows(rows, rank, coords, reducer)
                        if np.ndim(out["value"]) != 0:
                            raise ValueError(
                                f"{name} of {q!r} of {calc} leaves a profile, not one number per case; "
                                "pass axis='z' or axis='x' instead of versus= to draw it"
                            )
                        coordinate = coords.names[0] if coords is not None else "cell"
                    else:
                        out = {c: to_array(rows, 0, c) for c in COLUMNS}
                    pieces.append(DataFrame({**base, **out}, index=[0]))
    if not pieces:
        raise _nothing_selected(resolved, quantities)
    scalar = all(sweep.rank(q, c) == 0 for c in ofs for q in quantities)
    if axis is not None:
        dims, fixed, single = _dims(sweep, quantities, ofs, resolved, indices)
        kind = "profile"
    else:
        dims, fixed, single = _dims(sweep, quantities, ofs, resolved, indices, exclude=(versus,))
        kind = "curve"
    return Result(
        kind=kind,
        frame=pd.concat(pieces, ignore_index=True),
        x=xname,
        dims=dims,
        numeric=_numeric(sweep, dims),
        fixed=fixed,
        calculation=single,
        labels=sweep.labels,
        banded=sweep.banded and uncertainty,
        reducer=None if scalar else name,
        ylabel_source=tuple(quantities),
        sigma=sigma,
        level_space=_level_space(sweep),
        coordinate=coordinate if name.startswith("where_") else None,
    )
