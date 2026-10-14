"""The inverse question: the parameter value at which a quantity meets a target."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.interpolate import PchipInterpolator

from .envelope import ENVELOPES, INNER, reducer_name
from .results import Result, _dims, _level_space, _limits_of, _numeric, binder_column, check_units, reduce
from .sweep import Sweep, _case_text

COLUMNS = ENVELOPES + INNER
METHODS = ("linear", "pchip")


def crossings(x, y, target: float, method: str = "linear") -> list[float]:
    """Every ``x`` at which ``y`` crosses ``target``, ascending; ``x`` sorted."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return []
    if method == "pchip":
        spline = PchipInterpolator(x, y - target)
        return sorted(float(r) for r in spline.roots() if x[0] <= r <= x[-1])
    if method != "linear":
        raise ValueError(f"unknown method {method!r}; use 'linear' or 'pchip'")
    d = y - target
    out = []
    for k in range(len(x) - 1):
        if d[k] == 0:
            out.append(float(x[k]))
        elif d[k] * d[k + 1] < 0:
            out.append(float(x[k] + (x[k + 1] - x[k]) * d[k] / (d[k] - d[k + 1])))
    if d[-1] == 0:
        out.append(float(x[-1]))
    return sorted(set(out))


def first_crossing(x, y, target: float, method: str, context: str) -> float:
    """The lowest crossing, warning when there are several or none."""
    found = crossings(x, y, target, method)
    if not found:
        x = np.asarray(x, float)[np.isfinite(np.asarray(y, float))]
        rng = f"{x.min():g} to {x.max():g}" if len(x) else "no solved points"
        warnings.warn(f"{context}: the curve never reaches {target:g} over {rng}", stacklevel=4)
        return float("nan")
    if len(found) > 1:
        warnings.warn(
            f"{context}: {len(found)} crossings of {target:g} at {', '.join(f'{v:g}' for v in found)}; using the first",
            stacklevel=4,
        )
    return found[0]


def slice_at(curve: DataFrame, parameter: str, value: float, sweep: Sweep) -> DataFrame:
    """``curve`` at ``parameter == value``, blended between grid levels when off the grid."""
    levels = sweep.levels(parameter)
    hits = np.isclose(levels, value, rtol=1e-9, atol=0.0)
    if hits.any():
        return curve[np.isclose(curve[parameter], levels[hits][0], rtol=1e-9, atol=0.0)].copy()
    if not levels.min() < value < levels.max():
        raise ValueError(
            f"{parameter}={value:g} is outside the grid; the values of {parameter} are {', '.join(f'{v:g}' for v in levels)}"
        )
    lo, hi = levels[levels < value].max(), levels[levels > value].min()
    warnings.warn(f"{parameter}={value:g} is not on the grid; interpolating between {lo:g} and {hi:g}", stacklevel=4)
    keys = [c for c in curve.columns if c not in (parameter, "case", *COLUMNS)]
    a = curve[np.isclose(curve[parameter], lo)].drop(columns=["case"], errors="ignore").set_index(keys)
    b = curve[np.isclose(curve[parameter], hi)].drop(columns=["case"], errors="ignore").set_index(keys)
    w = (value - lo) / (hi - lo)
    joined = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    out = DataFrame({c: (1 - w) * joined[f"{c}_a"] + w * joined[f"{c}_b"] for c in COLUMNS}, index=joined.index)
    out[parameter] = value
    return out.reset_index()


def _per_quantity(spec, quantities: list[str], name: str, default) -> dict:
    if isinstance(spec, Mapping):
        if set(spec) != set(quantities):
            raise ValueError(
                f"{name}= has keys {', '.join(spec)} but the quantities are {', '.join(quantities)}; "
                "give one entry per quantity"
            )
        return {q: spec[q] for q in quantities}
    return {q: default(q) if spec is None else spec for q in quantities}


def _default_target(sweep: Sweep):
    def pick(q: str):
        if q not in sweep.limits:
            raise ValueError(f"no limit declared for {q!r}; pass equals= or declare it in limits=")
        return sweep.limits[q].value

    return pick


def _default_reducer(sweep: Sweep):
    return lambda q: sweep.limits[q].reducer if q in sweep.limits else "min"


def _safe_side(sweep: Sweep, q: str, g: DataFrame, solve_for: str, crossing: float, target) -> str | None:
    if q not in sweep.limits:
        return None
    limit = sweep.limits[q]
    level = 0.0 if isinstance(target, str) else float(target)
    xs, ys = g[solve_for].to_numpy(float), g["value"].to_numpy(float)
    finite = np.isfinite(ys)

    def ok(y):
        return y >= level if limit.side == "above" else y <= level

    if np.isnan(crossing):
        return "everywhere" if finite.any() and ok(ys[finite]).all() else "nowhere"
    before = np.flatnonzero((xs < crossing) & finite)
    if before.size:
        return "below" if ok(ys[before[-1]]) else "above"
    after = np.flatnonzero((xs > crossing) & finite)
    return "above" if ok(ys[after[0]]) else "below"


def required(
    sweep: Sweep, quantity, equals=None, solve_for=None, versus=None, reducer=None, of=None, at=None, where=None,
    method="linear", nearest=False, sigma=1.0, uncertainty=True, limiting=False,
) -> Result:
    """The value of ``solve_for`` at which the reduced quantity equals ``equals``, against ``versus``.

    Every envelope column is walked on its own; ``lower`` and ``upper`` hold the smaller and the
    larger of the two walked bounds, so they bracket ``value`` whether the quantity rises or falls
    with ``solve_for``, and both are missing when either walk finds no crossing.
    """
    at = dict(at or {})
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use {' or '.join(map(repr, METHODS))}")
    for role, name in (("solve_for", solve_for), ("versus", versus)):
        if name is None:
            raise ValueError(f"required() needs {role}=, a sweep parameter")
        if name not in sweep.parameters:
            raise ValueError(f"{role}={name!r} is not a sweep parameter; the parameters are {', '.join(sweep.parameters)}")
    for name in (solve_for, versus, *at):
        if name in sweep.categorical:
            raise ValueError(f"{name} is categorical; it can colour, style or panel a figure, not be an axis")
    if solve_for == versus:
        raise ValueError(f"solve_for and versus are both {solve_for!r}; they must be different parameters")
    for name in (solve_for, versus):
        if where and name in where:
            raise ValueError(f"{name} is a query axis and cannot also be fixed in where=")
        if name in at:
            raise ValueError(f"{name} is a query axis and cannot also be fixed in at=")
    twice = [name for name in at if where and name in where]
    if twice:
        raise ValueError(f"{', '.join(twice)} is fixed in both where= and at=; pass it in one of them")
    quantities, ofs = sweep.names(quantity, of)
    targets = _per_quantity(equals, quantities, "equals", _default_target(sweep))
    reducers = _per_quantity(reducer, quantities, "reduce", _default_reducer(sweep))
    resolved = sweep.resolve(where, nearest)
    free = [p for p in sweep.parameters if p != solve_for]
    group_cols = ["quantity", "calculation", *free]
    rows = []
    for q in quantities:
        target = targets[q]
        target_name = target if isinstance(target, str) else None
        if target_name is not None:
            sweep.names(target_name, ofs)
            check_units(sweep.labels, [q, target_name])
        reduced = reduce(
            sweep, [q] + ([target_name] if target_name else []), reducers[q], versus=solve_for, of=ofs,
            where=resolved, sigma=sigma, uncertainty=uncertainty,
        )
        curve = reduced.frame
        for p, v in at.items():
            curve = slice_at(curve, p, float(v), sweep)
        if target_name is not None:
            others = [c for c in curve.columns if c not in ("quantity", "case", "filled", *COLUMNS)]
            t = curve[curve.quantity == target_name].set_index(others)
            a = curve[curve.quantity == q].set_index(others)
            diff = DataFrame({c: a[c] - t[c] for c in COLUMNS}, index=a.index)
            diff["quantity"] = q
            if "filled" in a.columns:
                diff["filled"] = a["filled"] | t["filled"]
            curve, level = diff.reset_index(), 0.0
        else:
            level = float(target)
        for key, g in curve.groupby(group_cols, sort=True, observed=True):
            g = g.sort_values(solve_for)
            fixed_here = dict(zip(free, key[2:]))
            context = f"{key[0]} at {_case_text(fixed_here)}"
            n_solved = int(np.isfinite(g["value"]).sum())
            if n_solved < 2:
                raise ValueError(
                    f"{context}: the walk along {solve_for} has {n_solved} solved point(s); at least two are needed"
                )
            out = dict(zip(group_cols, key))
            for c in COLUMNS:
                out[c] = first_crossing(g[solve_for], g[c], level, method, context)
            _order_envelopes(out)
            out["safe"] = _safe_side(sweep, q, g, solve_for, out["value"], target)
            _warn_hole_in_bracket(sweep, g[solve_for].to_numpy(), out["value"], fixed_here, solve_for, context)
            rows.append(out)
    if not rows:
        picked = _case_text({**(where or {}), **at})
        raise ValueError(
            f"no solved cases of {', '.join(quantities)}"
            + (f" at {picked}" if picked else "")
            + f"; nothing to walk along {solve_for}"
        )
    frame = DataFrame(rows)[["quantity", "calculation", *free, *COLUMNS, "safe"]]
    indices = [k for k in sweep.select(resolved) if not sweep.is_hole(k)]
    at_values = {k: float(v) for k, v in at.items()}
    dims, fixed, single = _dims(sweep, quantities, ofs, {**resolved, **at_values}, indices, exclude=(solve_for, versus))
    sides = set(frame.loc[np.isfinite(frame["value"].to_numpy(float)), "safe"])
    reducer_names = {q: reducer_name(reducers[q]) for q in quantities}
    if limiting:
        frame, dims = _collapse_required(frame, dims, quantities, ofs, free, solve_for)
    return Result(
        kind="curve",
        frame=frame,
        x=versus,
        dims=dims,
        numeric=_numeric(sweep, dims),
        fixed=fixed,
        calculation=single,
        labels=sweep.labels,
        banded=sweep.banded and uncertainty,
        reducer=None,
        ylabel_source=(solve_for,),
        sigma=sigma,
        walk_reducer=reducer_names[quantities[0]] if reducer is not None or quantities[0] in sweep.limits else None,
        walk_reducers=reducer_names,
        level_space=_level_space(sweep),
        target=targets[quantities[0]],
        targets=dict(targets),
        walk_quantity=", ".join(quantities),
        limits=_limits_of(sweep, quantities),
        safe_side=sides.pop() if len(sides) == 1 and sides <= {"below", "above"} else None,
        binding=limiting,
    )


def _collapse_required(frame: DataFrame, dims: dict, quantities, ofs, free, solve_for: str) -> tuple[DataFrame, dict]:
    if "quantity" not in dims and "calculation" not in dims:
        raise ValueError(f"limiting=True but only {quantities[0]} of {ofs[0]} is walked; nothing to collapse")
    missing = sorted(set(frame.loc[frame["safe"].isna(), "quantity"]))
    if missing:
        raise ValueError(f"limiting needs the side of every limit; declare {', '.join(missing)} in limits=")
    frame = frame.assign(binder=binder_column(frame, "calculation" in dims))
    kept = []
    groups = frame.groupby(free, sort=True, observed=True) if free else [((), frame)]
    for _, g in groups:
        live = g[g["safe"] != "everywhere"]
        finite = live[np.isfinite(live["value"])]
        if finite["safe"].nunique() > 1:
            below = finite.loc[finite["safe"] == "below", "quantity"].iloc[0]
            above = finite.loc[finite["safe"] == "above", "quantity"].iloc[0]
            raise ValueError(
                f"{below} is safe below the required {solve_for} and {above} above it; "
                "the feasible region is an interval, not one curve"
            )
        blocked = live[live["safe"] == "nowhere"]
        if not blocked.empty:
            kept.append(blocked.iloc[[0]])
            continue
        if finite.empty:
            kept.append(g.iloc[[0]])
            continue
        pick = finite["value"].idxmin() if finite["safe"].iloc[0] == "below" else finite["value"].idxmax()
        kept.append(g.loc[[pick]])
    out = pd.concat(kept, ignore_index=True)
    dims = {k: v for k, v in dims.items() if k not in ("quantity", "calculation")}
    return out, dims


def _order_envelopes(out: dict) -> None:
    for low, high in (("lower", "upper"), ("inner_lower", "inner_upper")):
        a, b = out[low], out[high]
        if np.isnan(a) or np.isnan(b):
            out[low] = out[high] = float("nan")
        else:
            out[low], out[high] = min(a, b), max(a, b)


def _warn_hole_in_bracket(sweep: Sweep, xs: np.ndarray, crossing: float, fixed_here: dict, solve_for: str, context: str) -> None:
    if np.isnan(crossing) or any(k not in sweep.parameters for k in fixed_here):
        return
    on_grid = {}
    for k, v in fixed_here.items():
        levels = sweep.levels(k)
        if k in sweep.categorical:
            if v in levels:
                on_grid[k] = v
        elif np.isclose(levels, v, rtol=1e-9, atol=0.0).any():
            on_grid[k] = v
    if len(on_grid) != len(fixed_here):
        return
    for k in sweep.select(on_grid):
        if not sweep.is_hole(k):
            continue
        v = sweep.cases[k][solve_for]
        if (xs < v).any() and (xs > v).any() and xs[xs < v].max() <= crossing <= xs[xs > v].min():
            warnings.warn(
                f"{context}: the crossing is interpolated across missing case {k} ({_case_text(sweep.case_of(k))})",
                stacklevel=4,
            )
