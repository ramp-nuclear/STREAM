from inspect import signature
from typing import Callable

from networkx import DiGraph
from stream import CalculationGraph, Calculation
from stream.aggregator import vars_
from stream.utilities import just

__all__ = ["maximally_coupled"]


def maximally_coupled(
    *calculations: Calculation, exclude: Callable[[str], bool] = None
) -> CalculationGraph:
    r"""Connect calculations into an :class:`CalculationGraph` as fully as possible through
    inspection. This is an `opt-out` way of constructing such connections, if you will.

    Parameters
    ----------
    calculations: Calculation
        The calculations to be connected
    exclude: Callable[[str], bool], optional
        An exclusion strategy for variable names, that is ``True`` values are omitted.

    Returns
    -------
    agr: CalculationGraph
        Maximally coupled calculations

    Examples
    --------
    >>> from stream.composition import Calculation_factory as factory
    >>> Addition = factory(lambda y, *, x: y + x, [False], dict(y=0))
    >>> Multiplication = factory(lambda x, *, y: x * y, [False], dict(x=0))
    >>> add, mult = Addition(name="+"), Multiplication("*")
    >>> maximally_coupled(add, mult).to_aggregator().external
    {*: {'y': {+: 0}}, +: {'x': {*: 1}}}
    >>> maximally_coupled(add, mult, exclude=lambda s: s=="x").to_aggregator().external
    {*: {'y': {+: 0}}}
    """
    edges = []
    exclude = exclude or just(False)
    for v in calculations:
        expected = _expected_in_calculate(v)
        for u in calculations:
            if u is v:
                continue
            if variables := tuple(_indexable(u, v, expected, exclude)):
                edges.append((u, v, vars_(*variables)))
    g = DiGraph()
    g.add_nodes_from(calculations)
    return CalculationGraph(g) + CalculationGraph(DiGraph(edges))


def _indexable(u, v, variables, exclude):
    for var in variables:
        if exclude(var):
            continue
        try:
            ans = u.indices(var, v)
        except KeyError:
            continue
        if ans is None:
            continue
        yield var


def _expected_in_calculate(c):
    return list(signature(c.calculate).parameters)[1:]
