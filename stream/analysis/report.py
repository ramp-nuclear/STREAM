r"""
Reporting
=========

A set of utilities for analysing an :ref:`Aggregator` system before solving or as part
of a debugging process
"""
from enum import Enum
from inspect import Parameter, signature
from typing import Callable, Iterator, Literal

from IPython.display import display, Markdown
from rich.table import Table

from stream import Calculation
from stream.aggregator import Aggregator


def _filter_vars(agr: Aggregator, c: Calculation,
                 condition: Callable[[Aggregator, Calculation, str, Parameter], bool]):
    """Filters out variable names according to a given condition"""
    return [name for name, param in signature(c.calculate).parameters.items()
            if condition(agr, c, name, param)]


def _check_unset(agr: Aggregator, c: Calculation, name: str, param: Parameter) -> bool:
    """Checks whether an input (name, parameter) is included in the Aggregator. `kwargs`
    are regarded as set, since they are provisional.
    """
    return ((param.kind is not Parameter.VAR_KEYWORD)
            and (name not in agr.external.get(c, []))
            and (name not in agr.funcs.get(c, [])))


def _check_missing(agr: Aggregator, c: Calculation, name: str, param: Parameter) -> bool:
    return _check_unset(agr, c, name, param) and param.default is Parameter.empty


def _check_set_externally(agr: Aggregator, c: Calculation, name: str, _) -> bool:
    return name in agr.funcs.get(c, [])


def _entries(agr: Aggregator) -> Iterator:
    for c, section in agr.sections.items():
        yield (
            str(c),
            type(c).__name__,
            f"{section.start} - {section.stop}",
            ', '.join(_filter_vars(agr, c, _check_unset)[1:]),
            ', '.join(_filter_vars(agr, c, _check_set_externally)),
            ', '.join(_filter_vars(agr, c, _check_missing)[1:])
        )


def _md_table_format(agr: Aggregator) -> Iterator[str]:
    for e in _entries(agr):
        yield "| "
        yield " | ".join(e)
        yield " |\n"


def _rich_table_format(agr: Aggregator) -> Table:
    T = Table("Calculation", "Type", "Equation No.", "Unset", "Set Externally", "Missing")
    for e in _entries(agr):
        T.add_row(*e)
    return T


def description(agr: Aggregator) -> str:
    return (
        f"Contains {len(agr)} Equations, "
        f"{len(agr.graph.nodes)} Nodes, "
        f"{len(agr.graph.edges)} Edges.\n"
    )


def _markdown_report(agr: Aggregator) -> str:
    """Creates the data to report prettily using markdown"""

    header = (
        "| Calculation | Type | Equation No. | Unset | Set Externally | Missing |\n"
        "| :---------- | :--- | :----------- | :---- | :------------- | :------ |\n"
    )
    return "".join([description(agr), "\n", header, *_md_table_format(agr)])


class Printer(Enum):
    JUPYTER = "jupyter"
    TERMINAL = "terminal"
    RAW = "raw"


def report(agr: Aggregator,
           printer: Printer | Literal["jupyter", "terminal", "raw"] = Printer.JUPYTER
           ) -> None:
    r"""Print a report with useful information regarding an :ref:`Aggregator`

    Parameters
    ----------
    agr : Aggregator
        to be reported
    printer : Printer | Literal["jupyter", "terminal", "raw"]
        Which method to use for printing the report. By default, we try
        to use IPython to display the report nicely inside a Jupyter notebook.
        The ``terminal`` option uses ``rich`` to display the report in simpler REPLs
        such as the Python REPL.

    Examples
    --------
    >>> from stream import Aggregator
    >>> from stream.composition import Calculation_factory as Factory
    >>> A = Factory(lambda v, *, a = None, b: _, [True], dict(v=1))("A")
    >>> B = Factory(lambda v, *, c = None, d: _, [True], dict(a=1, b=2))("B")
    >>> report(Aggregator.from_decoupled(A, B, funcs={A: dict(b=0)}), "raw")
    Contains 2 Equations, 2 Nodes, 0 Edges.
    <BLANKLINE>
    | Calculation | Type | Equation No. | Unset | Set Externally | Missing |
    | :---------- | :--- | :----------- | :---- | :------------- | :------ |
    | A | Constructed | 0 - 1 | a | b |  |
    | B | Constructed | 1 - 2 | c, d |  | d |
    <BLANKLINE>

    Notes
    -----
    What do `Unset`, `Set Externally`, `Missing` mean?

    * **Unset**: Variables which are requested by the calculate method and not present
      in the graph or external functions
    * **Set Externally**: Variables which are not provided by the calculation graph,
      but by external functions.
    * **Missing**: Variables which are requested by the calculate method and not present
      in the graph or external functions, **and** do not have a default value.
      If an Aggregator has a non-empty `Missing` column, the simulation is
      expected to crash.
    """
    match printer:
        case Printer.JUPYTER | Printer.JUPYTER.value:
            display(Markdown(_markdown_report(agr)))
        case Printer.TERMINAL | Printer.TERMINAL.value:
            print(description(agr))
            display(_rich_table_format(agr))
        case Printer.RAW | Printer.RAW.value:
            print(_markdown_report(agr))
        case _:
            raise ValueError(f"Printer '{printer}' is not registered, please use one "
                             f"of the following: {list(v.value for v in Printer)}")
