r"""
The :class:`Aggregator` class defined below is the beating heart of **STREAM**.

It defines a methodology where several (separate) calculations can be simulated in a coupled manner.
Such coupling is defined through a graph, and is simulated through several backends (see :ref:`Backends`).
The main scheme is neatly presented in :ref:`Documentation`.

.. autosummary::
    :nosignatures:

    Aggregator.compute
    Aggregator.load
    Aggregator.save
    Aggregator.var_index
    Aggregator.at_times
    Aggregator.solve
    Aggregator.solve_steady
"""

from .aggregator import *
from .constraints import *
from .solution import Solution
from .utils import (
    VARS,
    BaseAgr,
    ExternalFunctions,
    add_variables,
    draw_aggregator,
    non_unique_calculations,
    vars_,
)
