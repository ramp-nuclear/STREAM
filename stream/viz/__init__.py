"""Figures and tidy data from steady-state sweeps.

A sweep is a set of steady states over a grid of design parameters, and the
figures that matter are the answers to a few questions about it: how a
quantity varies along the channel, how its worst value moves with a
parameter, and how far a parameter can go before a limit is crossed. The
workflow has three steps:

1. Build a :class:`Sweep` from the case dicts, the matching result frames
   (``State.to_dataframe()`` outputs, optionally with threshold rows and
   ``uq_attach`` bands) and the aggregator they came from, from your own loop
   or from :func:`cases` and :func:`run_cases`. ``print(sweep)`` says what can be asked of it.
2. Ask a question: :meth:`Sweep.profile`, :meth:`Sweep.field`,
   :meth:`Sweep.reduce` or :meth:`Sweep.required`. A map is ``reduce`` with
   two parameters on ``versus``.
3. Call ``.plot()`` on the answer, or read its ``frame``, a tidy DataFrame
   with ``lower``, ``value`` and ``upper`` columns from the uncertainty
   envelopes.

::

    cases_ = cases(power=[10e3, 20e3, 30e3, 40e3, 60e3, 80e3], mdot=[0.1, 0.2, 0.3], shape=["cosine", "flat"])
    frames = run_cases(model, cases_)
    sweep = Sweep(cases_, frames, agr,
                  labels={"power": ("Power per plate", "kW", 1e-3),
                          "mdot": ("Mass flow", "kg/s"),
                          "shape": ("Axial power shape", ""),
                          "OSVR": ("OSV ratio", ""),
                          "CHFR": ("CHF ratio", ""),
                          "T_wall, left": ("Clad surface temperature", "°C")},
                  limits={"OSVR": at_least(1.3), "CHFR": at_least(1.3), "T_wall, left": at_most(110.0)})
    print(sweep)
    sweep.profile(["T_cool", "T_wall, left"], where=dict(power=40e3, mdot=0.2, shape="cosine")).plot()
    sweep.reduce("OSVR", "min", versus="power", where=dict(shape="cosine")).plot(color="mdot", crossing="mark")
    sweep.reduce("T_wall, left", "max", versus="power").plot(color="mdot", col="shape", show_limit="shade")
    sweep.reduce("OSVR", "min", versus=("power", "mdot"), where=dict(shape="cosine")).plot(show_limit="shade")
    sweep.required(["OSVR", "CHFR", "T_wall, left"], solve_for="power", versus="mdot", limiting=True).plot(color="shape")

Customisation lives in three places. The :class:`Sweep` holds what is true of
the whole sweep: parameter and quantity labels, the declared limits and
whether holes are filled. The question holds what is asked: ``where``, ``of``,
the reducer, ``versus``, ``limiting`` and the target. :meth:`Result.plot` holds how
the answer is drawn: the encodings, bands, how the limit and its crossing are
shown, the style sheet and label overrides. A declared limit is drawn on every
figure that shows its quantity.

.. warning::

   This sub-package was written by an AI assistant, and it is a large amount of intricately interwoven code.
   Reading it, changing it or fixing it will not feel like the rest of STREAM. Before a change is safe, expect
   to spend real time tracing how :class:`Sweep`, its questions and :class:`Result` hand data to each other,
   and lean on the tests in ``tests/test_viz``.
"""

from ._envelope import at as at
from ._results import Result as Result
from .labels import Label as Label
from .labels import Labels as Labels
from .limits import at_least as at_least
from .limits import at_most as at_most
from .style import Style as Style
from .sweep import Sweep as Sweep
from .sweep import cases as cases
from .sweep import run_cases as run_cases

__all__ = ["Label", "Labels", "Result", "Style", "Sweep", "at", "at_least", "at_most", "cases", "run_cases"]
