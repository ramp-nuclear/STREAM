"""Figures and tidy data from steady-state sweeps.

Build a :class:`Sweep` from a list of case dicts and the matching list of
result frames (``State.to_dataframe()`` outputs, optionally with threshold
rows and ``uq_attach`` bands), then ask it a question (``profile``, ``field``,
``reduce``, ``required``) and plot the answer::

    sweep = Sweep(cases, frames, agr, labels={"power": ("Power", "MW", 1e-6)})
    sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot()
    sweep.reduce("CHFR", "min", versus="power").plot(color="mdot")
    sweep.required("CHFR", equals=1.3, reduce="min", solve_for="power", versus="mdot").plot()

Every answer carries ``frame``, a tidy DataFrame with ``lower``, ``value``
and ``upper`` columns from the uncertainty envelopes, and ``plot`` draws it
with generated labels and legends under the STREAM style.
"""

from .envelope import at as at
from .labels import Label as Label
from .labels import Labels as Labels
from .style import Style as Style
from .sweep import Sweep as Sweep
from .sweep import run_cases as run_cases
