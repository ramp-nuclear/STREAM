"""Figures and tidy data from steady-state sweeps.

Build a :class:`Sweep` from a list of case dicts and the matching list of
result frames (``State.to_dataframe()`` outputs, optionally with threshold
rows and ``uq_attach`` bands), declare the limits the design must respect,
then ask it a question (``profile``, ``field``, ``reduce``, ``required``)
and plot the answer::

    cases_ = cases(power=[2e6, 4e6, 6e6], mdot=[0.2, 0.3], wall_law=["Dittus", "regime"])
    cases_, frames = run_cases(model, cases_)
    sweep = Sweep(cases_, frames, agr,
                  labels={"power": ("Power", "MW", 1e-6)},
                  limits={"CHFR": at_least(1.3), "T_clad": at_most(900.0)})
    print(sweep)
    sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3, wall_law="regime")).plot()
    sweep.reduce("CHFR", "min", versus="power").plot(color="mdot", style="wall_law")
    sweep.reduce("CHFR", "min", versus="power", where=dict(wall_law="regime")).plot(crossing="mark")
    sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(col="wall_law", limit="shade")
    sweep.required(["CHFR", "T_clad"], solve_for="power", versus="mdot", limiting=True).plot()

Every answer carries ``frame``, a tidy DataFrame with ``lower``, ``value``
and ``upper`` columns from the uncertainty envelopes, and ``plot`` draws it
with generated labels and legends under the STREAM style. A declared limit
is drawn on every figure that shows its quantity, and ``limiting=True``
collapses several quantities or calculations to the one that binds.
"""

from .envelope import at as at
from .labels import Label as Label
from .labels import Labels as Labels
from .limits import Limit as Limit
from .limits import at_least as at_least
from .limits import at_most as at_most
from .style import Style as Style
from .sweep import Sweep as Sweep
from .sweep import cases as cases
from .sweep import run_cases as run_cases
