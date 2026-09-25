"""report() is readable off-notebook, and the Unset/Missing columns strip the
state parameter by kind (keyword-only wireable vars only), not by position — so a
real unset variable is never dropped."""

import numpy as np
import pytest

from stream.aggregator import Aggregator
from stream.analysis.report import Printer, report
from stream.composition import Calculation_factory


def _unset_column(capsys, agr) -> str:
    """The 'Unset' cell of the single data row of a 'raw' report."""
    report(agr, "raw")
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.startswith("| A "):
            return [c.strip() for c in line.strip("|").split("|")][3]
    raise AssertionError(f"no data row for 'A' in report:\n{out}")


@pytest.fixture()
def factory_A():
    # calculate(v, *, a=None, b): 'v' is the (positional) state slice; 'a'/'b' are
    # the wireable keyword-only variables.
    return Calculation_factory(lambda v, *, a=None, b: None, [True], dict(v=1))("A")


def test_report_default_printer_readable_off_notebook(capsys, factory_A):
    # With no live IPython frontend, the default (JUPYTER) printer must fall back to
    # the terminal renderer instead of printing "<...Markdown object>".
    report(Aggregator.from_decoupled(factory_A, funcs={factory_A: dict(b=0)}))
    out = capsys.readouterr().out
    assert "Calculation" in out
    assert "Markdown object" not in out


def test_report_terminal_printer_renders_readably(capsys, factory_A):
    # The terminal renderer draws the rich table (not its repr) in a plain process.
    report(Aggregator.from_decoupled(factory_A, funcs={factory_A: dict(b=0)}), Printer.TERMINAL)
    out = capsys.readouterr().out
    assert "Calculation" in out
    assert "rich.table.Table object" not in out


def test_report_unset_column_keeps_real_var_when_state_param_is_func_supplied(capsys, factory_A):
    # When the positional state slice 'v' is itself func-supplied, by-kind filtering
    # (keyword-only vars only) still keeps the real unset var 'a' and never lists 'v'.
    normal = _unset_column(capsys, Aggregator.from_decoupled(factory_A, funcs={factory_A: dict(b=0)}))
    patho = _unset_column(capsys, Aggregator.from_decoupled(factory_A, funcs={factory_A: dict(v=0, b=0)}))
    assert "a" in normal and "v" not in normal
    assert "a" in patho  # the real unset var survives even when 'v' is func-supplied
    assert "v" not in patho  # the state slice is never reported as unset


def test_report_state_slice_never_in_unset_for_real_calculation(capsys):
    # A real Calculation's state slice ('variables'/'y') and any positional 't' must
    # never appear in the Unset column — they are solver-supplied, not wireable.
    from stream.calculations import ChannelAndContacts
    from stream.pipe_geometry import EffectivePipe
    from stream.substances import light_water

    pipe = EffectivePipe.rectangular(0.6, 0.06, 0.003, 0.003)
    channel = ChannelAndContacts(np.linspace(0.0, 0.6, 5), light_water, pipe, stop_at_saturation=False)
    with pytest.warns(UserWarning):  # the Tin/Tin_minus heuristic warning
        agr = Aggregator.from_decoupled(channel, funcs={channel: dict(mdot=0.02, Tin=80.0)})
    report(agr, "raw")
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("| CC "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    unset = {v.strip() for v in cells[3].split(",")}
    assert "variables" not in unset and "y" not in unset and "t" not in unset
