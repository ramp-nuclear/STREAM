import pkgutil
from dataclasses import fields
from pathlib import Path

from conftest import make_case_frame

from stream import viz
from stream.viz import Result, Sweep

PUBLIC_MODULES = {"sweep", "labels", "limits", "style"}
PRIVATE_MODULES = {"_common", "_results", "_inverse", "_envelope", "_draw", "_encoding", "_maps"}


def test_the_package_exports_exactly_the_user_names():
    assert viz.__all__ == ["Label", "Labels", "Result", "Style", "Sweep", "at", "at_least", "at_most", "cases", "run_cases"]
    assert all(hasattr(viz, n) for n in viz.__all__)
    assert not hasattr(viz, "Limit")


def test_module_names_draw_the_line():
    names = {m.name for m in pkgutil.iter_modules(viz.__path__)}
    assert names == PUBLIC_MODULES | PRIVATE_MODULES


def test_public_modules_declare_all():
    labels, limits, style, sweep = viz.labels, viz.limits, viz.style, viz.sweep
    assert sweep.__all__ == ["Sweep", "cases", "run_cases"]
    assert labels.__all__ == ["Label", "Labels"]
    assert limits.__all__ == ["Limit", "at_least", "at_most"]
    assert style.__all__ == ["Style"]
    for mod in (sweep, labels, limits, style):
        public = {
            n
            for n in vars(mod)
            if not n.startswith("_") and callable(vars(mod)[n]) and getattr(vars(mod)[n], "__module__", None) == mod.__name__
        }
        assert public == set(mod.__all__)


def test_sweep_public_surface():
    public = {n for n in dir(Sweep) if not n.startswith("_")}
    assert public == {"profile", "field", "reduce", "required", "single", "levels", "case_of", "n_cases", "labels"}


def test_sweep_instance_data(agr, grid_cases):
    s = Sweep(grid_cases, [make_case_frame(c["power"], c["mdot"]) for c in grid_cases], agr)
    assert {n for n in vars(s) if not n.startswith("_")} == {
        "cases",
        "frames",
        "parameters",
        "table",
        "holes",
        "filled",
        "limits",
        "banded",
        "agr",
    }


def test_result_public_fields(agr, grid_cases):
    assert [f.name for f in fields(Result)] == ["frame", "x", "dims", "fixed", "calculation", "question", "sweep", "labels", "limits", "safe_side", "y", "holes", "limit_note"]
    assert {n for n in dir(Result) if not n.startswith("_") and n not in {f.name for f in fields(Result)}} == {"plot", "ylabel", "subtitle", "subtitle_with", "kind", "binding"}
    sweep = Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr)
    assert "sweep=" not in repr(sweep.profile("T_cool", where={"power": 4e6, "mdot": 0.3}))


def test_docs_page_carries_every_line_of_the_package_example():
    example = viz.__doc__.split("::\n\n", 1)[1].split("\n\n", 1)[0]
    page = (Path(viz.__file__).parents[2] / "docs/source/viz.rst").read_text()
    assert all(line.strip() in page for line in example.splitlines())
    assert "automodule" not in page
