import pkgutil

import stream.viz

PUBLIC_MODULES = {"sweep", "labels", "limits", "style"}
PRIVATE_MODULES = {"_results", "_inverse", "_envelope", "_draw", "_encoding", "_maps"}


def test_the_package_exports_exactly_the_user_names():
    assert stream.viz.__all__ == ["Label", "Labels", "Result", "Style", "Sweep", "at", "at_least", "at_most", "cases", "run_cases"]
    assert all(hasattr(stream.viz, n) for n in stream.viz.__all__)
    assert not hasattr(stream.viz, "Limit")


def test_module_names_draw_the_line():
    names = {m.name for m in pkgutil.iter_modules(stream.viz.__path__)}
    assert names == PUBLIC_MODULES | PRIVATE_MODULES


def test_public_modules_declare_all():
    from stream.viz import labels, limits, style, sweep

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
