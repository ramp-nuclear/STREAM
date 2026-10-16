import pytest

from stream.viz._encoding import LINESTYLES, MARKERS, Encoding
from stream.viz.labels import Label, Labels
from stream.viz.style import DEFAULT_STYLE

LABELS = Labels({"power": ("Power", "MW", 1e-6), "mdot": ("Mass flow", "kg/s"), "CHFR": Label("CHF ratio", "", color="#e67e22"), "OSVR": ("OSV ratio", "")})
NUM = frozenset({"power", "mdot", "depth"})


def test_nothing_varies():
    e = Encoding.assign({}, NUM, LABELS, DEFAULT_STYLE)
    assert e.channels == {} and e.legend == [] and e.col is None and e.row is None
    assert e.props({}) == {}


def test_quantity_takes_color_with_registry_colours():
    e = Encoding.assign({"quantity": ["OSVR", "CHFR"]}, NUM, LABELS, DEFAULT_STYLE)
    assert e.channels == {"color": "quantity"}
    assert e.maps["color"]["CHFR"] == "#e67e22"
    assert e.maps["color"]["OSVR"] == DEFAULT_STYLE.palette[0]
    assert e.legend[0].title == "Quantity"
    assert [t for t, _ in e.legend[0].entries] == ["OSV ratio", "CHF ratio"]
    assert e.props({"quantity": "CHFR"}) == {"color": "#e67e22"}


def test_numeric_dimension_gets_a_ramp_and_scaled_entries():
    e = Encoding.assign({"power": [2e6, 4e6, 6e6]}, NUM, LABELS, DEFAULT_STYLE)
    assert e.channels == {"color": "power"}
    assert list(e.maps["color"].values()) == DEFAULT_STYLE.ramp_colors(3)
    assert e.legend[0].title == "Power [MW]"
    assert [t for t, _ in e.legend[0].entries] == ["2", "4", "6"]


def test_numeric_ramp_position_comes_from_the_whole_sweep():
    space = {"power": [2e6, 4e6, 6e6, 8e6]}
    whole = Encoding.assign({"power": space["power"]}, NUM, LABELS, DEFAULT_STYLE, level_space=space)
    part = Encoding.assign({"power": [2e6, 6e6]}, NUM, LABELS, DEFAULT_STYLE, level_space=space)
    assert part.maps["color"] == {2e6: whole.maps["color"][2e6], 6e6: whole.maps["color"][6e6]}
    alone = Encoding.assign({"power": [2e6, 6e6]}, NUM, LABELS, DEFAULT_STYLE)
    assert alone.maps["color"][6e6] != part.maps["color"][6e6]


def test_a_level_keeps_its_colour_when_a_case_is_missing(holed_sweep):
    whole = holed_sweep.profile("CHFR", of="CC", where=dict(mdot=0.2))
    with pytest.warns(UserWarning, match="skipping unsolved"):
        gapped = holed_sweep.profile("CHFR", of="CC", where=dict(mdot=0.3))
    assert whole.dims["power"] == [2e6, 4e6, 6e6, 8e6] and gapped.dims["power"] == [2e6, 4e6, 6e6]
    a = Encoding.assign(whole.dims, whole._numeric, whole.sweep.labels, DEFAULT_STYLE, level_space=whole._level_space)
    b = Encoding.assign(gapped.dims, gapped._numeric, gapped.sweep.labels, DEFAULT_STYLE, level_space=gapped._level_space)
    assert b.maps["color"][6e6] == a.maps["color"][6e6]


def test_default_order_quantity_then_style_then_marker():
    e = Encoding.assign({"quantity": ["CHFR", "OSVR"], "mdot": [0.2, 0.3], "depth": [1.0, 2.0, 3.0]}, NUM, LABELS, DEFAULT_STYLE)
    assert e.channels == {"color": "quantity", "style": "mdot", "marker": "depth"}
    assert e.maps["style"] == {0.2: LINESTYLES[0], 0.3: LINESTYLES[1]}
    assert e.maps["marker"] == {1.0: MARKERS[0], 2.0: MARKERS[1], 3.0: MARKERS[2]}
    assert [b.title for b in e.legend] == ["Quantity", "Mass flow [kg/s]", "depth"]
    assert e.props({"quantity": "OSVR", "mdot": 0.3, "depth": 2.0}) == {"color": DEFAULT_STYLE.palette[0], "linestyle": "--", "marker": "s"}


def test_explicit_overrides():
    dims = {"quantity": ["CHFR", "OSVR"], "mdot": [0.2, 0.3]}
    e = Encoding.assign(dims, NUM, LABELS, DEFAULT_STYLE, color="mdot", style_="quantity")
    assert e.channels == {"color": "mdot", "style": "quantity"}
    e2 = Encoding.assign(dims, NUM, LABELS, DEFAULT_STYLE, col="mdot")
    assert e2.channels == {"color": "quantity"} and e2.col == "mdot"
    assert [b.title for b in e2.legend] == ["Quantity"]


def test_override_naming_a_fixed_dimension_raises():
    with pytest.raises(ValueError, match="marker.*'depth'.*does not vary.*quantity, mdot"):
        Encoding.assign({"quantity": ["CHFR", "OSVR"], "mdot": [0.2, 0.3]}, NUM, LABELS, DEFAULT_STYLE, marker="depth")


def test_same_dimension_twice_raises():
    with pytest.raises(ValueError, match="'mdot'.*color.*style"):
        Encoding.assign({"mdot": [0.2, 0.3]}, NUM, LABELS, DEFAULT_STYLE, color="mdot", style_="mdot")


def test_fourth_dimension_raises_and_suggests():
    dims = {"quantity": ["a", "b"], "mdot": [1.0, 2.0], "depth": [1.0, 2.0], "power": [1.0, 2.0]}
    with pytest.raises(ValueError, match="'power'.*col=.*row=.*where="):
        Encoding.assign(dims, NUM, LABELS, DEFAULT_STYLE)
    e = Encoding.assign(dims, NUM, LABELS, DEFAULT_STYLE, row="power")
    assert e.row == "power" and set(e.channels) == {"color", "style", "marker"}


def test_capacity_exceeded_raises():
    with pytest.raises(ValueError, match="5 levels.*style.*4"):
        Encoding.assign({"power": [1.0, 2.0, 3.0, 4.0, 5.0], "quantity": ["a", "b"]}, NUM, LABELS, DEFAULT_STYLE, style_="power")
    with pytest.raises(ValueError, match="7 levels.*color.*6"):
        Encoding.assign({"quantity": list("abcdefg")}, NUM, LABELS, DEFAULT_STYLE)


def test_categorical_palette_order_is_registry_then_alphabetical():
    labels = Labels({"OSVR": "OSV ratio", "CHFR": "CHF ratio"})
    e = Encoding.assign({"quantity": ["OFIR", "CHFR", "OSVR"]}, NUM, labels, DEFAULT_STYLE)
    assert e.maps["color"] == {"OSVR": DEFAULT_STYLE.palette[0], "CHFR": DEFAULT_STYLE.palette[1], "OFIR": DEFAULT_STYLE.palette[2]}


def test_numeric_levels_printing_alike_warn_and_keep_the_legend():
    labels = Labels({"power": ("Power", "MW", 1.0, ".1f")})
    with pytest.warns(UserWarning, match=r"power levels 1 and 1\.04 both print as '1\.0'; declare a finer format for power in labels="):
        e = Encoding.assign({"power": [1.0, 1.04, 2.0]}, NUM, labels, DEFAULT_STYLE)
    assert [t for t, _ in e.legend[0].entries] == ["1.0", "1.0", "2.0"]


def test_numeric_levels_printing_apart_do_not_warn():
    labels = Labels({"power": ("Power", "MW")})
    e = Encoding.assign({"power": [1.0, 1.04, 2.0]}, NUM, labels, DEFAULT_STYLE)
    assert [t for t, _ in e.legend[0].entries] == ["1", "1.04", "2"]


def test_legend_entries_follow_the_registry_default_format():
    e = Encoding.assign({"mdot": [0.2, 0.3]}, NUM, LABELS.with_default_format(".2f"), DEFAULT_STYLE)
    assert [t for t, _ in e.legend[0].entries] == ["0.20", "0.30"]
