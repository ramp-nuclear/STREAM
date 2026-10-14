import pytest

from stream.viz.labels import STREAM_LABELS, Label, Labels, as_label


def test_shorthands_become_labels():
    assert as_label("mdot", "Mass flow") == Label("Mass flow")
    assert as_label("mdot", ("Mass flow", "kg/s")) == Label("Mass flow", "kg/s")
    assert as_label("mdot", ("Mass flow", "g/s", 1e3)) == Label("Mass flow", "g/s", 1e3)
    assert as_label("band", None) == Label(None)
    assert as_label("x", Label("x", "mm", 1e3)) == Label("x", "mm", 1e3)


def test_bad_shorthand_names_the_key():
    with pytest.raises(TypeError, match="power"):
        as_label("power", 3.0)


def test_unknown_name_displays_itself_without_unit():
    labels = Labels()
    assert labels["CHFR"] == Label("CHFR")
    assert labels.axis("CHFR") == "CHFR"
    assert labels.unit("CHFR") is None
    assert not labels.known("CHFR")


def test_display_falls_back_to_the_name_itself():
    labels = Labels({"mdot": ("Mass flow", "kg/s"), "band": None})
    assert labels.display("mdot") == "Mass flow"
    assert labels.display("CHFR") == "CHFR"
    assert labels.display("band") == "band"
    assert labels.display(0.3) == "0.3"


def test_axis_uses_square_brackets_and_empty_unit_has_none():
    labels = Labels({"mdot": ("Mass flow", "kg/s"), "CHFR": ("CHF ratio", "")})
    assert labels.axis("mdot") == "Mass flow [kg/s]"
    assert labels.axis("CHFR") == "CHF ratio"
    assert labels.unit("CHFR") == ""


def test_value_is_scaled_then_formatted():
    labels = Labels({"mdot": ("Mass flow", "g/s", 1e3), "depth": ("Channel depth", "mm", 1e3)})
    assert labels.value("mdot", 0.3) == "300"
    assert labels.value("depth", 0.002) == "2"
    assert labels.value("power", 5e6) == "5e+06"


def test_merged_later_wins():
    base = Labels({"mdot": ("Mass flow", "kg/s"), "power": ("Power", "W")})
    over = Labels({"power": ("Power", "MW", 1e-6)})
    merged = base.merged(over)
    assert merged["mdot"] == Label("Mass flow", "kg/s")
    assert merged["power"] == Label("Power", "MW", 1e-6)
    assert base["power"] == Label("Power", "W")


def test_describe_lists_fixed_values_with_units():
    labels = Labels({"power": ("Power", "MW", 1e-6), "mdot": ("Mass flow", "kg/s")})
    assert labels.describe({"power": 4e6, "mdot": 0.3}) == "Power = 4 MW, Mass flow = 0.3 kg/s"
    assert labels.describe({}) == ""


def test_stream_defaults_cover_the_saved_keys():
    for name in ("T_cool", "q, left", "static_pressure", "h_left", "T_wall, right", "T", "z", "x"):
        assert STREAM_LABELS.known(name), name
    assert STREAM_LABELS.axis("q, left") == "Heat flux, left [W/m$^2$]"
    assert STREAM_LABELS.axis("T_cool") == "Coolant temperature [°C]"
    assert STREAM_LABELS.unit("Re") == ""
    assert STREAM_LABELS["band"] == Label("uncertainty")
    assert STREAM_LABELS["band_sys"] == Label("systematic")
    assert STREAM_LABELS["band_total"] == Label("systematic + statistical")
    assert STREAM_LABELS["quantity"] == Label("Quantity")
    assert STREAM_LABELS["calculation"] == Label("Calculation")


def test_labels_is_a_mapping_in_insertion_order():
    labels = Labels({"b": "B", "a": "A"})
    assert list(labels) == ["b", "a"]
    assert len(labels) == 2
    assert "a" in labels
