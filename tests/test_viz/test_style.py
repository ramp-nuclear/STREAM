import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stream.viz.style import DEFAULT_STYLE, Style


def test_default_palette_is_the_six_house_colours():
    assert DEFAULT_STYLE.palette == ("#2980b9", "#1abc9c", "#e67e22", "#8e44ad", "#7f8c8d", "#2c3e50")


def test_context_applies_and_restores_rc():
    before = plt.rcParams["axes.spines.top"]
    with DEFAULT_STYLE.context():
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["legend.frameon"] is False
        assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(DEFAULT_STYLE.palette)
    assert plt.rcParams["axes.spines.top"] == before


def test_ramp_colors_are_hex_and_ordered_light_to_dark():
    colors = DEFAULT_STYLE.ramp_colors(3)
    assert len(colors) == 3 and all(c.startswith("#") and len(c) == 7 for c in colors)
    lum = [sum(int(c[k : k + 2], 16) for k in (1, 3, 5)) for c in colors]
    assert lum[0] > lum[1] > lum[2]
    assert DEFAULT_STYLE.ramp_colors(1) == [DEFAULT_STYLE.ramp_colors(2)[1]]


def test_replace_returns_a_modified_copy():
    other = DEFAULT_STYLE.replace(band_alpha=0.5)
    assert other.band_alpha == 0.5
    assert DEFAULT_STYLE.band_alpha != 0.5
    assert isinstance(other, Style)
