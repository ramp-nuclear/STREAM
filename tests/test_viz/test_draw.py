import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from stream.viz import Style


def legend_texts(ax):
    leg = ax.get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def test_single_profile_has_no_legend_and_labels(sweep):
    fig, ax = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot()
    assert ax.get_legend() is None
    assert ax.get_xlabel() == "z [m]"
    assert ax.get_ylabel() == "Coolant temperature [°C]"
    assert ax.get_title() == "Power = 4e+06 W, Mass flow = 0.3 kg/s, CC"
    assert len(ax.get_lines()) == 1
    assert ax.get_lines()[0].get_marker() == "o"
    assert plt.rcParams["axes.spines.top"] is True
    plt.close(fig)


def test_family_profile_legend_block_titled_with_unit(sweep):
    sweep.labels = {"power": ("Power", "MW", 1e-6)}
    fig, ax = sweep.profile("CHFR", of="CC", where=dict(mdot=0.3)).plot()
    assert len(ax.get_lines()) == 4
    assert legend_texts(ax) == ["2", "4", "6", "8"]
    assert ax.get_legend().get_title().get_text() == "Power [MW]"
    colors = [ln.get_color() for ln in ax.get_lines()]
    assert len(set(colors)) == 4
    plt.close(fig)


def test_scale_applies_to_x_and_y(sweep):
    sweep.labels = {"z": ("z", "mm", 1e3), "T_cool": ("Bulk", "°C", 2.0)}
    r = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot()
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), [50, 150, 250, 350])
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), 2.0 * r.frame.value)
    assert ax.get_xlabel() == "z [mm]"
    plt.close(fig)


def test_where_reducer_y_carries_the_coordinate_scale(sweep):
    r = sweep.reduce("CHFR", "where_min", versus="power", where=dict(mdot=0.3))
    fig, ax = r.plot(labels={"z": ("z", "mm", 1e3)})
    assert ax.get_ylabel() == "where_min CHFR [mm]"
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), 1e3 * r.frame.value)
    plt.close(fig)


def test_where_reducer_with_an_axis_carries_the_collapsed_axis_scale(sweep):
    r = sweep.reduce("T", "where_max", axis="x", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot()
    assert ax.get_ylabel() == r.ylabel == "where_max Temperature [m]"
    plt.close(fig)
    fig, ax = r.plot(labels={"x": ("x", "mm", 1e3)})
    assert ax.get_ylabel() == "where_max Temperature [mm]"
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), 1e3 * r.frame.value)
    plt.close(fig)


def test_subtitle_follows_the_labels_given_to_plot(sweep):
    r = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot(labels={"power": ("Power", "MW", 1e-6)})
    assert ax.get_title() == "Power = 4 MW, Mass flow = 0.3 kg/s, CC"
    plt.close(fig)
    fig, axes = sweep.reduce("CHFR", "min", versus="power").plot(col="mdot", labels={"mdot": ("Flow", "g/s", 1e3)})
    assert [a.get_title() for a in axes.ravel()] == ["Flow = 200 g/s, CC", "Flow = 300 g/s, CC"]
    plt.close(fig)


def test_band_drawn_only_when_banded(sweep, banded_sweep):
    fig, ax = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot()
    assert not ax.collections
    plt.close(fig)
    fig, ax = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot()
    assert len(ax.collections) == 1
    assert legend_texts(ax) == ["uncertainty"]
    plt.close(fig)
    fig, ax = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3), sigma=2.0).plot()
    assert legend_texts(ax) == ["uncertainty (2σ)"]
    plt.close(fig)


def test_band_modes(banded_sweep):
    r = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot(band="none")
    assert not ax.collections and ax.get_legend() is None
    plt.close(fig)
    fig, ax = r.plot(band="split")
    assert len(ax.collections) == 2
    assert legend_texts(ax) == ["systematic", "systematic + statistical"]
    plt.close(fig)
    fig, ax = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3), sigma=2.0).plot(band="split")
    assert legend_texts(ax) == ["systematic", "systematic + statistical (2σ)"]
    plt.close(fig)
    fig, ax = r.plot(errorbars=True)
    assert ax.containers
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    plt.close(fig)
    with pytest.raises(ValueError, match="'thick'.*total, split, none"):
        r.plot(band="thick")


def test_bound_draws_envelope_without_band(banded_sweep):
    r = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot(bound="lower")
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), r.frame.lower)
    assert not ax.collections
    plt.close(fig)
    with pytest.raises(ValueError, match="'top'.*nominal, lower, upper"):
        r.plot(bound="top")


def test_split_band_with_no_systematic_width(agr):
    from conftest import channel_rows, frame

    from stream.viz.sweep import Sweep

    rows = channel_rows(np.array([40.0, 45.0, 50.0, 55.0]), np.array([3.0, 2.0, 1.5, 1.8]), 4e6)
    only = frame(rows, sys=np.zeros(len(rows)), stat=np.full(len(rows), 0.05))
    fig, ax = Sweep([{}], [only], agr).profile("T_cool").plot(band="split")
    assert len(ax.collections) == 1
    assert legend_texts(ax) == ["systematic + statistical"]
    plt.close(fig)


def test_band_legend_hidden_through_labels(banded_sweep):
    fig, ax = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3)).plot(labels={"band": None})
    assert len(ax.collections) == 1 and ax.get_legend() is None
    plt.close(fig)


def test_curve_three_channels_and_legend_blocks(agr):
    from conftest import make_case_frame

    from stream.viz.sweep import Sweep

    cases = [dict(power=p, mdot=m, depth=d) for p in (2e6, 4e6, 8e6) for m in (0.2, 0.3) for d in (1.0, 2.0)]
    s = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr)
    s.labels = {"CHFR": ("CHF ratio", ""), "T_cool": ("Bulk", "°C")}
    fig, ax = s.reduce("CHFR", "min", versus="power").plot(color="mdot", style="depth")
    assert len(ax.get_lines()) == 4
    assert legend_texts(ax) == ["mdot", "  0.2", "  0.3", "depth", "  1", "  2"]
    assert ax.get_ylabel() == "min CHF ratio"
    assert ax.get_xlabel() == "Power [W]"
    plt.close(fig)


def test_quantity_dimension_selects_rows_by_name(sweep):
    r = sweep.profile(["T_cool", "CHFR"], of="CC", where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot()
    assert len(ax.get_lines()) == 2
    assert legend_texts(ax) == ["Coolant temperature", "CHFR"]
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), r.frame.value[r.frame.quantity == "T_cool"])
    plt.close(fig)


def test_panel_titles_take_categorical_level_names(sweep):
    r = sweep.profile(["T_cool", "CHFR"], of="CC", where=dict(power=4e6, mdot=0.3))
    fig, axes = r.plot(col="quantity")
    assert axes.shape == (1, 2)
    assert [a.get_title() for a in axes.ravel()] == [f"Coolant temperature, {r.subtitle}", f"CHFR, {r.subtitle}"]
    plt.close(fig)


def test_grid_panels(sweep):
    fig, axes = sweep.reduce("CHFR", "min", versus="power").plot(col="mdot")
    assert axes.shape == (1, 2)
    assert [a.get_title() for a in axes.ravel()] == ["Mass flow = 0.2 kg/s, CC", "Mass flow = 0.3 kg/s, CC"]
    assert axes[0, 1].get_ylabel() == ""
    plt.close(fig)
    with pytest.raises(ValueError, match="ax=.*col="):
        sweep.reduce("CHFR", "min", versus="power").plot(col="mdot", ax=plt.gca())


def test_existing_axes_and_style_sheet(sweep):
    fig, (a, b) = plt.subplots(1, 2)
    out_fig, out_ax = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3)).plot(ax=b, style_sheet=Style(linewidth=3.0))
    assert out_fig is fig and out_ax is b
    assert b.get_lines()[0].get_linewidth() == 3.0
    assert not a.get_lines()
    plt.close(fig)


def test_field_map(sweep):
    f = sweep.field("T", where=dict(power=4e6, mdot=0.3))
    fig, ax = f.plot()
    assert ax.collections
    assert ax.get_xlabel() == "x [m]" and ax.get_ylabel() == "z [m]"
    assert fig.axes[-1].get_ylabel() == "Temperature [°C]"
    assert any(p.get_label() == "meat" for p in ax.patches)
    plt.close(fig)
    fig, ax = f.plot(bound="upper")
    np.testing.assert_allclose(ax.collections[0].get_array().reshape(2, 3), f.arrays["upper"])
    plt.close(fig)


def test_field_refuses_the_line_arguments(sweep):
    f = sweep.field("T", where=dict(power=4e6, mdot=0.3))
    with pytest.raises(ValueError, match="color='mdot'.*field"):
        f.plot(color="mdot")
    with pytest.raises(ValueError, match="row='mdot'.*field"):
        f.plot(row="mdot")
    with pytest.raises(ValueError, match="band='split'.*field.*bound="):
        f.plot(band="split")
    with pytest.raises(ValueError, match="errorbars=True.*field.*bound="):
        f.plot(errorbars=True)


def test_required_curve_plot(sweep):
    r = sweep.required("CHFR", equals=1.5, reduce="min", solve_for="power", versus="mdot")
    fig, ax = r.plot()
    assert ax.get_ylabel() == "Power [W]" and ax.get_xlabel() == "Mass flow [kg/s]"
    assert len(ax.get_lines()) == 1
    plt.close(fig)
    fig, ax = r.plot(labels={"power": ("Power", "MW", 1e-6)})
    assert ax.get_ylabel() == "Power [MW]"
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), 1e-6 * r.frame.value)
    plt.close(fig)
