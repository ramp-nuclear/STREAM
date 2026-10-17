import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import FakeChannel, channel_rows, frame, make_case_frame
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure

from stream.viz import Label, Labels, Style, Sweep, at, at_least, at_most


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
    assert [a.get_title() for a in axes.ravel()] == ["Flow = 200 g/s", "Flow = 300 g/s"]
    assert fig.get_suptitle() == "CC"
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
    assert [a.get_title() for a in axes.ravel()] == ["Coolant temperature", "CHFR"]
    assert fig.get_suptitle() == r.subtitle
    plt.close(fig)


def test_grid_panels(sweep):
    fig, axes = sweep.reduce("CHFR", "min", versus="power").plot(col="mdot")
    assert axes.shape == (1, 2)
    assert [a.get_title() for a in axes.ravel()] == ["Mass flow = 0.2 kg/s", "Mass flow = 0.3 kg/s"]
    assert fig.get_suptitle() == "CC"
    assert axes[0, 1].get_ylabel() == ""
    plt.close(fig)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="ax=.*col="):
        sweep.reduce("CHFR", "min", versus="power").plot(col="mdot", ax=ax)
    plt.close(fig)


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


def test_binding_markers_and_legend(agr, grid_cases):
    sweep = Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr,
                  limits={"CHFR": at_least(1.5), "T_cool": at_most(60.0)})
    r = sweep.required(["CHFR", "T_cool"], equals={"CHFR": 1.5, "T_cool": 60.0}, solve_for="power", versus="mdot", limiting=True)
    fig, ax = r.plot()
    lines = ax.get_lines()
    assert len(lines) == 2
    assert lines[1].get_linestyle() == "None" and lines[1].get_marker() == "o"
    assert legend_texts(ax) == ["CHFR"]
    assert ax.get_legend().get_title().get_text() == "binding"
    plt.close(fig)


def _marker_colors(ax):
    return [to_hex(line.get_color()) for line in ax.get_lines() if line.get_marker() == "o" and line.get_linestyle() == "None"]


def _legend_marker_colors(ax):
    leg = ax.get_legend()
    return {t.get_text(): to_hex(h.get_color()) for t, h in zip(leg.get_texts(), leg.legend_handles)}


def test_binding_markers_match_the_legend_colours(agr, grid_cases):
    sweep = Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr,
                  limits={"CHFR": at_least(1.5), "T_cool": at_most(60.0)})
    r = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot", limiting=True)
    fig, ax = r.plot()
    assert set(_marker_colors(ax)) == {_legend_marker_colors(ax)["CHFR"]}
    plt.close(fig)


def test_binding_markers_on_errorbars(agr, grid_cases):
    sweep = Sweep(grid_cases, [make_case_frame(**c, banded=True) for c in grid_cases], agr,
                  limits={"CHFR": at_least(1.5), "T_cool": at_most(60.0)})
    r = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot", limiting=True)
    fig, ax = r.plot(errorbars=True)
    assert len(_marker_colors(ax)) == 1
    plt.close(fig)


def test_binding_marker_colours_agree_across_panels(agr):
    def make(power, mdot, peak):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        rows += [("hot", "CHFR", 0, j, chfr[j] * peak) for j in range(4)]
        return frame(rows)

    agr["hot"] = FakeChannel([0.0, 0.1, 0.2, 0.3, 0.4])
    cases = [dict(power=p, mdot=m, peak=k) for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3) for k in (0.8, 1.25)]
    sweep = Sweep(cases, [make(**c) for c in cases], agr, limits={"CHFR": at_least(1.5)})
    r = sweep.required("CHFR", solve_for="power", versus="mdot", of=["CC", "hot"], limiting=True)
    assert set(r.frame[r.frame.peak == 0.8].binder) == {"CHFR of hot"}
    assert set(r.frame[r.frame.peak == 1.25].binder) == {"CHFR of CC"}
    fig, axes = r.plot(col="peak")
    legend = _legend_marker_colors(axes[0, -1])
    assert set(_marker_colors(axes[0, 0])) == {legend["CHFR of hot"]}
    assert set(_marker_colors(axes[0, 1])) == {legend["CHFR of CC"]}
    assert legend["CHFR of hot"] != legend["CHFR of CC"]
    plt.close(fig)


def _two_binder_sweep(agr, labels=None):
    def make(power, mdot, shape):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        k = {"cosine": 1.0, "peaked": 2.0}[shape]
        wall = 50.0 + k * power / 1e5 * np.array([0.7, 0.9, 1.0, 0.8])
        return frame(channel_rows(T, chfr, power) + [("CC", "T_wall, left", 0, j, wall[j]) for j in range(4)])

    cases = [{"power": p, "mdot": m, "shape": s} for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3) for s in ("cosine", "peaked")]
    sweep = Sweep(cases, [make(**c) for c in cases], agr, labels=labels,
                  limits={"CHFR": at_least(1.5), "T_wall, left": at_most(100.0)})
    return sweep.required(["CHFR", "T_wall, left"], solve_for="power", versus="mdot", limiting=True)


def test_binding_markers_keep_off_the_line_colours(agr):
    r = _two_binder_sweep(agr)
    assert set(r.frame.binder) == {"CHFR", "T_wall, left"}
    fig, ax = r.plot(color="shape")
    lines = {to_hex(ln.get_color()) for ln in ax.get_lines() if ln.get_linestyle() != "None"}
    markers = set(_marker_colors(ax))
    assert len(lines) == 2 and len(markers) == 2
    assert lines.isdisjoint(markers)
    plt.close(fig)


def test_binding_legend_shows_display_names(agr):
    r = _two_binder_sweep(agr, labels={"T_wall, left": ("Clad surface temperature", "°C")})
    fig, ax = r.plot(color="shape")
    texts = [t.strip() for t in legend_texts(ax)]
    assert "Clad surface temperature" in texts and "T_wall, left" not in texts
    assert "CHFR" in texts
    plt.close(fig)


def limited(agr, grid_cases, **limits):
    return Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr, limits=limits)


def dashed_lines(ax):
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]


def limit_texts(ax):
    return [t.get_text() for t in ax.texts if getattr(t, "xycoords", None) == ("axes fraction", "data")]


def limit_contour(ax):
    (cs,) = [c for c in ax.collections if isinstance(c, QuadContourSet) and not c.filled]
    return cs


def hatched(ax):
    collections = [c for c in ax.collections if c.get_hatch()]
    return collections + [p for p in ax.patches if p.get_hatch()]


def test_limit_line_on_a_curve(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(color="mdot")
    (line,) = dashed_lines(ax)
    assert line.get_ydata()[0] == pytest.approx(1.5)
    assert any(t.get_text() == "CHFR = 1.5" for t in ax.texts)
    plt.close(fig)
    fig, ax = sweep.reduce("T_cool", "max", versus="power").plot()
    assert dashed_lines(ax) == [] and len(ax.texts) == 0
    plt.close(fig)


def test_limit_line_scaled_and_coloured_by_quantity(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    r = sweep.profile(["CHFR", "T_cool"], where=dict(power=4e6, mdot=0.3))
    fig, ax = r.plot(labels={"CHFR": ("CHFR", "", 10.0)})
    (line,) = dashed_lines(ax)
    assert line.get_ydata()[0] == pytest.approx(15.0)
    chfr_colour = next(ln for ln in ax.get_lines() if ln.get_linestyle() == "-").get_color()
    assert line.get_color() == chfr_colour
    plt.close(fig)


def test_limit_shade_hatches_the_unsafe_side(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(color="mdot", show_limit="shade")
    (patch,) = hatched(ax)
    assert patch.get_hatch() == "////"
    lo, hi = ax.get_ylim()
    assert patch.get_window_extent(fig.canvas.get_renderer()).height > 0
    assert lo < 1.5 < hi
    plt.close(fig)
    sweep2 = limited(agr, grid_cases, T_cool=at_most(50.0))
    fig, ax = sweep2.reduce("T_cool", "max", versus="power").plot(show_limit="shade")
    assert len(hatched(ax)) == 1
    plt.close(fig)


def test_equal_limits_draw_one_line_and_one_text(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3), T_cool=at_least(1.3))
    fig, ax = sweep.reduce(["CHFR", "T_cool"], "min", versus="power", where={"mdot": 0.3}).plot()
    assert len(dashed_lines(ax)) == 1 and to_hex(dashed_lines(ax)[0].get_color()) == Style().limit_color
    assert limit_texts(ax) == ["limit = 1.3"]
    plt.close(fig)


def test_equal_limits_with_one_display_keep_the_name(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3), T_cool=at_least(1.3))
    fig, ax = sweep.reduce(["CHFR", "T_cool"], "min", versus="power").plot(labels={"T_cool": ("CHFR", "")})
    assert limit_texts(ax) == ["CHFR = 1.3"]
    plt.close(fig)


def test_equal_limits_format_with_the_first_quantity(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3), T_cool=at_least(1.3))
    fig, ax = sweep.reduce(["CHFR", "T_cool"], "min", versus="power").plot(labels={"CHFR": ("CHFR", "", 1.0, ".2f"), "T_cool": ("T", "", 1.0, ".0f")})
    assert limit_texts(ax) == ["limit = 1.30"]
    plt.close(fig)


def test_unequal_limits_keep_one_line_per_quantity(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3), T_cool=at_most(50.0))
    fig, ax = sweep.profile(["CHFR", "T_cool"], where={"power": 4e6, "mdot": 0.3}).plot()
    assert len(dashed_lines(ax)) == 2
    plt.close(fig)


def test_shade_is_allowed_when_the_limits_merge(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3), T_cool=at_least(1.3))
    fig, ax = sweep.reduce(["CHFR", "T_cool"], "min", versus="power").plot(show_limit="shade")
    assert len(hatched(ax)) == 1
    plt.close(fig)


def test_limit_shade_refusals(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    with pytest.raises(ValueError, match="shade needs one quantity per axes; put quantity on col= or row="):
        sweep.profile(["CHFR", "T_cool"], where=dict(power=4e6, mdot=0.3)).plot(show_limit="shade")
    with pytest.raises(ValueError, match="no limit declared for 'T_cool'; declare it in limits= to shade"):
        sweep.reduce("T_cool", "max", versus="power").plot(show_limit="shade")
    r = limited(agr, grid_cases).required("CHFR", equals=1.6, solve_for="power", versus="mdot")
    with pytest.raises(ValueError, match="shading needs the side of the limit; declare CHFR in limits="):
        r.plot(show_limit="shade")
    with pytest.raises(ValueError, match="show_limit='both'; use one of line, shade, none"):
        sweep.reduce("CHFR", "min", versus="power").plot(show_limit="both")


def test_limit_shade_allowed_with_quantity_on_panels(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_most(50.0))
    fig, axes = sweep.profile(["CHFR", "T_cool"], where=dict(power=4e6, mdot=0.3)).plot(col="quantity", show_limit="shade")
    assert all(len(hatched(ax)) == 1 for ax in axes.ravel())
    plt.close(fig)


def test_limit_none_and_required_shade(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(show_limit="none")
    assert dashed_lines(ax) == [] and hatched(ax) == []
    plt.close(fig)
    r = sweep.required("CHFR", solve_for="power", versus="mdot")
    fig, ax = r.plot()
    assert dashed_lines(ax) == []
    plt.close(fig)
    fig, ax = r.plot(show_limit="shade")
    assert len(hatched(ax)) == 1
    plt.close(fig)


def test_field_limit_contour(agr, grid_cases):
    sweep = limited(agr, grid_cases, T=at_most(60.0))
    fig, ax = sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3)).plot()
    assert any(type(c).__name__ == "QuadContourSet" for c in ax.collections)
    plt.close(fig)
    field = sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3))
    fig, ax = field.plot(show_limit="shade")
    (cells,) = hatched(ax)
    assert isinstance(cells, PolyCollection)
    assert len(cells.get_paths()) == at_most(60.0).unsafe(field.arrays["value"]).sum() == 2
    plt.close(fig)


def test_limit_shade_on_panels_keeps_every_limit_in_view(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_most(100.0))
    for order in (["CHFR", "T_cool"], ["T_cool", "CHFR"]):
        r = sweep.profile(order, where=dict(power=4e6, mdot=0.3))
        fig, axes = r.plot(col="quantity", show_limit="shade")
        for ax in axes.ravel():
            lo, hi = ax.get_ylim()
            assert lo < 1.5 and 100.0 < hi
            assert len(hatched(ax)) == 1
        plt.close(fig)


def test_limit_shade_refuses_a_free_parameter_on_a_binding_result(agr):
    def make(power, mdot, peak):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        rows += [("hot", "CHFR", 0, j, chfr[j] * peak) for j in range(4)]
        return frame(rows)

    agr["hot"] = FakeChannel([0.0, 0.1, 0.2, 0.3, 0.4])
    cases = [dict(power=p, mdot=m, peak=k) for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3) for k in (0.8, 1.25)]
    sweep = Sweep(cases, [make(**c) for c in cases], agr, limits={"CHFR": at_least(1.5)})
    r = sweep.required("CHFR", solve_for="power", versus="mdot", of=["CC", "hot"], limiting=True)
    with pytest.raises(ValueError, match="shade needs one curve per axes; put peak on col= or row="):
        r.plot(color="peak", show_limit="shade")
    fig, axes = r.plot(col="peak", show_limit="shade")
    assert all(len(hatched(ax)) == 1 for ax in axes.ravel())
    plt.close(fig)
    r = sweep.required("CHFR", solve_for="power", versus="mdot", of=["CC", "hot"])
    with pytest.raises(ValueError, match="shade needs one curve per axes; put calculation on col= or row="):
        r.plot(show_limit="shade")


def test_field_shade_reads_the_drawn_bound_and_skips_safe_fields(agr, grid_cases):
    sweep = Sweep(grid_cases, [make_case_frame(**c, banded=True) for c in grid_cases], agr, limits={"T": at_most(60.0)})
    field = sweep.field("T", of="plate", where=dict(power=4e6, mdot=0.3))
    fig, ax = field.plot(bound="upper", show_limit="shade")
    (cells,) = hatched(ax)
    assert len(cells.get_paths()) == at_most(60.0).unsafe(field.arrays["upper"]).sum() == 3
    plt.close(fig)
    safe = limited(agr, grid_cases, T=at_most(100.0)).field("T", of="plate", where=dict(power=4e6, mdot=0.3))
    fig, ax = safe.plot(show_limit="shade")
    assert hatched(ax) == []
    plt.close(fig)


def contour_sets(ax):
    return [c for c in ax.collections if isinstance(c, QuadContourSet)]


def test_map_draws_filled_contours_with_a_colour_bar(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    sweep.labels = {"power": ("Power", "MW", 1e-6), "mdot": ("Mass flow", "kg/s")}
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot()
    assert ax.get_xlabel() == "Power [MW]" and ax.get_ylabel() == "Mass flow [kg/s]"
    assert len(fig.axes) == 2 and fig.axes[1].get_ylabel() == "min CHFR"
    assert ax.get_title() == "CC"
    assert contour_sets(ax)
    assert any(t.get_text() == "CHFR = 1.5" for t in ax.texts)
    plt.close(fig)


def test_map_shade_and_none(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(show_limit="shade")
    assert any(h for c in contour_sets(ax) for h in c.hatches)
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(show_limit="none")
    assert len(contour_sets(ax)) == 1
    plt.close(fig)


def test_map_marks_holes_and_cases(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr)
    with pytest.warns(UserWarning):
        r = sweep.reduce("CHFR", "min", versus=("power", "mdot"))
    fig, ax = r.plot()
    crosses = [ln for ln in ax.get_lines() if ln.get_marker() == "x"]
    assert len(crosses) == 1 and crosses[0].get_xdata()[0] == pytest.approx(grid_cases[3]["power"])
    plt.close(fig)
    fig, ax = r.plot(style_sheet=Style(map_cases=True))
    dots = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert len(dots) == 1 and len(dots[0].get_xdata()) == 7
    plt.close(fig)


def test_map_on_a_ragged_sweep_draws_without_crosses(agr, grid_cases):
    cs = grid_cases[:-1]
    sweep = Sweep(cs, [make_case_frame(**c) for c in cs], agr)
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot()
    assert contour_sets(ax) and not [ln for ln in ax.get_lines() if ln.get_marker() == "x"]
    plt.close(fig)


def test_map_panels_share_levels_and_refuse_line_arguments(agr, grid_cases):
    sweep = limited(agr, grid_cases)
    r = sweep.reduce("CHFR", "min", versus=("power", "mdot"))
    for name in ("color", "style", "marker"):
        with pytest.raises(ValueError, match=f"{name}= does not apply to a map; the colour is the quantity; use col= or row= for a dimension"):
            r.plot(**{name: "mdot"})
    with pytest.raises(ValueError, match="band= does not apply to a map; use bound= to pick an envelope"):
        r.plot(band="none")
    with pytest.raises(ValueError, match="errorbars= does not apply to a map; use bound= to pick an envelope"):
        r.plot(errorbars=True)
    cs = [dict(power=p, mdot=m, Tin=t) for t in (300.0, 310.0) for p in (2e6, 4e6, 6e6) for m in (0.2, 0.3)]
    three = Sweep(cs, [make_case_frame(c["power"], c["mdot"]) for c in cs], agr)
    with pytest.raises(ValueError, match="'Tin' varies; a map takes it on col= or row="):
        three.reduce("CHFR", "min", versus=("power", "mdot")).plot()
    fig, axes = three.reduce("CHFR", "min", versus=("power", "mdot")).plot(col="Tin")
    assert axes.shape == (1, 2) and len(fig.axes) == 3
    sets = [contour_sets(ax)[0] for ax in axes.ravel()]
    np.testing.assert_allclose(sets[0].levels, sets[1].levels)
    plt.close(fig)


def test_filled_points_draw_open_markers_with_a_legend_entry(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3)).plot()
    open_ = [ln for ln in ax.get_lines() if ln.get_linestyle() == "None" and ln.get_markerfacecolor() == "white"]
    assert len(open_) == 1 and open_[0].get_xdata()[0] == pytest.approx(4e6)
    assert "interpolated" in legend_texts(ax)
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot()
    assert [ln for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"]
    plt.close(fig)


def test_where_reducer_carries_no_limit(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(0.2))
    r = sweep.reduce("CHFR", "where_min", versus="power")
    assert r.limits == {}
    fig, ax = r.plot()
    assert dashed_lines(ax) == [] and len(ax.texts) == 0
    plt.close(fig)
    with pytest.raises(ValueError, match=r"where_min gives a location, not a value of CHFR; use min, max, mean or at\(\) to shade"):
        r.plot(show_limit="shade")
    m = sweep.reduce("CHFR", "where_min", versus=("power", "mdot"))
    fig, ax = m.plot()
    assert len(contour_sets(ax)) == 1 and len(ax.texts) == 0
    plt.close(fig)
    with pytest.raises(ValueError, match="where_min gives a location, not a value of CHFR"):
        m.plot(show_limit="shade")


def test_custom_reducer_carries_no_limit(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5))

    def lowest(values, coords):
        return float(np.nanmin(values))

    r = sweep.reduce("CHFR", lowest, versus="power")
    fig, ax = r.plot()
    assert dashed_lines(ax) == []
    plt.close(fig)
    with pytest.raises(
        ValueError, match=r"a custom reducer gives no value of CHFR to compare with its limit; use min, max, mean or at\(\)"
    ):
        r.plot(show_limit="shade")


def test_required_shade_reads_each_panel_side(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.5), T_cool=at_least(45.0))
    r = sweep.required(["CHFR", "T_cool"], solve_for="power", versus="mdot")
    assert set(r.frame.safe) == {"below", "above"} and r.safe_side is None
    fig, axes = r.plot(col="quantity", show_limit="shade")
    assert all(len(hatched(ax)) == 1 for ax in axes.ravel())
    plt.close(fig)
    with pytest.raises(ValueError, match="CHFR is safe below the required power and T_cool above it; shade one quantity per axes"):
        r.plot(show_limit="shade")


def test_map_shade_at_most_hatches_above_the_limit(agr, grid_cases):
    sweep = limited(agr, grid_cases, T_cool=at_most(60.0))
    fig, ax = sweep.reduce("T_cool", "max", versus=("power", "mdot")).plot(show_limit="shade")
    (shade,) = [c for c in contour_sets(ax) if any(c.hatches)]
    assert shade.levels[0] == pytest.approx(60.0) and shade.levels[1] > 60.0
    plt.close(fig)


def test_map_cases_dots_skip_filled_cases(agr, grid_cases):
    frames = [make_case_frame(**c) for c in grid_cases]
    frames[3] = None
    with pytest.warns(UserWarning):
        sweep = Sweep(grid_cases, frames, agr, fill_holes=True)
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(style_sheet=Style(map_cases=True))
    dots = [ln for ln in ax.get_lines() if ln.get_marker() == "o" and ln.get_markerfacecolor() != "white"]
    opens = [ln for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"]
    assert len(dots) == 1 and len(dots[0].get_xdata()) == 7
    assert len(opens) == 1 and opens[0].get_xdata()[0] == pytest.approx(grid_cases[3]["power"])
    plt.close(fig)


def formatted_sweep(agr):
    cases = [dict(power=p, mdot=m) for p in (4.21e6, 6e6) for m in (0.2, 0.3)]
    return Sweep(cases, [make_case_frame(**c) for c in cases], agr, limits={"CHFR": at_least(1.5)})


def test_a_label_format_reaches_legend_subtitle_and_limit(agr):
    sweep = formatted_sweep(agr)
    labels = {"power": ("Power", "MW", 1e-6, ".1f"), "CHFR": ("CHFR", "", 1.0, ".2f")}
    fig, ax = sweep.profile("CHFR", of="CC", where=dict(mdot=0.3)).plot(labels=labels)
    assert legend_texts(ax) == ["4.2", "6.0"]
    plt.close(fig)
    fig, ax = sweep.profile("CHFR", of="CC", where=dict(power=4.21e6, mdot=0.3)).plot(labels=labels)
    assert ax.get_title() == "Power = 4.2 MW, mdot = 0.3, CC"
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus="mdot").plot(labels=labels)
    assert legend_texts(ax) == ["4.2", "6.0"]
    assert any(t.get_text() == "CHFR = 1.50" for t in ax.texts)
    plt.close(fig)


def test_style_number_format_sets_the_default_of_a_figure(sweep):
    r = sweep.reduce("CHFR", "min", versus="power")
    fig, ax = r.plot()
    assert legend_texts(ax) == ["0.2", "0.3"]
    plt.close(fig)
    fig, ax = r.plot(style_sheet=Style(number_format=".2f"))
    assert legend_texts(ax) == ["0.20", "0.30"]
    plt.close(fig)
    single = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    fig, ax = single.plot(style_sheet=Style(number_format=".2f"))
    assert ax.get_title() == "Power = 4000000.00 W, Mass flow = 0.30 kg/s, CC"
    plt.close(fig)
    assert single.subtitle == "Power = 4e+06 W, Mass flow = 0.3 kg/s, CC"


def test_style_number_format_reaches_sigma_and_the_reducer_position(sweep, banded_sweep):
    r = banded_sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3), sigma=2.0)
    fig, ax = r.plot(style_sheet=Style(number_format=".1f"))
    assert any("(2.0σ)" in t for t in legend_texts(ax))
    plt.close(fig)
    r = sweep.reduce("T_cool", at(z=0.35), versus="power", where=dict(mdot=0.3))
    fig, ax = r.plot(labels={"z": ("z", "m", 1.0, ".3f")})
    assert ax.get_ylabel() == "at z=0.350 Coolant temperature [°C]"
    plt.close(fig)
    fig, ax = r.plot(style_sheet=Style(number_format=".2f"))
    assert ax.get_ylabel() == "at z=0.35 Coolant temperature [°C]"
    plt.close(fig)
    assert r.ylabel == "at z=0.35 Coolant temperature [°C]"


def test_levels_printing_alike_warn_at_the_plot_call(agr):
    cases = [dict(power=p, mdot=0.3) for p in (1.0e6, 1.04e6)]
    sweep = Sweep(cases, [make_case_frame(**c) for c in cases], agr)
    r = sweep.reduce("CHFR", "min", versus="mdot")
    with pytest.warns(UserWarning, match=r"power levels 1e\+06 and 1\.04e\+06 both print as '1\.0'") as record:
        fig, ax = r.plot(labels={"power": ("Power", "MW", 1e-6, ".1f")})
    assert record[0].filename == __file__
    assert legend_texts(ax) == ["1.0", "1.0"]
    plt.close(fig)


def drop_lines(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def crossing_markers(ax):
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "None" and ln.get_markeredgecolor() == "white"]


def foot_annotations(ax):
    return [t for t in ax.texts if getattr(t, "xycoords", None) == ("data", "axes fraction")]


def foot_texts(ax):
    return [t.get_text() for t in foot_annotations(ax)]


def test_crossing_mark_draws_a_drop_line_a_marker_and_the_value(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    r = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    xc = 4.8e6
    fig, ax = r.plot(crossing="mark")
    (drop,) = drop_lines(ax)
    (segment,) = drop.get_segments()
    assert segment[0][0] == pytest.approx(xc) and segment[1][0] == pytest.approx(xc)
    assert segment[1][1] == pytest.approx(1.3) and segment[0][1] == pytest.approx(ax.get_ylim()[0])
    (marker,) = crossing_markers(ax)
    assert marker.get_xdata()[0] == pytest.approx(xc) and marker.get_ydata()[0] == pytest.approx(1.3)
    assert foot_texts(ax) == ["4.8e+06"]
    assert len(dashed_lines(ax)) == 1
    plt.close(fig)


def test_crossing_text_sits_under_the_ticks(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3}).plot(crossing="mark")
    (t,) = foot_annotations(ax)
    assert t.get_rotation() == 0 and t.xycoords == ("data", "axes fraction") and t.xy[1] == 0
    assert t.xyann[1] < 0 and t.get_ha() == "center" and t.get_va() == "top"
    plt.close(fig)


def test_crossing_mark_twice_on_one_axes_keeps_the_label_pad(agr, grid_cases):
    r = limited(agr, grid_cases, CHFR=at_least(1.3)).reduce("CHFR", "min", versus="power", where={"mdot": 0.3})
    fig, ax = r.plot(crossing="mark")
    pad = ax.xaxis.labelpad
    assert pad > plt.rcParams["axes.labelpad"]
    r.plot(ax=ax, crossing="mark")
    assert ax.xaxis.labelpad == pytest.approx(pad)
    plt.close(fig)


def test_crossing_mark_draws_on_a_figure_without_pyplot(agr, grid_cases):
    fig = Figure()
    ax = fig.subplots()
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3}).plot(ax=ax, crossing="mark")
    assert len(foot_annotations(ax)) == 1


def test_close_crossings_take_two_rows_and_far_ones_one(agr):
    cases = [{"power": p, "mdot": m} for p in (2e6, 4e6, 6e6, 8e6) for m in (0.28, 0.3, 0.45)]
    sweep = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr, limits={"CHFR": at_least(1.3)})
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="mark")
    rows = [t.xyann[1] for t in sorted(foot_annotations(ax), key=lambda t: t.xy[0])]
    assert rows[0] == rows[2] and rows[1] < rows[0]
    plt.close(fig)


def test_crossing_mark_on_a_shared_x_grid_shows_ticks_in_every_panel(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, axes = sweep.reduce("CHFR", "min", versus="power").plot(row="mdot", crossing="mark")
    assert all(ax.xaxis.get_ticklabels()[0].get_visible() for ax in axes.ravel())
    assert all(len(foot_annotations(ax)) == 1 for ax in axes.ravel())
    plt.close(fig)


def test_crossing_none_draws_nothing_new(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot()
    assert drop_lines(ax) == [] and crossing_markers(ax) == [] and foot_texts(ax) == []
    plt.close(fig)


def test_crossing_legend_appends_the_value_to_the_one_encoded_dimension(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="legend")
    assert legend_texts(ax) == ["0.2 (3.4e+06 W)", "0.3 (4.8e+06 W)"]
    assert ax.get_legend().get_title().get_text() == "mdot; Power at CHFR = 1.3"
    assert [d.get_segments()[0][0][0] for d in drop_lines(ax)] == pytest.approx([3.4e6, 4.8e6])
    assert foot_texts(ax) == []
    plt.close(fig)


def test_crossing_legend_with_two_encoded_dimensions_adds_a_block(agr):
    cases = [dict(power=p, mdot=m, wall_law=w) for p in (2e6, 4e6, 6e6, 8e6) for m in (0.2, 0.3) for w in ("Dittus", "regime")]
    sweep = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr, limits={"CHFR": at_least(1.3)})
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(color="mdot", style="wall_law", crossing="legend")
    texts = legend_texts(ax)
    block = texts[texts.index("Power at CHFR = 1.3") + 1:]
    assert block == ["  0.2, Dittus: 3.4e+06 W", "  0.2, regime: 3.4e+06 W", "  0.3, Dittus: 4.8e+06 W", "  0.3, regime: 4.8e+06 W"]
    assert "  0.2" in texts and "  Dittus" in texts
    plt.close(fig)


def test_crossing_legend_on_a_single_curve_titles_one_entry(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3)).plot(crossing="legend")
    assert ax.get_legend().get_title().get_text() == "Power at CHFR = 1.3"
    assert legend_texts(ax) == ["4.8e+06 W"]
    assert foot_texts(ax) == []
    plt.close(fig)


def test_crossing_skips_a_curve_that_never_reaches_the_limit(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(0.1))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="mark")
    assert drop_lines(ax) == [] and crossing_markers(ax) == [] and foot_texts(ax) == []
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="legend")
    assert legend_texts(ax) == ["0.2", "0.3"]
    plt.close(fig)


def test_crossing_marks_only_the_first_of_several(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.7))
    r = sweep.profile("CHFR", of="CC", where=dict(power=4e6, mdot=0.3))
    assert np.count_nonzero(np.diff(np.sign(r.frame.value.to_numpy() - 1.7))) == 2
    fig, ax = r.plot(crossing="mark")
    (drop,) = drop_lines(ax)
    (segment,) = drop.get_segments()
    assert segment[0][0] == pytest.approx(0.21)
    assert foot_texts(ax) == ["0.21"]
    plt.close(fig)


def test_crossing_follows_the_drawn_bound(agr, grid_cases):
    sweep = Sweep(grid_cases, [make_case_frame(**c, banded=True) for c in grid_cases], agr, limits={"CHFR": at_least(1.3)})
    r = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    fig, ax = r.plot(crossing="mark")
    (nominal,) = crossing_markers(ax)
    plt.close(fig)
    fig, ax = r.plot(bound="lower", crossing="mark")
    (lower,) = crossing_markers(ax)
    assert lower.get_xdata()[0] < nominal.get_xdata()[0]
    lows = r.frame.sort_values("power").lower.to_numpy()
    k = np.flatnonzero(lows < 1.3)[0]
    x0, x1 = r.frame.sort_values("power").power.to_numpy()[k - 1:k + 1]
    assert x0 < lower.get_xdata()[0] < x1
    plt.close(fig)


def test_crossing_without_the_limit_line(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3)).plot(show_limit="none", crossing="mark")
    assert len(drop_lines(ax)) == 1 and dashed_lines(ax) == []
    assert [t.get_text() for t in ax.texts] == ["4.8e+06"]
    plt.close(fig)


def test_crossing_refusals(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    with pytest.raises(ValueError, match="crossing= needs a horizontal limit; a required result has none, its y axis is the crossing"):
        sweep.required("CHFR", solve_for="power", versus="mdot").plot(crossing="mark")
    with pytest.raises(ValueError, match="crossing= does not apply to a map"):
        sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(crossing="mark")
    with pytest.raises(ValueError, match="crossing= does not apply to a field"):
        limited(agr, grid_cases, T=at_most(60.0)).field("T", of="plate", where=dict(power=4e6, mdot=0.3)).plot(crossing="mark")
    with pytest.raises(ValueError, match="where_min gives a location, not a value of CHFR"):
        sweep.reduce("CHFR", "where_min", versus="power").plot(crossing="legend")
    with pytest.raises(ValueError, match="no limit declared for 'T_cool'; declare it in limits= to mark its crossing"):
        sweep.reduce("T_cool", "max", versus="power").plot(crossing="mark")
    with pytest.raises(ValueError, match="crossing='both'; use one of none, mark, legend"):
        sweep.reduce("CHFR", "min", versus="power").plot(crossing="both")


def test_crossing_legend_on_panels_prints_at_the_foot(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, axes = sweep.reduce("CHFR", "min", versus="power").plot(col="mdot", crossing="legend")
    assert [foot_texts(ax) for ax in axes.ravel()] == [["3.4e+06"], ["4.8e+06"]]
    assert all(ax.get_legend() is None for ax in axes.ravel())
    plt.close(fig)


def test_crossing_value_prints_with_the_label_format(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    r = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    fig, ax = r.plot(crossing="mark", labels={"power": ("Power", "MW", 1e-6, ".1f")})
    assert foot_texts(ax) == ["4.8"]
    (marker,) = crossing_markers(ax)
    assert marker.get_xdata()[0] == pytest.approx(4.8)
    plt.close(fig)


def test_crossing_legend_names_each_limit_when_quantities_differ(agr):
    cases = [dict(power=p, mdot=0.3) for p in (2e6, 4e6)]
    sweep = Sweep(cases, [make_case_frame(**c) for c in cases], agr, limits={"CHFR": at_least(1.7), "T_cool": at_most(48.0)})
    r = sweep.profile(["CHFR", "T_cool"], of="CC", where=dict(mdot=0.3))
    fig, ax = r.plot(color="quantity", style="power", crossing="legend")
    texts = legend_texts(ax)
    assert texts[texts.index("z at the limit") + 1:] == [
        "  CHFR, 4e+06, CHFR = 1.7: 0.21 m",
        "  Coolant temperature, 2e+06, Coolant temperature = 48 °C: 0.17 m",
        "  Coolant temperature, 4e+06, Coolant temperature = 48 °C: 0.13 m",
    ]
    plt.close(fig)


def test_crossing_legend_keeps_the_order_of_a_categorical_dimension(agr):
    cases = [dict(power=p, mdot=0.3, wall_law=w) for w in ("regime", "Dittus") for p in (2e6, 4e6, 6e6, 8e6)]
    sweep = Sweep(cases, [make_case_frame(c["power"], c["mdot"]) for c in cases], agr, limits={"CHFR": at_least(1.3)})
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="legend")
    assert legend_texts(ax) == ["regime (4.8e+06 W)", "Dittus (4.8e+06 W)"]
    assert ax.get_legend().get_title().get_text() == "wall_law; Power at CHFR = 1.3"
    plt.close(fig)


def test_crossing_legend_leaves_curves_that_never_cross_unchanged(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(2.5))
    fig, ax = sweep.reduce("CHFR", "min", versus="power").plot(crossing="legend")
    assert legend_texts(ax) == ["0.2", "0.3 (2.667e+06 W)"]
    assert ax.get_legend().get_title().get_text() == "mdot; Power at CHFR = 2.5"
    assert len(drop_lines(ax)) == 1
    plt.close(fig)


def test_crossing_mark_on_a_binding_result(agr):
    def make(power, mdot):
        T = np.array([40.0, 45.0, 50.0, 55.0]) + power / 1e6
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(T, chfr, power)
        rows += [("hot", "CHFR", 0, j, chfr[j] * (1.2 if power < 5e6 else 0.8)) for j in range(4)]
        return frame(rows)

    agr["hot"] = FakeChannel([0.0, 0.1, 0.2, 0.3, 0.4])
    cases = [dict(power=p, mdot=0.3) for p in (2e6, 4e6, 6e6, 8e6)]
    sweep = Sweep(cases, [make(**c) for c in cases], agr, limits={"CHFR": at_least(1.3)})
    r = sweep.reduce("CHFR", "min", versus="power", of=["CC", "hot"], limiting=True)
    xc = 4e6 + 2e6 * 0.2 / 0.7
    fig, ax = r.plot(crossing="mark")
    (drop,) = drop_lines(ax)
    assert drop.get_segments()[0][0][0] == pytest.approx(xc)
    assert foot_texts(ax) == [f"{xc:.4g}"]
    plt.close(fig)


def test_crossing_mark_under_a_shaded_limit_keeps_the_axes(agr, grid_cases):
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    r = sweep.reduce("CHFR", "min", versus="power", where=dict(mdot=0.3))
    fig, ax = r.plot(show_limit="shade")
    shaded = ax.get_ylim()
    plt.close(fig)
    fig, ax = r.plot(show_limit="shade", crossing="mark")
    assert ax.get_ylim() == pytest.approx(shaded)
    (drop,) = drop_lines(ax)
    assert drop.get_segments()[0][0][1] == pytest.approx(shaded[0])
    plt.close(fig)


def test_a_sweep_keeps_the_default_format_of_its_labels(agr, grid_cases):
    labels = Labels({"mdot": ("Mass flow", "kg/s")}, default_format=".2f")
    sweep = Sweep(grid_cases, [make_case_frame(**c) for c in grid_cases], agr, labels=labels)
    r = sweep.profile("T_cool", where=dict(power=4e6, mdot=0.3))
    assert r.subtitle == "Power = 4000000.00 W, Mass flow = 0.30 kg/s, CC"
    fig, ax = r.plot()
    assert ax.get_title() == "Power = 4e+06 W, Mass flow = 0.3 kg/s, CC"
    plt.close(fig)


def test_limit_color_reaches_line_hatch_contour_and_crossing(agr, grid_cases):
    sheet = Style(limit_color="#123456")
    sweep = limited(agr, grid_cases, CHFR=at_least(1.3))
    fig, ax = sweep.reduce("CHFR", "min", versus="power", where={"mdot": 0.3}).plot(show_limit="shade", crossing="mark", style_sheet=sheet)
    assert {to_hex(ln.get_color()) for ln in dashed_lines(ax)} == {"#123456"}
    assert {to_hex(np.atleast_2d(h.get_edgecolor())[0]) for h in hatched(ax)} == {"#123456"}
    (drop,) = drop_lines(ax)
    assert to_hex(drop.get_color()[0]) == to_hex(ax.get_lines()[0].get_color()) != "#123456"
    plt.close(fig)
    fig, ax = sweep.reduce("CHFR", "min", versus=("power", "mdot")).plot(style_sheet=sheet)
    assert to_hex(limit_contour(ax).get_edgecolor()[0]) == "#123456"
    plt.close(fig)


def test_limit_color_reaches_the_field_contour_and_hatch(agr, grid_cases):
    field = limited(agr, grid_cases, T=at_most(60.0)).field("T", of="plate", where={"power": 4e6, "mdot": 0.3})
    fig, ax = field.plot(show_limit="shade", style_sheet=Style(limit_color="#123456"))
    assert to_hex(limit_contour(ax).get_edgecolor()[0]) == "#123456"
    (cells,) = hatched(ax)
    assert to_hex(np.atleast_2d(cells.get_edgecolor())[0]) == "#123456"
    plt.close(fig)


def ratio_sweep(agr, cases, labels=None):
    def make(power, mdot, **_):
        chfr = np.array([3.0, 2.0, 1.5, 1.8]) * mdot / 0.3 * (4e6 / power)
        rows = channel_rows(np.array([40.0, 45.0, 50.0, 55.0]), chfr, power)
        return frame(rows + [("CC", "OSVR", 0, j, 0.9 * chfr[j]) for j in range(4)])

    return Sweep(cases, [make(**c) for c in cases], agr, labels=labels)


def legend_colors(ax):
    leg = ax.get_legend()
    return {t.get_text(): to_hex(h.get_color()) for t, h in zip(leg.get_texts(), leg.legend_handles)}


def test_quantities_take_their_label_colour_then_the_palette_in_registry_order(agr, grid_cases):
    palette = Style().palette
    fixed = ratio_sweep(agr, grid_cases, {"OSVR": ("OSV ratio", ""), "CHFR": Label("CHF ratio", "", color="#123456")})
    fig, ax = fixed.reduce(["CHFR", "OSVR"], "min", versus="power", where={"mdot": 0.3}).plot()
    assert ax.get_legend().get_title().get_text() == "Quantity"
    assert legend_colors(ax) == {"CHF ratio": "#123456", "OSV ratio": palette[0]}
    plt.close(fig)
    ordered = ratio_sweep(agr, grid_cases, {"OSVR": ("OSV ratio", ""), "CHFR": ("CHF ratio", "")})
    fig, ax = ordered.reduce(["CHFR", "OSVR"], "min", versus="power", where={"mdot": 0.3}).plot()
    assert legend_colors(ax) == {"CHF ratio": palette[1], "OSV ratio": palette[0]}
    plt.close(fig)


def test_a_level_keeps_its_ramp_colour_when_a_case_is_missing(holed_sweep):
    fig, ax = holed_sweep.profile("CHFR", of="CC", where={"mdot": 0.2}).plot()
    whole = legend_colors(ax)
    plt.close(fig)
    with pytest.warns(UserWarning, match="skipping unsolved"):
        fig, ax = holed_sweep.profile("CHFR", of="CC", where={"mdot": 0.3}).plot()
    gapped = legend_colors(ax)
    assert list(whole) == ["2e+06", "4e+06", "6e+06", "8e+06"] and list(gapped) == ["2e+06", "4e+06", "6e+06"]
    assert all(gapped[k] == whole[k] for k in gapped)
    assert list(whole.values()) == Style().ramp_colors(4)
    plt.close(fig)


def test_dimensions_fill_colour_then_style_then_marker(agr):
    cases = [{"power": p, "mdot": m, "depth": d} for p in (2e6, 4e6) for m in (0.2, 0.3) for d in (1.0, 2.0, 3.0)]
    r = ratio_sweep(agr, cases).reduce(["CHFR", "OSVR"], "min", versus="power")
    fig, ax = r.plot()
    texts = legend_texts(ax)
    assert [t for t in texts if not t.startswith(" ")] == ["Quantity", "mdot", "depth"]
    assert {ln.get_linestyle() for ln in ax.get_lines()} == {"-", "--"}
    assert {ln.get_marker() for ln in ax.get_lines()} == {"o", "s", "^"}
    assert len({to_hex(ln.get_color()) for ln in ax.get_lines()}) == 2
    plt.close(fig)
    fig, ax = r.plot(color="mdot", style="quantity")
    assert {ln.get_linestyle() for ln in ax.get_lines()} == {"-", "--"}
    assert [t for t in legend_texts(ax) if not t.startswith(" ")] == ["mdot", "Quantity", "depth"]
    plt.close(fig)


def test_encoding_refusals(agr):
    cases = [{"power": p, "mdot": m, "depth": d, "peak": k} for p in (2e6, 4e6) for m in (0.2, 0.3) for d in (1.0, 2.0)
             for k in (1.0, 1.2)]
    r = ratio_sweep(agr, cases).reduce(["CHFR", "OSVR"], "min", versus="power")
    with pytest.raises(ValueError, match="'peak' varies but colour, line style and marker are all taken; put it on a panel grid"):
        r.plot()
    fig, axes = r.plot(row="peak")
    assert axes.shape == (2, 1)
    plt.close(fig)
    one = ratio_sweep(agr, [{"power": p, "mdot": m} for p in (2e6, 4e6) for m in (0.2, 0.3)])
    curve = one.reduce("CHFR", "min", versus="power")
    with pytest.raises(ValueError, match="marker='depth' does not vary in this result; the varying dimensions are mdot"):
        curve.plot(marker="depth")
    with pytest.raises(ValueError, match="'mdot' is assigned to both color and style"):
        curve.plot(color="mdot", style="mdot")
    five = ratio_sweep(agr, [{"power": p, "mdot": m} for p in (1e6, 2e6, 3e6, 4e6, 5e6) for m in (0.2, 0.3)])
    with pytest.raises(ValueError, match="'power' has 5 levels, more than style can show \\(4\\)"):
        five.reduce("CHFR", "min", versus="mdot").plot(style="power")
