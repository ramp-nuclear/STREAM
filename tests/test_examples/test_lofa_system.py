import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stream.examples.lofa import params, system
from stream.examples.lofa.params import Params


def test_build_closes_gravity_and_counts_edges():
    _, K, refs = system.build(Params())
    assert refs["pk"] is None
    assert len(refs["channels"]) == 4
    assert set(refs["pumps"]) == {"A", "B"}
    assert K.g.number_of_edges() == 4 + 1 + 1 + 1 + 1 + 1 + 2 + 1


def test_build_horizontal_tank_has_no_gravity():
    _, _, refs = system.build(Params())
    tank = refs["tank"]
    dp = tank.pressure(T=np.full(tank.n, 40.0), Tw=np.full(tank.n, 40.0), mdot=0.0)
    assert np.allclose(dp, 0.0)


def test_riser_change_keeps_gravity_closed():
    for drop in (5.3, 7.3, 9.3):
        system.build(Params(basement_drop=drop))


def test_unknown_guess_kind_is_a_value_error():
    agr, K, refs = system.build(Params())
    with pytest.raises(ValueError, match="ballpark"):
        system.guess(agr, K, refs, "toolkit")


def test_unknown_wall_heat_transfer_law_is_a_value_error():
    with pytest.raises(ValueError, match="htc"):
        system.make_channel("HotChannel", 0.0022, True, "regime", "x", True)


def _hot_channel_h(htc: str, mdot: float) -> float:
    ch = system.make_channel("HotChannel", 0.0022, True, "regime", htc, True)
    h = ch.h_wall(T_cool=np.full(ch.n, 40.0), T_wall=np.full(ch.n, 60.0), mdot=mdot, pressure=1.8e5)
    return np.asarray(h, dtype=float)


def test_regime_wall_heat_transfer_keeps_natural_convection_at_low_flow():
    regime, dittus = _hot_channel_h("regime", 1e-4), _hot_channel_h("dittus", 1e-4)
    assert np.all(np.isfinite(regime)) and np.all(regime > 0)
    assert np.all(np.isfinite(dittus)) and np.all(dittus > 0)
    assert np.all(regime > dittus)
    for htc in ("regime", "dittus"):
        h = _hot_channel_h(htc, params.PER_PLATE_REFERENCE)
        assert np.all(np.isfinite(h)) and np.all(h > 0)


def test_decay_heat_power_law_values():
    dh = system.decay_heat(10e6)
    assert dh(0.0) == pytest.approx(0.066 * 10e6)
    assert dh(10.0) < dh(0.0)
    assert dh(3600.0) / 10e6 == pytest.approx(0.066 * 3601.0 ** -0.2, rel=1e-9)


def test_draw_labels_every_edge():
    _, K, _ = system.build(Params())
    fig, ax = plt.subplots()
    out = system.draw(K, ax=ax)
    assert out is ax
    labels = {t.get_text() for t in ax.texts}
    assert {"J_pool_top", "J_plenum", "J_split", "J_merge"} <= labels
    assert {"HotChannel", "NormalChannel", "LowChannel", "BypassChannel", "Flapper", "Pool"} <= labels
    plt.close(fig)


@pytest.mark.slow
def test_steady_from_ballpark_passes_hand_checks():
    agr, K, refs = system.build(Params())
    y = system.steady(agr, system.guess(agr, K, refs, "ballpark"))
    chk = system.steady_checks(agr, K, refs, y)
    assert chk["T_out_error_max"] < 0.5
    assert chk["split_error_max"] < 0.15
    assert chk["pool_flow"] * params.NC_SIGNS["pool"] < 0
    assert chk["inlet_matches_pool"]
    assert chk["wall_above_coolant"]
    assert 2e4 < chk["heads"]["A"] < 3e5
    assert chk["heads"]["A"] == pytest.approx(chk["heads"]["B"], rel=1e-6)


@pytest.mark.slow
def test_expert_and_perturbed_guesses_land_on_the_ballpark_state():
    agr, K, refs = system.build(Params())
    ref = system.steady(agr, system.guess(agr, K, refs, "ballpark"))
    for kind in ("expert", "perturbed"):
        y = system.steady(agr, system.guess(agr, K, refs, kind, seed=3))
        assert np.max(np.abs(y - ref)) < 1e-6 * (1 + np.max(np.abs(ref))), kind


@pytest.mark.slow
def test_operating_point_is_steady_and_sums_to_the_core_power():
    p = Params()
    agr, _, refs, y = system.operating_point(p)
    residual = agr.compute(y, 0.0)
    scale = agr.scaled_atol(1.0)
    assert np.max(np.abs(residual) / scale) < 1e-6
    st = agr.save(y)
    pk = refs["pk"]
    assert float(np.asarray(st[pk.name]["power"])) == pytest.approx(p.power, rel=1e-9)
    assert float(np.asarray(st[pk.name]["pk_power"])) == pytest.approx(p.power - system.decay_heat(p.power)(0.0), rel=1e-9)
    core = sum(params.CORE[k].plates * params.power_share(k) * float(np.asarray(st[pk.name]["power"])) for k in params.HEATED)
    assert core == pytest.approx(p.power, rel=1e-9)


@pytest.mark.slow
def test_worths_carry_plate_counts():
    _, _, refs, _ = system.operating_point(Params())
    pk = refs["pk"]
    total_coolant = sum(float(np.sum(w)) for c, w in pk.temp_worth.items() if c in refs["channels"].values())
    total_fuel = sum(float(np.sum(w)) for c, w in pk.temp_worth.items() if c in refs["fuels"].values())
    assert total_coolant == pytest.approx(-1e-4, rel=1e-9)
    assert total_fuel == pytest.approx(-2e-5, rel=1e-9)
    w_avg = float(np.sum(pk.temp_worth[refs["channels"]["normal"]]))
    w_hot = float(np.sum(pk.temp_worth[refs["channels"]["hot"]]))
    assert w_avg / w_hot == pytest.approx(300 * 1.0 / (1 * 1.6), rel=1e-9)


def test_axial_cells_resize_channels_and_plates():
    assert Params().axial_cells == 10
    p = Params(axial_cells=5)
    agr_th, K_th, refs_th = system.build(p)
    assert all(ch.n == 5 for ch in refs_th["channels"].values())
    assert all(tuple(fuel.shape) == (5, params.FUEL_N + 2 * params.CLAD_N) for fuel in refs_th["fuels"].values())
    ref_state = agr_th.save(system.guess(agr_th, K_th, refs_th, "ballpark"))
    _, _, refs = system.build(p, kinetics=True, ref_state=ref_state)
    pk = refs["pk"]
    for k in params.HEATED:
        assert pk.temp_worth[refs["channels"][k]].shape == (5,)
        assert pk.temp_worth[refs["fuels"][k]].shape == (5 * params.FUEL_N,)
    _, _, refs_default = system.build(Params())
    assert all(ch.n == 10 for ch in refs_default["channels"].values())
    assert all(fuel.m == 10 for fuel in refs_default["fuels"].values())
