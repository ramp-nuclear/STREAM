import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stream.examples.lofa import params, system
from stream.examples.lofa.params import Params


def test_build_closes_gravity_and_counts_edges():
    agr, K, refs = system.build(Params())
    assert refs["pk"] is None
    assert len(refs["channels"]) == 4
    assert set(refs["pumps"]) == {"A", "B"}
    assert K.g.number_of_edges() == 4 + 1 + 1 + 1 + 1 + 1 + 2 + 1


def test_build_horizontal_tank_has_no_gravity():
    agr, K, refs = system.build(Params())
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


def test_decay_heat_power_law_values():
    dh = system.decay_heat(10e6)
    assert dh(0.0) == pytest.approx(0.066 * 10e6)
    assert dh(10.0) < dh(0.0)
    assert dh(3600.0) / 10e6 == pytest.approx(0.066 * 3601.0 ** -0.2, rel=1e-9)


def test_draw_labels_every_edge():
    agr, K, refs = system.build(Params())
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
    agr, K, refs, y = system.operating_point(p)
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
    agr, K, refs, y = system.operating_point(Params())
    pk = refs["pk"]
    total_coolant = sum(float(np.sum(w)) for c, w in pk.temp_worth.items() if c in refs["channels"].values())
    total_fuel = sum(float(np.sum(w)) for c, w in pk.temp_worth.items() if c in refs["fuels"].values())
    assert total_coolant == pytest.approx(-1e-4, rel=1e-9)
    assert total_fuel == pytest.approx(-2e-5, rel=1e-9)
    w_avg = float(np.sum(pk.temp_worth[refs["channels"]["normal"]]))
    w_hot = float(np.sum(pk.temp_worth[refs["channels"]["hot"]]))
    assert w_avg / w_hot == pytest.approx(300 * 1.0 / (1 * 1.6), rel=1e-9)
