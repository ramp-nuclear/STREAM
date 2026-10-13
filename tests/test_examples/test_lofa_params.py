import pytest

from stream.examples.lofa import params


def test_core_table_shares_sum_to_one_over_the_core():
    total = sum(params.CORE[k].plates * params.power_share(k) for k in params.HEATED)
    assert total == pytest.approx(1.0)
    assert params.power_share("hot") > params.power_share("normal") > params.power_share("low")
    assert all(params.power_share(k) == 0.0 for k in params.UNHEATED)


def test_plate_geometry_is_one_plate():
    pipe = params.plate_pipe(0.0022)
    assert pipe.area == pytest.approx(0.07 * 0.0022)
    assert pipe.hydraulic_diameter == pytest.approx(4 * pipe.area / (2 * (0.07 + 0.0022)))


def test_nominal_flows_scale_with_plate_counts():
    p = params.Params()
    m = params.nominal_flows(p)
    total = sum(params.CORE[k].plates * m[k] for k in params.CORE)
    assert m["total"] == pytest.approx(total)
    assert m["bypass"] > m["normal"] and m["hot"] == m["normal"] == m["low"]


def test_params_default_to_negative_temperature_coefficients():
    p = params.Params()
    assert p.coolant_worth < 0 and p.fuel_worth < 0
    assert p.scenario == "both" and p.flapper_stuck is False


def test_params_default_to_the_regime_dependent_wall_heat_transfer():
    assert params.Params().htc == "regime"
