from copy import deepcopy

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from stream.calculations import (
    Gravity,
    HeatExchanger,
    LocalPressureDrop,
    Pump,
    Resistor,
    ResistorSum,
    Bend,
)
from stream.substances import light_water
from stream.utilities import summed, just

from .conftest import medium_floats, normal_floats, pos_medium_floats


@settings(deadline=None)
@given(medium_floats, medium_floats, medium_floats)
def test_pump_as_ideal_dp_source(p, T, mdot):
    P = Pump(pressure=p)
    result = P.calculate([T, p], Tin=T, mdot=mdot)
    assert np.allclose(result, [0, 0])
    result = P.calculate([T, 1.0], Tin=T, mdot=mdot)
    assert np.allclose(result, [0, 1.0 - p])


@given(medium_floats, medium_floats, medium_floats)
def test_pump_as_ideal_current_source(p, T, mdot):
    P = Pump(mdot0=mdot)
    result = P.calculate([T, p], Tin=T, mdot=mdot)
    assert np.allclose(result, [0, 0])
    result = P.calculate([T, p], Tin=T, mdot=1.0)
    assert np.allclose(result, [0, 1.0 - mdot])


def test_pump_errors_on_impossibly_imposed_dp_and_mdot():
    """One cannot impose both dp and mdot as ideal sources"""
    with pytest.raises(ValueError):
        Pump(pressure=1, mdot0=2)


def test_pump_errors_on_when_no_source_type_was_imposed():
    """The pump source type (and value) must be set"""
    p = Pump()
    with pytest.raises(ValueError):
        p.calculate([0, 0], Tin=3, mdot=5)


@given(medium_floats, normal_floats, medium_floats)
def test_resistor(r, T, mdot):
    R = Resistor(resistance=r)
    result = R.calculate([T, -mdot * r], Tin=T, mdot=mdot)
    assert np.allclose(result, [0, 0])


temps = st.floats(min_value=25, max_value=50)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_resistor_factor_just_multiplies(r, factor, mdot):
    fric = Resistor(r)
    fricfac = factor * fric
    p0 = factor * fric.dp_out(mdot=mdot, Tin=25.0)
    p1 = fricfac.dp_out(mdot=mdot, Tin=25.0)
    assert np.allclose(p0, p1)


@given(pos_medium_floats, pos_medium_floats)
def test_resistor_multiplication_is_symmetric(f, r):
    res = Resistor(r)
    assert res * f == f * res


@given(
    st.sampled_from(
        [
            Resistor(resistance=100),
            Gravity(light_water, disposition=10.0),
            LocalPressureDrop(fluid=light_water, A1=1.0, A2=2.0),
        ]
    ),
    pos_medium_floats,
)
def test_resistor_mul_can_be_deepcopied(r, f):
    assert deepcopy(f * r)


@given(*(5 * [normal_floats]))
def test_hx(outlet, pressure, T, mdot, Tin):
    HX = HeatExchanger(outlet=outlet)
    result = HX.calculate([T, pressure], mdot=mdot, Tin=Tin)
    assert np.allclose(result, [T - outlet, pressure])


@settings(deadline=None)
@given(
    st.lists(
        st.sampled_from(
            [
                Resistor(resistance=100),
                Gravity(light_water, disposition=10.0),
                LocalPressureDrop(fluid=light_water, A1=1.0, A2=2.0),
            ]
        ),
        max_size=30,
    )
)
def test_resistor_sum_calculates_additions_of_different_resistors(rs):
    RS = ResistorSum(*rs)
    result = RS.calculate([0, 0], mdot=5, Tin=0)
    additions = sum([np.array(c.calculate([0, 0], mdot=5, Tin=0)) for c in rs])
    assert np.allclose(result, additions)
    assert RS.should_continue([0, 0], mdot=5, Tin=0)


@settings(deadline=None)
@given(
    st.lists(
        st.sampled_from(
            [
                Resistor(resistance=100),
                Gravity(light_water, disposition=10.0),
                LocalPressureDrop(fluid=light_water, A1=1.0, A2=2.0),
            ]
        ),
        min_size=2,
        max_size=30,
    )
)
def test_resistor_sum_from_a_sum_of_resistor_sums(rs):
    added_RS = ResistorSum(rs[0], name="AddedRS") + summed(tuple(map(ResistorSum, rs[1:])))
    RS = ResistorSum(*rs)
    result = RS.calculate([0, 0], mdot=5, Tin=0)

    assert added_RS.name == "AddedRS"
    assert np.allclose(result, added_RS.calculate([0, 0], mdot=5, Tin=0))


@given(st.lists(medium_floats, min_size=1), medium_floats)
def test_arbitrary_resistors_in_resistor_sum(resistances, mdot):
    Rs = [Resistor(r) for r in resistances]
    Rs_dp = sum(r.dp_out(Tin=0, mdot=mdot) for r in Rs)

    R_sum = ResistorSum(*Rs)
    assert Rs_dp == R_sum.dp_out(Tin=1, mdot=mdot)

    R_summed = summed(ResistorSum(r) for r in Rs)
    assert Rs_dp == R_summed.dp_out(Tin=2, mdot=mdot)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_local_pressure_drop_is_always_non_positive(A1, A2, mdot):
    calc = LocalPressureDrop(light_water, A1, A2)
    dp = calc.dp_out(Tin=25.0, mdot=mdot)
    assert dp <= 0.0


@settings(deadline=None)
@given(pos_medium_floats)
def test_local_pressure_drop_for_expansion_to_infinity(mdot):
    expansion = LocalPressureDrop(light_water, 1.0, np.inf)
    t = 25.0
    rho = light_water.density(t)
    v = mdot / rho / 1.0
    precalc = 0.5 * rho * v**2
    assert np.isclose(expansion.dp_out(Tin=t, mdot=mdot), -precalc)


@settings(deadline=None)
@given(pos_medium_floats)
def test_zero_bend_angle_returns_zero_pressure_drop(mdot):
    bend = Bend(
        light_water,
        hydraulic_diameter=1.0,
        area=np.pi**2 / 4,
        bend_radius=1.0,
        bend_angle=0.0,
        friction_func=just(1.0),
    )
    assert np.equal(bend.dp_out(mdot=mdot, Tin=25.0), 0.0)
