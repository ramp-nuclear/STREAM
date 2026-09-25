from copy import deepcopy

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from stream.calculations import (
    Bend,
    Gravity,
    HeatExchanger,
    LevelHead,
    LocalPressureDrop,
    Pump,
    Resistor,
    ResistorSum,
)
from stream.calculations.ideal.resistors import ResistorMul, Screen
from stream.substances import light_water
from stream.units import g
from stream.utilities import just, summed

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


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, medium_floats)
def test_resistor_mul_factor_enters_residual(r, factor, mdot):
    """The multiplication factor must scale the residual the solver actually sees,
    not just the standalone dp_out."""
    base = Resistor(r)
    scaled = factor * base
    T = 25.0
    base_res = np.array(base.calculate([T, 0.0], mdot=mdot, Tin=T))
    scaled_res = np.array(scaled.calculate([T, 0.0], mdot=mdot, Tin=T))
    # out[1] = variables[1] - dp_out, so the factored residual is factor * base's.
    assert np.allclose(scaled_res, factor * base_res)


@given(st.integers(min_value=-1000, max_value=1000).filter(bool), pos_medium_floats)
def test_resistor_mul_accepts_int_factor(n, r):
    """The docstring advertises `2 * resistor`; an int factor must be accepted and
    stored as a float."""
    scaled = n * Resistor(r)
    assert isinstance(scaled.factor, float)
    assert scaled.factor == float(n)


def test_resistor_mul_rejects_non_numeric_factor():
    with pytest.raises(TypeError):
        ResistorMul("x", Resistor(10.0))


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


def test_resistor_mul_deepcopy_is_a_distinct_graph_node():
    """Deep-copying a subsystem containing a scaled resistor must produce a distinct
    graph node rather than silently collapsing onto the original."""
    import networkx as nx

    rm = 2.0 * Resistor(100)
    rm_copy = deepcopy(rm)
    assert rm_copy.resistor is not rm.resistor
    g = nx.DiGraph()
    g.add_node(rm)
    g.add_node(rm_copy)
    assert g.number_of_nodes() == 2


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


def test_screen_is_finite_at_zero_flow():
    """A Screen must not raise ZeroDivisionError at mdot=0 (the zero-flow steady
    guess, and the reversal crossing) — its dp is linear and vanishes there."""
    screen = Screen(clear_area=0.5, total_area=1.0, wire_diameter=0.001, fluid=light_water)
    assert screen.dp_out(mdot=0.0, Tin=50.0) == 0.0
    # The residual path the solver evaluates must not raise either.
    assert np.allclose(screen.calculate([50.0, 0.0], mdot=0.0, Tin=50.0), [0.0, 0.0])
    # Continuity of the physical limit: dp -> 0 as mdot -> 0.
    assert np.isclose(screen.dp_out(mdot=1e-6, Tin=50.0), 0.0, atol=1e-6)


def test_bilinear_inertia_stays_positive_for_reversed_flow():
    """Inertance is a positive geometric quantity for either flow direction. Reversed
    flow must not make L negative (anti-dissipative) or unbounded, and L must stay
    strictly positive through mdot = 0."""
    from stream.calculations.ideal.inertia import Inertia, bilinear

    L0, mdot0 = 100.0, 1.0
    L = bilinear(L0, mdot0)
    assert L(mdot=-1.0) == pytest.approx(L0)  # was -100 (negative inertance)
    assert L(mdot=-2.0) == pytest.approx(L0)  # was -200 (grows unbounded in reverse)
    assert 0.0 < L(mdot=0.0) <= L0  # was exactly 0 (singular mdot2 = dp/L)
    assert L(mdot=0.5) == pytest.approx(L0 * 0.5)  # forward taper preserved
    # dp = -L*mdot2 must oppose the acceleration, not assist it.
    assert Inertia(L).dp_out(mdot=-1.0, mdot2=-0.1) == pytest.approx(L0 * 0.1)  # was -10


def test_local_pressure_drop_reynolds_uses_hydraulic_diameter():
    """Re must be built from the equivalent-circle diameter D = 2*sqrt(A/pi), not the
    radius sqrt(A/pi); at low flow the wrong Dh mis-reads the Idelchik table."""
    from stream.physical_models.dimensionless import Re_mdot
    from stream.physical_models.pressure_drop import local_pressure_by_mdot, local_pressure_factor
    from stream.physical_models.pressure_drop.local import sudden_contraction_factor, sudden_expansion_factor

    A1, A2 = 0.02, 0.008  # A2 < A1 -> forward flow is a contraction
    Tin, mdot = 25.0, 0.05  # low flow keeps Re in the Re-sensitive table region
    calc = LocalPressureDrop(light_water, A1, A2)

    A = min(A1, A2)
    aratio = min(A1 / A2, A2 / A1)
    Dh = 2 * np.sqrt(A / np.pi)
    re = Re_mdot(mdot, A, Dh, light_water.viscosity(Tin))
    assert 50 < re < 10000  # guard: the fix only matters where the table depends on Re
    f = local_pressure_factor(
        mdot=mdot,
        aratio=aratio,
        re=re,
        positive_flow=sudden_contraction_factor,
        negative_flow=sudden_expansion_factor,
    )
    expected = -local_pressure_by_mdot(mdot, light_water.density(Tin), f, A)
    assert np.isclose(calc.dp_out(Tin=Tin, mdot=mdot), expected)


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


def test_levelhead_defaults_to_initial_level_when_unrouted():
    lh = LevelHead(light_water, 0.5, 4.0)
    rho = float(light_water.density(np.array([30.0])))
    assert lh.dp_out(Tin=np.array([30.0])) == pytest.approx(rho * g * 3.5)


def test_levelhead_tracks_routed_level_and_sign():
    lh = LevelHead(light_water, 0.0, 4.0, sign=-1.0)
    rho = float(light_water.density(np.array([30.0])))
    assert lh.dp_out(Tin=np.array([30.0]), level=2.0) == pytest.approx(-rho * g * 2.0)
