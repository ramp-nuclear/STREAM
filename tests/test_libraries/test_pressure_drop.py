"""
Regression / comparison testing the pressure drop functions and correlations
"""
from functools import partial

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings, assume
from scipy.constants import g

from stream import EffectivePipe
from stream.calculations import LocalPressureDrop
from stream.physical_models.pressure_drop import (
    pressure_diff, Darcy_Weisbach_pressure_by_mdot, friction_factor,
    )
from stream.physical_models.pressure_drop.friction import (
    turbulent_friction, viscosity_correction, rectangular_laminar_correction,
    laminar_friction, regime_dependent_friction, _DARCY_NAMES,
    )
from stream.physical_models.pressure_drop.local import (
    mdot_by_local_pressure, expansion_factor, sudden_contraction_factor, sudden_expansion_factor,
    local_pressure_by_mdot, contraction_factor, bend_factor,
    )
from stream.substances import light_water
from stream.substances.mocks import mock_liquid_funcs
from stream.utilities import lin_interp
from .conftest import pos_medium_floats, mock_pipe, normal_floats


@pytest.mark.parametrize(("re", "epsilon", "turbulent_pd"),
                         [(4.5e3, 0, 0.03847),
                          (1e4, 0, 0.03087),
                          (2e5, 0, 0.01565),
                          (1e6, 0, 0.01165)])
def test_turbulent_friction_for_precalculated_case(re, epsilon, turbulent_pd):
    assert np.isclose(turbulent_friction(re, epsilon), turbulent_pd, rtol=1e-3)


@pytest.mark.parametrize(("heat_wet_ratio", "mu_ratio", "K_H"),
                         [(0.90065264684554, 0.212973, 0.46662),
                          (0.90065264684554, 0.24774516, 0.50029),
                          (0.90065264684554, 0.28151305, 0.53113),
                          (0.90065264684554, 0.3139354, 0.55931)])
def test_viscosity_correction_for_precalculated_case(heat_wet_ratio, mu_ratio, K_H):
    assert np.isclose(viscosity_correction(heat_wet_ratio, mu_ratio), K_H)


@pytest.mark.parametrize(("a_dp", "K_R"),
                         [(0.05, 0.71742),
                          (0.1, 0.76565),
                          (X.XX, 0.81110),
                          (0.2, 0.85346)])
def test_rectangular_laminar_correction_for_precalculated_case(a_dp, K_R):
    assert np.isclose(rectangular_laminar_correction(a_dp), K_R)


@pytest.mark.parametrize(
    ("mdot", "rho", "f", "L", "Dh", "A", "pdd"),
    [
        (0.3, 970, 0.01, 0.0256, 0.004539811457578, 0.00015651, 106.79692),
        (0.4, 980, 0.03, 0.0256, 0.004539811457578, 0.00015651, 563.77152),
        (0.5, 990, 0.04, 0.0256, 0.004539811457578, 0.00015651, 1162.66012),
        (0.6, 1000, 0.05, 0.0256, 0.004539811457578, 0.00015651, 2071.86033),
        ],
    )
def test_Darcy_Weisbach_pressure_by_mdot_for_precalculated_case(mdot, rho, f, L, Dh, A, pdd):
    assert np.isclose(Darcy_Weisbach_pressure_by_mdot(mdot, rho, f, L, Dh, A), pdd)


@settings(deadline=None)
@given(pmf := pos_medium_floats, pmf, pmf, pmf)
def test_local_pressure_inverse_methods_are_inverse(mdot, rho, f, A):
    func = partial(local_pressure_by_mdot, rho=rho, f=f, A=A)
    func_inv = partial(mdot_by_local_pressure, rho=rho, f=f, A=A)

    assert np.isclose(mdot, func_inv(dp=func(mdot=mdot)))


pos_areas = st.floats(1e-5, 10, allow_nan=False)
large_re = st.floats(5e3, 1e6, allow_nan=False)


@settings(deadline=None)
@given(pos_areas, pos_areas, large_re)
def test_idelchik_expansion_is_analytic_for_large_re(a1, a2, re):
    a1, a2 = sorted((a1, a2))
    f1 = sudden_expansion_factor(a1 / a2, re)
    f2 = expansion_factor(a1 / a2)
    assert np.isclose(f1, f2), (f1, f2)


@settings(deadline=None)
@given(normal_floats)
def test_expansion_when_ratio_is_1_asserts_no_pressure(re):
    assert sudden_expansion_factor(1, re) == 0.


@settings(deadline=None)
@given(normal_floats)
def test_contraction_when_ratio_is_1_asserts_no_pressure(re):
    assert sudden_contraction_factor(1, re) == 0.


@settings(deadline=None)
@given(pos_medium_floats)
def test_contraction_when_ratio_is_0_has_factor_half(re):
    assert sudden_contraction_factor(0, re) == 0.5


@settings(deadline=None)
@given(pos_medium_floats)
def test_expansion_when_ratio_is_0_has_factor_one(re):
    assert sudden_expansion_factor(0, re) == 1.0


mdots = (st.floats(-10, -1e-8, allow_nan=False, allow_subnormal=False)
         | st.floats(1e-8, 10, allow_nan=False, allow_subnormal=False))


@settings(deadline=None)
@given(pos_areas, pos_areas, mdots)
def test_idelchik_local_pd_computes_nonpositive_dp(a1, a2, mdot):
    assume(a1 != a2)
    calc = LocalPressureDrop(light_water, a1, a2)
    dp = calc.dp_out(mdot=mdot, Tin=40.)
    assert np.sign(dp) == -np.sign(mdot), (dp, mdot)


very_high_re = st.floats(1e4, 1e6, allow_nan=False)


@settings(deadline=None)
@given(pos_areas, pos_areas, very_high_re)
def test_idelchik_contraction_at_high_reynolds_follows_known_form(a1, a2, re):
    a1, a2 = sorted((a1, a2))
    f1 = sudden_contraction_factor(a1 / a2, re)
    f2 = contraction_factor(a1 / a2)
    assert np.isclose(f1, f2), (f1, f2)


def test_pressure_difference_for_a_few_simple_cases():
    """Pressure difference (by gravity and friction) becomes greater as
    current decreases, since gravity offsets friction"""
    pipe = EffectivePipe.circular(1, 0.5)
    pressure = partial(pressure_diff, fluid=light_water,
                       pipe=pipe, dz=pipe.length, Tw=10000)

    assert np.isclose(pressure(T=20, mdot=0),
                      light_water.density(20.) * g * pipe.length)
    assert pressure(T=20, mdot=1e4) < 0
    assert pressure(T=20, mdot=0.2) > 0


def test_friction_factor_identifies_the_required_names():
    for name in _DARCY_NAMES:
        assert friction_factor(name)


def test_friction_factor_raises_value_error_for_unknown_names():
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        friction_factor("Hello")


def test_friction_factor_keyword_passing():
    args = T, Tw, mdot, fluid, pipe = 60., 80., 0.5, light_water, mock_pipe
    assert (friction_factor('turbulent', epsilon=1.)(*args)
            != friction_factor('turbulent', epsilon=0.)(*args))
    with pytest.raises(TypeError):
        friction_factor("laminar", epsilon=1.0)(*args)


@settings(deadline=None)
@given(st.tuples(st.floats(min_value=1e2, allow_nan=False, allow_infinity=False),
                 st.floats(min_value=1e2, allow_nan=False, allow_infinity=False))
       .filter(lambda x: x[0] != x[1]),
       pos_medium_floats)
def test_regime_dependent_friction_assigns_regimes_correctly(re, md):
    re_bounds = (min(re), max(re))
    f = regime_dependent_friction(
        T_cool=np.array([20.0]),
        T_wall=np.array([20.0]),
        mdot=md,
        fluid=mock_liquid_funcs,
        pipe=EffectivePipe(1, 4, 4, 1),
        re_bounds=re_bounds,
        k_R=1.0
        )
    if md < re_bounds[0]:
        assert np.allclose(f, laminar_friction(md))
    elif md > re_bounds[1]:
        assert np.allclose(f, turbulent_friction(md))
    else:
        assert np.allclose(f, lin_interp(
            *re_bounds, laminar_friction(md), turbulent_friction(md), md))


@pytest.mark.parametrize(
    ("angle", "relative_curvature", "re", "k_bend"),
    [
        (np.deg2rad(30), 1.0, 1.0e5, 1.3 * 0.45 * 0.21),
        (np.deg2rad(90), 1.0, 1.0e5, 1.3 * 1.00 * 0.21),
        (np.deg2rad(90), 0.6, 1.0e5, 1.19 * 1.00 * 0.753080095095887),
        (np.deg2rad(90), 0.6, 0.2e5, 1.49 * 1.00 * 0.753080095095887),
        (np.deg2rad(130), 1.0, 1.0e5, 1.3 * 1.205555555555556 * 0.21),
        ],
    )
def test_bend_factor_for_precalculated_cases(angle, relative_curvature, re, k_bend):
    assert np.isclose(bend_factor(angle, relative_curvature, re), k_bend)


angles_in_range = st.floats(min_value=0, max_value=np.pi, allow_nan=False, allow_subnormal=False)
angles_out_of_range = (st.floats(min_value=np.pi, max_value=1e3, allow_nan=False,
                                 allow_subnormal=False, exclude_min=True)
                       | st.floats(max_value=0, min_value=-1e3, allow_nan=False,
                                   allow_subnormal=False, exclude_max=True))
relative_curvature_in_range = st.floats(min_value=0.5, allow_nan=False, allow_subnormal=False)
relative_curvature_out_of_range = st.floats(max_value=0.5, allow_nan=False, allow_subnormal=False,
                                            exclude_max=True)
re_in_range = st.floats(min_value=1e4, allow_nan=False, allow_subnormal=False)


@settings(deadline=None)
@given(relative_curvature_in_range, re_in_range)
def test_bend_factor_for_zero_bend_angle_returns_zero(relative_curvature, re):
    assert np.equal(bend_factor(angle=0.0, relative_curvature=relative_curvature, re=re), 0.0)


@settings(deadline=None)
@given(angles_out_of_range, relative_curvature_in_range, re_in_range)
def test_bend_factor_raises_error_when_angle_out_of_range(angle, relative_curvature, re):
    with pytest.raises(ValueError):
        bend_factor(angle=angle, relative_curvature=relative_curvature, re=re)


@settings(deadline=None)
@given(angles_in_range, relative_curvature_out_of_range, re_in_range)
def test_bend_factor_raises_error_when_relative_curvature_out_of_range(angle, relative_curvature, re):
    with pytest.raises(ValueError):
        bend_factor(angle=angle, relative_curvature=relative_curvature, re=re)
