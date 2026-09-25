"""Physical-models chokepoint behavior: NaN-init instead of uninitialised garbage,
no aliased-strategy mutation, a domain-named shape guard, a translated zero-flow
friction error, a scalar friction return, and warning-free healthy solves.
"""

import warnings
from functools import partial

import numpy as np
import pytest

from stream.errors import StreamError
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water
from stream.units import mm
from stream.utilities import just

pipe = EffectivePipe.rectangular(length=1, edge1=2 * mm, edge2=70 * mm, heated_edge=70 * mm)


# --- NaN-init instead of np.empty garbage -----------------------

def test_regime_dependent_h_spl_nan_re_returns_all_nan():
    """A NaN driving input (mdot) leaves every regime mask False; the result must
    propagate NaN, not return uninitialised memory (finite garbage)."""
    from stream.physical_models.heat_transfer_coefficient.single_phase import (
        regime_dependent_h_spl,
    )

    # Prime the arena with recognisable junk so a bare np.empty would echo it.
    _junk = np.full(2, -12345.678)
    del _junk

    cool = light_water.to_properties(np.array([40.0, 40.0]))
    h = regime_dependent_h_spl(
        coolant=cool, mdot=np.nan, Dh=pipe.hydraulic_diameter, A=pipe.area,
        T_cool=np.array([40.0, 40.0]), T_wall=np.array([50.0, 50.0]),
        re_bounds=(2100, 4000), coolant_funcs=light_water,
        develop_length=np.array([0.3, 0.6]), aspect_ratio=0.03,
    )
    assert np.isnan(h).all()


def test_regime_dependent_q_scb_nan_re_returns_all_nan():
    """The subcooled-boiling sibling has the same np.empty hazard: NaN re must
    yield an all-NaN q, not garbage."""
    from stream.physical_models.heat_transfer_coefficient import regime_dependent_q_scb

    T_wall = np.array([140.0, 145.0])
    p = np.full_like(T_wall, 1e5)
    q = regime_dependent_q_scb(
        T_wall, light_water.to_properties(T_wall, p),
        re=np.full_like(T_wall, np.nan), re_bounds=(2000, 4000),
    )
    assert np.isnan(q).all()


def test_regime_dependent_h_spl_healthy_run_is_finite():
    """The NaN-init must not perturb a healthy run: every finite-re cell is written."""
    from stream.physical_models.heat_transfer_coefficient.single_phase import (
        regime_dependent_h_spl,
    )

    cool = light_water.to_properties(np.array([40.0, 40.0]))
    h = regime_dependent_h_spl(
        coolant=cool, mdot=0.08, Dh=pipe.hydraulic_diameter, A=pipe.area,
        T_cool=np.array([40.0, 40.0]), T_wall=np.array([50.0, 50.0]),
        re_bounds=(2100, 4000), coolant_funcs=light_water,
        develop_length=np.array([0.3, 0.6]), aspect_ratio=0.03,
    )
    assert np.all(np.isfinite(h)) and np.all(h > 0)


# --- copy-on-boiling (no aliased-strategy mutation) --------------

_kw = dict(mdot=0.08, pressure=np.array([2e5, 2e5]), coolant_funcs=light_water,
           Dh=pipe.hydraulic_diameter, A=pipe.area)


def test_just_strategy_array_not_mutated_by_boiling_eval():
    """wall_heat_transfer_coeff with a just(arr) strategy must not corrupt the
    user's array in place, and a repeat non-boiling eval must be history-independent."""
    from stream.physical_models.heat_transfer_coefficient import wall_heat_transfer_coeff

    arr = np.array([2.0e4, 2.0e4])
    strat = just(arr)

    h_before = wall_heat_transfer_coeff(
        T_wall=np.array([110.0, 110.0]), T_cool=np.array([100.0, 100.0]), h_spl=strat, **_kw
    ).copy()
    # a boiling evaluation with the same aliased strategy
    wall_heat_transfer_coeff(
        T_wall=np.array([180.0, 180.0]), T_cool=np.array([110.0, 110.0]), h_spl=strat, **_kw
    )
    assert np.array_equal(arr, [2.0e4, 2.0e4])  # user's array untouched

    h_after = wall_heat_transfer_coeff(
        T_wall=np.array([110.0, 110.0]), T_cool=np.array([100.0, 100.0]), h_spl=strat, **_kw
    ).copy()
    assert np.allclose(h_before, h_after)  # F is not history-dependent


def test_copy_preserves_boiling_enhancement_for_nonaliased_strategy():
    """The copy must not sever the in-place *= : a non-aliased strategy's boiling
    result still carries the partial-SCB enhancement over the bare SPL value."""
    from stream.physical_models.heat_transfer_coefficient import (
        Dittus_Boelter_h_spl,
        wall_heat_transfer_coeff,
    )

    T_wall = np.array([180.0, 180.0])
    T_cool = np.array([110.0, 110.0])
    cool = light_water.to_properties(T_cool, _kw["pressure"])
    h_spl_bare = Dittus_Boelter_h_spl(
        coolant=cool, mdot=_kw["mdot"], Dh=pipe.hydraulic_diameter, A=pipe.area
    )
    h = wall_heat_transfer_coeff(T_wall=T_wall, T_cool=T_cool, h_spl=Dittus_Boelter_h_spl, **_kw)
    # boiling was detected and the SCB factor (>1) was applied
    assert np.all(h > h_spl_bare)


# --- shape guard ------------------------------------------------

def test_wall_htc_shape_mismatch_raises_named_error():
    """A sized T_wall whose length differs from T_cool must raise a StreamError-
    family error naming T_wall and both lengths, not a bare IndexError."""
    from stream.physical_models.heat_transfer_coefficient import (
        Dittus_Boelter_h_spl,
        wall_heat_transfer_coeff,
    )

    with pytest.raises(StreamError) as ei:
        wall_heat_transfer_coeff(
            T_wall=np.array([200.0]),  # size 1
            T_cool=np.array([40.0, 41.0]),  # 2 cells
            h_spl=Dittus_Boelter_h_spl, mdot=0.08,
            pressure=np.array([2e5, 2e5]), coolant_funcs=light_water,
            Dh=pipe.hydraulic_diameter, A=pipe.area,
        )
    msg = str(ei.value)
    assert "T_wall" in msg and "1" in msg and "2" in msg
    assert not isinstance(ei.value, IndexError)


def test_wall_htc_matched_and_scalar_wall_do_not_raise():
    """Matched-length arrays and a scalar T_wall broadcast against array T_cool
    must both stay allowed (the previously-working callers)."""
    from stream.physical_models.heat_transfer_coefficient import (
        Dittus_Boelter_h_spl,
        wall_heat_transfer_coeff,
    )

    common = dict(h_spl=Dittus_Boelter_h_spl, mdot=0.08,
                  pressure=np.array([2e5, 2e5]), coolant_funcs=light_water,
                  Dh=pipe.hydraulic_diameter, A=pipe.area)
    # matched
    wall_heat_transfer_coeff(T_wall=np.array([110.0, 110.0]), T_cool=np.array([100.0, 100.0]), **common)
    # scalar wall, array cool
    wall_heat_transfer_coeff(T_wall=110.0, T_cool=np.array([100.0, 100.0]), **common)


# --- friction zero-flow translation -----------------------------

def test_turbulent_friction_factor_zero_flow_translates():
    """friction_factor('turbulent') at mdot=0 must raise the translated error,
    catchable as StreamError AND ZeroDivisionError, naming the correlation + remedy."""
    from stream.physical_models.pressure_drop import friction_factor

    f = friction_factor("turbulent")
    with pytest.raises(ZeroDivisionError) as ei:  # ZeroDivisionError lineage preserved
        f(T_cool=40.0, T_wall=50.0, mdot=0.0, fluid=light_water, pipe=pipe)
    assert isinstance(ei.value, StreamError)  # also a StreamError
    msg = str(ei.value)
    assert "turbulent_friction" in msg
    assert "regime_dependent" in msg


def test_turbulent_friction_direct_kernel_stays_bare():
    """The @njit kernel stays bare by design: a direct call raises a plain
    ZeroDivisionError with no StreamError lineage."""
    from stream.physical_models.pressure_drop.friction import turbulent_friction

    with pytest.raises(ZeroDivisionError) as ei:
        turbulent_friction(0.0)
    assert not isinstance(ei.value, StreamError)


def test_turbulent_friction_array_path_unchanged():
    """The array path is asymmetric by design: it returns inf, no error."""
    from stream.physical_models.pressure_drop.friction import turbulent_friction

    out = turbulent_friction(np.array([0.0, 5.0, 20.0]))
    assert np.isinf(out[0]) and np.all(np.isfinite(out[1:]))


# --- scalar friction return -------------------------------------

def test_regime_dependent_friction_scalar_returns_float():
    """Scalar input returns a python float equal to the size-1 array result, and
    assigning it into a scalar slot emits no DeprecationWarning."""
    from stream.physical_models.pressure_drop.friction import regime_dependent_friction

    common = dict(T_cool=40.0, T_wall=50.0, fluid=light_water, pipe=pipe,
                  re_bounds=(2100, 4000), k_R=1.0)
    scalar = regime_dependent_friction(mdot=0.05, **common)
    arr = regime_dependent_friction(mdot=np.array([0.05]), **common)  # old shape-(1,)
    assert isinstance(scalar, float)
    assert scalar == float(arr[0])

    out = np.zeros(2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out[1] = scalar
    assert not any(issubclass(wi.category, DeprecationWarning) for wi in w)


def test_regime_dependent_friction_array_input_keeps_shape():
    """Array input is unchanged: a size-1 array mdot still returns a shape-(1,) array."""
    from stream.physical_models.pressure_drop.friction import regime_dependent_friction

    common = dict(T_cool=40.0, T_wall=50.0, fluid=light_water, pipe=pipe,
                  re_bounds=(2100, 4000), k_R=1.0)
    out = regime_dependent_friction(mdot=np.array([0.05]), **common)
    assert isinstance(out, np.ndarray) and out.shape == (1,)


# --- warning hygiene -----------------------------------

def _build_channel_fuel():
    from stream.calculations import ChannelAndContacts, Fuel
    from stream.composition import x_boundaries
    from stream.composition.mtr_geometry import symmetric_plate
    from stream.physical_models.heat_transfer_coefficient import wall_heat_transfer_coeff
    from stream.physical_models.heat_transfer_coefficient.single_phase import spl_htc
    from stream import Solid
    from stream.utilities import normalize

    z_N, clad_N, fuel_N = 6, 1, 2
    meat_depth, clad_depth, meat_width = 0.5 * mm, 0.4 * mm, 70 * mm
    shape = (z_N, fuel_N + 2 * clad_N)
    meat = np.zeros(shape, bool)
    meat[:, clad_N:-clad_N] = True
    materials = np.empty(shape, object)
    materials[meat] = Solid(density=3000, specific_heat=800, conductivity=100)
    materials[~meat] = Solid(density=2700, specific_heat=900, conductivity=250)
    zb = np.linspace(0, 1, z_N + 1)
    F = Fuel(z_boundaries=zb, x_boundaries=x_boundaries(clad_N, fuel_N, clad_depth, meat_depth),
             material=Solid.from_array(materials), meat_indices=meat,
             power_shape=normalize(np.ones(z_N * fuel_N)), y_length=meat_width)
    cd, cw = 2 * mm, 70 * mm
    chan_pipe = EffectivePipe.rectangular(length=1, edge1=cd, edge2=cw, heated_edge=meat_width)
    hwf = partial(wall_heat_transfer_coeff,
                  h_spl=spl_htc("regime_dependent", re_bounds=(2100, 4000), aspect_ratio=cd / cw, Lh=1.0))
    C = ChannelAndContacts(z_boundaries=zb, fluid=light_water, pipe=chan_pipe, h_wall_func=hwf)
    return symmetric_plate(C, F, funcs={C: dict(mdot=0.08, Tin=40.0, p_abs=2e5),
                                        F: dict(power=1000.0)}).to_aggregator()


def test_healthy_regime_dependent_solve_emits_zero_warnings():
    """A healthy converged steady solve with the regime_dependent HTC must emit NO
    warnings — warnings in successful runs are defects by convention."""
    agr = _build_channel_fuel()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        sol = agr.solve_steady(np.ones(len(agr)), globalize=True)
    resid = float(np.linalg.norm(agr.compute(sol, 0)))
    assert resid < 1e-6, f"solve did not converge: ||F|| = {resid:.2e}"
    assert list(wlist) == [], f"healthy solve emitted {len(wlist)} warning(s): " + "; ".join(
        f"{w.category.__name__} @ {w.filename.split('/')[-1]}:{w.lineno}" for w in wlist
    )


def test_htc_errstate_wrapping_is_scoped_not_global():
    """The surgical errstate wraps must be context-scoped, not process-global: a
    call that passes through the wrapped phi expression must not disable numpy's
    error handling for a genuine invalid divide right afterwards."""
    from stream.physical_models.heat_transfer_coefficient.single_phase import (
        regime_dependent_h_spl,
    )

    cool = light_water.to_properties(np.array([40.0, 40.0]))
    # Healthy call: passes unconditionally through the errstate-wrapped phi block.
    regime_dependent_h_spl(
        coolant=cool, mdot=0.08, Dh=pipe.hydraulic_diameter, A=pipe.area,
        T_cool=np.array([40.0, 40.0]), T_wall=np.array([50.0, 50.0]),
        re_bounds=(2100, 4000), coolant_funcs=light_water,
        develop_length=np.array([0.3, 0.6]), aspect_ratio=0.03,
    )
    with warnings.catch_warnings(record=True) as w_outside:
        warnings.simplefilter("always")
        np.array([0.0]) / np.array([0.0])  # genuine invalid divide, must still warn
    assert any(issubclass(wi.category, RuntimeWarning) for wi in w_outside)
