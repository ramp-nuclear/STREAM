"""Discontinuity scanner: ``scan_1d`` asserts that a 1-D residual map is continuous
with non-exploding difference quotients. It is deliberately permissive of C0 kinks
(lin_interp / np.interp seams are allowed); the strict C1 guarantees of the
individual laws are pinned by their own slope-cap tests.

Any new hydraulic LumpedComponent must be added to the ``LUMPED_COMPONENTS``
parametrization of ``test_scan_lumped_component``.
"""

import numpy as np
import pytest

from stream import smoothing
from stream.calculations.channel import ChannelAndContacts
from stream.calculations.flapper import Flapper
from stream.calculations.ideal.heat_exchangers import HeatExchanger
from stream.calculations.ideal.pumps import Pump
from stream.calculations.ideal.resistors import (
    Bend,
    Friction,
    Gravity,
    LocalPressureDrop,
    RegimeDependentFriction,
    Resistor,
    Screen,
    VolumetricFlowResistor,
)
from stream.calculations.kirchhoff import Junction
from stream.physical_models.heat_transfer_coefficient.single_phase import regime_dependent_h_spl
from stream.physical_models.pressure_drop.friction import regime_dependent_friction
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water
from stream.utilities import directed_Tin, just

EPS = smoothing.DEFAULT_MDOT_EPS


def scan_1d(f, lo, hi, n=801, jump_ratio=0.35, quotient_growth=3.0, label=""):
    """Assert ``x -> f(x)`` (scalar or residual-vector valued) is continuous with
    non-exploding difference quotients on ``[lo, hi]``.

    Evaluate on uniform grids of step ``h`` and ``h/10``:
      (1) finiteness on both grids;
      (2) jump test:     ``max|Δf|(h/10) <= jump_ratio * max|Δf|(h) + atol``;
      (3) quotient test: ``max|Δf/Δx|(h/10) <= quotient_growth * max|Δf/Δx|(h) + atol``.
    ``atol = 1e-9 * max(1, max|f|)``, applied per residual row (max over rows).
    """

    def evaluate(m):
        x = np.linspace(lo, hi, m)
        y = np.array([np.atleast_1d(np.asarray(f(xi), dtype=float)).ravel() for xi in x])
        return x, y  # y shape (m, rows)

    x1, y1 = evaluate(n)
    x2, y2 = evaluate(10 * (n - 1) + 1)
    assert np.all(np.isfinite(y1)), f"{label}: non-finite on coarse grid"
    assert np.all(np.isfinite(y2)), f"{label}: non-finite on fine grid"

    atol = 1e-9 * max(1.0, float(np.max(np.abs(y1))))
    # per-row maxima
    dj1, dj2 = np.max(np.abs(np.diff(y1, axis=0)), axis=0), np.max(np.abs(np.diff(y2, axis=0)), axis=0)
    dq1 = np.max(np.abs(np.diff(y1, axis=0) / np.diff(x1)[:, None]), axis=0)
    dq2 = np.max(np.abs(np.diff(y2, axis=0) / np.diff(x2)[:, None]), axis=0)
    bad_jump = np.where(dj2 > jump_ratio * dj1 + atol)[0]
    assert bad_jump.size == 0, f"{label}: jump in rows {bad_jump.tolist()} ({dj2}/{dj1})"
    bad_quot = np.where(dq2 > quotient_growth * dq1 + atol)[0]
    assert bad_quot.size == 0, f"{label}: derivative blow-up in rows {bad_quot.tolist()} ({dq2}/{dq1})"


# --- 1. Junction residual vs loop mdot -----------------------------------


class _Comp:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return hash(self.n)


def _junction_2edge():
    A, B = _Comp("A"), _Comp("B")
    j = Junction("J2e")
    return lambda m: j.calculate([0.0], Tin={A: 80.0}, Tin_minus={B: 30.0}, mdot={A: m, B: m})


def _junction_3edge_weighted():
    c1, c2, c3 = _Comp("1"), _Comp("2"), _Comp("3")
    j = Junction("J3e", weights={c2: 2.0})
    return lambda m: j.calculate(
        [0.0], Tin={c1: 40.0, c2: 60.0}, Tin_minus={c3: 90.0}, mdot={c1: m, c2: 0.7 * m, c3: -m}
    )


@pytest.mark.parametrize("builder", [_junction_2edge, _junction_3edge_weighted])
@pytest.mark.parametrize("lo,hi,n", [(-10 * EPS, 10 * EPS, 801), (-2.0, 2.0, 8001)])
def test_scan_junction(builder, lo, hi, n):
    scan_1d(builder(), lo, hi, n=n, label="junction")


# --- 2. directed_Tin ------------------------------------------------------


@pytest.mark.parametrize("lo,hi,n", [(-10 * EPS, 10 * EPS, 801), (-2.0, 2.0, 8001)])
def test_scan_directed_tin(lo, hi, n):
    scan_1d(lambda m: directed_Tin(80.0, 55.0, m), lo, hi, n=n, label="directed_Tin")


# --- 3. every concrete LumpedComponent ------------------------------------


def _lumped_components():
    dens = light_water.density
    return {
        "Resistor": Resistor(resistance=0.5),
        "VolumetricFlowResistor": VolumetricFlowResistor(k=1.0, name="VFR", density_func=dens, klow=0.1),
        "Friction": Friction(f=0.02, fluid=light_water, length=1.0, hydraulic_diameter=0.01, area=1e-3),
        "Gravity": Gravity(fluid=light_water, disposition=1.0),
        "LocalPressureDrop": LocalPressureDrop(fluid=light_water, A1=1e-3, A2=2e-3),
        "Bend": Bend(fluid=light_water, hydraulic_diameter=0.01, area=1e-3, bend_radius=0.03, bend_angle=np.pi / 2),
        "HeatExchanger": HeatExchanger(outlet=70.0),
        "Pump_p": Pump(pressure=1e4, name="Pp"),
        "Pump_mdot": Pump(mdot0=1.0, name="Pm"),
        "RegimeDependentFriction": RegimeDependentFriction(
            pipe=EffectivePipe(length=1.0, heated_perimeter=0.04, wet_perimeter=0.04, area=1e-4),
            fluid=light_water, re_bounds=(2000.0, 4000.0), k_R=1.0,
        ),
        # wire diameter placing Re=50 (~m=0.14) and Re=1000 (~m=2.8) near the sweep
        "Screen": Screen(clear_area=0.3, total_area=1.0, wire_diameter=0.05, fluid=light_water),
    }


LUMPED_COMPONENTS = list(_lumped_components().items())


# RegimeDependentFriction.dp_out assigns a shape-(1,) array into out[1], a numpy>=1.25 deprecation.
@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim")
@pytest.mark.parametrize("name,comp", LUMPED_COMPONENTS)
def test_scan_lumped_component(name, comp):
    # Window deliberately straddles but never lands exactly on mdot=0: some
    # components have an exact-zero singularity (e.g. Bend's friction factor at
    # re=0, which lacks the re==0 guard Screen has). The reversal is still densely
    # sampled, testing continuity across it.
    scan_1d(
        lambda m: comp.calculate([80.0, 0.0], mdot=m, Tin=80.0, Tin_minus=55.0),
        -2.0, 2.000173, n=8001, label=name,  # odd offset: no grid node lands on exact 0
    )


# --- 4. Flapper open state ------------------------------------------------


def _open_flapper():
    F = Flapper(open_at_current=1.0, f=2.0, fluid=light_water, area=1e-3, open_rate=1.0)
    F.t_open = 0.0
    return F


def test_scan_flapper_vs_dp():
    F = _open_flapper()
    scan_1d(lambda dp: F.calculate([0.0, dp], mdot=0.3, Tin=80.0, Tin_minus=55.0, t=100.0),
            -50.0, 50.0, n=801, label="flapper_dp")


def test_scan_flapper_vs_mdot():
    F = _open_flapper()
    scan_1d(lambda m: F.calculate([0.0, 5.0], mdot=m, Tin=80.0, Tin_minus=55.0, t=100.0),
            -10 * EPS, 10 * EPS, n=801, label="flapper_mdot")


def test_scan_flapper_vs_t():
    F = _open_flapper()  # opens at t=0, ramp ends at t=1/open_rate=1
    scan_1d(lambda t: F.calculate([0.0, 5.0], mdot=0.3, Tin=80.0, Tin_minus=55.0, t=t),
            0.0, 2.0, n=801, label="flapper_t")


# --- 5. regime_dependent_h_spl --------------------------------------------

_HTC_KW = dict(Dh=5e-3, A=1e-3, re_bounds=(2000.0, 4000.0), coolant_funcs=light_water,
               depth=0.003, Lh=0.6, develop_length=np.array([0.3]), aspect_ratio=0.1)


def _h_of_mdot(m):
    film = light_water.to_properties(np.array([75.0]))
    return regime_dependent_h_spl(coolant=film, mdot=m, T_cool=np.array([60.0]),
                                  T_wall=np.array([90.0]), **_HTC_KW)


def _h_of_twall(tw):
    film = light_water.to_properties(np.array([(60.0 + tw) / 2]))
    return regime_dependent_h_spl(coolant=film, mdot=5e-4, T_cool=np.array([60.0]),
                                  T_wall=np.array([tw]), **_HTC_KW)


def test_scan_regime_dependent_h_vs_mdot():
    scan_1d(_h_of_mdot, 1e-4, 5e-2, n=2001, label="htc_mdot")


def test_scan_regime_dependent_h_vs_twall():
    # crosses the band and T_wall = T_cool (the |gr| path) at small fixed mdot
    scan_1d(_h_of_twall, 40.0, 80.0, n=2001, label="htc_twall")


# --- 6. regime_dependent_friction ----------------------


def test_scan_regime_dependent_friction():
    """Guards the lam<->turb lin_interp blend, scanned across the re_bounds
    transition (Re ~ 1076..6453 over the window), where the friction factor is
    bounded; near mdot=0 the laminar 64/Re factor diverges by design (cancelled
    by mdot*|mdot| in dp), so the raw factor is not scanned through 0."""
    pipe = EffectivePipe(length=1.0, heated_perimeter=0.04, wet_perimeter=0.04, area=1e-4)  # Dh=0.01
    f = lambda m: regime_dependent_friction(
        T_cool=60.0, T_wall=90.0, mdot=m, fluid=light_water, pipe=pipe, re_bounds=(2000.0, 4000.0), k_R=1.0
    )
    scan_1d(f, 0.005, 0.03, n=2001, label="friction")


# --- 7. ChannelAndContacts advection (guards the upwind scheme) --------


@pytest.mark.slow
def test_scan_channel_and_contacts_advection():
    pipe = EffectivePipe.rectangular(0.6, 0.05, 0.003, 0.06)
    cc = ChannelAndContacts(np.linspace(0, 0.6, 4), light_water, pipe, h_wall_func=just(5e3))
    n = cc.n
    variables = np.concatenate([np.full(n, 80.0), np.full(n, 5e3), np.full(n, 5e3), [0.0]])
    f = lambda m: cc.calculate(
        variables, T_left=90.0, T_right=90.0, Tin=80.0, Tin_minus=55.0, mdot=m, p_abs=1e5
    )
    scan_1d(f, -10 * EPS, 10 * EPS, n=801, label="channel_and_contacts")
