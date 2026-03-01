import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, sampled_from
from numpy import allclose

from stream.aggregator import Aggregator
from stream.calculations.heat_diffusion import (
    Fuel,
    Solid,
    Walls,
    xz_diffusion,
    cylindrical_areas_volumes,
    rz_diffusion,
    r_diffusion,
    x_diffusion,
)
from stream.state import State
from stream.substances.mocks import mock_solid
from stream.units import cm, mm
from stream.utilities import concat, pair_mean_1d
from .conftest import are_close, pos_medium_floats


def test_Fuel_at_constant_temperature_has_derivative_0():
    z = np.arange(3)
    dz = np.diff(z)
    x = np.arange(6)
    dx = np.diff(x)
    material = mock_solid
    power_shape = np.zeros((len(dz), (len(dx))))

    fuel = Fuel(z, x, material, y_length=1.0, power_shape=power_shape)
    T0 = 30
    T = T0 * np.ones(len(dz) * (len(dx) + 2))
    result = fuel.calculate(
        T,
        power=0,
        T_left=T0 * np.ones_like(dz),
        T_right=T0 * np.ones_like(dz),
        h_left=np.ones_like(dz),
        h_right=np.ones_like(dz),
    )
    are_close(result, np.zeros_like(T))


@pytest.mark.implementation
@settings(deadline=None)
@given(pm_flt := pos_medium_floats, pm_flt, pm_flt)
def test_derivative_of_one_cell_follows_the_x_diffusion_kernel(T, T_left, T_right):
    z = np.arange(2)
    x = np.arange(2)
    dx = 1
    power_shape = np.zeros((len(z) - 1, len(x) - 1))

    fuel = Fuel(z, x, mock_solid, y_length=1, power_shape=power_shape)
    input = np.array((T, T_left, T_right))
    result = fuel.calculate(
        input,
        power=0,
        T_left=np.array([T_left]),
        T_right=np.array([T_right]),
        h_left=np.array([np.inf]),
        h_right=np.array([np.inf]),
    )

    k = fuel.material.conductivity
    dT = fuel.save(result)["T"]

    are_close(dT, (T_left + T_right - T * 2) * 2 * k / dx)


@pytest.mark.implementation
@given(pm_flt := pos_medium_floats, pm_flt, pm_flt)
def test_not_equispaced(T0, T_l, T_r):
    z = np.arange(2)
    x = np.array([0, 1, 3, 4])
    power_shape = np.zeros((len(z) - 1, len(x) - 1))

    T = np.array((T_l, T0, T_r))
    T_left, T_right = np.full(1, T_l), np.full(1, T_r)
    input = concat(T, T_left, T_right)

    fuel = Fuel(z, x, mock_solid, y_length=1, power_shape=power_shape)
    result = fuel.calculate(
        input,
        power=0,
        T_left=T_left,
        T_right=T_right,
        h_left=np.full(1, np.inf),
        h_right=np.full(1, np.inf),
    )
    dT = fuel.save(result)["T"]

    are_close(dT, [(T0 - T_l) / 1.5, (T_l + T_r - 2 * T0) / 3, (T0 - T_r) / 1.5])


@pytest.fixture(scope="module")
def multiple_regions():
    z = np.arange(3)
    x = np.array([0, 1, 3, 4])
    y = 1

    meat = np.ones((len(z) - 1, len(x) - 1))
    meat[:, 0] = 0

    power_shape = np.ones((len(z) - 1, len(x) - 2))
    return Fuel(
        z,
        x,
        mock_solid,
        x_contacts=np.tile(np.array([np.inf, 2e4, np.inf, np.inf]), (2, 1)),
        meat_indices=meat,
        y_length=y,
        power_shape=power_shape,
    )


def test_specific_multi_cell_has_the_right_dimensions(multiple_regions):
    f = multiple_regions
    assert (f.m, f.n) == (2, 3)
    assert len(f) == f.m * f.n + 2 * f.m


def test_specific_multi_cell_has_the_right_meat(multiple_regions):
    f = multiple_regions
    assert allclose(f.meat, np.array([[0, 1, 1], [0, 1, 1]]))


def test_specific_multi_cell_has_the_right_power_shape(multiple_regions):
    f = multiple_regions
    assert allclose(f.power_shape, np.array([1, 1, 1, 1]))


def test_specific_multi_cell_has_the_right_contacts(multiple_regions):
    f = multiple_regions
    assert allclose(
        f.x_contacts,
        np.array([[np.inf, 2e4, np.inf, np.inf], [np.inf, 2e4, np.inf, np.inf]]),
    )


def test_specific_multi_cell_has_the_right_dTdt(multiple_regions):
    f = multiple_regions
    T = np.full(len(f), 10)
    T_wall = np.full(f.m, 10)

    result = f.calculate(
        T,
        power=100.0,
        T_left=T_wall,
        T_right=T_wall,
        h_left=np.full(2, 1e-9),
        h_right=np.full(2, 1e-9),
    )
    dT = f.save(result)["T"].reshape(f.m, f.n)
    assert allclose(dT, np.array([0, 50, 100]))


@pytest.fixture(scope="module")
def _some_config():
    z = np.arange(3) * 64 * cm / 2
    x = np.array([0, 1, 3, 4]) * 0.38 * mm
    y = 51.4 * mm
    dz = np.diff(z)
    dx = np.diff(x)
    power_shape = np.ones((len(z) - 1, len(x) - 1))

    T = np.full((len(dz)) * (len(dx)) + 2 * len(dz), 100)
    T_cool = np.array((20, 40))
    h_left, h_right = np.array([1e5]), np.array([1e5])

    fuel = Fuel(
        z,
        x,
        Solid(density=3000, specific_heat=700, conductivity=240),
        y_length=y,
        power_shape=power_shape,
    )
    agr = Aggregator.from_decoupled(
        fuel,
        funcs={fuel: dict(T_left=T_cool, T_right=T_cool, h_left=h_left, h_right=h_right)},
    )
    return agr, fuel, T, T_cool


def test_steady_state_with_zero_power(_some_config):
    agr, fuel, T, T_cool = _some_config
    agr.funcs[fuel]["power"] = 0.0
    # Zero Power Steady State. Boundary conditions rule
    sol = agr.solve_steady(T)
    steady = agr.save(sol)[fuel.name]
    assert np.allclose(steady["T_wall_left"], T_cool)
    assert np.allclose(steady["T_wall_right"], T_cool)
    assert np.allclose(steady["T"], np.repeat(T_cool, fuel.n).reshape(fuel.shape))


def test_steady_state_for_a_configuration_with_uniform_power(_some_config):
    agr, fuel, T, T_cool = _some_config
    # Uniform Power. Outgoing flux must equal power at every elevation
    agr.funcs[fuel]["power"] = 100.0
    steady = agr.save(agr.solve_steady(T))[fuel.name]

    h_left, h_right = (agr.funcs[fuel][s] for s in ("h_left", "h_right"))

    outgoing_flux = (
        fuel.y_length
        * fuel.dz
        * (h_left * (steady["T_wall_left"] - T_cool) + h_right * (steady["T_wall_right"] - T_cool))
    )
    power_by_z = np.sum(np.full(6, 100.0).reshape(fuel.m, fuel.n), axis=1)
    assert np.allclose(outgoing_flux, power_by_z), outgoing_flux - power_by_z


def test_initialization_of_Fuel_with_one_known_example():
    z_N, x_N = 9, 3
    cells = z_N * x_N
    z = np.arange(z_N + 1)
    x = np.arange(x_N + 1)
    power_shape = np.ones((z_N, x_N))
    F = Fuel(z, x, mock_solid, y_length=1, power_shape=power_shape)
    assert np.all(F.x_contacts == np.inf)
    assert F.variables == dict(
        T=slice(0, cells),
        T_wall_left=slice(cells, cells + z_N),
        T_wall_right=slice(cells + z_N, cells + 2 * z_N),
    )
    assert np.all(F.mass_vector[F.variables["T"]] == 1.0)
    assert np.all(F.mass_vector[F.variables["T_wall_left"]] == 0.0)
    assert np.all(F.mass_vector[F.variables["T_wall_right"]] == 0.0)


@settings(deadline=None)
@given(
    pos_medium_floats,
    integers(1, 10),
    integers(1, 10),
    sampled_from([x_diffusion, xz_diffusion, r_diffusion, rz_diffusion]),
)
def test_diffusion_gives_0_for_uniform_temperatures(T0, z_N, x_N, func):
    """
    If all temperatures are uniform (with zero power),
    no heat flux should be generated, and the system should be at steady-state.
    This property is tested over all available kernels.
    """
    x = np.arange(x_N + 1)
    z = np.arange(z_N + 1)

    T = np.full((z_N, x_N), T0)
    T_walls = Walls(
        left=np.full(z_N, T0),
        right=np.full(z_N, T0),
        top=np.full(x_N, T0),
        bottom=np.full(x_N, T0),
    )
    x_contacts = np.full((z_N, x_N + 1), np.inf)
    z_contacts = np.full((z_N + 1, x_N), np.inf)
    contacts = x_contacts, z_contacts

    dTdt = func(
        T=T,
        T_walls=T_walls,
        material=mock_solid,
        power=0.0,
        x=x,
        z=z,
        contacts=contacts,
        y=1,
    )
    assert np.allclose(dTdt, 0.0)


radii = st.floats(min_value=1e-3, max_value=1e5)
small_dr_change_relative = st.floats(min_value=1e-2, max_value=1)


@pytest.mark.implementation
@settings(deadline=None)
@given(pm_flt := pos_medium_floats, pm_flt, pm_flt, radii, small_dr_change_relative)
def test_derivative_of_one_cell_follows_the_r_diffusion_kernel(T, T_left, T_right, r1, dr_rel):
    r2 = r1 + r1 * dr_rel
    x = np.array([r1, r2])
    dr = r2 - r1

    z = np.arange(2)
    power_shape = np.zeros((len(z) - 1, len(x) - 1))
    input = np.array((T, T_left, T_right))

    fuel = Fuel(z, x, mock_solid, y_length=1, heat_func=r_diffusion, power_shape=power_shape)
    result = fuel.calculate(
        input,
        power=0,
        T_left=np.array([T_left]),
        T_right=np.array([T_right]),
        h_left=np.array([np.inf]),
        h_right=np.array([np.inf]),
    )

    k = fuel.material.conductivity
    dT = fuel.save(result)["T"]
    c = 2 * k * (2 / dr) / (r2**2 - r1**2)
    are_close(dT, c * ((T_left - T) * r1 + (T_right - T) * r2))


dz = st.floats(min_value=1e-3, max_value=0.1)


@pytest.mark.implementation
@settings(deadline=None)
@given(
    pm_flt := pos_medium_floats,
    pm_flt,
    pm_flt,
    pm_flt,
    pm_flt,
    radii,
    small_dr_change_relative,
    pm_flt,
    dz,
)
def test_derivative_of_one_cell_follows_the_rz_diffusion_kernel(
    T, T_left, T_right, T_top, T_bottom, r1, dr_rel, z1, dz
):
    r2 = r1 + r1 * dr_rel
    x = np.array([r1, r2])
    dr = r2 - r1

    z = np.array([z1, z1 + dz])
    dz = z[1] - z[0]
    power_shape = np.zeros((len(z) - 1, len(x) - 1))
    input = np.array((T, T_left, T_right))

    fuel = Fuel(z, x, mock_solid, y_length=1, heat_func=rz_diffusion, power_shape=power_shape)
    result = fuel.calculate(
        input,
        power=0,
        T_left=np.array([T_left]),
        T_right=np.array([T_right]),
        T_bottom=np.array([T_bottom]),
        T_top=np.array([T_top]),
        h_left=np.array([np.inf]),
        h_right=np.array([np.inf]),
        h_top=np.array([np.inf]),
        h_bottom=np.array([np.inf]),
    )

    k = fuel.material.conductivity
    dT = fuel.save(result)["T"]
    c = 2 / dr / (r2**2 - r1**2)
    a_left, a_right = x * c
    a_top = a_bottom = 1 / dz**2
    a_iter = (a_left, a_right, a_top, a_bottom)
    t_iter = (T_left, T_right, T_top, T_bottom)
    expected_dT = 2 * k * sum(a * (t - T) for a, t in zip(a_iter, t_iter))
    are_close(dT, expected_dT)


def test_annulus_given_wall_temperatures(temps=(45, 75), edges=(1, 3)):
    r"""
    For an axially symmetric annulus of radii :math:`r_1, r_2`, given inner and
    outer boundary temperatures :math:`T_{1}, T_{2}`, the radial temperature
    dependence is [#Incropera]_:

    .. math:: T(r) = (T_{1} - T_{2})\frac{\ln(r / r_2)}{\ln(r_1 / r_2)} + T_{2}

    References
    ----------
    .. [#Incropera] Fundamentals of Heat and Mass Transfer (6th edition), Incropera, p. 116-117.
    """
    cells = 200
    Ts1, Ts2 = temps
    r1, r2 = edges
    r_boundaries = np.linspace(*edges, num=cells + 1)
    r = pair_mean_1d(r_boundaries)
    z_boundaries = np.arange(2)
    power_shape = np.ones((len(z_boundaries) - 1, len(r_boundaries) - 1))

    T_exp = (Ts1 - Ts2) * np.log(r / r2) / np.log(r1 / r2) + Ts2

    fuel = Fuel(
        z_boundaries,
        r_boundaries,
        mock_solid,
        y_length=1,
        heat_func=r_diffusion,
        power_shape=power_shape,
    )
    d = dict(T_left=Ts1, T_right=Ts2, h_left=np.inf, h_right=np.inf, power=0)
    agr = Aggregator.from_decoupled(fuel, funcs={fuel: d})
    steady = agr.save(agr.solve_steady(State.uniform(agr.graph, (Ts1 + Ts2) / 2)))
    T_num = steady["Fuel"]["T"][0]
    assert np.allclose(T_num, T_exp)


def test_cylinder_given_heat_production_and_wall_temperature(Twall=45, r0=3):
    r"""
    Temperature distribution calculated analytically matches the numeric result,
    in a polar system with heat production per volume :math:`\dot{q}`, radius :math:`r_{0}`
    and fixed outer temperature :math:`T_{s}`.
    The analytic solution is [#Incropera]_:

    .. math:: T(r) = T_s + \dot{q} (r_0 ^2 - r^2)/4k

    """
    total_cylinder_power = 100
    cells = 100
    radii = np.linspace(0, r0, cells + 1)
    r = pair_mean_1d(radii)
    z = np.arange(2)
    volumes = cylindrical_areas_volumes(r=radii, z=z)[2]
    normalized_volumes = volumes / np.sum(volumes)

    qdot = total_cylinder_power / (np.pi * r0**2)
    k = mock_solid.conductivity
    T_exp = qdot * (r0**2 - r**2) / (4 * k) + Twall

    fuel = Fuel(
        z,
        radii,
        mock_solid,
        y_length=1,
        heat_func=r_diffusion,
        power_shape=normalized_volumes,
    )
    inp = dict(T_right=Twall, h_left=np.inf, h_right=np.inf, power=total_cylinder_power)
    agr = Aggregator.from_decoupled(fuel, funcs={fuel: inp})
    steady = agr.save(agr.solve_steady(State.uniform(agr.graph, Twall)))
    T_num = steady["Fuel"]["T"][0]
    assert np.allclose(T_num, T_exp)


def test_annulus_given_heat_production_and_wall_temperatures(temps=(45, 75), edges=(1, 3.123)):
    r"""For an axially symmetric annulus of radii :math:`r_1, r_2`, given inner and
    outer boundary temperatures :math:`T_{1}, T_{2}`, and heat production of
    :math:`\dot{q}` the radial temperature dependence is:

    .. math:: T(r) = \frac{\dot{q}}{4k}\left[r_1^2 - r^2
        + (r_2^2 - r_1^2)\frac{\ln(r / r_1)}{\ln(r_2 / r_1)}\right]
        + (T_2 - T_1)\frac{\ln(r / r_1)}{\ln(r_2 / r_1)} + T_1
    """
    cells = 210
    total_rod_power = 100
    r1, r2 = edges
    Ts1, Ts2 = temps

    r_boundaries = np.linspace(*edges, cells + 1)
    r = pair_mean_1d(r_boundaries)
    z_boundaries = np.arange(2)
    volumes = cylindrical_areas_volumes(r=r_boundaries, z=z_boundaries)[2]
    normalized_volumes = volumes / np.sum(volumes)
    k = mock_solid.conductivity

    qdot = total_rod_power / (np.pi * (r2**2 - r1**2))

    ln_ratio = np.log(r / r1) / np.log(r2 / r1)
    T_exp = qdot / (4 * k) * (r1**2 - r**2 + ln_ratio * (r2**2 - r1**2)) + (Ts2 - Ts1) * ln_ratio + Ts1

    fuel = Fuel(
        z_boundaries,
        r_boundaries,
        mock_solid,
        y_length=1,
        heat_func=r_diffusion,
        power_shape=normalized_volumes,
    )
    inp = dict(T_left=Ts1, T_right=Ts2, h_left=np.inf, h_right=np.inf, power=total_rod_power)
    agr = Aggregator.from_decoupled(fuel, funcs={fuel: inp})
    steady = agr.save(agr.solve_steady(State.uniform(agr.graph, np.mean(temps))))
    T_num = steady["Fuel"]["T"][0]
    assert np.allclose(T_num, T_exp)
