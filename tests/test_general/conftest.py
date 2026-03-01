import numpy as np
from hypothesis.strategies import floats, integers

from stream import Solid, EffectivePipe, light_water
from stream.calculations import Fuel, ChannelAndContacts
from stream.composition import x_boundaries
from stream.units import mm
from stream.utilities import normalize

normal_floats = floats(allow_infinity=False, allow_nan=False)
medium_floats = floats(allow_infinity=False, allow_nan=False, max_value=1e6, min_value=-1e6)
pos_medium_floats = floats(allow_infinity=False, allow_nan=False, max_value=1e6, min_value=1e-6)
non_neg_medium_integers = integers(max_value=int(1e6), min_value=0)
pos_floats = floats(0, allow_nan=False, allow_infinity=False, exclude_min=True)


def are_close(a, b, rtol=1.0e-5, atol=1.0e-8):
    """Asserts a, b are np.allclose with rtol, atol"""
    assert np.allclose(a, b, rtol=rtol, atol=atol), _relative_percentage(a, b)


def _relative_percentage(a, b):
    return f"{np.max(np.abs(a - b) / np.abs(b)):.2%}"


def MTR_fuel_and_channel(z_N: int, fuel_N: int, clad_N: int) -> tuple[Fuel, ChannelAndContacts]:
    """Creating an example of an MTR Fuel-Channel duo, for testing purposes"""
    meat_depth = 0.5 * mm
    clad_depth = 0.4 * mm
    meat_width = 70 * mm
    shape = np.array((z_N, fuel_N + 2 * clad_N))
    meat = np.zeros(shape, dtype=bool)
    meat[:, clad_N:-clad_N] = True
    materials = np.empty(shape, dtype=object)
    materials[meat] = Solid(density=3000, specific_heat=800, conductivity=100)
    materials[~meat] = Solid(density=2700, specific_heat=900, conductivity=250)
    material = Solid.from_array(materials)

    F = Fuel(
        z_boundaries=(zb := np.linspace(0, 1, z_N + 1)),
        x_boundaries=x_boundaries(clad_N, fuel_N, clad_depth, meat_depth),
        material=material,
        meat_indices=meat,
        power_shape=normalize(np.ones(z_N * fuel_N)),
        y_length=meat_width,
    )

    channel_depth = 2 * mm
    channel_width = 70 * mm
    pipe = EffectivePipe.rectangular(length=1, edge1=channel_depth, edge2=channel_width, heated_edge=meat_width)
    C = ChannelAndContacts(z_boundaries=zb, fluid=light_water, pipe=pipe)

    return F, C
