"""
MTR: Material Test Reactors
===========================
In this context, MTR refers to a typical type of fuels, which comprise of several rectangular parallel plates in each
rod, with coolant flowing between the plates.

The combination of calculations into an MTR-like setup, containing:

1. Fuel Plates
2. Coolant (T, incompressible scheme) channels on both sides.

The created objects are CalculationGraph objects which may be combined to eventually create an executable scheme.
"""

from copy import deepcopy
from typing import Sequence, Iterable, Literal, Callable

import numpy as np
from more_itertools import interleave_longest, windowed
from networkx import DiGraph

from stream.aggregator import CalculationGraph, vars_, ExternalFunctions
from stream.calculations import ChannelAndContacts, Fuel
from stream.units import Array1D, Meter, Name, Array2D
from stream.utilities import concat, cosine_shape

__all__ = [
    "chain_fuels_channels",
    "plate",
    "rod",
    "one_sided_connection",
    "symmetric_plate",
    "x_boundaries",
    "uniform_x_power_shape",
]


def plate(channel_l: ChannelAndContacts, channel_r: ChannelAndContacts, fuel: Fuel) -> CalculationGraph:
    """
    Build a fuel "plate" setup and its surroundings using the following components:

    Parameters
    ----------
    channel_l: ChannelAndContacts
        Left coolant channel
    channel_r: ChannelAndContacts
        Right coolant channel
    fuel: Fuel
        (Central) Fuel region

    Returns
    -------
    agr: CalculationGraph
        An CalculationGraph whose time-funcs are empty
    """

    return CalculationGraph(
        DiGraph(
            [
                (channel_l, fuel, vars_("T_left", "h_left")),
                (channel_r, fuel, vars_("T_right", "h_right")),
                (fuel, channel_l, vars_("T_right")),
                (fuel, channel_r, vars_("T_left")),
            ]
        )
    )


def chain_fuels_channels(channels: Sequence[ChannelAndContacts], fuels: Sequence[Fuel]) -> CalculationGraph:
    """
    Build a setup of interchanging fuel plates and channels by order of appearance.
    This function is to be used for the following geometry only::

        ... Channel, Fuel, Channel, Fuel ...

    Which is the case for MTR-like reactor rods.

    Parameters
    ----------
    channels: Sequence[ChannelAndContacts]
        Channels
    fuels: Sequence[Fuel]
        Fuel Plates

    Returns
    -------
    agr: CalculationGraph
        whose time-functions are empty

    Raises
    ------
    ValueError
        if the lengths of sequences does not allow interchanging
        (if difference is not +-1).
    """
    diff = len(channels) - len(fuels)
    g = DiGraph()
    if diff == 1:
        joint = interleave_longest(channels, fuels)
        for channel_l, fuel, channel_r in windowed(joint, 3, step=2):
            g.add_edge(channel_l, fuel, variables=("T_left", "h_left"))
            g.add_edge(channel_r, fuel, variables=("T_right", "h_right"))
            g.add_edge(fuel, channel_l, variables=("T_right",))
            g.add_edge(fuel, channel_r, variables=("T_left",))
        return CalculationGraph(g)
    elif diff == -1:
        joint = interleave_longest(fuels, channels)
        for fuel_l, channel, fuel_r in windowed(joint, 3, step=2):
            g.add_edge(fuel_l, channel, variables=("T_left",))
            g.add_edge(fuel_r, channel, variables=("T_right",))
            g.add_edge(channel, fuel_r, variables=("T_left", "h_left"))
            g.add_edge(channel, fuel_l, variables=("T_right", "h_right"))
        return CalculationGraph(g)
    raise ValueError(
        f"Interchanging Fuel-Channel not possible since the difference: #Channels - #Fuels = {diff}, not +-1"
    )


def _pair_connection(
    fuel: Fuel,
    channel: ChannelAndContacts,
    funcs: ExternalFunctions | None,
    fvars: Iterable[Name],
    cvars: Iterable[Name],
) -> CalculationGraph:
    """Connects a single fuel to a single channel by given variables

    Parameters
    ----------
    channel: ChannelAndContacts
        The channel object to connect.
    fuel: Fuel
        The fuel object to connect.
    fvars: Iterable[Name]
        Fuel variables to send to the channel.
    cvars: Iterable[Name]
        Channel variables to send to the fuel.
    funcs: ExternalFunctions
        Functions of time to connect to fuel or channel.

    Returns
    -------
    CalculationGraph

    """
    return CalculationGraph(
        DiGraph(
            (
                (fuel, channel, vars_(*fvars)),
                (channel, fuel, vars_(*cvars)),
            )
        ),
        funcs,
    )


def one_sided_connection(
    channel: ChannelAndContacts,
    fuel: Fuel,
    fuel_side: Literal["left", "right"],
    funcs: ExternalFunctions | None = None,
) -> CalculationGraph:
    """Connects a single fuel to a single channel on one side

    .. important::
     In this connection, the fuel assumes it is connected on both sides to the channel,
     but the channel is connected only to one chosen ``fuel_side``

    Parameters
    ----------
    channel: ChannelAndContacts
        The channel object to connect.
    fuel: Fuel
        The fuel object to connect.
    fuel_side: 'left' or 'right'
        The side of the channel at which the fuel is placed.
        The following diagrams might help. Assume there are no previous connections for the fuel or channel.
        Therefore, a `left` case looks like this::

         ||          |            ||
         ||   Fuel   |   Channel  ||
         ||          |            ||

        and a `right` case looks like this::

         ||             |         ||
         ||   Channel   |   Fuel  ||
         ||             |         ||

    funcs: ExternalFunctions
        Functions of time to connect to fuel or channel.

    Returns
    -------
    CalculationGraph

    """
    fuel_var = {"left": "T_left", "right": "T_right"}[fuel_side]
    channel_vars = ("T_right", "h_right", "T_left", "h_left")
    return _pair_connection(fuel, channel, funcs, (fuel_var,), channel_vars)


def symmetric_plate(
    channel: ChannelAndContacts, fuel: Fuel, funcs: ExternalFunctions | None = None
) -> CalculationGraph:
    """Create a symmetric plate.
    This is convenient if symmetry or low calculation costs are relevant.
    The objects must already describe the desired geometry.

    The fuel and channel are combined such that they're simultaneously on both sides of each other.

    Parameters
    ----------
    channel: ChannelAndContacts
        The channel object to connect.
    fuel: Fuel
        The fuel object to connect.
    funcs: ExternalFunctions
        External functions relating to the components

    Returns
    -------
    CalculationGraph
    """
    return _pair_connection(
        fuel,
        channel,
        funcs,
        ("T_left", "T_right"),
        ("T_left", "h_left", "T_right", "h_right"),
    )


def rod(
    channel: ChannelAndContacts, channels: int, fuel: Fuel, plates: int
) -> (CalculationGraph, Sequence[ChannelAndContacts], Sequence[Fuel]):
    r"""
    Create a rod of interleaving channels and fuel plates, in which all channels, fuel plates are the same.
    This function utilizes :func:`chain_fuels_channels`.

    Parameters
    ----------
    channel: ChannelAndContacts
        calculation which is to be constructed several times.
    channels: int
        # of channels.
    fuel: Fuel
        calculation which is to be constructed several times.
    plates: int
        # of fuel plates.

    Returns
    -------
    out: tuple[CalculationGraph, Sequence[ChannelAndContacts], Sequence[Fuel]]
        The connected CalculationGraph, created channels, created fuels.

    See Also
    --------
    chain_fuels_channels
    """
    chs = [deepcopy(channel) for _ in range(channels)]
    fuels = [deepcopy(fuel) for _ in range(plates)]
    return chain_fuels_channels(channels=chs, fuels=fuels), chs, fuels


def x_boundaries(clad_N: int, fuel_N: int, clad_w: Meter, meat_w: Meter) -> Array1D:
    """
    Construct the plate boundaries perpendicular to the plate's large plane (denoted `x`).

    Parameters
    ----------
    clad_N: int
        Into how many cladding cells to partition (symmetrically on each side).
    fuel_N: int
        Into how many fuel (meat) cells to partition (centered).
    clad_w: Meter
        Width of cladding (on each side). This value is not used if clad_N = 0.
    meat_w: Meter
        Width of Meat.

    Returns
    -------
    x_bounds: Meter
        An array containing the cells boundary placements, beginning with zero.
    """
    if clad_N == 0:
        x = np.linspace(0, meat_w, fuel_N + 1)
    else:
        x = concat(
            np.linspace(0, clad_w, clad_N, endpoint=False),
            np.linspace(clad_w, clad_w + meat_w, fuel_N, endpoint=False),
            np.linspace(clad_w + meat_w, 2 * clad_w + meat_w, clad_N + 1),
        )
    return x


def uniform_x_power_shape(
    z_N: int,
    fuel_N: int,
    clad_N: int,
    clad_w: Meter,
    meat_w: Meter,
    meat_h: Meter,
    z_shaper: Callable[[Meter], Meter] = cosine_shape,
) -> Array2D:
    r"""Create a ``power_shape`` for :class:`~.Fuel` in which the distribution along the x-axis is uniform,
    and controlled by ``z_shaper`` along the z-axis. Utilizes :func:`x_boundaries`.

    Parameters
    ----------
    z_N : int
        Number of cells in the direction of flow across the plate (denoted `z`)
    fuel_N : int
        Into how many fuel (meat) cells to partition (centered).
    clad_N : int
        Into how many cladding cells to partition (symmetrically on each side).
    clad_w: Meter
        Width of cladding (on each side). This value is not used if clad_N = 0.
    meat_w: Meter
        Width of meat.
    meat_h : Meter
        Height of meat (assumed to be the total modelled plate)
    z_shaper : Callable[[Meter], Meter]
        Profiler function, which receives cell boundaries and returns weights for each cell.
        Default is :func:`~.cosine_shape`.

    Returns
    -------
    Array2D
    """
    power_shape_in_rod = z_shaper(np.linspace(0, meat_h, z_N + 1))
    dx_meat = np.diff(x_boundaries(clad_N, fuel_N, clad_w, meat_w))
    dx_meat = dx_meat if clad_N == 0 else dx_meat[clad_N:-clad_N]
    area_fraction = dx_meat / meat_w
    return np.outer(power_shape_in_rod, area_fraction)
