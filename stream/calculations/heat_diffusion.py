r"""
A calculation depicting heat diffusion in a solid
(supposedly material or cladding).


Discretizing the Heat Equation
==============================

Calculate the temporal derivative of temperatures
according to the heat equation, which is, in this case:

.. math::
    \rho c_p \frac{\partial T}{\partial t} - \nabla\cdot(k\nabla T) = q'''

The medium is considered 2D in both the Cartesian and the Polar case.

Cartesian Discretization:
-------------------------

Integrating and writing the terms, while assuming y-symmetry.
Cell :math:`(i,j)` reads :math:`i` th cell from the top and
:math:`j` th cell from the left.

1. Temporal derivative
    .. math::
        \int_{V_{ij}} \rho c_p\frac{\partial T}{\partial t} dV
        = \rho_{ij} c_{p,ij}\frac{\partial T_{ij}}{\partial t} V_{ij}

2. Power Source
    .. math::
        \int_{V_{ij}} q'''dV = P_{ij}

3. Diffusive Term
    .. math::
        \int_{V_{ij}} \nabla\cdot(k\nabla T)dV
        = \oint_{\partial V_{ij}}(k\nabla T)\cdot dA

    Cells are boxes, so in the x-direction:

    .. math::
        = \left[(k\nabla T)_{i,j+1/2} - (k\nabla T)_{i,j-1/2}\right]A_i

    The thermal resistance going from one cell center to the other is just

    .. math::
        r_{ij, i(j+1)} = \frac{\Delta x_{ij}}{2k_{ij}}
        + \frac{\Delta x_{i(j+1)}}{2k_{i(j+1)}} + 1 / h_{ij, i(j+1)}

    Where :math:`h` is contact conductance.
    Then, the flux is taken as:

    .. math::
        (k\nabla T)_{i,j+1/2} = (T_{i,j+1} - T_{i,j}) / r_{ij, i(j+1)}

    At the boundaries, the wall temperatures are used in a similar fashion,
    such that the temperature gradient is taken as one-sided.

Polar Discretization:
---------------------

For the polar case, azimuthal symmetry is assumed, and the temperature is a function
of the radius :math:`r` and height :math:`z`. Similar to the Cartesian case,
here cell :math:`(i,j)` reads :math:`i` th cell from the top and :math:`j` th cell
from the smallest radius (0 for a cylinder). The integration is the same (since the
:math:`\nabla_\hat{r}` and :math:`\nabla_\hat{z}` components are just
:math:`\partial_r` and :math:`\partial_z`, respectively), but the volumes and cell areas
are :math:`r`-dependent. Specifically:

.. math::
    \begin{align}
    A_{i, j-1/2} &= 2\pi r_{j-1/2}\Delta z_i \\
    A_{i-1/2, j} = A_{i+1/2, j} &= \pi (r_{j+1/2} ^ 2 - r_{j-1/2} ^ 2) \\
    V_{ij} &= \pi (r_{j+1/2} ^ 2 - r_{j-1/2} ^ 2)\Delta z_i \\
    \end{align}

The thermal resistance is then computed in the same manner using the distances
between cell centers :math:`\Delta r, \Delta z`.
"""
import logging
import pickle
from dataclasses import astuple, dataclass
from functools import partial
from typing import Callable, Sequence, Type, TypeVar

import numpy as np
from cachetools import cached
from more_itertools import chunked, interleave
from numba import njit

from stream.calculation import Calculation, unpacked, CalcState
from stream.physical_models.heat_transfer_coefficient import wall_temperature
from stream.units import (
    Array1D, Array2D, Celsius, CPerS, JPerKgK, KgPerM3, M2KPerW,
    Meter, Meter2, Meter3, Name, Place, Value, Watt, WPerM2, WPerM2K, WPerMK,
    )
from stream.utilities import (
    dataclass_map, harmonic_mean as in_parallel, if_is, pair_mean, STREAM_DEBUG,
    )

logger = logging.getLogger("stream.fuel")

__all__ = ["cylindrical_areas_volumes", "Fuel",
           "r_diffusion", "rz_diffusion",
           "x_diffusion", "xz_diffusion",
           "Solid", "Walls",
           ]


@dataclass(frozen=True, slots=True)
class Walls:
    """
    Simple 2D walls container
    """

    left: Value = None
    right: Value = None
    top: Value = None
    bottom: Value = None

    @property
    def x(self):
        """The x (lateral) values of the walls
        """
        return self.left, self.right

    @property
    def z(self):
        """The z (axial) values of the walls.
        """
        return self.top, self.bottom


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Solid:
    """
    Simple bulk properties of a material

    Parameters
    ----------
    density: KgPerM3
    specific_heat: JPerKgK
    conductivity: WPerMK
    """

    density: KgPerM3
    specific_heat: JPerKgK
    conductivity: WPerMK

    @classmethod
    def from_array(cls: Type[_T], array: np.array) -> _T:
        """Create a Solid from an array of Solid scalars

        Parameters
        ----------
        array: np.array
            An array of Solid objects, where each instance has scalar values

        Returns
        -------
        Solid
            An instance with array-shaped fields.

        Examples
        --------
        >>> Solid.from_array(np.array([Solid(1, 2, 3), Solid(4, 5, 6)]))
        Solid(density=array([1, 4]), specific_heat=array([2, 5]), conductivity=array([3, 6]))
        """
        fields = map(np.array, zip(*map(astuple, array.flat)))
        return cls(*(f.reshape(array.shape) for f in fields))


def x_diffusion(T: Celsius, T_walls: Walls,
                material: Solid, power: Watt,
                x: Meter, z: Meter,
                contacts: Sequence[WPerM2K], y: Meter) -> CPerS:
    """1D heat diffusion (`x`) in a 2D mesh (`x`, `z`)

    Calculates the temporal derivative of temperatures
    according to the heat equation.

    Parameters
    ----------
    T: Celsius
        Material temperature
    T_walls: Walls
        Wall temperatures, left and right
        (in the case of polar coordinates, inner and outer radii)
    material: Solid
        a Solid object for thermal properties
    power: Watt
        Power, a source term.
    x: Meter
        Boundaries crossing the x-axis.
    z: Meter
        Boundaries crossing the z-axis.
    y: Meter
        In the Cartesian case, the length of the non-described
        (meaning symmetric) dimension.
    contacts: Sequence[WPerM2K]
        indices and transfer coefficients for Fuel-Clad contacts

    Returns
    -------
    dT/dt: CPerS
        for the heat equation in the material. Only x-diffusion
        is assumed.
    """
    dx = np.diff(x)
    dz = np.diff(z)
    x_areas = y * np.outer(dz, np.ones(len(x)))
    volumes = y * np.outer(dz, dx)

    return generic_2d_diffusion(
        T=T, T_walls=T_walls, s=material, power=power, dr=(dx, dz),
        areas=(x_areas,), volumes=volumes, contacts=contacts)


def xz_diffusion(T: Celsius, T_walls: Walls,
                 material: Solid, power: Watt,
                 x: Meter, z: Meter,
                 contacts: Sequence[WPerM2K],
                 y: Meter) -> CPerS:
    """2D heat diffusion in a 2D mesh (`x`, `z`)

    Calculates the temporal derivative of temperatures
    according to the heat equation.

    Parameters
    ----------
    T: Celsius
        Material temperature
    T_walls: Walls
        Wall temperatures, left and right
        (in the case of polar coordinates, inner and outer radii)
    material: Solid
        a Solid object for thermal properties
    power: Watt
        Power, a source term.
    x: Meter
        Boundaries crossing the x-axis.
    z: Meter
        Boundaries crossing the z-axis.
    y: Meter
        In the Cartesian case, the length of the non-described
        (meaning symmetric) dimension.
    contacts: Sequence[WPerM2K]
        indices and transfer coefficients for Fuel-Clad contacts

    Returns
    -------
    dT/dt: CPerS
        for the heat equation in the material. xz-diffusion is assumed.

    See Also
    --------
    x_diffusion
    """
    dx = np.diff(x)
    dz = np.diff(z)
    x_areas = y * np.outer(dz, np.ones(len(x)))
    z_areas = y * np.outer(np.ones(len(z)), dx)
    volumes = y * np.outer(dz, dx)

    return generic_2d_diffusion(
        T=T, T_walls=T_walls, s=material, power=power, dr=(dx, dz),
        areas=(x_areas, z_areas), volumes=volumes, contacts=contacts)


@njit
def cylindrical_areas_volumes(r: Meter, z: Meter) -> tuple[Meter2, Meter2, Meter3]:
    """Calculate areas and cell volumes for a cylindrical mesh

    Parameters
    ----------
    r: Meter
        Boundaries crossing the r-axis.
    z: Meter
        Boundaries crossing the z-axis.

    Returns
    -------
        Areas and volumes for rz diffusion

    Examples
    --------
    >>> r = np.array([0, 1, 4, 14]); z = np.array([0, 3, 5, XX])
    >>> r_areas, z_areas, volumes = cylindrical_areas_volumes(r=r, z=z)
    >>> r_areas / (2 * np.pi)
    array([[  0.,   3.,  12.,  42.],
           [  0.,   2.,   8.,  28.],
           [  0.,  10.,  40., 140.]])
    >>> z_areas / np.pi
    array([[  1.,  XX., 180.],
           [  1.,  XX., 180.],
           [  1.,  XX., 180.],
           [  1.,  XX., 180.]])
    >>> volumes / np.pi
    array([[   3.,   45.,  540.],
           [   2.,   30.,  360.],
           [  10.,  XXX., 1800.]])
    """
    dz = np.diff(z)
    cum_areas = np.pi * np.diff(np.asarray(r) ** 2)
    r_areas = 2 * np.pi * np.outer(dz, r)
    z_areas = np.outer(np.ones(len(z)), cum_areas)
    volumes = np.outer(dz, cum_areas)
    return r_areas, z_areas, volumes


def rz_diffusion(T: Celsius, T_walls: Walls,
                 material: Solid, power: Watt,
                 x: Meter, z: Meter,
                 contacts: Sequence[WPerM2K], **_) -> CPerS:
    """2D heat diffusion in a 2D cylindrical mesh (`r`, `z`).
    Assumes azimuthal symmetry.

    Calculates the temporal derivative of temperatures
    according to the heat equation.

    Parameters
    ----------
    T: Celsius
        Material temperature
    T_walls: Walls
        Wall temperatures, inner and outer radii.
    material: Solid
        a Solid object for thermal properties
    power: Watt
        Power, a source term.
    x: Meter
        Boundaries crossing the r-axis.
    z: Meter
        Boundaries crossing the z-axis.
    contacts: Sequence[WPerM2K]
        indices and transfer coefficients for Fuel-Clad contacts

    Returns
    -------
    dT/dt: CPerS
        for the heat equation in the material. rz diffusion is assumed.
    """
    dx = np.diff(x)
    dz = np.diff(z)
    r_areas, z_areas, volumes = cylindrical_areas_volumes(r=x, z=z)
    return generic_2d_diffusion(
        T=T, T_walls=T_walls, s=material, power=power, dr=(dx, dz),
        areas=(r_areas, z_areas), volumes=volumes, contacts=contacts)


def r_diffusion(T: Celsius, T_walls: Walls,
                material: Solid, power: Watt,
                x: Meter, z: Meter,
                contacts: Sequence[WPerM2K], **_) -> CPerS:
    """1D heat diffusion (`r`) in a 2D cylindrical mesh (`r`, `z`).
    Assumes azimuthal symmetry.

    Calculates the temporal derivative of temperatures
    according to the heat equation.

    Parameters
    ----------
    T: Celsius
        Material temperature
    T_walls: Walls
        Wall temperatures, inner and outer radii.
    material: Solid
        a Solid object for thermal properties
    power: Watt
        Power, a source term.
    x: Meter
        Boundaries crossing the r-axis.
    z: Meter
        Boundaries crossing the z-axis.
    contacts: Sequence[WPerM2K]
        indices and transfer coefficients for Fuel-Clad contacts

    Returns
    -------
    dT/dt: CPerS
        for the heat equation in the material. r diffusion is assumed.
    """
    dx = np.diff(x)
    dz = np.diff(z)
    r_areas, z_areas, volumes = cylindrical_areas_volumes(r=x, z=z)
    return generic_2d_diffusion(
        T=T, T_walls=T_walls, s=material, power=power, dr=(dx, dz),
        areas=(r_areas,), volumes=volumes, contacts=contacts)


def generic_2d_diffusion(T: Celsius, T_walls: Walls, s: Solid, power: Watt,
                         dr: Sequence[Meter], areas: Sequence[Meter2],
                         volumes: Meter3,
                         contacts: Sequence[WPerM2K]) -> CPerS:
    """
    Calculate the temporal derivative of temperatures
    according to the heat equation.

    .. note:: This is the generic 2D function. Given correct areas and volumes,
        An azimuthal symmetry is also possibly portrayed, albeit still using a
        cartesian version of the heat equation.

    Parameters
    ----------
    T: Celsius
    T_walls: Walls
        Left, Right, Top, Bottom walls
    s: Solid
    power: Watt
    dr: Sequence[Meter]
    areas: Sequence[Meter2]
    volumes: Meter3
    contacts: WPerM2K

    Returns
    -------
    dT/dt: CPerS
    """
    # noinspection PyCallingNonCallable
    resistances = _resistances(dr=dr, contacts=contacts, k=s.conductivity)
    fluxes = _fluxes(T=T, T_walls=T_walls, resistances=resistances)
    flow = np.sum(_flows(fluxes, areas), axis=0)

    return (flow + power) / (s.density * s.specific_heat * volumes)


stacks = x_stack, z_stack = np.column_stack, np.row_stack
diffs = x_diff, z_diff = np.diff, partial(np.diff, axis=0)
pair_means = x_pair_mean, z_pair_mean = pair_mean, partial(pair_mean, axis=0)


def _x_bulk(a): return a[:, 1:-1]


def _z_bulk(a): return a[1:-1, :]


bulks = _x_bulk, _z_bulk
left = dict(indices=0, axis=1)
right = dict(indices=-1, axis=1)
top = dict(indices=0, axis=0)
bottom = dict(indices=-1, axis=0)
edges = (left, right, top, bottom)


@cached(cache={}, key=lambda *args, **kwargs: pickle.dumps((args, kwargs)))
def _resistances(dr: Sequence[Meter], contacts: Sequence[WPerM2K], k: WPerMK
                 ) -> Sequence[M2KPerW]:
    """
    Compute heat resistance at each cell face, given medium conductivity
    at each cell and contact heat transfer coefficient at each cell face.

    Parameters
    ----------
    dr: Sequence[Meter]
        Distances in the problem dimensionality (probably (dx, dz))
    contacts: Sequence[WPerM2K]
        Local heat transfer coefficient at each cell face ("contact")
    k: WPerMK
        Material conductivity

    Returns
    -------
    r: Sequence[M2KPerW]
        Heat transfer resistance at each cell face, divided into directions.
    """
    # noinspection PyTypeChecker
    rs = [ds / k for ds in np.meshgrid(*dr)]
    cs = contacts

    faces = (r.take(**edge) / 2 + 1 / c.take(**edge) for edge, r, c
             in zip(edges, interleave(rs, rs), interleave(cs, cs)))

    edges_in_bulk = (p_mean(r) + 1 / bulk(c) for r, c, p_mean, bulk
                     in zip(rs, cs, pair_means, bulks))

    return [stack((face1, bulk_faces, face2))
            for bulk_faces, (face1, face2), stack
            in zip(edges_in_bulk, chunked(faces, 2), stacks)]


def _fluxes(T: Celsius, T_walls: Walls, resistances: Sequence[M2KPerW]
            ) -> Sequence[WPerM2]:
    """
    Given Temperatures at medium and boundaries, and resistances
    Parameters
    ----------
    T: Celsius
        Medium temperatures.
    T_walls: Walls (Celsius)
        Wall temperatures.
    resistances: Sequence[M2KPerW]

    Returns
    -------
    fluxes: Sequence[WPerM2]
        Heat flux at each cell face, divided into directions (x, z)
    """
    return [diff(stack((wall1, T, wall2))) / r
            for diff, stack, (wall1, wall2), r
            in zip(diffs, stacks, (T_walls.x, T_walls.z), resistances)]


def _flows(fluxes: Sequence[WPerM2], areas: Sequence[Meter2]) -> Sequence[Watt]:
    """
    Given fluxes and cross-sections of cell faces (assumed same lengths,
    normals align with fluxes), compute the overall flow into the cell.

    Parameters
    ----------
    fluxes: Sequence[WPerM2]
        flux (WPerM2) in each direction. Each entry in the sequence
        should be an array_like of all cells in the direction matching areas.
    areas: Sequence[Meter2]
        areas of cell faces in each direction.

    Returns
    -------
    flows: Sequence[Watt]
        Energy flow (Watt) in each direction (same length as inputs),
        where each entry's length is the number of cells in that dimension
        (that is, 1 less than areas' entries).
    """
    return [diff(flux * a) for diff, flux, a, in zip(diffs, fluxes, areas)]


def _fill(val, shape): return np.full(shape, val)


wall_or_default = dataclass_map(Walls, if_is)
in_par_walls = dataclass_map(Walls, in_parallel)
fill_solid = dataclass_map(Solid, _fill)


class Fuel(Calculation):
    r"""Represents a solid component in which heat is generated and/or transferred.
    An internal volumetric heat source may be supplied and heat is conducted in up to
    two dimensions.

    Geometry is introduced through 2 explicit dimensions, termed :math:`x` and :math:`z`
    and an auxiliary dimension termed :math:`y` in which symmetry is assumed. Providing
    ``heat_func`` may allow one to support generally many continuous 2D structures.
    Explicitly supported geometries are:

    - Rectangular (:func:`x_diffusion` or :func:`xz_diffusion`).
    - Cylindrical (:func:`r_diffusion` or :func:`rz_diffusion`) - in this case
      the radial dimension is :math:`x`, and :math:`y` represents an azimuthal symmetry.

    See more in :ref:`Discretizing the Heat Equation` for the according Cartesian and
    Polar discretizations of the heat equation.

    Boundary conditions are supplied through extraneous temperature and conductance
    pairs, which are termed ``left``, ``right``, ``top``, and ``bottom``, and couple
    to the 2D mesh :math:`z \times x` accordingly. Be aware that there is an
    inherent difference between (``left``, ``right``) and (``top``, ``bottom``).
    While this calculation provides its calculated wall temperatures at the
    (``left``, ``right``) walls (so that other calculations may transfer heat from it),
    the (``top``, ``bottom``) wall temperatures are calculated but not provided.

    **Defaults:**

    - Geometry is rectangular
    - ``Contacts`` are assumed infinitely conductive.
    - ``Meat_indices`` describes the entire plate as meat.
    - ``heat_func`` describes x-diffusion only.
    - ``T_wall_func`` assumes zero-inertia at wall.
    - ``power_shape`` is assumed uniform.
    """

    def __init__(self, z_boundaries: Meter, x_boundaries: Meter,
                 material: Solid, y_length: Meter,
                 heat_func: Callable = x_diffusion,
                 T_wall_func: Callable = wall_temperature,
                 x_contacts: WPerM2K = None, z_contacts: WPerM2K = None,
                 meat_indices: Array2D = None,
                 power_shape: Array2D = np.ones(1), name: str = 'Fuel'):
        """
        Parameters
        ----------
        z_boundaries: Meter
            Boundaries crossing the z-axis
        x_boundaries: Meter
            Boundaries crossing the x-axis
        material: Solid
            Bulk properties. Can be of shape: (z_cells, x_cells) matrix
        heat_func: Callable
            A function computing the temporal derivative of bulk temperatures.
        T_wall_func: Callable
            A function computing wall temperatures.
        x_contacts: WPerM2K
            Contact conductivity of x_axis contacts. This means its shape
            should be (z_cells, x_cells + 1), i.e. including outer boundaries.
        z_contacts: WPerM2K
            Contact conductivity of z_axis contacts. This means its shape
            should be (z_cells + 1, x_cells), i.e. including outer boundaries.
        meat_indices: Array2D
            Meat placements, specifically where power would be
            deposited (at meat_indices = 1).
        y_length : Meter
            Length of the symmetric dimension.
        power_shape: Array2D
            Shape of power distribution over the fuel meat
        """
        self.name = name
        # Geometry
        self.x_bounds = x_boundaries
        self.y_length = y_length
        self.z_bounds = z_boundaries
        self.dx = np.diff(x_boundaries)
        self.dz = np.diff(z_boundaries)
        self.x_centers = pair_mean(x_boundaries)
        self.z_centers = pair_mean(z_boundaries)
        self.n, self.m = n, m = len(self.dx), len(self.dz)
        self.shape = shape = np.array((m, n))
        self._indices = np.arange(m * n).reshape(shape)

        # Physical Properties
        self.material = fill_solid(material, shape=shape)
        self.x_contacts = if_is(x_contacts, np.full(shape + [0, 1], np.inf))
        self.z_contacts = if_is(z_contacts, np.full(shape + [1, 0], np.inf))
        self.meat = if_is(meat_indices, np.ones(shape))

        last_contacts = Walls(left=self.x_contacts[:, 0],
                              right=self.x_contacts[:, -1],
                              top=self.z_contacts[0, :],
                              bottom=self.z_contacts[-1, :])
        k = self.material.conductivity
        to_walls_conductivity = Walls(left=2 * k[:, 0] / self.dx[0],
                                      right=2 * k[:, -1] / self.dx[-1],
                                      top=2 * k[0, :] / self.dz[0],
                                      bottom=2 * k[-1, :] / self.dz[-1])
        self.h_to_wall = in_par_walls(to_walls_conductivity, last_contacts)
        self.power_shape = power_shape.flatten()

        # Equations
        self.heat_eq = partial(heat_func,
                               material=self.material,
                               x=self.x_bounds, z=self.z_bounds,
                               contacts=(self.x_contacts, self.z_contacts),
                               y=self.y_length)
        self.walls_eq = dataclass_map(Walls, T_wall_func)

        # Variables
        self._vars = dict(
            T=slice(0, n * m),
            T_wall_left=slice(n * m, (1 + n) * m),
            T_wall_right=slice((1 + n) * m, (2 + n) * m),
            )
        logger.log(STREAM_DEBUG, f"New {self.name}, {m} by {n}")

    @unpacked
    def calculate(self, variables: np.array, *, power: Watt,
                  T_left: Celsius = None, T_right: Celsius = None,
                  T_top: Celsius = None, T_bottom: Celsius = None,
                  h_left: WPerM2K = None, h_right: WPerM2K = None,
                  h_top: WPerM2K = None, h_bottom: WPerM2K = None,
                  ) -> Array1D:
        r"""
        Calculating temperatures inside fuel and at the edges.
        If wall temperatures are not given

        Parameters
        ----------
        variables: Celsius
            Temperatures inside (and on edges) of Fuel element
        power: Watt
            Generated power at meat.
        T_left: Celsius
            Temperature just outside the left edge
        T_right: Celsius
            Temperature just outside the right edge
        T_top: Celsius
            Temperature just outside the top edge
        T_bottom: Celsius
            Temperature just outside the bottom edge
        h_left: WPerM2K
            Left wall conductance
        h_right: WPerM2K
            Right wall conductance
        h_top: WPerM2K
            Top wall conductance
        h_bottom: WPerM2K
            Bottom wall conductance

        Returns
        -------
        Functional output: CPerS or C
            Temporal derivative (for inner temperatures) and error for walls
        """
        out = np.empty(len(self))
        _v = self._vars
        T = variables[_v["T"]].reshape(self.shape)
        power_mat = np.zeros(shape=self.shape)
        power_mat[self.meat == 1] = power * self.power_shape

        T_last_cell = Walls(left=T[:, 0], right=T[:, -1], top=T[0, :], bottom=T[-1, :])
        h_extraneous = wall_or_default(
            Walls(left=h_left, right=h_right, top=h_top, bottom=h_bottom)
            )
        T_extraneous = wall_or_default(
            Walls(left=T_left, right=T_right, top=T_top, bottom=T_bottom), T_last_cell
            )

        T_walls = self.walls_eq(T_extraneous, T_last_cell, h_extraneous, self.h_to_wall)

        out[_v["T"]] = self.heat_eq(T=T, T_walls=T_walls, power=power_mat).flatten()
        out[_v["T_wall_left"]] = T_walls.left - variables[_v["T_wall_left"]]
        out[_v["T_wall_right"]] = T_walls.right - variables[_v["T_wall_right"]]
        return out

    def indices(self, variable: Name, asking=None) -> Place:
        return dict(
            T_left=self._vars["T_wall_right"],
            T_right=self._vars["T_wall_left"],
            T_top=self._indices[-1, :],
            T_bottom=self._indices[0, :],
            T=self._indices[self.meat == 1],
            )[variable]

    @property
    def mass_vector(self) -> Sequence[bool]:
        mass = np.ones(len(self), dtype=bool)
        mass[self.n * self.m:] = False
        return mass

    def __len__(self) -> int:
        """N (columns) by M (rows) temperature cells + 2*M wall temperatures"""
        return (2 + self.n) * self.m

    @property
    def variables(self) -> dict[str, Place]: return self._vars

    def save(self, vector: Sequence[float], **_) -> CalcState:
        state = super().save(vector, **_)
        state['T'] = state['T'].reshape(self.shape)
        return state

    def load(self, state: CalcState) -> Array1D:
        return super().load(state=state | {'T': np.asarray(state['T']).flatten()})

    load.__doc__ = Calculation.load.__doc__
