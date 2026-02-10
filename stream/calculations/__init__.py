r"""Includes several predefined thermohydraulic calculations.
These calculations are derived from :class:`~.Calculation`.

Currently, included here are mostly calculations regarding the incompressible
1D coolant flow scheme. Other notable members are :class:`.Fuel`,
the heat diffusion descriptor, and :class:`.PointKinetics`,
the point-reactor neutronics descriptor.
"""

from .channel import Channel, ChannelAndContacts
from .flapper import Flapper
from .heat_diffusion import Fuel, Solid
from .ideal import *
from .kirchhoff import Junction, Kirchhoff, KirchhoffWDerivatives
from .point_kinetics import PointKinetics, PointKineticsWInput
