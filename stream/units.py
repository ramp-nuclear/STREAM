"""
Physical, abstract and data units for type hinting.


Naming convention
~~~~~~~~~~~~~~~~~
Whenever a unit is on its own, its full name is used.
A number following a unit denotes its power (e.g. Meter3 is cubic meters).
The word "Per" separates the unit into  **[numerator]Per[denominator]**.

.. note::
    Physical Units use mostly the SI units.

    Kelvin and Celsius are used interchangeably whenever they appear
    at the denominator.
"""

from typing import Callable

import numpy as np
from scipy.constants import g as standard_acceleration


# Data Sets
Array1D = np.ndarray
Array2D = np.ndarray
Array3D = np.ndarray
Array = np.ndarray
Value = float | np.ndarray

# Physical Units
Celsius = Value
CPerS = Value
Fahrenheit = Value
Joule = Value
JPerKg = Value
JPerKgK = Value
JPerKgK2 = Value
Kelvin = Value
Kg = Value
KgPerM3 = Value
KgPerM7 = Value
KgPerM4S = Value
KgPerM3C = Value
KgPerS = Value
KgPerS2 = Value
KgPerSM2 = Value
M2KPerW = Value
M3PerS = Value
Meter = Value
Meter2 = Value
Meter3 = Value
MeV = Value
MeVPerFission = Value
MeVPerS = Value
MPerS = Value
MPerS2 = Value
Newton = Value
NPerM = Value
PaPerM = Value
PaS = Value
Pascal = Value
PaSPerKg = Value
Pcm = Value
PerC = Value
PerKgM = Value
PerM = Value
PerMS = Value
PerS = Value
Radians = Value
Second = Value
Watt = Value
WPerK = Value
WPerM = Value
WPerM2 = Value
WPerM2K = Value
WPerMK = Value
WPerMK2 = Value
WPerS = Value

# Abstract Units
Name = str
Place = int | slice | np.ndarray
FunctionOfTime = Callable[[Second], Value] | Value
Functional = Callable[[Array1D, Second], Array1D]
DecayHeatFunction = Callable[[Second, Second], Value]

# SI labels
meter = second = 1

# Conversions to SI
cm = 1e-2 * meter
mm = 1e-3 * meter
hour = 3600 * second
gram = 1e-3
cm3 = cm ** 3

# Factors
mega = 1e6
kilo = 1e3
milli = 1e-3
micro = 1e-6
pcm = 1e-5

# Constants
g: MPerS2 = standard_acceleration
