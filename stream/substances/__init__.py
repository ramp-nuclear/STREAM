r"""
This package contains several substances (read: correlations or effective
state equations).

For example:

>>> from numpy import array as ar
>>> light_water.density(T=ar([20, 100]))
array([997.95471288, 959.13959928])

References
----------
.. [#simantov] A. Crabtree and M. Siman-Tov,
 "Thermophysical Properties of Saturated Light and Heavy Water for Advanced Neutron Source Applications",
 ORNL/TM-12322, 1993.
"""
from .liquid import Liquid, LiquidFuncs
from .heavy_water import heavy_water
from .light_water import light_water
