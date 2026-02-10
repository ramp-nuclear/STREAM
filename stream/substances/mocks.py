"""
Mock Substances, whose properties are mostly just 1's
"""
from functools import partial

import numpy as np

from stream.calculations.heat_diffusion import Solid
from stream.substances import Liquid, LiquidFuncs
from stream.units import Celsius, Pascal
from stream.utilities import uniform_dataclass

mock_liquid_funcs = uniform_dataclass(LiquidFuncs, np.ones_like)
mock_liquid = uniform_dataclass(Liquid, 1.0)
mock_solid = uniform_dataclass(Solid, 1.0)


def constant_LiquidFuncs(fluid: LiquidFuncs, T: Celsius, p: Pascal) -> LiquidFuncs:
    r"""Create a constant-valued :class:`.LiquidFuncs` based on a given
    instance at a (T, p) phase-space point.

    T, p are taken as at_least1d to allow compatibility with vector operations.

    Parameters
    ----------
    fluid: LiquidFuncs
        Model fluid to be evaluated.
    T: Celsius
        Temperature.
    p: Pascal
        Pressure.

    Returns
    -------
    const_fluid: LiquidFuncs
        A LiquidFuncs which always returns the same value for each parameter.

    Examples
    --------
    >>> from stream.substances import light_water
    >>> clf = constant_LiquidFuncs(light_water, T=20., p=1e5)
    >>> clf.density(20.) == clf.density(80.) == light_water.density(20.)
    True
    >>> clf.density(np.zeros(4)) / clf.density(20)
    array([1., 1., 1., 1.])
    """
    evaluated = fluid.to_properties(np.atleast_1d(T), np.atleast_1d(p))
    # noinspection PyUnresolvedReferences
    return LiquidFuncs(
        **{
            fld: partial(np.full_like, fill_value=getattr(evaluated, fld), dtype=float)
            for fld in LiquidFuncs.__dataclass_fields__
        }
    )
