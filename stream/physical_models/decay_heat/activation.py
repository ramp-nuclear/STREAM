import numpy as np

from stream.units import (DecayHeatFunction, PerS, Second, Value)


def _activation(t: Second, T: Second, lamda: PerS) -> Value:
    l = np.atleast_1d(lamda)
    return (-np.expm1(-l * T)) @ np.exp(-np.outer(l, t))


def profile(lamda: PerS) -> DecayHeatFunction:
    r"""

    Parameters
    ----------
    lamda: PerS
        Activated isotope decay rate

    Returns
    -------
    DecayHeatFunction
        Decay profile from initially activated isotope

    Examples
    --------
    >>> f = profile(lamda_U239 := 4.91e-04)
    >>> f(0, np.inf)
    array([1.])
    >>> f(1000, np.inf)
    array([0.61201407])
    >>> f(0, 0)
    array([0.])
    """

    def _profile(t: Second, T: Second = np.inf) -> Value:
        return _activation(t, T, lamda)

    return _profile


def double_decay_profile(lamda1: PerS, lamda2: PerS) -> DecayHeatFunction:
    r"""

    Parameters
    ----------
    lamda1: PerS
        Activated isotope decay rate
    lamda2: PerS
        Interim isotope decay rate

    Returns
    -------
    DecayHeatFunction
        Double decay profile from initially activated isotope

    Examples
    --------
    >>> f = double_decay_profile(lamda_U239 := 4.91e-04, lamda_Np := 3.41e-06)
    >>> f(0, np.inf)
    array([1.])
    >>> f(1000, np.inf)
    array([0.99928541])
    >>> f(0, 0)
    array([0.])
    """

    def _profile(t: Second, T: Second = np.inf) -> Value:
        charge1 = _activation(t, T, lamda1)
        charge2 = _activation(t, T, lamda2)
        return (lamda1 * charge2 - lamda2 * charge1) / (lamda1 - lamda2)

    return _profile
