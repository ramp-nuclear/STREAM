from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from stream.units import DecayHeatFunction, MeVPerFission, MeVPerS, PerS, Second

__all__ = ["read", "Standard", "Source", "contribution", "fp_inner_"]

STANDARDS = Path(__file__).parent / "standards"


class Standard(Enum):
    ANS14 = "ans14"
    ANS73 = "ans73"
    JAERI91 = "jaeri91"


class Source(Enum):
    U235 = "U235"
    U235_beta = "U235_beta"
    U235_gamma = "U235_gamma"
    U238 = "U238"
    U238_gamma = "U238_gamma"


def _to_filepath(standard: Standard, source: Source) -> Path:
    return STANDARDS / f"{source.value}_{standard.value}.csv"


def read(standard: Standard, source: Source) -> pd.DataFrame:
    filepath = _to_filepath(standard, source)
    try:
        return pd.read_csv(filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        raise FileNotFoundError(f"Does not vendor {source.name} from {standard.name}") from None


def write(standard: Standard, source: Source, data: pd.DataFrame):
    data.to_csv(_to_filepath(standard, source), index=False)


def _process_standard(standard: Standard, source: Source) -> tuple[PerS, MeVPerS]:
    df = read(standard, source)
    alpha: MeVPerS = df.alpha.to_numpy()
    lamda: PerS = df.lamda.to_numpy()
    return alpha, lamda


def contribution(standard: Standard, source: Source) -> DecayHeatFunction:
    r"""A DecayHeatFunction for fission products` decay, given a :class:`~Standard`, :class:`~Source`.

    Parameters
    ----------
    standard : Standard
        The decay heat standard from which the data should be taken.
    source : Source
        Chosen source.

    Returns
    -------
    DecayHeatFunction
        Whose return value has units of MeV / fission event

    Examples
    --------
    >>> import pytest
    >>> try:
    ...     fp = contribution(Standard.ANS14, Source.U235)
    ... except FileNotFoundError:
    ...     pytest.skip("This doctest requires the standards which are not there")
    >>> Q: MeV = 200 # Approx. total recoverable energy associated with one fission
    >>> fp(t=0., T=np.inf) / Q
    array([0.06728006])
    >>> fp(t=1., T=0.)
    array([0.])
    """
    alpha, lamda = _process_standard(standard, source)

    def _contribution(t: Second, T: Second = np.inf) -> MeVPerFission:
        return fp_inner_(t, T, lamda, alpha)

    return _contribution


def fp_inner_(
    t: Second,
    T: Second,
    lamda: PerS,
    alpha: MeVPerS,
) -> MeVPerFission:
    r"""Implements :eq:`FP`, the decay heat stemming from fission products decays.

    Parameters
    ----------
    t: Second
        Times after shutdown
    T: Second
        Operation time
    lamda: PerS
        Decay rates
    alpha: MeVPerS
        Fractional fission power contributions

    Returns
    -------
    MeVPerFission
        At times `t` after shutdown, :math:`F(t,T)`

    Examples
    --------
    >>> fp_inner_(
    ... t=0., T=np.inf, lamda=np.ones(1), alpha=np.ones(1))
    array([1.])
    """

    return (alpha / lamda) * (-np.expm1(-lamda * T)) @ np.exp(-np.outer(lamda, t))
