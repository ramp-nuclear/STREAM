"""Laminar Single Phase Heat Transfer Coefficient"""
import numpy as np

from stream.physical_models.dimensionless import Re_mdot, Pr
from stream.substances import Liquid
from stream.units import Value, WPerM2K, Meter, KgPerS, Meter2

#: For parallel plates in laminar flow, the Nusselt number is constant,
#: depending on the boundary conditions. See
#: J. H. Lienhard IV, J. H. Lienhard V, "A HEAT TRANSFER TEXTBOOK",
#: 3rd ed., ch. 7.5, p. 373
FIXED_TEMPS, FIXED_FLUXES, ONE_FIXED_FLUX_ONE_ADIABATIC = 7.541, 8.235, 5.385


def constant_Nusselt_h_spl(*, coolant: Liquid, Dh: Meter, nu: float = FIXED_FLUXES,
                           **_) -> WPerM2K:
    r"""A :class:`~.SinglePhaseLiquidHTC` for constant :func:`~.Nu` Number: math:`h = \text{Nu}k/D_h`

    Parameters
    ----------
    coolant: Liquid
        Properties of the coolant.
    Dh: Meter
        Hydraulic diameter.
    nu: float
        Nusselt Number. Default is :const:`FIXED_FLUXES`

    Returns
    -------
    Watts per Meter squared degrees Kelvin
        Heat transfer coefficient

    """
    return coolant.conductivity * nu / Dh


def Marco_Han_Nusselt(aspect_ratio: float) -> Value:
    """
    For the case of laminar flow in rectangular ducts, an approximation to an analytical solution by Marco and Han is
    provided (reported error is +-0.03%). [#kakac]_
    This is useful for ducts of case 1, where the channel is heated from all sides.

    Parameters
    ----------
    aspect_ratio: float
        channel_depth / channel_width (less than 1)

    Returns
    -------
    Nu: float
        The Nusselt number (Nu)

    Examples
    --------
    >>> Marco_Han_Nusselt(0.)
    8.235
    >>> Marco_Han_Nusselt(0.2)
    5.991134842079999

    References
    ----------
    .. [#kakac] S. Kakac, R. K. Shah, W. Wung,
       "Handbook of Single-Phase Convective Heat-transfer" ch. 3
    """
    assert 0.0 <= aspect_ratio <= 1.0, f"{aspect_ratio = } must be non-negative and less than 1"
    return FIXED_FLUXES * (
            1.0
            - 2.0421 * aspect_ratio
            + 3.853 * aspect_ratio ** 2
            - 2.4765 * aspect_ratio ** 3
            + 1.0578 * aspect_ratio ** 4
            - 0.1861 * aspect_ratio ** 5
    )


def two_sided_heating_nusselt(aspect_ratio: Value, nu0: float = FIXED_FLUXES) -> Value:
    """Nusselt number for channels heated from two sides.

    For the case of laminar flow in rectangular ducts, an approximation to an analytical solution is given
    in case 3 of Kakac [#kakac_two_side]_

    Parameters
    ----------
    aspect_ratio: float
        channel_depth / channel_width (0<=x<=1)
    nu0: float
        The nusselt value for an infinitely wide channel compared to its depth.
        Defaults to the fully developed value.

    Returns
    -------
    Nu: float
        The Nusselt number (Nu)

    Examples
    --------
    >>> two_sided_heating_nusselt(0.)
    8.235

    The answer should conform with the results Shah & London give from Schmidt in their table 44 [#Shah_two_side]_

    >>> alpha = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    >>> vtable44 = np.array([8.235, 7.248, 6.561, 5.997, 5.555, 5.203, 4.662, 4.094])
    >>> np.allclose(two_sided_heating_nusselt(alpha), vtable44, rtol=1.1e-2)
    True


    References
    ----------
    .. [#kakac_two_side] S. Kakac, R. K. Shah, W. Wung, "Handbook of Single-Phase Convective Heat-transfer" ch. 3
    .. [#Shah_two_side] R.K. Shah and A.L. London, Laminar flow forced convection in ducts,
        A source book for compact heat exchanger analytical data, Academic Press 1978,
        New York, San Francisco, London,
        Chapter VII, Rectangular ducts, Section 3, Page 206, Table 44.
    """
    return nu0 * (
            1.
            - 1.4122 * aspect_ratio
            + 2.3473 * aspect_ratio ** 2
            - 2.8983 * aspect_ratio ** 3
            + 2.0629 * aspect_ratio ** 4
            - 0.6077 * aspect_ratio ** 5
    )


def fully_developed_laminar_h_spl(*, coolant: Liquid, Dh: Meter, aspect_ratio: float,
                                  **_) -> WPerM2K:
    """A :class:`~.SinglePhaseLiquidHTC` relying on :func:`Marco_Han_Nusselt`

    Parameters
    ----------
    coolant: Liquid
        Properties of the coolant.
    Dh: Meter
        Hydraulic diameter.
    aspect_ratio: float in [0,1]
        The aspect ratio of the channel.

    Returns
    -------
    Watts per Meter squared degrees Kelvin
        Heat transfer coefficient

    """
    return coolant.conductivity * two_sided_heating_nusselt(aspect_ratio) / Dh


def _nusselt_coefficient_developing(x: Value) -> Value:
    r"""Developing flow nusselt number at infinitely wide rectangular channels.
    Taken from Shah & London [#Shah_develop]_ , who recommend them for being convenient apporixmations.

    Also appears in TERMIC [#TERMIC_dev]_.

    According to Shah & London, the errors compared to their Table 34 of directly computed values should be
    less than 0.2% for :math:`x \leq 6\cdot 10^{-5}`, 0.6% for :math:`6\cdot 10{^-5} < x \leq 2\cdot 10^{-4}`,
    0.8% for :math:`2\cdot 10^{-4} < x \leq 10^{-3}` and 0.6% for :math:`x > 10^{-3}`.
    It turns out to be slightly larger than those values overall, and about 10x times higher for
    :math:`2\cdot 10^{-4} < x \leq 10^{-3}`

    This is tested in the examples for this function.

    Examples
    --------
    >>> x = _xstar_table34
    >>> v = _nusselt_coefficient_developing(x)
    >>> r1 = x <= 6e-5
    >>> r2 = (6e-5 < x) & (x <= 2e-4)
    >>> r3 = (2e-4 < x) & (x <= 1e-3)
    >>> r4 = x > 1e-3
    >>> areclose = [np.allclose(v[m], _nuxh_table34[m], rtol=rtol)  # type: ignore
    ...             for m, rtol in zip((r1, r2, r3, r4), (3e-3, 7e-3, 6.2e-2, 7e-3))]
    >>> all(areclose)
    True


    References
    ----------
    .. [#Shah_develop] R.K. Shah and A.L. London, Laminar flow forced convection in ducts,
        A source book for compact heat exchanger analytical data, Academic Press 1978,
        New York, San Francisco, London,
        Chapter VI, Parallel plates, Section C, Page 182, Equations 317-319.
    .. [#TERMIC_dev] P. Abbate, TERMIC V4.1: A Program for
        the Thermal-Hydraulic Analysis of a MTR Core in Forced Convection,
        Vol 1: Models and Correlations, Rev 3, Pages 14-15,
        Nuclear Engineering Division, INVAP, December 2003

    """
    return np.where(x <= 2e-4,
                    1.49 * x ** (-1 / 3),
                    np.where(x <= 1e-3,
                             (1.49 * (x ** (-1 / 3))) - 0.4,
                             8.235 + 8.68 * np.exp(-164 * x) * ((1e3 * x) ** -0.506)
                             )
                    )


_xstar_table34 = np.array(sum([[j * 10 ** (-i) for j in (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9)]
                               for i in [6, 5, 4, 3, 2]], start=[]) + [0.1, 0.15, 0.2])
_nuxh_table34 = np.array([148.773,
                          129.944,
                          118.049,
                          103.110,
                          93.673,
                          86.954,
                          81.824,
                          77.724,
                          74.339,
                          71.477,
                          69.011,
                          60.292,
                          54.787,
                          47.880,
                          43.521,
                          40.419,
                          38.054,
                          36.165,
                          34.607,
                          33.290,
                          32.153,
                          28.154,
                          25.636,
                          22.488,
                          20.512,
                          19.133,
                          18.050,
                          17.205,
                          16.511,
                          15.928,
                          15.427,
                          13.681,
                          12.604,
                          11.299,
                          10.516,
                          9.9878,
                          9.6085,
                          9.3249,
                          9.1073,
                          8.9374,
                          8.8031,
                          8.4393,
                          8.3107,
                          8.2458,
                          8.2368,
                          8.2355,
                          ]
                         + 7 * [8.2353])


def _worsoe_schmidt_leveque_type(x):
    r"""Worsoe-Schmidt's Leveque-type similarity solution for the Nusselt number at low xstar.

    Shah [#Shah_worsoe]_ computed those for :math:`10^{-6} \leq x \leq 10^{-3}`,
    and they should be up to 5 digits accurate with the proper solution for
    :math:`5\cdot 10^{-4} \leq x \leq 5\cdot 10^{-3}` and at most 0.08% off for
    :math:`5\cdot 10^{-5} \leq x \leq 10^{-2}`.

    Examples
    --------
    Table 34 (for :math:`x \leq 10^{-4}`) was supposed to be computed using this equation,
    so it should be accurate there.

    >>> m = _xstar_table34 <= 1e-4
    >>> np.allclose(_worsoe_schmidt_leveque_type(_xstar_table34[m]), _nuxh_table34[m], rtol=1e-4)
    True

    References
    ----------
    .. [#Shah_worsoe] R.K. Shah and A.L. London, Laminar flow forced convection in ducts,
        A source book for compact heat exchanger analytical data, Academic Press 1978,
        New York, San Francisco, London,
        Chapter VI, Parallel plates, Section C, Page 182, Equation 316

    """
    xt = x ** (1 / 3)
    repv = (0.670960978 * xt + 0.159064137 * (xt ** 2) + 0.12012 * x + 0.12495 * (xt ** 4) + 0.15602 * (xt ** 5)
            + 0.22176 * (x ** 2) + 0.34932 * (xt ** 7) - 4 * x)
    return 1 / repv


def _nusselt_coefficient_interp_developing(x: Value) -> Value:
    r"""Nusselt coefficient for developing heated flow and fully develop hydraulics, by interpolation of data.

    In Shah & London [#Shah_interp]_, table 34 includes pre-computed data for :math:`10^{-6} \leq x \leq 0.2`.
    These are supposed to be accurate representations of an analytic solution for :math:`x > 10^{-4}`,
    and of Equation 316 (an approximate solution, see :func:`._worsoe_schmidt_leveque_type`) otherwise.

    This function is more exact than :func:`._nusselt_coefficient_developing` but slightly slower.
    The other one is used by default in our code, and its error is checked against the tabulated values directly in
    its tests.

    See Also
    --------
    :func:`._nusselt_coefficient_developing`
        An analytic approximation to this function.
    :func:`._worsoe_schmidt_leveque_type`
        The underlying function for vanishingly small x.


    References
    ----------
    .. [#Shah_interp] R.K. Shah and A.L. London, Laminar flow forced convection in ducts,
        A source book for compact heat exchanger analytical data, Academic Press 1978,
        New York, San Francisco, London,
        Chapter VI, Parallel plates, Section C, Page 181, Table 34

    """
    return np.where(x < _xstar_table34[0],
                    _worsoe_schmidt_leveque_type(x),
                    np.interp(x, _xstar_table34, _nuxh_table34, right=8.2353))


def developing_laminar_h_spl(*, coolant: Liquid, mdot: KgPerS, A: Meter2,
                             Dh: Meter, develop_length: Meter, aspect_ratio: float,
                             **_
                             ) -> WPerM2K:
    """A :class:`~.SinglePhaseLiquidHTC` relying on A Nusselt number definition
    for thermally developing flow. [#Shah]_

    This includes a correction for the channel being finite in size.
    These equations are valid for Reynolds<2100. [#TERMIC]_

    References
    ----------
    .. [#Shah] R.K. Shah and A.L. London, Laminar flow forced convection in ducts,
        A source book for compact heat exchanger analytical data, Academic Press 1978,
        New York, San Francisco, London,
        Chapter VI, Parallel plates, Section C, Page 181, Table 34.
    .. [#TERMIC] P. Abbate, TERMIC V4.1: A Program for
        the Thermal-Hydraulic Analysis of a MTR Core in Forced Convection,
        Vol 1: Models and Correlations, Rev 3, Pages 14-15,
        Nuclear Engineering Division, INVAP, December 2003

    Parameters
    ----------
    coolant: Liquid
        Properties of the coolant.
    mdot: KgPerS
        Mass flow rate.
    A: Meter2
        Flow area.
    Dh: Meter
        Hydraulic diameter.
    develop_length: Meter
        Distance from channel entrance.
    aspect_ratio: float in [0,1]
        The aspect ratio of the channel.

    Returns
    -------
    Watts per Meter squared degrees Kelvin
        Heat transfer coefficient

    """
    re = Re_mdot(mdot=mdot, A=A, L=Dh, mu=coolant.viscosity)
    pr = Pr(coolant.specific_heat, coolant.viscosity, coolant.conductivity)
    x_star = develop_length / Dh / re / pr / (6 - 5 * np.exp(-0.75 * aspect_ratio / 0.3257))
    nudev = _nusselt_coefficient_developing(x_star)
    nusselt = two_sided_heating_nusselt(aspect_ratio, nudev)
    return nusselt * coolant.conductivity / Dh
