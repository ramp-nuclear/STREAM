from collections.abc import Sequence
from enum import Enum
from functools import partial, reduce
from typing import Literal, Protocol

import numpy as np

from stream.physical_models.dimensionless import Re_mdot, flow_regimes
from stream.physical_models.heat_transfer_coefficient.laminar import (
    constant_Nusselt_h_spl,
    developing_laminar_h_spl,
    fully_developed_laminar_h_spl,
)
from stream.physical_models.heat_transfer_coefficient.natural_convection import (
    Elenbaas_h_spl,
)
from stream.physical_models.heat_transfer_coefficient.turbulent import (
    Dittus_Boelter_h_spl,
)
from stream.substances import Liquid, LiquidFuncs
from stream.units import Celsius, KgPerS, Meter, Meter2, MPerS2, Pascal, Value, WPerM2K
from stream.units import g as local_gravity
from stream.utilities import lin_interp


class SinglePhaseLiquidHTCExArgs(Protocol):
    def __call__(
        self,
        *,
        coolant: Liquid,
        mdot: KgPerS,
        Dh: Meter,
        A: Meter2,
        T_cool: Celsius,
        T_wall: Celsius,
        coolant_funcs: LiquidFuncs,
        pressure: Pascal,
        # The following are here to weaken the constraint only, use with caution.
        h_spl=None,
        q_scb=None,
        film=None,
        incipience=None,
        partial_scb=None,
        develop_length=None,
        **_,
    ) -> WPerM2K:
        """Same as :class:`~.SinglePhaseLiquidHTC` except it accepts any additional
        keyword parameters.
        """
        ...


def regime_dependent_h_spl(
    coolant: Liquid,
    mdot: KgPerS,
    Dh: Meter,
    A: Meter2,
    T_cool: Celsius,
    T_wall: Celsius,
    re_bounds: tuple[Value, Value],
    coolant_funcs: LiquidFuncs,
    Lh: Meter,
    g: MPerS2 = local_gravity,
    laminar: SinglePhaseLiquidHTCExArgs = developing_laminar_h_spl,
    turbulent: SinglePhaseLiquidHTCExArgs = Dittus_Boelter_h_spl,
    natural: SinglePhaseLiquidHTCExArgs = Elenbaas_h_spl,
    **kwargs,
) -> WPerM2K:
    r"""A flow-regime-dependent single phase heat transfer coefficient function.

    The forced-convection part interpolates linearly on the bulk-evaluated Reynolds
    number between the ``laminar`` and ``turbulent`` functions, with ``re_bounds``
    setting the transition band. The laminar function is passed bulk properties,
    the turbulent one film properties.

    The natural-convection contribution is added to the forced value by the cube norm
    :math:`h = (h_f^3 + h_n^3)^{1/3}`, the assisting form of Churchill [#Churchill]_.
    It is a smooth maximum of the two: the plain maximum used by RELAP5 and TRACE is
    its large-exponent limit. In a channel narrower than about 4 mm the natural term
    adds under 1 %; from 6 mm up it matters.

    At zero flow the laminar law returns its fully developed value. That value lies
    below the conduction limit of a stagnant heated slab (Nu 12 on the hydraulic
    diameter for uniform flux), so it is a floor and the composition needs no separate
    stagnant-channel limit. Upflow data in narrow heated channels place the coefficient
    at the laminar value when buoyancy aids the flow [#Sudo]_.

    Every switch here is fed by the flow (Re), never by the wall superheat the result
    feeds back into, and the cube norm only adds, so :math:`h(T_{wall}) \cdot \Delta T`
    is monotone and a cell's wall balance has a unique root.

    Limitations: laminar flow opposed by buoyancy (heated downflow) is over-predicted,
    since buoyancy slows the fluid at the wall, unless a downflow law is given.
    Buoyancy effects on turbulent flow are not modelled. The natural term is an
    open-chimney correlation evaluated on bulk properties with the absolute Rayleigh
    number.

    Parameters
    ----------
    coolant: Liquid
        Coolant `film` properties. See in :func:`~.wall_heat_transfer_coeff`
    mdot: KgPerS
        Coolant mass flow
    Dh: Meter
        Hydraulic diameter
    A: Meter2
        Flow area
    T_cool: Celsius
        Coolant bulk temperature
    T_wall: Celsius
        Wall temperature
    re_bounds: tuple[Value, Value]
        Boundaries depicting transition between laminar, interim, and turbulent regimes.
    coolant_funcs: LiquidFuncs
        Coolant properties functions.
    Lh: Meter
        Heated length, passed on to the ``laminar``, ``turbulent`` and ``natural``
        functions.
    g: MPerS2
        Gravitational acceleration along the channel, passed on to ``natural``. It is
        the same ``g`` the channel's pressure law (:func:`~.pressure_diff`) uses, so
        positive flow runs along gravity when ``g > 0``; a riser passes the negated
        gravity to both laws. A horizontal channel passes ``0.0``, which drops the
        natural term and returns the forced value.
    laminar: SinglePhaseLiquidHTCExArgs
        Laminar heat transfer coefficient. It is evaluated with bulk coolant properties.
    turbulent: SinglePhaseLiquidHTCExArgs
        Turbulent heat transfer coefficient
    natural: SinglePhaseLiquidHTCExArgs
        Natural convection heat transfer coefficient, evaluated with bulk properties.

    Returns
    -------
    h: WPerM2K
        Heat transfer coefficient

    Raises
    ------
    TypeError
        If ``gz_band`` is passed.

    References
    ----------
    .. [#Churchill] S. W. Churchill, "A comprehensive correlating equation for laminar,
       assisting, forced and free convection", AIChE J. 23(1):10-16, 1977.
    .. [#Sudo] Y. Sudo, K. Miyata, H. Ikawa, M. Ohkawara, M. Kaminaga, "Experimental
       study of differences in single-phase forced-convection heat transfer
       characteristics between upflow and downflow for narrow rectangular channel",
       J. Nucl. Sci. Technol. 22(3):202-212, 1985.
    """
    if "gz_band" in kwargs:
        raise TypeError(
            "regime_dependent_h_spl no longer takes gz_band: the natural term is combined by the cube norm"
            " at every flow and the laminar law holds at zero flow"
        )
    re_bulk = Re_mdot(mdot, A, Dh, coolant_funcs.viscosity(T_cool))

    lam, inter, turb = flow_regimes(re_bulk, re_bounds)

    inp = (
        {
            "coolant": coolant,
            "mdot": mdot,
            "Dh": Dh,
            "A": A,
            "Lh": Lh,
            "T_cool": T_cool,
            "T_wall": T_wall,
            "coolant_funcs": coolant_funcs,
        }
        | kwargs
    )
    h = np.empty(len(T_cool))

    bulk = coolant_funcs.to_properties(T_cool)
    h_turb = turbulent(**inp)
    h[turb] = h_turb[turb]
    if np.any(lam + inter):
        h_lam = laminar(**(inp | {"coolant": bulk}))
        h[inter] = lin_interp(*re_bounds, y1=h_lam, y2=h_turb, x=re_bulk)[inter]
        h[lam] = h_lam[lam]

    h_nat = natural(**(inp | {"coolant": bulk, "g": g})) if g != 0.0 else np.zeros_like(h)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.cbrt(h**3 + h_nat**3)


def maximal_h_spl(
    hs: Sequence[SinglePhaseLiquidHTCExArgs] = (
        Elenbaas_h_spl,
        Dittus_Boelter_h_spl,
        developing_laminar_h_spl,
    ),
) -> SinglePhaseLiquidHTCExArgs:
    """Creates a new SinglePhaseLiquidHTCExArgs function, which returns the maximal value out of the given functions.

    Parameters
    ----------
    hs: Sequence[SinglePhaseLiquidHTCExArgs]
        Functions to evaluate

    Returns
    -------
    SinglePhaseLiquidHTCExArgs
        A SPL HTC function with maximal values
    """

    def _max_h(
        *,
        coolant: Liquid,
        mdot: KgPerS,
        Dh: Meter,
        A: Meter2,
        T_cool: Celsius,
        T_wall: Celsius,
        coolant_funcs: LiquidFuncs,
        **kwargs,
    ) -> WPerM2K:
        return reduce(
            np.maximum,
            (
                h(
                    coolant=coolant,
                    mdot=mdot,
                    Dh=Dh,
                    A=A,
                    T_cool=T_cool,
                    T_wall=T_wall,
                    coolant_funcs=coolant_funcs,
                    **kwargs,
                )
                for h in hs
            ),
        )

    return _max_h


_SPL = {
    "natural": Elenbaas_h_spl,
    "laminar": developing_laminar_h_spl,
    "laminar_constant_nu": constant_Nusselt_h_spl,
    "laminar_developed": fully_developed_laminar_h_spl,
    "turbulent": Dittus_Boelter_h_spl,
    "regime_dependent": regime_dependent_h_spl,
    "maximal": maximal_h_spl(),
}


class SPLMethod(Enum):
    NATURAL = "natural"
    LAMINAR = "laminar"
    LAMINAR_CONSTANT_NU = "laminar_constant_nu"
    LAMINAR_DEVELOPED = "laminar_developed"
    TURBULENT = "turbulent"
    REGIME_DEPENDENT = "regime_dependent"
    MAXIMAL = "maximal"


def spl_htc(
    name: SPLMethod
    | Literal[
        "natural",
        "laminar",
        "laminar_constant_nu",
        "laminar_developed",
        "turbulent",
        "regime_dependent",
        "maximal",
    ],
    **kwargs,
) -> SinglePhaseLiquidHTCExArgs:
    r"""Create a Single Phase Liquid Heat Transfer Coefficient function chosen from the
    list below with `almost` uniform signatures.
    The main usage of this function is as input for :func:`~.wall_heat_transfer_coeff`.

    Available functions:

    .. list-table::
        :widths: 20, 80

        * - **regime_dependent**
          - :func:`regime_dependent_h_spl`: forced part interpolated on the Re number
            over ``re_bounds``, natural part added by the cube norm; requires ``Lh``.
        * - **laminar**
          - :func:`~.laminar_h_spl`. Requires the ``aspect_ratio = channel_depth / channel_width`` parameter.
        * - **laminar_constant_nu**
          - :func:`~.laminar_h_spl`.
        * - **laminar_developed**
          - :func:`~.laminar_developed`.
        * - **turbulent**
          - :func:`~.Dittus_Boelter_h_spl` which employs :func:`~.Dittus_Boelter`.
        * - **natural**
          - :func:`~.Elenbaas_h_spl`. Requires the ``Lh = heated_length`` parameter.
        * - **maximal**
          - Computes the natural, laminar and turbulent HTCs and selects the highest at each cell.


    Parameters
    ----------
    name: SPLMethod | Literal["natural", "laminar", "turbulent", "regime_dependent", "maximal"]
        Method name
    kwargs: Dict
        Options to pass onto the given method

    Returns
    -------
    SinglePhaseLiquidHTCExArgs
        Single Phase Liquid Heat Transfer Coefficient function
    """
    name = name.value if isinstance(name, SPLMethod) else name
    f: SinglePhaseLiquidHTCExArgs = partial(_SPL[name], **kwargs)  # type: ignore
    f.__doc__ = _SPL[name].__doc__
    return f
