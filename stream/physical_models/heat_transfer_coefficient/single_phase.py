from enum import Enum
from functools import reduce, partial
from typing import Literal, Protocol, Sequence

import numpy as np

from stream.physical_models.dimensionless import flow_regimes, Re_mdot, Gr
from stream.physical_models.heat_transfer_coefficient.laminar import (
    developing_laminar_h_spl,
    constant_Nusselt_h_spl,
    fully_developed_laminar_h_spl,
)
from stream.physical_models.heat_transfer_coefficient.natural_convection import (
    Elenbaas_h_spl,
)
from stream.physical_models.heat_transfer_coefficient.turbulent import (
    Dittus_Boelter_h_spl,
)
from stream.substances import Liquid, LiquidFuncs
from stream.units import Celsius, KgPerS, Meter, Meter2, Value, WPerM2K, Pascal
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
    laminar: SinglePhaseLiquidHTCExArgs = developing_laminar_h_spl,
    turbulent: SinglePhaseLiquidHTCExArgs = Dittus_Boelter_h_spl,
    natural: SinglePhaseLiquidHTCExArgs = Elenbaas_h_spl,
    **kwargs,
) -> WPerM2K:
    r"""A flow-regime-dependent single phase heat transfer coefficient function.

    Given laminar, turbulent and natural regimes heat transfer functions, this function
    interpolates between turbulent-laminar values, where ``re_bounds`` determines their
    respective area of applicability, and applies the natural function
    when :math:`\text{Gr}/\text{Re}^2_\text{film} > 1`.

    The interpolation is done linearly on the bulk-evaluated Reynolds number.
    The laminar regime function is passed bulk values, whereas the turbulent function is
    passed film values.

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
    laminar: SinglePhaseLiquidHTCExArgs
        Laminar heat transfer coefficient. It is evaluated with bulk coolant properties.
    turbulent: SinglePhaseLiquidHTCExArgs
        Turbulent heat transfer coefficient
    natural: SinglePhaseLiquidHTCExArgs
        Natural convection heat transfer coefficient

    Returns
    -------
    h: WPerM2K
        Heat transfer coefficient
    """
    re_bulk = Re_mdot(mdot, A, Dh, coolant_funcs.viscosity(T_cool))

    lam, inter, turb = flow_regimes(re_bulk, re_bounds)

    inp = (
        dict(
            coolant=coolant,
            mdot=mdot,
            Dh=Dh,
            A=A,
            T_cool=T_cool,
            T_wall=T_wall,
            coolant_funcs=coolant_funcs,
        )
        | kwargs
    )
    h = np.empty(len(T_cool))

    h_turb = turbulent(**inp)
    h[turb] = h_turb[turb]
    if np.any(lam + inter):
        h_lam = laminar(**(inp | dict(coolant=coolant_funcs.to_properties(T_cool))))
        h[inter] = lin_interp(*re_bounds, y1=h_lam, y2=h_turb, x=re_bulk)[inter]
        h[lam] = h_lam[lam]

    gr = Gr(
        coolant.density,
        mu := coolant.viscosity,
        coolant.thermal_expansion,
        T_cool,
        T_wall,
        Dh,
    )
    re_film = Re_mdot(mdot, A, Dh, mu)
    nat = gr / (re_film**2) > 1
    if np.any(nat):
        h[nat] = natural(**(inp | dict(coolant=coolant_funcs.to_properties(T_cool))))[nat]

    return h


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
          - :func:`regime_dependent_h_spl`, which depends on the
            :func:`~.Re` No., given ``re_bounds``. Laminar, Turbulent and Natural
            :class:`~.SinglePhaseLiquidHTCExArgs` functions are required.
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
