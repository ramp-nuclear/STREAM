from enum import Enum
from functools import partial, reduce
from typing import Literal, Protocol, Sequence

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
from stream.smoothing import smooth_step
from stream.substances import Liquid, LiquidFuncs
from stream.units import Celsius, KgPerS, Meter, Meter2, Pascal, Value, WPerM2K
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
    gz_band: tuple[float, float] = (0.01, 0.1),
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

    Buoyancy enters by **superposition**, not by substitution: the natural-convection
    contribution is combined with the forced value through the Churchill cube norm
    :math:`h = (h_f^3 + h_n^3)^{1/3}` (aiding internal mixed convection only ever
    *enhances* heat transfer). Finally, when the through-flow renewal genuinely dies —
    Graetz number :math:`\text{Gz} = \text{Pe}\,D_h/L_h` below ``gz_band`` — the value
    hands over to the pure ``natural`` (stagnant-channel) function, so a flow-reversal
    instant recovers the buoyant-cavity limit.

    Every switch here is fed by the flow (Re, Gz), never by the wall superheat the
    result feeds back into, and the superposition only adds — so :math:`h(T_{wall})
    \cdot \Delta T` is monotone and a cell's wall balance has a unique root.

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
        Heated length, used for the Graetz-number stagnation handover (and passed on
        to the ``natural`` function).
    gz_band: tuple[float, float]
        Graetz-number band over which the composition hands over to the pure
        ``natural`` function as the through-flow vanishes. The default engages only
        at genuine stagnation scale (Pe below ~0.1·Lh/Dh); every circulating steady
        state sits far above it.
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
            Lh=Lh,
        )
        | kwargs
    )
    # nan (not empty): a NaN re leaving every regime mask False propagates NaN detectably, not uninitialised memory.
    h = np.full(len(T_cool), np.nan)

    bulk = coolant_funcs.to_properties(T_cool)
    h_turb = turbulent(**inp)
    h[turb] = h_turb[turb]
    if np.any(lam + inter):
        h_lam = laminar(**(inp | dict(coolant=bulk)))
        h[inter] = lin_interp(*re_bounds, y1=h_lam, y2=h_turb, x=re_bulk)[inter]
        h[lam] = h_lam[lam]

    h_nat = natural(**(inp | dict(coolant=bulk)))
    # Scoped: extreme solver iterates drive inf/0*inf through the cube norm and blend; the NaN/inf result is the caller's rejection signal, not a healthy-path event.
    with np.errstate(invalid="ignore", over="ignore"):
        h = np.cbrt(h**3 + h_nat**3)
        pe = np.abs(re_bulk) * bulk.viscosity * bulk.specific_heat / bulk.conductivity
        with np.errstate(divide="ignore"):  # log10(0) -> -inf -> weight 0 at true stagnation
            w_flow = smooth_step(np.log10(pe * Dh / Lh), np.log10(gz_band[0]), np.log10(gz_band[1]))
        return w_flow * h + (1.0 - w_flow) * h_nat


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
          - :func:`regime_dependent_h_spl`: forced part interpolated on the
            :func:`~.Re` No. over ``re_bounds``, natural part added by Churchill
            cube-norm superposition, with a Graetz-number handover to the pure
            natural function at stagnation. Requires ``Lh``.
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

    .. note::
        ``'natural'`` (and ``'maximal'``, which includes it) use the Elenbaas
        correlation, whose ``h -> 0`` as ``Ra -> 0`` (wall superheat -> 0). There is
        **no conduction floor by design** — at a near-stagnation, near-isothermal
        state the natural-convection coupling collapses to ~0, so a channel using
        this HTC decouples from its wall there. This is correct physics, not a bug;
        if a non-zero floor is needed, select a different HTC or add one explicitly.


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
