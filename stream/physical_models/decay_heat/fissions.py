r"""Functions for creating a fission-following decay heat profile"""

import numpy as np

from stream import Aggregator
from stream.calculations import PointKinetics
from stream.calculations.point_kinetics import ReactivityController
from stream.composition import point_kinetics_steady_state
from stream.units import Array1D, DecayHeatFunction, PerS, Second, Value
from stream.utilities import identity


__all__ = ["profile", "profile_from_pk"]


def profile(
    time: Second,
    generation_time: Second,
    delayed_neutron_fractions: Array1D,
    delayed_groups_decay_rates: PerS,
    controls: ReactivityController | None = None,
) -> DecayHeatFunction:
    r"""
    Parameters
    ----------
    time: Second
        Times on which the calculated profile should rely.
    generation_time: Second
        mean time between neutron generations.
    delayed_neutron_fractions: Array1D
        the fractional yield of neutrons in
        defined delay groups. 1$ worth is the total.
    delayed_groups_decay_rates: PerS
        each group's decay rate.
    controls: ReactivityController
        Reactor reactivity controls

    Returns
    -------
    DecayHeatFunction
        The fission time profile :math:`\text{PK}(t, T)`
    """
    pk = PointKinetics(
        generation_time=generation_time,
        delayed_neutron_fractions=delayed_neutron_fractions,
        delayed_groups_decay_rates=delayed_groups_decay_rates,
        controls=controls,
    )
    agr = Aggregator.from_decoupled(pk, funcs={pk: dict(T=0, t=identity)})
    sol = agr.solve(
        y0 := agr.load(point_kinetics_steady_state(pk, 1e6)),
        time,
        yp0=agr.compute(y0, time[0]),
        eq_type="DAE",
    )
    prompt = agr.at_times(sol, pk, "power") / 1e6
    CUTOFF = 1e-12
    prompt[prompt < CUTOFF] = 0.0

    def _fissions(t: Second, T: Second = np.inf) -> Value:
        """
        Parameters
        ----------
        t: Second
            Times after shutdown
        T: Second
            Operation time

        Returns
        -------
        Value
            Profile at times `t` after shutdown
        """
        return np.interp(t, time, prompt)

    return _fissions


def profile_from_pk(time: Second, pk: PointKinetics) -> DecayHeatFunction:
    r"""

    Parameters
    ----------
    time: Second
        Times on which the calculated profile should rely.
    pk: PointKinetics
        Calculation on which the calculated profile should rely.

    Returns
    -------
    DecayHeatFunction
        The fission time profile :math:`\text{PK}(t, T)`
    """
    return profile(
        time,
        generation_time=pk.Lambda,
        delayed_groups_decay_rates=pk.lambdak,
        delayed_neutron_fractions=pk.betak,
        input_reactivity_func=pk.input_reactivity,
    )
