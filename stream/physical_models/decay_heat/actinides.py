import numpy as np

from stream.physical_models.decay_heat import activation
from stream.units import DecayHeatFunction, MeV, MeVPerFission, Second

__all__ = ["U239_profile", "Np239_profile", "contribution"]

CapturesPerFission = float

U239_profile = activation.profile(U239_decay_rate := 4.91e-04)
Np239_profile = activation.double_decay_profile(U239_decay_rate, Np239_decay_rate := 3.41e-06)

E_U: MeV = 0.460
E_Np: MeV = 0.405


def contribution(R: CapturesPerFission) -> DecayHeatFunction:
    r"""Gives a profile of the actinides contribution as described in
    [#ANS2014]_ and :eq:`actinides_contribution`.

    Parameters
    ----------
    R: CapturesPerFission
        Ratio of neutron captures in :math:`^{238}U` to fission events at operation time.

    Returns
    -------
    DecayHeatFunction
        Actinides (U239, Np239) total contribution

    Examples
    --------
    >>> contribution(0.005)(0, np.inf).item()
    0.004325
    """

    def _profile(t: Second, T: Second = np.inf) -> MeVPerFission:
        r"""

        Parameters
        ----------
        t: Second
            Times after shutdown
        T: Second
            Operation time

        Returns
        -------
        Value
            At times `t` after shutdown, :math:`(F_{U}(t, T)+F_{Np}(t, T))`
        """
        return (E_U * U239_profile(t, T) + E_Np * Np239_profile(t, T)) * R

    return _profile
