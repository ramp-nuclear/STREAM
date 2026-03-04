"""Tools for jiggling power shapes or outright making them up."""

import numpy as np

from stream.units import Array2D


def local_power_shift(
    dist: Array2D,
    relative_shift: float = 1.0,
    percentile: float = 90.0,
) -> np.array:
    """Raise the q-th percentile power by a given quantity

    Parameters
    ----------
    dist: Array2D
        The power distribution. We assume a 2D array, where the first axis is the
        length (z) axis, and the second is depth.
        Power does not fluctuate along the depth (probably), but the solution
        of the temperature gradient does, so we need to shift the entire row.
    relative_shift: float
        The relative amount by which the given percentile should be raised
        compared to its current value.
        A value of 1 means we perform no shift.
    percentile: float
        Which percentile of the summed-row distribution should be shifted.
        Must be in the range [0, 100]. The default is 90, which means the top
        10% of rows are shifted and then the distribution is normalized.
        To further stress this point, the 0 percentile
        is equivalent to doing nothing, and the 100 percentile is equivalent
        to shifting the max row. Note that if the entire non-zero distribution
        is within the percentile, no change is made.

    Returns
    -------
    The new, shifted power shape.

    """
    assert dist.ndim == 2
    zpower = np.sum(dist, axis=1)
    p = np.percentile(zpower, percentile)
    maxrowp, maxrow = zpower[zpower >= p].sum(), zpower >= p
    s = np.sum(dist)
    if np.all(maxrow[np.flatnonzero(zpower)]):
        return dist
    factor = (s - maxrowp * relative_shift) / (s - maxrowp)
    res = factor * dist
    res[maxrow, :] = relative_shift * dist[maxrow, :]
    return res
