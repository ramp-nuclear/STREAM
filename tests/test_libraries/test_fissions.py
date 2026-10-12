"""Fission decay-heat profile tests.

Kept separate from test_decay_heat.py, which skips at import when the decay-heat
standards data is unavailable; profile_from_pk needs none of that data.
"""

import numpy as np

from stream.calculations import PointKinetics
from stream.physical_models.decay_heat import fissions


def test_profile_from_pk_builds_a_profile_from_an_existing_point_kinetics():
    """profile_from_pk must run end-to-end from a PointKinetics object, forwarding
    its real attributes (generation time, decay rates, fractions, controls)."""
    pk = PointKinetics(
        generation_time=5e-5,
        delayed_neutron_fractions=np.array([0.0065]),
        delayed_groups_decay_rates=np.array([0.08]),
    )
    time = np.linspace(0.0, 10.0, 5)
    prof = fissions.profile_from_pk(time, pk)
    assert callable(prof)
    assert np.all(np.isfinite(prof(time, np.inf)))
