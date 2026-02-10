"""Testing functions relating to decay heat"""
from functools import partial

import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import SearchStrategy
from scipy.constants import day, hour, week

from stream.physical_models.decay_heat import (actinides, activation, fission_products, fissions)
from stream.physical_models.decay_heat.fission_products import (Source, Standard)
from stream.utilities import factor, just
from .conftest import pos_medium_floats


Q = 200.0
fp = fission_products.contribution(Standard.ANS14, Source.U235)
al28 = activation.profile(5.16e-3)
ac = actinides.contribution(R=1)


def test_decay_heat_contributions_are_monotonous_in_time():
    decays = (al28, ac, fp)
    time = np.logspace(-8, 8)
    for contribution in decays:
        decay_power = contribution(time, np.inf)
        assert np.all(np.diff(decay_power) <= 0), (
            f"{contribution} is not monotonous in time")


def test_fission_products_timed_contributions_against_known_values():
    """Fission products' beta decay contribute (traditionally) about 6.5%
    immediately after shutdown, and some lower values forth"""
    fp_percent = factor(partial(fp, T=np.inf), 100 / Q)
    assert np.allclose(fp_percent(
        [0., hour, day, week]),
        [6.5, 1.3, 0.5, 0.3],
        rtol=1e-1)


_lengths = st.shared(st.integers(min_value=1, max_value=50), key='arr_size')


def pos_med_arr(floats: SearchStrategy[float] = pos_medium_floats) -> SearchStrategy:
    return (_lengths
            .flatmap(lambda n: st.lists(floats, min_size=n, max_size=n))
            .map(np.array))


@settings(deadline=None)
@given(pos_medium_floats, pos_med_arr(), pos_med_arr())
def test_fission_decay_heat_is_constant_for_critical_core(L, ls, bs):
    time = np.linspace(0, 10)
    assert np.allclose(
        fissions.profile(time=time, generation_time=L, delayed_groups_decay_rates=ls,
                         delayed_neutron_fractions=bs,
                         input_reactivity_func=just(0.0))(time, np.inf), 1.0)


@settings(deadline=None)
@given(st.floats(1e-5, 1e-2, allow_nan=False, allow_infinity=False),
       pos_med_arr(st.floats(1e-3, 1e2, allow_nan=False, allow_infinity=False)),
       pos_med_arr(st.floats(1e-6, 1e-2, allow_nan=False, allow_infinity=False)))
def test_fission_decay_heat_is_decreasing_for_subcritical_core(L, ls, bs):
    time = np.linspace(0, 1, 10)

    ff = fissions.profile(time=time, generation_time=L, delayed_groups_decay_rates=ls,
                          delayed_neutron_fractions=bs,
                          input_reactivity_func=just(-0.001))

    assert np.all(np.diff(ff(time, np.inf)) <= 0)
