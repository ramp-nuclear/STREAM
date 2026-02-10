"""Tests for the UQ analysis

"""

from inspect import currentframe
from itertools import chain

import hypothesis.strategies as st
import numpy as np
import pytest
from dask import visualize, delayed
from hypothesis import given, settings
from pandas import DataFrame
from scipy.constants import atmosphere
from scipy.optimize import root_scalar

from stream.aggregator import Aggregator
from stream.analysis.UQ import Uncertainty, UQModel, DASKUQModel
from stream.calculations import ChannelAndContacts, Pump, HeatExchanger
from stream.composition import flow_graph_to_agr_and_k, flow_graph, flow_edge as edge
from stream.physical_models.thresholds import Sudo_Kaminaga_CHF
from stream.pipe_geometry import EffectivePipe
from stream.state import to_dataframe, State
from stream.substances import light_water
from stream.units import Array1D, cm, mm
from stream.utilities import just


def _gen_records(key, value):
    yield from ([dict(i=0, j=0, calculation='foo', variable=key, value=value)]
                if key != 'v'
                else [dict(i=0, j=j, calculation='moo', variable=key, value=v)
                      for j, v in enumerate(value)])


def _simplemodel(vec: Array1D, v1: float, v2: float) -> DataFrame:
    res = dict(a=v1 + v2, m=v1 * v2, p=v1 ** v2, s=vec.sum(), v=np.sin(vec))
    records = list(chain(*(_gen_records(k, v) for k, v in res.items())))
    return DataFrame.from_records(records)


@delayed(pure=True)
def _search_simple(model, vec, v1, v2):
    def _f(v):
        res = model(vec=vec, v1=v, v2=v2)
        just_foo = res[res.variable == 'a']
        foo_max = np.max(just_foo.value.values)
        return foo_max - 1

    root = root_scalar(_f, x0=v1, x1=v2)
    record = dict(i=0, j=0, calculation='root', variable='add', value=root.root)
    return DataFrame.from_records([record])


@delayed(pure=True)
def _stupid_feature(model, **p):
    return model(**p)


def test_simplemodel_returns_known_dataframe():
    df = _simplemodel(np.array([0., 0.]), 2, 2)
    assert np.allclose(df.value.values, [4, 4, 4, 0, 0, 0])


def test_uq_of_simplemodel_by_known_result():
    model = UQModel({'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.},
                    _simplemodel)
    sys, stat = model.uq(vec=Uncertainty(systematic_relative=np.array([0.1, 0.1, 0.1]),
                                         statistical_absolute=np.array([0.2, 0.2, 0.2])),
                         v1=Uncertainty(systematic_absolute=1.),
                         v2=Uncertainty(statistical_absolute=1.)
                         )
    assert np.allclose(sys[:2], [1, 3])
    assert np.allclose(stat[0], 1.)


non_negative = st.floats(min_value=0., max_value=3., allow_subnormal=False)
uncertainties = st.builds(Uncertainty, *(4 * [non_negative]))
positive = st.floats(min_value=1e-3, max_value=3.)
one_positive = (st.tuples(*(3 * [non_negative]), positive)
                .flatmap(lambda tpl: st.permutations(tpl)))
positive_uncertainty = one_positive.map(lambda tpl: Uncertainty(*tpl))


@given(uncertainties, positive_uncertainty)
def test_uq_of_multiple_sources_strictly_larger_than_one(
        u1: Uncertainty, u2: Uncertainty):
    model = UQModel({'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.},
                    _simplemodel)
    sys1, stat1 = model.uq(v1=u1)
    sys2, stat2 = model.uq(v1=u1, v2=u2)
    d1, d2 = sys2 - sys1, stat2 - stat1
    assert np.all(d1 >= 0)
    assert np.all(d2 >= 0)
    assert np.sum(d1 + d2) > 0


shared_non_neg = st.shared(non_negative, key='1')
dbl = shared_non_neg.map(lambda x: 2 * x)
uncertainties_smaller = st.builds(Uncertainty, *(4 * [shared_non_neg])).filter(bool)
uncertainties_larger = st.builds(Uncertainty, *(4 * [dbl])).filter(bool)


@given(uncertainties_smaller, uncertainties_larger)
def test_uq_of_larger_uncertainty_is_larger(small: Uncertainty, large: Uncertainty):
    model = UQModel({'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.},
                    _simplemodel)
    sys1, stat1 = model.uq(v1=small)
    sys2, stat2 = model.uq(v1=large)
    d1, d2 = sys2 - sys1, stat2 - stat1
    assert np.all(d1 >= 0)
    assert np.all(d2 >= 0)
    assert np.sum(d1 + d2) > 0


def test_uq_of_no_uncertainty_is_zero():
    model = UQModel({'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.},
                    _simplemodel)
    sys, stat = model.uq()
    assert sys == 0 and stat == 0


# noinspection PyPep8Naming
@settings(deadline=None)
@given(uncertainties)
def test_uq_of_DASKUQModel_same_as_UQModel(u):
    params = {'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.}
    model = UQModel(params, _simplemodel)
    dmodel = DASKUQModel(params, _simplemodel, _stupid_feature)
    sys_ground_truth, stat_ground_truth = model.uq(v1=u)
    sys, stat = dmodel.uq(v1=u)
    assert np.allclose(sys_ground_truth, sys.compute())
    assert np.allclose(stat_ground_truth, stat.compute())


@given(uncertainties)
def test_uq_with_multiple_features_is_longer_than_1_feature(u):
    params = {'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.}
    dmodel = DASKUQModel(params, _simplemodel, _stupid_feature, _search_simple)
    candidate = dmodel.uq(v1=u)
    assert len(candidate[0].compute()) > len(_simplemodel(**params))


@pytest.mark.parametrize("persist", (True, False))
def test_dask_uq_graph_by_regression(persist, image_regression):
    params = {'vec': np.array([1., 2., 3.]), 'v1': 2., 'v2': 3.}
    dmodel = DASKUQModel(params, _simplemodel, _stupid_feature, _search_simple)
    taskgraph = dmodel.uq(v1=Uncertainty(1., 1., 3., 5.), persist=persist)
    ext = 'png'
    fname = f'{currentframe().f_code.co_name}_{persist}'
    visualize(taskgraph, filename=fname, format=ext)
    with open(fname + '.png', 'rb') as f:
        image_regression.check(f.read(), diff_threshold=1)


def test_uq_improves_with_mdot_for_simple_channel():
    hx = HeatExchanger(20.)
    z = np.linspace(0, 50, 101) * cm
    pipe = EffectivePipe.rectangular(50 * cm, 6 * cm, 2.5 * mm, 6 * cm)
    c = ChannelAndContacts(z, light_water, pipe)

    def _make_agr(mdot: float, tlow: float) -> tuple[Aggregator, State, np.array]:
        t = np.linspace(tlow, tlow + 15, 100)
        p = Pump(mdot0=mdot)
        g = flow_graph(edge(('In', 'Out'), c), edge(('Out', 'In'), hx, p))
        agr, _ = flow_graph_to_agr_and_k(
            g,
            funcs={c: {'T_left': just(t), 'T_right': just(t),
                       'Tin': just(20), 'Tin_minus': just(20)}},
            reference_node=("In", 2 * atmosphere),
            abs_pressure_comps=(c,),
            )
        guess = State.merge(
            State.uniform(agr.graph, 20., "Tin", "T", "T_wall_left", "T_wall_right", "T_cool"),
            State.uniform(agr.graph, 0., "pressure"),
            {'Kirchhoff': {
                '(In -> Out, 0)': mdot,
                '(Out -> In, 0)': mdot,
                '(p_abs of CC)': 2 * atmosphere
                },
                'CC': {'h_left': 1e4, 'h_right': 1e4}
                }
            )
        return agr, guess, t

    def _calc(agr: Aggregator, guess, t, mdot):
        sol = agr.save(agr.solve_steady(guess))
        return to_dataframe({c.name: c.save(
            c.load(sol['CC']), T_left=t, T_right=t, Tin=20, Tin_minus=20,
            mdot=mdot, p_abs=2 * atmosphere
            )})

    def _chf(mdot, tlow):
        agr, guess, t = _make_agr(mdot, tlow)
        state = State.from_dataframe(_calc(agr, guess, t, mdot))
        tb = state[c.name]['T_cool']
        stat_p = state[c.name]['static_pressure']
        ts = light_water.sat_temperature(stat_p)
        sat_cool = light_water.to_properties(ts, stat_p)
        chfd = Sudo_Kaminaga_CHF(tb, sat_cool, mdot, pipe)
        state[c.name]['CHF'] = chfd
        df = state.to_dataframe()
        return df[df.variable == 'CHF']

    mdots = np.linspace(0.04, 0.2, 10)
    duq = [UQModel({'mdot': mdot, 'tlow': 70.}, _chf) for mdot in mdots]
    nominals = [uq.nominal for uq in duq]
    uqv = [uq.uq(mdot=Uncertainty(statistical_relative=0.05),
                 tlow=Uncertainty(statistical_relative=0.7))
           for uq in duq]
    res = np.array([np.min(nom.value.values - u) for nom, u in zip(nominals, uqv)])
    assert np.all(np.diff(res) >= 0), res
