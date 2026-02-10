"""Tests that calculations are serializable.

Serializability is used for distributed work and for human-readable debugging.

"""

import cloudpickle
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from networkx import MultiDiGraph

import stream.calculations as calcs
from stream.calculations.heat_diffusion import Solid
from stream.pipe_geometry import EffectivePipe
from stream.substances import light_water as lw


light_water = st.just(lw)
reg_floats = st.floats(min_value=1e-2, max_value=10, allow_nan=False,
                       allow_infinity=False, allow_subnormal=False)
solids = st.builds(Solid, reg_floats, reg_floats, reg_floats)
pipes = st.builds(EffectivePipe, reg_floats, reg_floats, reg_floats, reg_floats)
boundaries = (st.lists(elements=reg_floats, min_size=2, max_size=40)
              .map(np.array).map(np.cumsum)
              )
delayed_groups = st.shared(st.integers(min_value=1, max_value=6), key='moo')
b1 = delayed_groups.flatmap(lambda x: st.lists(reg_floats, min_size=x,
                                               max_size=x).map(np.array))
names = st.text('abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=50)
calc_float_dicts = st.shared(st.dictionaries(names, reg_floats), key='dicts')
specific_calc_float_dicts = calc_float_dicts.flatmap(
    lambda d: st.fixed_dictionaries({key: reg_floats for key in d}))

channels = st.builds(calcs.Channel,
                     z_boundaries=boundaries,
                     fluid=light_water,
                     pipe=pipes,
                     name=names)
channels_and_contacts = st.builds(calcs.ChannelAndContacts,
                                  z_boundaries=boundaries,
                                  fluid=light_water,
                                  pipe=pipes,
                                  name=names
                                  )
flappers = st.builds(calcs.Flapper,
                     open_at_current=reg_floats,
                     f=reg_floats,
                     fluid=light_water,
                     area=reg_floats,
                     open_rate=reg_floats)
fuels = st.builds(calcs.Fuel,
                  z_boundaries=boundaries,
                  x_boundaries=boundaries,
                  material=solids,
                  y_length=reg_floats)
frictions = st.builds(calcs.Friction, reg_floats, light_water, reg_floats, reg_floats,
                      reg_floats, names)
gravities = st.builds(calcs.Gravity, light_water, reg_floats, name=names)
pdrops = st.builds(calcs.LocalPressureDrop, light_water, reg_floats, reg_floats,
                   name=names)
regfrics = st.builds(calcs.RegimeDependentFriction, pipes, light_water,
                     st.tuples(st.floats(min_value=100, max_value=2000),
                               st.floats(min_value=2500, max_value=6000)),
                     reg_floats, name=names)
resistors = st.builds(calcs.Resistor, reg_floats, names)
res_sums = (st.lists(resistors, min_size=1, max_size=50)
            .map(lambda x: calcs.ResistorSum(*x)))
heat_exchangers = st.builds(calcs.HeatExchanger, reg_floats)
inductors = st.builds(calcs.Inertia, reg_floats, name=names)
junctions = st.builds(calcs.Junction, names)
point_kinetics = st.builds(calcs.PointKinetics, reg_floats, b1, b1,
                           specific_calc_float_dicts, specific_calc_float_dicts,
                           name=names)
pumps = st.builds(calcs.Pump, reg_floats, name=names)


def _gmake(e: list[tuple[str, str, str, tuple[str, ...]]]) -> MultiDiGraph:
    g = MultiDiGraph()
    for u, v, key, c in e:
        g.add_edge(u, v, key, comps=c)
    return g


edges = st.tuples(names, names, names, st.lists(names, min_size=1).map(tuple))
graphs = st.builds(_gmake, st.lists(edges, min_size=1))
kirchoffs = st.builds(calcs.Kirchhoff, graphs, name=names)
kirchoffs_w_deriv = st.builds(calcs.KirchhoffWDerivatives, graphs, name=names)


@given(st.one_of(channels, channels_and_contacts, flappers, fuels, frictions, gravities,
                 pdrops, regfrics, resistors, res_sums, heat_exchangers, inductors,
                 junctions, point_kinetics, pumps, kirchoffs,
                 kirchoffs_w_deriv))
def test_calculations_are_cloud_pickleable(calc):
    try:
        cloudpickle.dumps(calc)
    except TypeError:
        print(calc)
        raise
