"""Testing different constructors"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from

from stream.aggregator import Aggregator
from stream.composition.constructors import Calculation_factory, ResistorFromKnownPoint
from stream.substances import light_water

from .conftest import are_close, pos_medium_floats


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_parabolic_FlowResistor_intersects_set_point(dp, mdot, Tin):
    r = ResistorFromKnownPoint(dp=-dp, mdot=mdot, behavior="parabolic", fluid=light_water, Tin=Tin)
    are_close(r.calculate([Tin, -dp], mdot=mdot, Tin=Tin), 0.0)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_constant_mdot_FlowResistor_intersects_set_point(Tin, dp, mdot):
    r = ResistorFromKnownPoint(mdot=mdot, behavior="constant")
    are_close(r.calculate([Tin, dp], mdot=mdot, Tin=Tin), 0.0)
    are_close(r.calculate([Tin, 2 * dp + 1], mdot=mdot, Tin=Tin), 0.0)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_constant_dp_FlowResistor_intersects_set_point(Tin, dp, mdot):
    r = ResistorFromKnownPoint(dp=dp, behavior="constant")
    are_close(r.calculate([Tin, dp], mdot=mdot, Tin=Tin), 0.0)


@settings(deadline=None)
@given(pos_medium_floats, pos_medium_floats, pos_medium_floats)
def test_linear_FlowResistor_intersects_set_point(Tin, dp, mdot):
    r = ResistorFromKnownPoint(dp=dp, mdot=mdot, behavior="linear")
    are_close(r.calculate([Tin, dp], mdot=mdot, Tin=Tin), 0.0)
    are_close(r.calculate([Tin, 2 * dp], mdot=2 * mdot, Tin=Tin), 0.0)


def test_constant_FlowResistor_errors_when_superimposed():
    dp, mdot = 1.0, 1.0
    with pytest.raises(ValueError):
        ResistorFromKnownPoint(dp=dp, mdot=mdot, behavior="constant")


def test_constant_FlowResistor_errors_when_under_imposed():
    with pytest.raises(AssertionError):
        ResistorFromKnownPoint(behavior="constant")


@given(sampled_from(["linear", "parabolic"]))
def test_FlowResistor_errors_when_under_imposed(b):

    with pytest.raises(AssertionError):
        ResistorFromKnownPoint(behavior=b)

    with pytest.raises(AssertionError):
        ResistorFromKnownPoint(mdot=1.0, behavior=b)

    with pytest.raises(AssertionError):
        ResistorFromKnownPoint(dp=1.0, behavior=b)


def test_Calculation_factory_given_a_simple_example_is_set_up_and_calculates_correctly():
    A = Calculation_factory(
        calculate=lambda y, **kwargs: np.ones(1),
        mass_vector=np.ones(1),
        variables=dict(y=0),
    )

    a = A("Hi")

    assert a.name == "Hi"
    assert a.calculate(np.array([1])) == np.array([1])
    assert a.variables == {"y": 0}
    assert a.mass_vector == np.ones(1)

    agr = Aggregator.from_decoupled(a)
    time = np.linspace(0, 1, 11)
    sol = agr.solve({a.name: dict(y=0.0)}, time)
    assert np.allclose(sol[:, 0], time)
