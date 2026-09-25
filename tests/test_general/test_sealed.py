"""Attribute contracts.

The ``@sealed`` decorator makes a Calculation reject *new* attribute names after
construction, so a stale private name (``_flag`` for a renamed ``_latched``) or a
typo of a real knob (``mdot_0`` for ``mdot0``) raises instead of silently creating
a dead attribute. The two pokes raise, naming the class and suggesting the intended
name; every existing (construction- and event-time) mutation is untouched; user
subclasses are entirely exempt.
"""

import numpy as np
import pytest

import stream
import stream.calculation
from stream.calculation import sealed
from stream.calculations import Flapper, PointKinetics, PointKineticsWInput, Pump
from stream.substances import light_water


def _flapper() -> Flapper:
    return Flapper(open_at_current=0.1, f=1.0, fluid=light_water, area=1e-3, open_rate=1.0)


# --- importability ----------------------------------------------------------


def test_sealed_is_importable_from_calculation_and_package():
    assert stream.calculation.sealed is sealed
    assert stream.sealed is sealed  # re-exported via stream.calculation.__all__


# --- stale and typo attribute names raise --------------------


def test_stale_private_name_raises_naming_flapper_and_suggests_latched():
    fl = _flapper()
    with pytest.raises(AttributeError) as exc:
        fl._flag = True  # dead name from before the _latched rename
    msg = str(exc.value)
    assert "Flapper has no attribute '_flag'" in msg
    assert "Did you mean '_latched'?" in msg
    assert not hasattr(fl, "_flag")  # the dead attribute was never created


def test_typo_of_real_knob_raises_and_suggests_mdot0():
    fl = _flapper()
    real = fl.mdot0
    with pytest.raises(AttributeError) as exc:
        fl.mdot_0 = 5.0  # typo of the real threshold mdot0
    assert "Did you mean 'mdot0'?" in str(exc.value)
    assert fl.mdot0 == real  # the physics knob is untouched
    assert not hasattr(fl, "mdot_0")


def test_rejection_message_carries_the_contract_sentence():
    fl = _flapper()
    with pytest.raises(AttributeError, match=r"reject new attributes after construction; set them in __init__"):
        fl.some_brand_new_name = 1


def test_no_close_match_omits_the_did_you_mean_clause():
    fl = _flapper()
    with pytest.raises(AttributeError) as exc:
        fl.qqzzxxww = 1  # nothing on the instance is remotely close
    msg = str(exc.value)
    assert "Did you mean" not in msg
    assert "Flapper has no attribute 'qqzzxxww'." in msg


# --- existing names keep working (construction- and event-time mutation) ----


def test_existing_name_reassignment_still_works():
    fl = _flapper()
    fl.mdot0 = 0.2  # a real, pre-existing knob
    assert fl.mdot0 == 0.2


def test_change_state_latching_path_still_mutates_existing_names():
    fl = _flapper()
    assert fl._latched is False
    # ref_mdot <= mdot0 latches: change_state sets the pre-existing t_open/_latched.
    fl.change_state(np.array([300.0, 0.0]), ref_mdot=0.05, t=1.0)
    assert fl._latched is True
    assert fl.t_open == 1.0
    # open()/close() likewise mutate only pre-existing names.
    fl.open(2.5)
    assert fl._latched is True and fl.t_open == 2.5
    fl.close()
    assert fl._latched is False and fl.t_open == np.inf


def test_class_attribute_default_stays_settable_post_seal():
    # mdot_eps is a class-level default (not set in __init__ for ideal components);
    # the seal must still allow assigning it, else per-instance overrides break.
    p = Pump(pressure=1.0)
    p.mdot_eps = 0.01
    assert p.mdot_eps == 0.01
    with pytest.raises(AttributeError):
        p.mdot_epss = 0.01  # a typo of it is still rejected


# --- a sealed class's own post-init addition of a new name is rejected -------


def test_own_post_init_new_name_is_rejected():
    fl = _flapper()
    with pytest.raises(AttributeError, match="Flapper has no attribute 'genuinely_new'"):
        fl.genuinely_new = object()


# --- subclass exemption -----------------------------------------------------


def test_user_subclass_is_entirely_exempt():
    class MySubFlapper(Flapper):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.added_after_super = 42  # added post-super().__init__ -> must be allowed

    sub = MySubFlapper(open_at_current=0.1, f=1.0, fluid=light_water, area=1e-3, open_rate=1.0)
    assert sub.added_after_super == 42
    # A non-decorated subclass never arms the seal:
    assert not hasattr(sub, "_sealed_")
    sub.a_brand_new_post_init_attr = 99  # entirely exempt, no raise
    assert sub.a_brand_new_post_init_attr == 99


def test_in_tree_subclass_chain_arms_at_its_own_exact_type():
    # PointKineticsWInput extends PointKinetics and decorates itself; both arm.
    pk = PointKinetics(1e-4, np.array([0.0065]), np.array([0.08]))
    pkw = PointKineticsWInput(1e-4, np.array([0.0065]), np.array([0.08]))
    assert pk._sealed_ is True
    assert pkw._sealed_ is True
    with pytest.raises(AttributeError):
        pkw.not_a_real_attribute = 1
