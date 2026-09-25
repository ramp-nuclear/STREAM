"""Tests for the ``stream.errors`` root and the re-parented domain exceptions:
a single ``StreamError`` root that ``except StreamError`` can catch, a generic
``StreamConstructionError``, the ``hint_block`` message helper, and the
backward-compatible MRO of the six existing domain exceptions.
"""

import pytest

from stream.aggregator.aggregator import NonUniqueCalculationNameError
from stream.calculations.channel import SaturationReachedError
from stream.composition.subsystems import GravityMismatchError, MissingFlowError
from stream.errors import StreamConstructionError, StreamError, hint_block
from stream.solvers import AlgRuntimeError, TransientRuntimeError


def _domain_instances():
    """One instance of every domain exception, built with its real signature."""
    return [
        TransientRuntimeError(None, None, None, "msg"),
        AlgRuntimeError("msg"),
        SaturationReachedError("CC", [1], [100.0], [90.0]),
        MissingFlowError("msg"),
        GravityMismatchError("msg"),
        NonUniqueCalculationNameError("msg"),
    ]


def test_errors_module_exposes_public_api():
    assert issubclass(StreamError, Exception)
    assert callable(hint_block)


def test_construction_error_is_stream_error_and_value_error():
    assert issubclass(StreamConstructionError, StreamError)
    assert issubclass(StreamConstructionError, ValueError)


def test_hint_block_exact_format():
    assert hint_block("do x", "do y") == "\n  → try: do x\n  → try: do y"


def test_hint_block_single_hint():
    assert hint_block("do x") == "\n  → try: do x"


def test_hint_block_no_hints_is_empty():
    assert hint_block() == ""


@pytest.mark.parametrize("exc", _domain_instances())
def test_domain_exceptions_caught_by_stream_error(exc):
    """The one-stop catch contract: ``except StreamError`` catches every one."""
    assert isinstance(exc, StreamError)
    try:
        raise exc
    except StreamError as caught:
        assert caught is exc


def test_runtime_error_mro_preserved():
    for exc in (
        TransientRuntimeError(None, None, None, "msg"),
        AlgRuntimeError("msg"),
        SaturationReachedError("CC", [1], [100.0], [90.0]),
    ):
        assert isinstance(exc, RuntimeError)


def test_value_error_mro_preserved():
    for exc in (GravityMismatchError("msg"), NonUniqueCalculationNameError("msg")):
        assert isinstance(exc, ValueError)


def test_missing_flow_error_mro_preserved():
    assert isinstance(MissingFlowError("msg"), Exception)
