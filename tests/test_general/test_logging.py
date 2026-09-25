"""Library logging etiquette.

STREAM is a library: it attaches a :class:`logging.NullHandler` to its ``"stream"``
logger at import and never a formatting handler, so a consuming application owns all
output. ``stream.enable_rich_logging`` opts back into the pretty Rich output on request.

Logging state is process-global, so every test here saves and restores the handler
list and level it touches — order between tests must not matter, and ``stream`` is
already imported by the session (conftest), so we assert on the *current* handler
state rather than re-importing.
"""

import io
import logging

import pytest
from rich.logging import RichHandler

import stream
from stream.utilities import STREAM_DEBUG


@pytest.fixture
def restored_stream_logger():
    """Yield the ``"stream"`` logger, restoring its handlers and level afterwards."""
    logger = logging.getLogger("stream")
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    try:
        yield logger
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)


def test_import_leaves_only_a_nullhandler_on_the_stream_logger():
    """No RichHandler at import; a NullHandler is present (the library contract)."""
    logger = logging.getLogger("stream")
    assert not any(isinstance(h, RichHandler) for h in logger.handlers)
    assert [type(h) for h in logger.handlers] == [logging.NullHandler]


def test_stream_record_is_not_emitted_twice_into_a_user_logging_setup(restored_stream_logger):
    """With the NullHandler default, one STREAM record reaches a user's root handler
    exactly once and no library-owned handler produces a second copy."""
    buf = io.StringIO()
    user_handler = logging.StreamHandler(buf)
    user_handler.setFormatter(logging.Formatter("APP %(name)s %(levelname)s: %(message)s"))
    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    saved_root_level = root.level
    root.addHandler(user_handler)
    root.setLevel(logging.INFO)
    try:
        logging.getLogger("stream.aggregator").warning("At t = 12.00000, stopped by [Flapper]")
    finally:
        root.handlers[:] = saved_root_handlers
        root.setLevel(saved_root_level)
    # delivered once (via propagation) and not a second time by a library formatter
    assert buf.getvalue().count("stopped by [Flapper]") == 1
    assert not any(isinstance(h, RichHandler) for h in restored_stream_logger.handlers)


def test_enable_rich_logging_attaches_is_idempotent_and_sets_level(restored_stream_logger):
    logger = restored_stream_logger
    stream.enable_rich_logging(level=logging.INFO)
    assert sum(isinstance(h, RichHandler) for h in logger.handlers) == 1
    assert logger.level == logging.INFO
    # a second call must not stack a second RichHandler; it re-sets the level
    stream.enable_rich_logging(level=STREAM_DEBUG)
    assert sum(isinstance(h, RichHandler) for h in logger.handlers) == 1
    assert logger.level == STREAM_DEBUG
