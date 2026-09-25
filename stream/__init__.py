r"""
The STREAM package includes underlying modules creating a simulation and
running it. Inherently, it allows the construction and solution of

.. math:: M\vec{\dot{y}} = \vec{F} \left(\vec{y}, t\right)

Which is a Differential Algebraic Equation (DAE).

It also includes some predefined thermohydraulic components in the
:mod:`.calculations` package and correlations for those components
in :mod:`.substances` and :mod:`.physical_models`.
"""

import logging

from .aggregator import *
from .calculation import *
from .state import State as State
from .jacobians import *
from .pipe_geometry import *
from .physical_models import *
from .analysis import *
from .substances import *
from .calculations import Solid as Solid

# Libraries attach only a NullHandler so importing STREAM emits nothing on its own; call enable_rich_logging to opt in.
logging.getLogger("stream").addHandler(logging.NullHandler())


def enable_rich_logging(level: int = logging.INFO):
    """Route the ``"stream"`` logger through a Rich handler and set its level.

    The library default is silent (a :class:`logging.NullHandler`); call this to
    restore the timestamped, colorized console output. It is idempotent -- a
    repeat call re-uses the single Rich handler rather than stacking another, and
    just updates the level.

    Parameters
    ----------
    level : int
        Logger level to set. ``logging.INFO`` (the default) surfaces events,
        restarts and stops; ``stream.utilities.STREAM_DEBUG`` (11) additionally
        opens the per-construction / per-solve chatter.
    """
    from rich.logging import RichHandler

    logger = logging.getLogger("stream")
    for handler in [h for h in logger.handlers if isinstance(h, RichHandler)]:
        logger.removeHandler(handler)
    logger.addHandler(RichHandler(log_time_format="[%X]"))
    logger.setLevel(level)
