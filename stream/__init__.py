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

from rich.logging import RichHandler

from .aggregator import *
from .calculation import *
from .state import State
from .jacobians import *
from .pipe_geometry import *
from .physical_models import *
from .analysis import *
from .substances import *
from .calculations import Solid

logger = logging.getLogger("stream")
logger.addHandler(RichHandler(log_time_format="[%X]"))
