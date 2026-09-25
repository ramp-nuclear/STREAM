"""Per-variable nominal magnitudes ("typical scales") keyed by variable name.

The registry supplies an order-of-magnitude value for each kind of state
variable. :func:`scale_vector` expands it into a length-``N`` vector aligned to
the aggregator's state vector, which downstream code consumes for scaled
finite-difference steps, per-variable absolute tolerances, and diagnostics.
"""

import logging

import numpy as np

from stream.units import Array1D
from stream.utilities import STREAM_DEBUG, offset

__all__ = ["DEFAULT_SCALES", "scale_vector"]

_logger = logging.getLogger("stream.scales")

DEFAULT_SCALES: dict[str, float] = {
    "T_cool": 3e2, "Tin": 3e2, "T": 3e2, "T_wall_left": 3e2, "T_wall_right": 3e2,
    "pressure": 1e4,          # component dp-residual (Channel/ideal) — NOT absolute
    "h_left": 1e4, "h_right": 1e4,
    "power": 1e6, "pk_power": 1e6,
    # Kirchhoff owned vars are opaque per-edge strings -> keyed by TYPE, not name:
    "mdot": 1.0, "abs_pressure": 1e5, "mdot2": 1.0,
    "ck": 1.0,
}


def scale_vector(
    agr,
    registry: dict[str, float] = DEFAULT_SCALES,
    overrides: dict[tuple[str, str], float] | None = None,
) -> Array1D:
    """Build the length-``N`` vector of nominal scales aligned to the state vector.

    Parameters
    ----------
    agr : Aggregator
        The aggregator whose state vector is to be scaled. Only duck-typed
        access to ``agr.sections`` and ``len(agr)`` is used.
    registry : dict[str, float]
        Variable name -> typical magnitude. Defaults to :data:`DEFAULT_SCALES`.
    overrides : dict[tuple[str, str], float] or None
        Optional ``(node.name, varname) -> scale`` overrides that win over
        ``registry`` for owned variables (e.g. a hotter fuel ``T``).

    Returns
    -------
    Array1D
        A float vector of length ``len(agr)``. Positions default to ``1.0``;
        each is set from the override, then the registry, then ``1.0``.

    Notes
    -----
    Iterates ``agr.sections`` (graph-node order) crossed with each node's local
    variables, composing local->global with :func:`~stream.utilities.offset`.
    This is the exact ``sections x variables`` iteration that ``load``/``save``
    rely on, so the returned vector is index-aligned to the state vector.
    Kirchhoff-family nodes expose opaque per-edge names, so they are routed by
    ``variables_by_type`` (``mdot``/``abs_pressure``/``mdot2``) instead. Names
    absent from the registry (with no override) default to ``1.0`` and are
    logged once each.
    """
    typ = np.ones(len(agr))
    overrides = overrides or {}
    unregistered: set[str] = set()
    for node, section in agr.sections.items():
        if hasattr(node, "variables_by_type"):  # Kirchhoff / KirchhoffWDerivatives
            for tname, place in node.variables_by_type.items():
                typ[offset(place, section.start)] = registry.get(tname, 1.0)
                if tname not in registry:
                    unregistered.add(tname)
        else:
            for vname, place in node.variables.items():
                key = (getattr(node, "name", None), vname)
                if key in overrides:
                    typ[offset(place, section.start)] = overrides[key]
                else:
                    typ[offset(place, section.start)] = registry.get(vname, 1.0)
                    if vname not in registry:
                        unregistered.add(vname)
    for name in sorted(unregistered):
        _logger.log(STREAM_DEBUG, "no nominal scale registered for %r; defaulting to 1.0", name)
    return typ
