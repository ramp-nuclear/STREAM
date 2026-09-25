"""Root of STREAM's exception hierarchy.

Every exception STREAM raises deliberately derives from :class:`StreamError`, so
``except StreamError`` is the one-stop catch for any STREAM-originated failure.
Only the root and the generic :class:`StreamConstructionError` live here; the
domain exceptions stay defined in their owning modules (e.g.
:class:`~stream.solvers.TransientRuntimeError` in ``solvers.py``) and simply add
:class:`StreamError` to their bases.

This is a leaf module — it imports nothing from ``stream`` — so it can be
imported anywhere (``solvers``, ``aggregator``, ``channel``, ``subsystems``)
without cycles.
"""


class StreamError(Exception):
    """Root of all STREAM-raised domain errors. Marker class — no behavior."""


class StreamConstructionError(StreamError, ValueError):
    """A system was built wrong and this was provable at construction time."""


def hint_block(*hints: str) -> str:
    """Render playbook hints as message-tail lines.

    Parameters
    ----------
    *hints : str
        One actionable suggestion per argument.

    Returns
    -------
    str
        A ``"\\n  → try: <hint>"`` line per hint (empty string for no hints),
        meant to be appended to an error message at the raise site.
    """
    return "".join(f"\n  → try: {hint}" for hint in hints)
