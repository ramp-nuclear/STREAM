r"""Wiring a free surface into a flow network.

A pool is a :class:`~stream.calculations.tank.Tank` sitting on the flow graph with one
hydrostatic head per leg it feeds or receives: the head is what turns the tank's level
into the driving pressure of that leg, and it is what falls away as the tank drains.
:func:`pool` writes those edges, and :func:`break_to_ambient` writes the edge through
which the inventory is lost. :func:`siphon_leg` writes the one arrangement in which a
pool empties past its own connection: a line that climbs over a crest before falling
below the pool.
"""

from stream.calculations import Friction, Gravity, Junction, LevelHead, Orifice, Tank
from stream.calculations.tank import Environment
from stream.composition.cycle import FlowEdge, flow_edge
from stream.errors import StreamConstructionError
from stream.pipe_geometry import EffectivePipe
from stream.substances import LiquidFuncs
from stream.units import Meter

__all__ = ["break_to_ambient", "pool", "siphon_leg"]


def pool(
    tank: Tank,
    *,
    outflows: dict[Junction, Meter] | None = None,
    inflows: dict[Junction, Meter] | None = None,
    fluid: LiquidFuncs | None = None,
) -> tuple[FlowEdge, ...]:
    r"""The flow edges connecting a free-surface ``tank`` to the legs it serves.

    Each leg gets its own :class:`~.LevelHead`, elevated at the connection and reading
    the tank's level, with the sign the free-surface flow solver requires: ``+1`` on an
    edge leaving the tank, ``-1`` on an edge entering it.

    Parameters
    ----------
    tank: Tank
        The pool.
    outflows: dict[Junction, Meter] or None
        Junctions the tank feeds, mapped to the elevation of the connection.
    inflows: dict[Junction, Meter] or None
        Junctions that feed the tank, mapped to the elevation of the connection.
    fluid: LiquidFuncs or None
        Liquid whose density the heads use; the tank's own by default.

    Returns
    -------
    tuple[FlowEdge, ...]
        The outflow edges followed by the inflow edges, ready for a
        :class:`~.FlowGraph`.

    Examples
    --------
    >>> from stream.calculations import Junction
    >>> from stream.substances import light_water
    >>> tank = Tank(light_water, 2.0, 4.0, z_uncovery=1.0, name="pool")
    >>> j = Junction(name="j")
    >>> (u, v, data), = pool(tank, outflows={j: 0.0})
    >>> u is tank, v is j, data["comps"][0].sign
    (True, True, 1.0)

    See Also
    --------
    break_to_ambient, ~stream.calculations.ideal.resistors.LevelHead,
    ~stream.calculations.tank.Tank
    """
    liquid = tank.fluid if fluid is None else fluid
    edges = [
        flow_edge((tank, j), LevelHead(liquid, z, tank.level0, sign=1.0, name=f"head_{tank.name}_{j.name}"))
        for j, z in (outflows or {}).items()
    ]
    edges += [
        flow_edge((j, tank), LevelHead(liquid, z, tank.level0, sign=-1.0, name=f"head_{j.name}_{tank.name}"))
        for j, z in (inflows or {}).items()
    ]
    return tuple(edges)


def break_to_ambient(node: Junction, orifice: Orifice, env: Environment) -> FlowEdge:
    r"""The flow edge on which ``node`` discharges into ``env`` through ``orifice``.

    Parameters
    ----------
    node: Junction
        The node that loses its inventory.
    orifice: Orifice
        The break. Sealed until it opens, so the intact system solves on this graph.
    env: Environment
        The ambient discharged into.

    Returns
    -------
    FlowEdge

    See Also
    --------
    pool, ~stream.calculations.break_flow.Orifice
    """
    return flow_edge((node, env), orifice)


def siphon_leg(
    tank: Tank,
    env: Environment,
    *,
    z_intake: Meter,
    z_crest: Meter,
    z_outlet: Meter,
    pipe: EffectivePipe,
    fluid: LiquidFuncs,
    orifice: Orifice,
    friction_factor: float = 0.02,
    z_breaker: Meter | None = None,
    crest_junction_name: str = None,
) -> tuple[tuple[FlowEdge, ...], Orifice]:
    r"""The flow edges of a line that leaves ``tank`` at ``z_intake``, climbs over a crest
    at ``z_crest`` and discharges into ``env`` at ``z_outlet``.

    What the climb buys is a drain the pool cannot stop by itself: the driving head is
    :math:`\rho g (L - z_\text{outlet})`, set by the outlet, so the pool keeps emptying
    after its surface has fallen past ``z_intake`` — down to uncovery, unless something
    shuts the line. What the climb costs is pressure: the crest carries the lowest
    absolute pressure in the leg, :math:`p_\text{surface} - \rho g (z_\text{crest} - L)`
    less the friction of the climb, and that is where a siphon flashes and breaks.

    ``pipe`` is the whole line; its length is split between the two legs in proportion to
    their rise and fall, and its cross-section is used for both.

    Parameters
    ----------
    tank: Tank
        The pool being emptied.
    env: Environment
        The ambient the line discharges into.
    z_intake: Meter
        Elevation at which the line leaves the pool, in the same datum as ``tank``'s level.
    z_crest: Meter
        Elevation of the high point the liquid has to be lifted over.
    z_outlet: Meter
        Elevation of the discharge. Below ``z_intake`` is what sustains the siphon.
    pipe: EffectivePipe
        Geometry of the line, for the friction of both legs.
    fluid: LiquidFuncs
        Liquid the line carries.
    orifice: Orifice
        The discharge. Sealed until it opens, so the intact system solves on this graph.
    friction_factor: float
        Darcy-Weisbach friction factor of the line.
    z_breaker: Meter or None
        Level at which a breaker is expected to arrest the drain. Giving it only
        *checks* that ``orifice`` latches there: ``closes_below`` is constructor state on
        a sealed class, so this helper cannot set it.
    crest_junction_name: str or None
        Name of the crest junction; the intake junction and the line's components are
        named after it. Defaults to ``f"{tank.name}_crest"``.

    Returns
    -------
    tuple[tuple[FlowEdge, ...], Orifice]
        The edges, ready for a :class:`~.FlowGraph`, and the component to hand to its
        ``abs_pressure_comps`` to watch the crest.

    Notes
    -----
    The crest component returned is the ``orifice`` itself, written at the head of the
    downcomer: the absolute pressure routed to a component is the one at its inlet, so
    that placement grounds ``p_abs`` at the crest, and
    :func:`~stream.analysis.thresholds.cavitation_crossings` reads the pressure that
    actually limits the siphon. The gravity and friction components that otherwise meet
    at the crest carry no ``fluid`` of their own, so saturation cannot be evaluated on
    them. The consequence is that an ``orifice`` armed with ``cc`` measures its flashing
    margin from the crest rather than from just upstream of its throat — the lower of the
    two, so it warns early rather than late.

    Examples
    --------
    >>> from stream.pipe_geometry import EffectivePipe
    >>> from stream.substances import light_water
    >>> tank = Tank(light_water, 2.0, 4.0, z_uncovery=0.2, name="pool")
    >>> env = Environment(name="ambient")
    >>> hole = Orifice(light_water, 5e-4, 0.61, name="break")
    >>> edges, crest = siphon_leg(tank, env, z_intake=1.0, z_crest=5.0, z_outlet=-2.0,
    ...                           pipe=EffectivePipe.circular(10.0, 0.05),
    ...                           fluid=light_water, orifice=hole)
    >>> len(edges), crest is hole
    (3, True)

    See Also
    --------
    pool, break_to_ambient, ~stream.calculations.break_flow.Orifice,
    ~stream.analysis.thresholds.cavitation_crossings
    """
    rise, fall = z_crest - z_intake, z_crest - z_outlet
    if rise < 0.0 or fall < 0.0 or rise + fall == 0.0:
        raise StreamConstructionError(
            f"A siphon over a crest at {z_crest} m cannot serve an intake at {z_intake} m and an "
            f"outlet at {z_outlet} m: the crest has to be the highest point of the leg, and above "
            f"at least one of its ends. Raise z_crest above both."
        )
    if z_breaker is not None and orifice.closes_below != (tank, z_breaker):
        raise StreamConstructionError(
            f"{orifice} is expected to arrest this siphon at {z_breaker} m, but it latches on "
            f"{orifice.closes_below!r}. The latching level is constructor state on a sealed class "
            f"and cannot be set here — build the break as "
            f"Orifice(..., closes_below=({tank}, {z_breaker}))."
        )

    crest_name = crest_junction_name or f"{tank.name}_crest"
    j_intake, j_crest = Junction(name=f"{crest_name}_intake"), Junction(name=crest_name)
    l_riser = pipe.length * rise / (rise + fall)
    l_downcomer = pipe.length * fall / (rise + fall)
    edges = (
        *pool(tank, outflows={j_intake: z_intake}, fluid=fluid),
        flow_edge(
            (j_intake, j_crest),
            Friction(
                friction_factor, fluid, l_riser, pipe.hydraulic_diameter, pipe.area,
                name=f"{crest_name}_riser_friction",
            ),
            Gravity(fluid, -rise, name=f"{crest_name}_riser"),
        ),
        flow_edge(
            (j_crest, env),
            orifice,
            Gravity(fluid, fall, name=f"{crest_name}_downcomer"),
            Friction(
                friction_factor, fluid, l_downcomer, pipe.hydraulic_diameter, pipe.area,
                name=f"{crest_name}_downcomer_friction",
            ),
        ),
    )
    return edges, orifice
