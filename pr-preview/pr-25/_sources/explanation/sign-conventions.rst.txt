================
Sign conventions
================

Every sign in a loop follows from a few rules about the drawn edges. This page states the rules,
then lists the names that carry a sign, an owner or a mirror. :doc:`/howto/wire-a-flow-network`
wires a loop under them, and :doc:`/LOFABuildingTheLoop` draws the pool reactor by them.

The rules
=========

**A flow's sign is along the drawn edge.** :func:`~stream.composition.flow_edge` places its
components in series from the edge's tail to its head, and a positive flow runs that way. Nothing
else fixes a direction: a reversed edge that still closes its loop builds a different plant, with
its pump pushing the other way.

**The pressure is a gain.** Each component's ``pressure`` is the rise from its inlet to its outlet along
the edge, and around every loop the gains sum to zero. A :class:`~stream.calculations.Pump` holds
its head, positive; a :class:`~stream.calculations.Resistor`, a friction element or a check valve is
negative for forward flow; an :class:`~stream.calculations.Inertia` is minus its inertance times the
flow's derivative.

**Down is positive.** A :class:`~stream.calculations.Gravity` returns :math:`\rho g \Delta h` for a
``disposition`` :math:`\Delta h`, a gain for a positive one, so a positive ``disposition`` is a drop
in height along the edge and a riser takes a negative one.
:func:`~stream.physical_models.pressure_drop.pressure_diff`, the channel's pressure law, takes
positive flow as downward, and a channel's physics reads only the cell lengths of its ``z_boundaries``: a
channel is always modelled as running down its edge, so heated channels are drawn top to bottom
and a rising leg is written with a ``Gravity``, not by reversing the boundaries, which on uneven
cells only reverses their order.

**Heat flux is positive into the fluid.** A channel computes its wall fluxes as the coefficient
times the wall temperature less the bulk temperature, so a wall hotter than the water heats it.

**Left and right are mirrored across a contact.** The consumer names the side it sees and the
supplier answers with the face that looks at it: a :class:`~stream.calculations.Fuel` asked for
``T_left`` answers with its own ``T_wall_right``, and a
:class:`~stream.calculations.ChannelAndContacts` asked for ``h_left`` answers with its own
``h_right``.

The names
=========

.. list-table::
   :header-rows: 1
   :widths: 14 24 10 52

   * - Name
     - Owner
     - Unit
     - Sign and meaning
   * - ``mdot``
     - :class:`~stream.calculations.Kirchhoff`, one row per edge
     - kg/s
     - Positive from the edge's tail to its head. A junction receives a dict, one flow per adjacent
       edge.
   * - ``mdot2``
     - :class:`~stream.calculations.KirchhoffWDerivatives`, one row per edge
     - kg/s²
     - The time derivative of the same edge's ``mdot``, same orientation.
   * - ``pressure``
     - every component on an edge
     - Pa
     - Gain from inlet to outlet along the edge; one number for a whole channel.
   * - ``disposition``
     - :class:`~stream.calculations.Gravity`, a constructor argument
     - m
     - Positive when the outlet sits below the inlet; the gain is then positive.
   * - ``p_abs``
     - ``Kirchhoff``, one row per entry of ``abs_pressure_comps``
     - Pa
     - Absolute pressure at the component's inlet in edge orientation: the reference pressure plus
       the gains on the path to it. It follows the edge, not the flow.
   * - ``Tin``
     - lumped components and junctions; a channel serves its last cell under the name
     - °C
     - The temperature handed downstream along the edge: a lumped component's outlet, a channel's
       last cell, a junction's mixed temperature. As an input, the temperature from the upstream
       neighbour, used for positive flow.
   * - ``Tin_minus``
     - served, never owned
     - °C
     - The temperature handed upstream: a channel's first cell, a lumped component's single
       temperature, a junction's mixed temperature. As an input, the temperature from the downstream neighbour, used for negative
       flow.
   * - ``T_wall_left``, ``T_wall_right``
     - ``Fuel``, algebraic rows
     - °C
     - The plate's own wall temperatures, reached by a neighbour as ``T_right`` and ``T_left``.
   * - ``h_left``, ``h_right``
     - ``ChannelAndContacts``, algebraic rows
     - W/m²K
     - The channel's wall coefficients, non-negative, served swapped. The flux's sign comes from the
       temperature difference.
   * - ``power``
     - :class:`~stream.calculations.PointKinetics`, its first row
     - W
     - Positive. On :class:`~stream.calculations.PointKineticsWInput` it is the total power, fission
       power being ``pk_power``.
   * - ``reactivity``
     - written by ``PointKinetics.save``
     - Δk/k
     - The control worth plus the temperature feedback. ``temp_worth`` is negative for negative
       feedback.

A lumped component holds one temperature, its outlet, and serves it as both ``Tin`` and
``Tin_minus``; :func:`~stream.utilities.directed_Tin` picks between the two inputs by the sign of the
flow. :doc:`natural-circulation` explains what that choice does at a reversal, and
:doc:`steady-states-and-guesses` why a guess with a wrong sign fails.
