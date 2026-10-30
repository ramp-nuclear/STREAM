.. _STREAM:

STREAM: `System Thermohydraulics for Reactor Evaluation, Analysis & Modeling`
=============================================================================

STREAM builds and solves systems of coupled differential algebraic equations,
:math:`M\dot{y} = F(y, t)`, put together from calculations that each own a slice of the state vector
:math:`y`. It ships thermal-hydraulic components, from pumps and channels to fuel plates and point
kinetics, with the correlations behind them, for modelling the heat removal of research reactors.

Where to look
-------------

The documentation has four parts:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - learn the package from a first loop to a loss of flow
     - the :doc:`Examples </Examples>`, in order
   * - get a specific task done
     - the :doc:`How-to guides </howto/index>`
   * - understand the physics and the reasoning behind the library
     - the :doc:`Explanation </explanation/index>` pages
   * - look up a name, its arguments, or the correlation behind a physical model
     - the :doc:`Documentation </stream>` of each module

Conventions
-----------

The library holds to a few conventions throughout; :doc:`/explanation/sign-conventions` collects the signs.

- Temperatures are in °C: the property functions of ``light_water`` take °C.
- Absolute pressure is in pascals, wherever ``p_abs`` is routed to a component, such as a heated channel,
  through ``abs_pressure_comps`` on :class:`~stream.composition.FlowGraph`.
- A component's ``pressure`` is the gain from its inlet to its outlet: a pump's head is positive, a loss is
  negative, and the gains cancel around every loop.
- A flow is positive along its drawn edge, from the first junction of :func:`~stream.composition.flow_edge`'s ``(u, v)`` to the second.
- A :class:`~stream.calculations.Gravity`'s ``disposition`` is positive when the outlet is below the inlet, so a riser takes a negative one.
- A plate and its channel share the same ``z_boundaries``, as :doc:`/howto/build-a-plate-or-rod` states.

.. toctree::
   :maxdepth: 2
   :hidden:

   Installation
   Examples
   howto/index
   explanation/index
   stream
   Contributing
   tests
