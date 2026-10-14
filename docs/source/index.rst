.. _STREAM:

STREAM: `System Thermohydraulics for Reactor Evaluation, Analysis & Modeling`
=============================================================================

STREAM builds and solves systems of coupled differential algebraic equations,
:math:`M\dot{y} = F(y, t)`, put together from calculations that each own a slice of the state vector
:math:`y`. It ships thermal-hydraulic components, from pumps and channels to fuel plates and point
kinetics, with the correlations behind them, for modelling the heat removal of research reactors.

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - write your first loop and check it by hand
     - :doc:`A first loop </quickstart>`
   * - write a component of your own as a calculation
     - :doc:`Write a calculation </howto/write-a-calculation>`
   * - put components on the edges of a loop and solve its flows
     - :doc:`Wire a flow network </howto/wire-a-flow-network>`
   * - pick the coolant channel past a plate and its wall and friction laws
     - :doc:`Choose a channel and its laws </howto/choose-a-channel-and-its-laws>`
   * - put a fuel plate behind a channel and multiply it into a rod
     - :doc:`Build a plate, an assembly or a rod </howto/build-a-plate-or-rod>`
   * - feed a calculation a constant, a step or a table in time
     - :doc:`Bind inputs and time schedules </howto/bind-inputs-and-schedules>`
   * - add reactor kinetics with a reactivity controller and a scram
     - :doc:`Add point kinetics, feedback and a trip </howto/add-point-kinetics-and-a-trip>`
   * - add the heat that lingers after shutdown
     - :doc:`Add decay heat </howto/add-decay-heat>`
   * - seed a steady solve that fails or lands on the wrong flow direction
     - :doc:`Get a steady solve to converge </howto/get-a-steady-solve-to-converge>`
   * - run a transient from an operating point and choose its tolerances
     - :doc:`Run a transient and set tolerances </howto/run-a-transient-and-set-tolerances>`
   * - stop a run at an event the solver locates, and restart it
     - :doc:`Trip, stop and restart </howto/trip-stop-and-restart>`
   * - find why a solve dies or lands somewhere wrong
     - :doc:`Diagnose a failing solve </howto/diagnose-a-failing-solve>`
   * - read a steady state or a transient by name, as a table or a history
     - :doc:`Read results </howto/read-results>`
   * - add boiling limits to a solution, at steady state and over a transient
     - :doc:`Compute margins </howto/compute-margins>`
   * - see a whole reactor built and run through a loss of flow
     - :doc:`Building the Pool Reactor Loop </LOFABuildingTheLoop>` and the three pages after it, or
       :doc:`/Examples`
   * - learn the physics behind a heat transfer, friction or boiling law
     - :doc:`Physical Models </stream.physical_models>`
   * - understand why a solve converges or a flow reverses
     - :doc:`Explanation </explanation/index>`
   * - look a name up
     - :doc:`Documentation </stream>`

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
   quickstart
   howto/index
   Examples
   explanation/index
   stream
   Contributing
   tests
