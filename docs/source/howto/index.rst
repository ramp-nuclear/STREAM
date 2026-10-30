=============
How-to guides
=============

A how-to is one task, with code the documentation build runs.

Building
========

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - write a component of your own as a calculation
     - :doc:`write-a-calculation`
   * - put components on the edges of a loop and solve its flows
     - :doc:`wire-a-flow-network`
   * - pick the coolant channel past a plate and its wall and friction laws
     - :doc:`choose-a-channel-and-its-laws`
   * - put a fuel plate behind a channel and multiply it into a rod
     - :doc:`build-a-plate-or-rod`
   * - feed a calculation a constant, a step or a table in time
     - :doc:`bind-inputs-and-schedules`
   * - add reactor kinetics with a reactivity controller and a scram
     - :doc:`add-point-kinetics-and-a-trip`
   * - add the heat that lingers after shutdown
     - :doc:`add-decay-heat`

.. toctree::
   :maxdepth: 1
   :hidden:

   write-a-calculation
   wire-a-flow-network
   choose-a-channel-and-its-laws
   build-a-plate-or-rod
   bind-inputs-and-schedules
   add-point-kinetics-and-a-trip
   add-decay-heat

Solving
=======

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - seed a steady solve that fails or lands on the wrong flow direction
     - :doc:`get-a-steady-solve-to-converge`
   * - run a transient from an operating point and choose its tolerances
     - :doc:`run-a-transient-and-set-tolerances`
   * - stop a run at an event the solver locates, and restart it
     - :doc:`trip-stop-and-restart`
   * - find why a solve dies or lands somewhere wrong
     - :doc:`diagnose-a-failing-solve`

.. toctree::
   :maxdepth: 1
   :hidden:

   get-a-steady-solve-to-converge
   run-a-transient-and-set-tolerances
   trip-stop-and-restart
   diagnose-a-failing-solve

Reading
=======

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - read a steady state or a transient by name, as a table or a history
     - :doc:`read-results`
   * - add boiling limits to a solution, at steady state and over a transient
     - :doc:`compute-margins`

.. toctree::
   :maxdepth: 1
   :hidden:

   read-results
   compute-margins

Plotting
========

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You want to
     - Read
   * - draw a sweep of steady states as profiles, curves, maps and required values
     - :doc:`plot-a-sweep`

.. toctree::
   :maxdepth: 1
   :hidden:

   plot-a-sweep
