=============
How-to guides
=============

A how-to is one task, with code the documentation build runs.

Building
========

- :doc:`write-a-calculation`: when no component in the library does what a part of the system does.
- :doc:`wire-a-flow-network`: when components must sit on the edges of a loop and share its flows.
- :doc:`choose-a-channel-and-its-laws`: when coolant flows past a heated wall.
- :doc:`build-a-plate-or-rod`: when the heat comes from a fuel plate, an assembly of plates or a rod.
- :doc:`bind-inputs-and-schedules`: when an input is a constant, a step or a table in time.
- :doc:`add-point-kinetics-and-a-trip`: when the power follows reactivity and the rods drop on a trip.
- :doc:`add-decay-heat`: when the power after shutdown matters.

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

- :doc:`get-a-steady-solve-to-converge`: when the steady solve fails or lands on the wrong flow direction.
- :doc:`run-a-transient-and-set-tolerances`: when a transient starts from an operating point.
- :doc:`trip-stop-and-restart`: when a run stops at an event and continues after it.
- :doc:`diagnose-a-failing-solve`: when a solve dies or its answer is wrong.

.. toctree::
   :maxdepth: 1
   :hidden:

   get-a-steady-solve-to-converge
   run-a-transient-and-set-tolerances
   trip-stop-and-restart
   diagnose-a-failing-solve

Reading
=======

- :doc:`read-results`: when a quantity is wanted out of a steady state or a transient.
- :doc:`compute-margins`: when the question is how far the plate is from boiling.

.. toctree::
   :maxdepth: 1
   :hidden:

   read-results
   compute-margins

Plotting
========

- :doc:`plot-a-sweep`: when the answer is a figure of a steady-state sweep over design parameters.

.. toctree::
   :maxdepth: 1
   :hidden:

   plot-a-sweep
