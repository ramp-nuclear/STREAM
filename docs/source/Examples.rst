Examples
========

The examples are meant to be read in the order below, from a loop of three components to a pool reactor through a
loss of flow.

From a first loop to a fuel rod
-------------------------------

- :doc:`quickstart` builds a pump, a pipe and a cooler into a loop, solves its steady state and checks the flow by
  hand.
- :doc:`SeriesRLCCircuit` writes the resistor, the inductor and the capacitor as three :ref:`Calculation` classes
  of its own and solves the circuit in time.
- :doc:`TheCubeProblem` wires twelve resistors on the edges of a cube with junctions and recovers the cube's
  equivalent resistance.
- :doc:`PCSCoastdown` adds a flywheel to a pump and a resistor and compares the coastdown transient with its
  analytic solution.
- :doc:`SampleFuelPlateAndChannel` couples a fuel plate to the coolant channel that cools it.
- :doc:`MultiplePlatesInRod` stacks such plates and channels into a rod and solves it in a loop with a pump and a
  heat exchanger.

.. toctree::
   :hidden:

   quickstart.ipynb
   SeriesRLCCircuit.ipynb
   TheCubeProblem.ipynb
   PCSCoastdown.ipynb
   SampleFuelPlateAndChannel.ipynb
   MultiplePlatesInRod.ipynb

More calculations
-----------------

- :doc:`CoupledSprings` writes one calculation for a mass between two springs and couples two of them.
- :doc:`PlanarPendulum` writes a pendulum as one calculation with an algebraic constraint for its string.

.. toctree::
   :hidden:

   CoupledSprings.ipynb
   PlanarPendulum.ipynb

Loss of flow
------------

A pool reactor assembled step by step from the library's components, shipped as :doc:`the examples package <stream.examples>`, then taken
from its operating point through a loss of flow into natural circulation.

- :doc:`LOFABuildingTheLoop` puts the reactor together one part at a time and solves a first steady state.
- :doc:`LOFAPoolReactor` solves the operating point and checks it against hand estimates.
- :doc:`LOFABothPumpsTrip` follows both pumps tripping, through the reactor trip and the flapper opening, into
  natural circulation.
- :doc:`LOFAVariations` trips one pump, keeps the flapper shut, and measures how far the reported times and
  temperatures can be trusted.

.. toctree::
   :hidden:

   LOFABuildingTheLoop.ipynb
   LOFAPoolReactor.ipynb
   LOFABothPumpsTrip.ipynb
   LOFAVariations.ipynb
