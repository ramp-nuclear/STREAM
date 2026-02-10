Calculations
============
.. automodule:: stream.calculations
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2

Summary of Available Calculations
---------------------------------

.. inheritance-diagram::
  Channel
  ChannelAndContacts
  Flapper
  Friction
  Fuel
  Gravity
  HeatExchanger
  Inertia
  Junction
  Kirchhoff
  KirchhoffWDerivatives
  LocalPressureDrop
  PointKinetics
  PointKineticsWInput
  Pump
  RegimeDependentFriction
  Resistor
  ResistorSum
  ConicalContraction
  Bend
  :parts: 1
  :top-classes: stream.calculation.Calculation

.. autosummary::
    :nosignatures:

    Channel
    ChannelAndContacts
    Flapper
    Friction
    Fuel
    Gravity
    HeatExchanger
    Inertia
    Junction
    Kirchhoff
    KirchhoffWDerivatives
    LocalPressureDrop
    PointKinetics
    PointKineticsWInput
    Pump
    RegimeDependentFriction
    Resistor
    ResistorSum
    ConicalContraction
    Bend

Heat Diffusion
--------------

.. automodule:: stream.calculations.heat_diffusion
   :members:
   :undoc-members:
   :show-inheritance:

Point Kinetics
--------------

.. automodule:: stream.calculations.point_kinetics
   :members:
   :undoc-members:
   :show-inheritance:

Channel
-------

.. automodule:: stream.calculations.channel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: stream.calculations.channel.coolant_first_order_upwind_dTdt


LumpedComponent
---------------

.. automodule:: stream.calculations.ideal.ideal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: stream.calculations.ideal
   :members:
   :undoc-members:
   :show-inheritance:

Pumps
^^^^^

.. automodule:: stream.calculations.ideal.pumps
   :members:
   :undoc-members:
   :show-inheritance:

Resistors
^^^^^^^^^

.. automodule:: stream.calculations.ideal.resistors
   :members:
   :undoc-members:
   :show-inheritance:

Heat Exchangers
^^^^^^^^^^^^^^^

.. automodule:: stream.calculations.ideal.heat_exchangers
   :members:
   :undoc-members:
   :show-inheritance:

Inertia
^^^^^^^

.. automodule:: stream.calculations.ideal.inertia
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: stream.calculations.ideal.inertia.bilinear

Flapper
-------
.. automodule:: stream.calculations.flapper
   :members:
   :undoc-members:
   :show-inheritance:

Kirchhoff
---------

.. automodule:: stream.calculations.kirchhoff
   :members:
   :undoc-members:
   :show-inheritance:

At one time, we thought Kirchhoff would be a passing craze,
and several more advanced models were deemed worthy candidates for its replacement.
For this reason, Aviv wrote a Eulogy:

Eulogy for The Kirchhoff Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. line-block::
    `So merry were the times we spent,`\
    `in cloudless night and peaceful day.`\
    `So many ways we came and went,`\
    `in questions deep the answers lay.`

    `No song or rhyme could captivate,`\
    `In simpler terms the laws of flow.`\
    `While rules complex quickly abate,`\
    `Thy legend will forever grow.`

    `Alas! Our days came to ends abrupt,`\
    `and tomorrow is always clad in veil.`\
    `No companion could have been more apt,`\
    `and by thy works we shall prevail!`

However, this was not the case, and Kirchhoff lives on.
