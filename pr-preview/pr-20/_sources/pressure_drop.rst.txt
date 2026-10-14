=============
Pressure Drop
=============

The flow of coolants through the core, pumps, heat exchangers and
other coolant cycle components is determined by means of pressure differences.
These differences (drops) depend on the cycle itself, and can be divided into several categories.

First and foremost, there are pressure drops due to :ref:`friction <Friction>` of the coolant with the surrounding walls
and due to :ref:`local <Local>` changes in flow area, direction and more.
These kinds of pressure drops tend to be close to
:math:`\Delta p\propto \rho v^2`, and are usually described by an effective
relation following this reasoning:

.. math:: \Delta p = f \frac{\rho u^2}{2}

Where :math:`f` (sometimes denoted :math:`K`) may be dependent non-trivially on
geometry, flow properties and regime.

Other contributions come from :ref:`gravity <Gravitational Pressure>` and :ref:`inertia <Inertial Pressure>`.

.. [#Blasius] Blasius, H., "Das Aehnlichkeitsgesetz bei Reibungsvorgängen in
    Flüssigkeiten", Mitteilungen über Forschungsarbeiten auf dem Gebiete
    des Ingenieurwesens, vol.134, VDI-Verlag Berlin (1913).
.. [#KAERI] KAERI, "Development of Research Reactor Technology",
    Korea Atomic Energy Research Institute (KAERI), KAERI/RR-3818/2014, 2014.


Friction
========

.. automodule:: stream.physical_models.pressure_drop.friction
   :members:

Local
=====

.. automodule:: stream.physical_models.pressure_drop.local
   :members:


Other Pressure Drop Components
==============================

Gravitational Pressure
----------------------
.. autofunction:: stream.physical_models.pressure_drop.gravity_pressure

.. seealso::
    :class:`~stream.calculations.ideal.resistors.Gravity` - a corresponding :ref:`Calculation`.

Inertial Pressure
-----------------
.. autofunction:: stream.physical_models.pressure_drop.inertia_pressure

.. seealso::
    :class:`~stream.calculations.ideal.inertia.Inertia` - a corresponding :ref:`Calculation`.


Helper Functions
================
A ready made :class:`~stream.calculations.channel.Channel`-compatible pressure function is provided:

.. autofunction:: stream.physical_models.pressure_drop.pressure_diff

.. autofunction:: stream.physical_models.pressure_drop.static_pressure
