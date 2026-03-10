=========================
Heat Transfer Coefficient
=========================

For more information regarding limits, please see :ref:`Thresholds`.

**Some explanation regarding the jargon of this module:**

- SPL - Single Phase Liquid region.
- INC - Incipience Point, beyond which boiling occurs.
- SCB - Subcooled Boiling region.


In this module several functions relating to the calculation of the
heat transfer coefficient between coolant and walls are defined.
The most significant point of contact between this subpackage and the rest of :ref:`STREAM <STREAM>`
is through the :func:`~stream.physical_models.heat_transfer_coefficient.wall_heat_transfer_coeff` function:

.. autofunction:: stream.physical_models.heat_transfer_coefficient.wall_heat_transfer_coeff

The last five arguments are functions used to control the general behavior of this function. We shall walk through those
options one by one:

#. :ref:`Single Phase Liquid (SPL)`
#. :ref:`Subcooled Boiling (SCB)`
#. :func:`Partial SCB <stream.physical_models.heat_transfer_coefficient.subcooled_boiling.Bergles_Rohsenhow_partial_SCB>`
#. :func:`Film <stream.physical_models.heat_transfer_coefficient.temperatures.film_temperature>`
#. :func:`Incipience <stream.physical_models.heat_transfer_coefficient.temperatures.Bergles_Rohsenow_T_ONB>`


.. [#BR] A. E. Rohsenow, W. M. Bergles, "The Determination of Forced Convection
        Surface Boiling Heat Transfer", Journal of Heat Transfer, 1964, p. 385
.. [#Collier] J. G. Thome, J. R. Collier, "Convective Boiling and Condensation",
    Oxford Science, 1994


Single Phase Liquid (SPL)
=========================

.. autofunction:: stream.physical_models.heat_transfer_coefficient.single_phase.spl_htc

.. autoclass:: stream.physical_models.heat_transfer_coefficient.SinglePhaseLiquidHTC
    :special-members: __call__
    :undoc-members:
    :show-inheritance:

.. autoclass:: stream.physical_models.heat_transfer_coefficient.single_phase.SinglePhaseLiquidHTCExArgs
    :special-members: __call__
    :undoc-members:
    :show-inheritance:

Turbulent Heat Transfer Coefficient
-----------------------------------

.. automodule:: stream.physical_models.heat_transfer_coefficient.turbulent
   :undoc-members:
   :members:


Laminar Heat Transfer Coefficient
---------------------------------

.. automodule:: stream.physical_models.heat_transfer_coefficient.laminar
   :undoc-members:
   :members:

Natural (Free) Convection Heat Transfer Coefficient
---------------------------------------------------

.. autofunction:: stream.physical_models.heat_transfer_coefficient.natural_convection.Elenbaas_h_spl

Composed HTC Functions
----------------------

.. autofunction:: stream.physical_models.heat_transfer_coefficient.single_phase.regime_dependent_h_spl
.. autofunction:: stream.physical_models.heat_transfer_coefficient.single_phase.maximal_h_spl


Incipience (INC)
================

.. autoclass:: stream.physical_models.heat_transfer_coefficient.IncipienceTemperatureFunction
    :special-members: __call__
    :undoc-members:
    :show-inheritance:

.. autofunction:: stream.physical_models.heat_transfer_coefficient.temperatures.Bergles_Rohsenow_T_ONB

Subcooled Boiling (SCB)
=======================

The subcooled boiling region is identified with a fully subcooled flux function, whose protocol is

.. autoclass:: stream.physical_models.heat_transfer_coefficient.SubCooledBoilingFluxFunction
    :special-members: __call__
    :undoc-members:
    :show-inheritance:

Several SCB flux correlations are available:

.. autofunction:: stream.physical_models.heat_transfer_coefficient.subcooled_boiling.McAdams_SCB_heat_flux
.. autofunction:: stream.physical_models.heat_transfer_coefficient.subcooled_boiling.Bergles_Rohsenhow_SCB_heat_flux
.. autofunction:: stream.physical_models.heat_transfer_coefficient.subcooled_boiling.regime_dependent_q_scb

and an interpolation scheme, which is
:func:`~stream.physical_models.heat_transfer_coefficient.subcooled_boiling.Bergles_Rohsenhow_partial_SCB`.

.. autofunction:: stream.physical_models.heat_transfer_coefficient.subcooled_boiling.Bergles_Rohsenhow_partial_SCB

In protocol form:

.. autoclass:: stream.physical_models.heat_transfer_coefficient.PartialSCBFactorFunction
    :special-members:
    :undoc-members:
    :show-inheritance:

Wall Temperature Closure
========================

.. autofunction:: stream.physical_models.heat_transfer_coefficient.temperatures.wall_temperature

Note that the wall temperature is also used to determine the film temperature:

.. autoclass:: stream.physical_models.heat_transfer_coefficient.FilmFunction
    :special-members: __call__
    :undoc-members:
    :show-inheritance:

.. autofunction:: stream.physical_models.heat_transfer_coefficient.temperatures.film_temperature
