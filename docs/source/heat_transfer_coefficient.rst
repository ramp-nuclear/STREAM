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

.. _htc-low-flow:

Low flow, reversal and natural circulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The problem
^^^^^^^^^^^

In a loss-of-flow transient the pump coasts down, and the flow in a heated channel passes through zero,
reverses and settles into natural circulation. The wall heat transfer coefficient :math:`h` must follow it,
but its correlations are forced-flow ones. The laminar law is keyed on the Graetz number
:math:`\text{Gz} = \text{Re}\,\text{Pr}\,D_h/x`, with :math:`x` the distance from the inlet. A large Gz
means a thin thermal boundary layer and a high :math:`h`. As the flow dies Gz goes to zero and the law
settles on its fully developed value, Nu about 8 in a wide plate channel.

That zero-flow limit lies outside the data the laminar law was fitted to, so holding it at stagnation has
to be justified. Handing over instead to a natural-convection correlation is tempting, for example
Elenbaas's correlation for natural convection between vertical heated plates. In the LOFA example's hot
channel, a 2.2 mm plate gap, at 10 K superheat, though, Elenbaas gives 136 W/m²K against a laminar
1161 W/m²K (Nu about 8), so :math:`h` drops almost an order of magnitude in the narrow band around zero
flow where such a handover acts, within a few units of Re of zero.

Each cell's wall temperature is the root of its wall balance: the plate's heat flux equals
:math:`h\,\Delta T`, with :math:`\Delta T` the wall superheat. When :math:`h` jumps, the root jumps, and the
solver must cut its step to cross it at every reversal. A switch to natural convection keyed on the
superheat is worse. :math:`h\,\Delta T` can then fall while :math:`\Delta T` rises, and one heat flux has
three wall temperatures. Newton lands on whichever root its guess leads to, so a steady solve depends on
its start and a transient can jump between branches.

What the law does
^^^^^^^^^^^^^^^^^

:func:`~.regime_dependent_h_spl` holds the laminar law at its fully developed value through zero flow, so
:math:`h` is the same on both sides of reversal. This is safe. Fluid at rest between two plates heated at
uniform flux conducts with Nu 12 on :math:`D_h`, above the laminar value of about 8, so the held value
understates :math:`h` at stagnation. Sudo's upflow lower limit at low Graetz number in a narrow channel,
described below, is of the same order, so the held value is not out of line with measurements.

The Elenbaas natural value :math:`h_n` is added to the forced value :math:`h_f` at every flow by the cube
norm :math:`h = (h_f^3 + h_n^3)^{1/3}`, a smooth maximum that only adds. Its share grows with the superheat
and about as the fourth power of the gap. In a 2 mm gap at up to 30 K superheat it adds under 1 %; from
about 4 mm it lifts the low-flow value.

Both switches in the law, the laminar to turbulent interpolation and the direction weight of the downflow
option below, are fed by the flow, never by the wall superheat. The superheat enters only through the
natural term and the film properties, the coolant properties at the film temperature (by default the mean
of the wall and bulk temperatures) that the turbulent law uses. For water both raise :math:`h`, so
:math:`h\,\Delta T` rises monotonically and the balance has one root.

.. plot:: htc_figures.py h_versus_re
   :width: 100%
   :alt: Wall HTC against signed Re for three gaps, flat through zero flow in the 2 mm gap.

   Wall HTC of a 70 mm wide heated channel at 10 K superheat against signed Re, ``re_bounds`` band shaded.
   The 2 mm gap is flat through zero flow; wider gaps rise on the natural term.

.. plot:: htc_figures.py wall_balance
   :width: 100%
   :alt: Wall heat flux against superheat at Re 100 and Re 20, both rising monotonically.

   Wall heat flux against superheat in a 2 mm gap at Re 100 and Re 20. Both rise monotonically and coincide
   on the held laminar value, so the 15 kW/m² line crosses them once.

Orientation
^^^^^^^^^^^

The law takes the direction of gravity as ``g``, with the convention of :func:`~.pressure_diff`: positive
flow runs along gravity when ``g > 0``. A channel whose positive flow runs down passes ``g``, a riser
passes it negated, and a horizontal channel passes ``0.0``, which drops the natural term and the downflow
option.

Heated downflow and the Sudo option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When flow past a heated wall runs along gravity, the warm fluid at the wall is buoyant and pushes against
the flow. In laminar flow this slows the fluid at the wall and lowers :math:`h`, which the forced law cannot
see. Sudo et al. measured it in a 2.25 mm gap heated on both sides: at Re 250 to 400 the downflow
coefficient was about half the laminar value. They give lower envelopes of their laminar data for each
direction as laws in the local Gz, for Gz above 16, with no data below.

``downflow="sudo"`` multiplies the laminar value in flow along gravity by the ratio of the downflow to the
upflow envelope, clipped to :math:`[2/3, 1]` (:func:`~.Sudo_downflow_ratio`). It fades out across
``re_bounds`` with the laminar part. A lower :math:`h` means a hotter wall, the conservative side. The
option is off by default because it rests on lower envelopes from one experiment in one gap.

Below Gz 16 there are no data, and at stagnation the held laminar value is already conservative, so the
law returns the ratio to 1 between Gz 16 and Gz 2, about one decade; the upper end is the data limit. The
lower end is bounded by construction: the ratio is clipped to :math:`[2/3, 1]`, so wherever the return ends,
the band lowers the laminar value by at most a third, and only below Gz 16, at the lowest flows of the
laminar range. In the LOFA example with both pumps tripped, for instance, ending the return anywhere from
Gz 0.002 to Gz 8, or holding the plateau to stagnation, moves the wall temperature of the hot plate against
the default by at most 3 K for less than twenty seconds around the reversal of the flow, and the peak clad
temperature over every band from stagnation up to Gz 2000 spans 0.2 K. Starting the return above Gz 16 is
not a choice of fade, because it cuts Sudo's law inside its data; in the same example it lowers the wall
temperature during the coastdown by up to 7 K for half a minute to more than two minutes. Against the
option off, the option itself raises the wall temperature of the hot plate in the same example by up to
8 K during the laminar downflow of the coastdown, and the peak clad temperature by 0.2 K. The band can be
moved only by calling :func:`~.Sudo_downflow_ratio` directly with its ``fade`` argument.

The local Gz needs ``develop_length``, which a channel supplies per cell, counted from the end the flow
enters.

Using it
^^^^^^^^

.. code-block:: python

   h_spl = spl_htc("regime_dependent", re_bounds=(2000.0, 5000.0), aspect_ratio=gap / width,
                   Lh=heated_length, g=g, downflow="sudo")

- ``re_bounds``: where laminar flow ends and turbulent flow begins in this geometry; :math:`h` is
  interpolated linearly between them. Sudo's narrow-channel data put the transition at 2000 to 4000; the
  figures and the LOFA example use 2000 to 5000.
- ``aspect_ratio``: the gap over the width, used by the laminar law.
- ``Lh``: the heated length, used by Elenbaas.
- ``g``: the gravity signed by the channel's orientation, as above.
- ``downflow``: ``"sudo"`` for a heated channel with laminar flow along gravity when the conservative wall
  temperature is wanted.
- ``develop_length``: not passed here; the channel supplies it per cell.


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
