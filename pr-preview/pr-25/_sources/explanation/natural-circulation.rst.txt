=====================================
Natural circulation and flow reversal
=====================================

A pool reactor that loses its pumps keeps its core cooled by buoyancy: the water in the heated
channels is lighter than the water around them, and the difference in weight drives a flow with no
pump in the loop. This page explains where that head comes from, why a core cooled downward reverses, what a
reversal asks of the model, and why the wall law matters most once the forced flow is gone. :doc:`/LOFABothPumpsTrip` shows the whole sequence on a pool reactor,
and :doc:`/howto/get-a-steady-solve-to-converge` solves a small natural-circulation loop.

Where the head comes from
=========================

Around a closed loop the hot column of water pushes up and the cold one pushes down, and the only pressure source left
when the pumps are gone is that difference:

.. math:: \Delta p_b = g\,\Delta h\,(\rho_c - \rho_h)

It is small: in the both-pumps case the
hot plate's buoyancy head at the end of the hour is about 39 Pa, against about 6807 Pa for the
weight of its 0.7 m column. The loop settles at the flow where friction spends that head, which is
why a pool reactor offers buoyancy a short path: the flapper joins the plenum below the core to the
pool beside it, so the natural-circulation loop is the core, the pool and the flapper, not the pump
legs and the heat exchanger.

Why the core reverses
=====================

In forced flow the pumps drive the water down through the core, and each heated channel's buoyancy
pushes up against them as a small correction. As the flywheels run down, the downward head falls below the buoyancy of the
hottest channels, their flow passes through zero and turns upward.

Parallel channels do not reverse together. The one with the most power per unit flow has the most
buoyancy and turns first, and for as long as some channels have turned and others have not, the
plena carry the difference: water goes down the cold channels and up the hot ones, a recirculation
inside the core that no single channel describes. :doc:`/LOFABothPumpsTrip` times each group's
reversal.

What a reversal does to the model
=================================

A channel is differenced upwind, so its inlet is the end the water comes from. When the flow
reverses, the inlet becomes the outlet, and every component needs the temperature it would be fed
from the other side. That is what ``Tin`` and ``Tin_minus`` are: ``Tin`` is the temperature from
the neighbour upstream along the drawn edge, ``Tin_minus`` from the neighbour downstream, and
:func:`~stream.utilities.directed_Tin` picks one by the sign of the flow, blending the two
within ``1e-3`` kg/s of zero so that the residual has no jump at the reversal. A
:class:`~stream.calculations.Junction` mixes its streams the same way: each stream's weight is a
smooth positive part of its flow into the junction, so a stream that turns to leave the junction
stops feeding it heat, to within that smoothing.

Two consequences follow for how a loop is drawn. A channel is always modelled as running downward
along its edge: its physics reads only the cell lengths of ``z_boundaries``, so reversing the
boundaries builds the same channel when the cells are uniform and the same cells in reverse order
when they are not, and a leg that rises is a
:class:`~stream.calculations.Gravity` with a negative ``disposition``. And a gravity leg weighs the
water it is fed: with one heat exchanger at one end, reversed flow reaches it hot from the other end
and the loop loses the cold column it circulates on. A
:class:`~stream.calculations.HeatExchanger` sets its outlet temperature whatever its inlet, so
placing one on each side of the leg keeps the column cold whichever way the water runs.

The wall law at low flow
========================

A purely turbulent correlation loses the plate's heat transfer as the flow dies;
:doc:`/LOFAPoolReactor` measures what that costs the plate. The regime-dependent law keeps a floor
by construction. Its forced part interpolates on the Reynolds number between a laminar and a
turbulent correlation; natural convection is then added to it by superposition, the cube root of the
sum of the cubes, so buoyancy only ever raises the coefficient; and once the through-flow no longer
renews the water in the channel, measured by the Graetz number, the law hands over to the
natural-convection value of a stagnant channel. Every switch reads the flow, never the wall
superheat the coefficient feeds back into, so the coefficient stays finite through a reversal. It is the default of ``stream.examples.lofa``.

The flows and their signs are also what a natural-circulation steady solve needs from its guess;
:doc:`steady-states-and-guesses` explains why, and :doc:`sign-conventions` collects the signs.
