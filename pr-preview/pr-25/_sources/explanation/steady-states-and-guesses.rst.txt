==========================
Steady states and guesses
==========================

A transient in this library starts from an operating point, and an operating point is a root of
the system's equations with every time derivative at zero. This page explains why, what a steady
solve needs from its guess, what the library builds of it, and what the two steps of the solve do. :doc:`/howto/get-a-steady-solve-to-converge`
does the task, and :doc:`/LOFABuildingTheLoop` seeds a full pool reactor by hand.

Why a transient starts from a steady state
==========================================

Started anywhere other than an operating point, a transient first shows the plant relaxing toward its own steady state,
mixed into the response to the change it was meant to show. The steady state also
gives the transient a consistent start: the differential rows of the initial vector, the coolant and plate temperatures
and, on a loop with inertia, the flows, are where IDA begins, and by default it makes only the algebraic rows and the
derivatives consistent with them, so a differential row that is off is kept as given.

What a guess has to get right
=============================

A steady solve is a Newton-type iteration, and it converges from a starting point close enough to
the root. Most of the vector is cheap to get close: a pressure follows from a flow through each
component's own law, a coolant temperature from the power, the flow and the heat capacity, a wall
coefficient from the wall law at a guessed temperature. The flows are the part that has to be right,
because every other row is computed from them, and of the flows the sign matters most. A guess with
the water going the wrong way through a heated channel feeds the channel from the wrong end and
sets its buoyancy against the flow it should drive. The natural-circulation loop of the how-to
fails from a forced-flow guess and converges from two upward guesses that differ threefold in
magnitude: there the magnitude does not matter and the sign does.

What the library builds and what it does not
============================================

:meth:`~stream.composition.FlowGraph.guess_steady_state` builds the hydraulic part from flows it is
given. It needs a flow on one component of every edge and raises
:class:`~stream.composition.MissingFlowError` naming the edges without one: it does not complete the
flows from the junction balances, and it does not choose a sign. From those flows and one
temperature it computes each component's pressure, sets the temperatures uniform, evaluates the
wall coefficients of each wall-coupled channel and adds up the absolute pressures from the reference. A pump that
imposes its flow, or a closed flapper, has no pressure that follows from the flow, and its guess is
zero. :func:`~stream.composition.symmetric_plate_steady_state` solves a plate and its channel at a
given flow, and :func:`~stream.composition.point_kinetics_steady_state` puts the kinetics at a given
power and zero reactivity; :meth:`~stream.state.State.merge` lets the later seed win where two overlap.
For a network coupled to plates, :func:`~stream.composition.steady_state.guess_steady_state` builds the whole guess in
one call: it completes the flows the junction balances and the known flows fix, walks the temperatures
around the loop, then solves the hydraulics and the plates one block at a time. It still raises
:class:`~stream.composition.subsystems.MissingFlowError` for a heated channel whose flow nothing on its loop fixes.
Where nothing on a loop fixes its flow, the flow and its sign are what one brings.

The two steps of a steady solve
===============================

:meth:`~stream.aggregator.Aggregator.solve_steady` first calls scipy's ``root`` with the options it
is given. When ``root`` reports that it did not converge, the
library raises an :class:`~stream.solvers.AlgRuntimeError`, and with the default
``globalize="auto"`` that error, and only that one, sends the solve on to
:func:`~stream.solvers.scaled_newton`; any other exception from scipy propagates.
``globalize=True`` skips scipy and ``globalize=False`` lets the error propagate. The fallback exists for guesses far from the root, where the
globalisation of scipy's default method stalls.

:func:`~stream.solvers.scaled_newton` is Newton's method with a line search: each step is halved
until the norm of the residual falls. The
damping is what makes it converge. Each linear solve is equilibrated, rows then columns, as insurance for a
badly scaled system, pressures of about 10⁵ Pa beside temperatures of about 100 °C, and a small
Tikhonov term keeps a rank-deficient step finite. It stops when the residual norm falls
below ``1e-6``, and raises after 100 iterations or when no step decreases the residual, with the
best iterate attached as the error's ``y`` for diagnosis. ``scales`` sets the nominal magnitudes
behind its finite-difference step and is read only when no ``jac`` is passed; with a ``jac`` both
steps use it.

A failure of both steps says the guess is outside the basin or the equations have no root there. A
wrong flow sign is the first suspect. A missing edge flow fails earlier, at the guess; :meth:`~stream.composition.FlowGraph.check_gravity_mismatch` finds a loop
whose heights do not close at zero flow;
:doc:`/howto/diagnose-a-failing-solve` goes through them, and :doc:`natural-circulation` explains
why a natural-circulation loop is the case where the sign decides.
