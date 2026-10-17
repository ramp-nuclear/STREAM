Visualization
=============

The reference for :mod:`stream.viz`. :doc:`/howto/plot-a-sweep` shows how to use it, one task at a time.

.. autofunction:: stream.viz.cases

.. autofunction:: stream.viz.run_cases

.. autoclass:: stream.viz.Sweep
   :members: profile, field, reduce, required
   :member-order: bysource

.. automethod:: stream.viz.Sweep.single

.. automethod:: stream.viz.Sweep.levels

.. automethod:: stream.viz.Sweep.case_of

.. autofunction:: stream.viz.at_least

.. autofunction:: stream.viz.at_most

A reducer, the second argument of :meth:`~stream.viz.Sweep.reduce` and the ``reduce`` argument of :meth:`~stream.viz.Sweep.required`, collapses the cells of each case:

=================================================  ==========================================
Reducer                                            Value
=================================================  ==========================================
``"min"``                                          the smallest cell value
``"max"``                                          the largest cell value
``"mean"``                                         the mean weighted by cell size
``at(z=...)``, ``at(x=...)`` or ``at(cell=...)``   the value of one cell
``"where_min"``                                    the coordinate of the smallest value
``"where_max"``                                    the coordinate of the largest value
a callable ``f(values, coords)``                   whatever it returns
=================================================  ==========================================

.. autofunction:: stream.viz.at

.. autoclass:: stream.viz.Result()
   :members: plot, ylabel, subtitle

.. autoclass:: stream.viz.Label

.. autoclass:: stream.viz.Labels

.. autoclass:: stream.viz.Style
