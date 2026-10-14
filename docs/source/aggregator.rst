Aggregator
----------

.. automodule:: stream.aggregator
   :members:
   :no-undoc-members:
   :show-inheritance:

Example
~~~~~~~
Consider three calculations, *A*, *B* and *C*, where *A* depends on *B* and on *C*:
the graph of the Aggregator has an edge from *B* to *A* and an edge from *C* to *A*.
*A* owns a subset of the entire vector :math:`\vec{y}_A\in\vec{y}`, which it
receives as input. It also reads values owned by the other two:
:math:`\vec{y}_i` owned by *B* and :math:`\vec{y}_j` owned by *C*.
Thus, *B* (e.g.) must be able to show the Aggregator the **place** in which
it stores :math:`\vec{y}_i`.
In this manner, there exists no communication between Calculations, and *B*
remains undisturbed regardless of other coupling to :math:`\vec{y}_i`.

The above analysis is done at construction time. At evaluation time, the
Aggregator divides the vector according to the references it was given, and
collects the results.

Required variables which are not present on the graph must be provided as
time-only functions or constants.

This can easily become a rather complex system:

.. autoclass:: stream.aggregator.Aggregator
   :members:
   :no-undoc-members:
   :show-inheritance:

Solvers
~~~~~~~

.. automodule:: stream.solvers
   :members:


Jacobians
~~~~~~~~~

.. automodule:: stream.jacobians
   :members:


State
~~~~~
The solution of a steady state problem is represented as a State object, which is mostly a fancy dict-type.

.. automodule:: stream.state
   :members:

Solution
~~~~~~~~
The solution at different times can be represented bya bunch of States and times, but a compact equivalent object
is the Solution object.

.. automodule:: stream.aggregator.solution
   :members:


Constraints
~~~~~~~~~~~
When solving using the IDA solver, it is possible to constrain the solution. Tools for this can be found in
:file:`stream.aggregator`
See the `sundials` package documentation for more details.

.. automodule:: stream.aggregator.constraints
   :members:
