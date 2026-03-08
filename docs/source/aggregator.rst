Aggregator
----------

.. automodule:: stream.aggregator
   :members:
   :no-undoc-members:
   :show-inheritance:

Example
~~~~~~~
.. image:: /_static/sample_agr.png
    :align: left

Consider the attached image. This graph describes the **dependencies** of each
calculation: *A*, *B*, *C*. Let's consider the Calculation *A*. *A* owns a subset
of the entire vector :math:`\vec{y}_A\in\vec{y}`, which it receives as input.
However, *A* depends on values owned by other Calculations:
:math:`\left(\vec{y}_i, \vec{y}_j\right)`, namely *B* and *C*.
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

.. image:: /_static/big_agr.png
    :align: center

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