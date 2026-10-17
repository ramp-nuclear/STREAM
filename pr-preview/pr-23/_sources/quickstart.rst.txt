Quickstart Guide
================

In this guide we'll go through a typical simulation with which :ref:`STREAM <Documentation>` may be of use.

Creating Calculations
~~~~~~~~~~~~~~~~~~~~~
Consider the physical system you would like to simulate, and look at the :ref:`Calculations` already implemented
- you may find just the components needed. If not, please refer to :ref:`Calculation` to see how one goes about
creating a custom component.
If all you want is a simple component, consider using :func:`~stream.composition.constructors.Calculation_factory`.

Composition
~~~~~~~~~~~
Assuming you have all of your calculations instantiated, the next step is combining these interdependent
calculations, usually with the end result being an :class:`~stream.aggregator.CalculationGraph`. Aptly named, the
:ref:`Composition` subpackage offers functionality for composing standard calculations.

External Input variables
~~~~~~~~~~~~~~~~~~~~~~~~
Any variables required by the calculations which are not provided by some other calculation must be provided by the
user in the form of :attr:`~stream.aggregator.CalculationGraph.funcs`. These inputs may be time-dependent functions or any other
value.

Next, an :class:`~stream.aggregator.Aggregator` instance may be created::

    cg = CalculationGraph(...)
    agr = cg.to_aggregator()

Initial Guess / Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to run the simulation, either finding a steady state solution or a transient based on given, one must provide
an initial guess / conditions. :class:`~stream.state.State` is useful in that regard, and especially the following may be of help:

.. autosummary::
    :nosignatures:

    ~stream.state.State.merge
    ~stream.state.State.uniform
    ~stream.aggregator.Aggregator.save
    ~stream.aggregator.Aggregator.load
    ~stream.composition.subsystems

Note that a `State` is basically a ``dict[Calculation, dict[Any, Value]]``

Introspection of the Created System / State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hooray! You now have a system of equations in the shape of an :class:`~stream.aggregator.Aggregator`
and your :class:`~stream.state.State`.
Some tools may be useful to see what you actually have, mostly in the :mod:`~stream.analysis` module.

.. autosummary::
   :nosignatures:

    ~stream.analysis.report.report
    ~stream.aggregator.draw_aggregator
    ~stream.analysis.debugging.debug_derivatives

Steady State Solution
~~~~~~~~~~~~~~~~~~~~~
Given the initial conditions, one may find a root of the functional :math:`\vec{F}(\vec{y}, t)`::

    steady_state = agr.solve_steady(initial_conditions)

A graph-aware Jacobian may be used as ``jac=``
:func:`~stream.jacobians.ALG_jacobian`.
Applying ``agr.save(steady_state)`` would yield a :class:`~stream.state.State`, with added information computed post-mortum
(this can be avoided by setting ``strict=True``). 
Adding computed thermo-hydraulic critical phenomena thresholds may be done using :func:`~stream.analysis.thresholds.threshold_analysis`.

Transient Simulation
~~~~~~~~~~~~~~~~~~~~
Running a transient solution is done similarly, with the addition of a time parameter. 
Here is an example with some important parameters, assuming (as typicali, and can be controlled using ``eq_type``) that the Aggregator forms a DAE::

    time = np.linspace(0, 100)
    solution = agr.solve(
        initial_state,
        time,
        rtol=1e-6,
        atol=absolute_tolerance_vector,
        jacfn=DAE_jacobian(agr),
        continuous=True,
        max_step_size=1e-1,
        constraints_type=contraints_vector,
    )

Here the graph-aware Jacobian is :func:`~stream.jacobians.DAE_jacobian`, 
the absolute tolerance may be set for each equation separately (using :class:`~stream.state.State` and loading is a good way to do that),
while the relative tolerance is global. The constraints vector controls the sign of the variables (See SUNDIALS IDA),
and can limit the search space somewhat. It follows the following mapping:: 

    dict(NONE=0.0, NON_NEG=1.0, POS=2.0, NON_POS=-1.0, NEG=-2.0)

For most use cases, the defaults are really good enough, but know that these options can be controlled.

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~
A common use case for a system code like STREAM is when one needs a relatively fast and simple
model that can be used for parametric studies, as well as uncertainty quantification (UQ) problems.
In STREAM, the :mod:`~stream.analysis.UQ` module is used for UQ problems, and it provides two
similar but distinct classes: The :class:`~stream.analysis.UQ.UQModel` class and the
:class:`~stream.analysis.UQ.DASKUQModel`.
Both classes are based on a Jacobian approach to UQ, where numeric partial derivatives of a model
are computed to get a linear approximation to the uncertainty of each model feature to its inputs.
Both models base their uncertainty propagation on the concepts of statistical and systematic
uncertainties, where statistical contributions are added in 2-norm, and systematic contributions
are added in 1-norm.

For interactive and simple use cases, we suggest you use the UQModel class.
The UQModel class is simpler, and its use requires just initialization and a call to its
:meth:`~stream.analysis.UQ.UQModel.uq` method::

    uqmod = UQModel(parameters=parameters, model=model)
    systematic, statistical = uqmod.uq(**uncertainty_dictionary)

This method call will return the total statistical and systematic uncertainty as vectors for the
model features.

For high throughput cases, one will most likely prefer the DASKUQModel class.
This class is designed around lazy execution, and calls to its methods most often return a DASK
Delayed object, which is a lazy object that takes a little effort to wrap one's head around.
You will likely prefer this class when you need to run multiple models, or the same model in a
large parametric search.
Like the simple case above, most users will initialize their DASKUQModel and then call its
:meth:`~stream.analysis.UQ.DASKUQModel.uq` method.

Analyzing the Output
~~~~~~~~~~~~~~~~~~~~
After running the simulation, one is free to manipulate the output, which is just an array (1D for a steady state
solution and 2D for a transient, wrapped with its time as :class:`~stream.aggregator.Solution`). In that regard, the following may be helpful for extracting data and manipulating it:


.. autosummary::
    :nosignatures:

    ~stream.aggregator.Aggregator.save
    ~stream.aggregator.Aggregator.var_index
    ~stream.aggregator.Aggregator.at_times
    ~stream.state.State
    ~stream.state.to_dataframe

When saving (``agr.save``) a steady state solution, you get :class:`~stream.state.State`.
When saving a transient solution, you get :const:`~stream.state.StateTimeseries` mapping from timepoint to State.
If you want to use a ``pandas.DataFrame``, those can be created using :func:`~stream.state.to_dataframe`, containing the following columns: calculation, variable, indices (i, j), and finally value.


-------

.. seealso:: You may find the :ref:`Examples` a more concrete down-to-earth approach to learning `STREAM`.
