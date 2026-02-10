Tests
=====
Tests are an inherent requirement of any serious software development project. :ref:`STREAM <Documentation>` is
especially dependent on tests (specifically analytic and unit tests) since nuclear reactor thermohydraulic experiments
are difficult and scarce, and many existing commercial software projects (e.g. ``RELAP``) are notoriously user-variable
and themselves cannot fully simulate core behaviors.

For this reason, the entire test suite is included in the documentation, except for **doctests**, which are part of the
:ref:`documentation <Documentation>` and included therein.

.. toctree::
   :maxdepth: 1

   tests.test_general
   tests.test_calculations
   tests.test_composition
   tests.test_libraries
   tests.integration

Running the Test Suite
----------------------
We use Pytest and Hypothesis to run our tests. To run all tests, in the ``tests`` directory simply run::

    pytest

To avoid running tests marked `slow`, one can run::

    pytest -k 'not slow'

Controlling the Number of Generated Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hypothesis generates examples to run against many of our tests. As a matter of convenience,
two profile are defined, controlling the number of examples: ``fast`` and ``thorough``. Run them using::

    pytest --hypothesis-profile=thorough

which can be mixed and matched with other pytest options.
