General Framework Tests
=======================

.. automodule:: test_general
   :members:
   :undoc-members:
   :show-inheritance:


Aggregator Tests
----------------

.. currentmodule:: test_general.test_aggregator

.. autosummary::
   :nosignatures:

    test_example_aggregator_has_known_shape
    test_save
    test_collision_of_calculations_raises_by_example
    test_load
    test_compute_of_a_graph_vs_known_implementation
    test_composition_of_specific_agrs_yields_known_agr
    test_aggregator_input_works_as_expected
    test_ida_root_functions
    test_agr_input_connect
    test_ida_continuous_mode
    test_associated_calculations_for_a_known_example

.. automodule:: test_general.test_aggregator
   :members:
   :undoc-members:


Data Representation Tests
-------------------------

.. currentmodule:: test_general.test_dataframes

.. autosummary::
   :nosignatures:

    test_dataframe_of_vertical_vector_has_multiple_i_single_j
    test_state_of_vertical_vector_is_vertical
    test_state_to_dataframe_and_back_is_same
    test_statetimeseries_to_dataframe_and_back_is_same
    test_solution_to_statetimeseries_and_back_is_same_by_example

.. automodule:: test_general.test_dataframes
   :members:
   :undoc-members:


Calculation (Framework) Tests
-----------------------------

.. currentmodule:: test_general.test_calculation

.. autosummary::
   :nosignatures:

    test_unpack_correctly_unpacks_data
    test_concat_is_at_most_1d
    test_concat_of_dictionaried_arrays_is_the_same_as_their_numpy_concat
    test_default_save_has_correct_output_for_one_structure
    test_default_save_is_compatible_with_calc_variables
    test_default_load_for_one_structure
    test_default_load_is_inverse_of_default_save

.. automodule:: test_general.test_calculation
   :members:
   :undoc-members:


Solver Backend Tests
--------------------

.. currentmodule:: test_general.test_solvers

.. autosummary::
    :nosignatures:

    test_dae_solver_using_planar_pendulum_against_scikits_odes_documented_solution
    test_dae_with_undetermined_var_fails_as_TransientRuntimeError
    test_dae_solver_accepts_all_algebraic_calculations

.. automodule:: test_general.test_solvers
   :members:
   :undoc-members:

Utilities Tests
---------------

.. currentmodule:: test_general.test_utilities

.. autosummary::
    :nosignatures:

    test_pair_mean_for_one_simple_case
    test_just
    test_offset
    test_offset_of_an_unknown_type_raises_TypeError
    test_cosine_shape_with_ppf_1_is_uniform
    test_cosine_shape_achieves_maximum_at_center

.. automodule:: test_general.test_utilities
   :members:
   :undoc-members:
