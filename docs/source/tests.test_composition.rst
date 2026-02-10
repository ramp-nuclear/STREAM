Composition Tests
=================

.. automodule:: test_composition
   :members:
   :undoc-members:
   :show-inheritance:

Cycle Tests
-----------

.. currentmodule:: test_composition.test_cycle

.. autosummary::
    :nosignatures:

    test_in_series_with_no_arguments_gives_an_empty_graph_and_funcs
    test_in_series_two_components_creates_a_graph_with_known_edges
    test_in_series_cyclic_for_3_calculations_has_the_right_edges
    test_in_parallel_of_with_no_comps_gives_empty_graph_and_funcs
    test_in_parallel_for_two_paths_with_a_comp_each_has_correct_edges
    test_kirchhoffify_of_empty_graphs_makes_empty_graph_and_funcs
    test_kirchhoffify_w_2_hydraulic_comps_and_empty_graph_sets_comps_edges_correctly
    test_kirchhoffify_given_inertial_components_and_regular_kirchoff_throws
    test_kirchhoffify_w_inertial_comps_adds_mdot_and_mdot2
    test_kirchhoffify_one_junction_connects_k_to_junction_but_not_vice_versa
    test_kirchhoffify_with_ref_mdots_gives_them_the_right_comps_by_example
    test_flow_edge_example
    test_flow_graph_to_agr_for_2_junctions_2_comps_has_correct_edges
    test_flow_graph_to_agr_for_2_junction_and_a_virtual_junction_has_correct_edges

.. automodule:: test_composition.test_cycle
   :members:
   :undoc-members:

Partial States Tests
--------------------

.. currentmodule:: test_composition.test_partial_states

.. autosummary::
    :nosignatures:

    test_uniform_with_an_example
    test_uniform_sets_only_to_given_value
    test_uniform_sets_variables_it_is_told_to
    test_merge_with_an_example
    test_merge_has_union_keys
    test_merge_values_appear_in_originals
    test_filter_values_with_an_example
    test_filter_calculations_with_an_example

.. automodule:: test_composition.test_partial_states
   :members:
   :undoc-members:

Subsystems Tests
----------------

.. currentmodule:: test_composition.test_subsystems

.. autosummary::
    :nosignatures:

    test_symmetric_plate_steady_state_has_zero_diff_in_power
    test_symmetric_plate_steady_state_has_zero_diff_in_low_mdot
    test_point_kinetics_steady_state_follows_analytic_formula
    test_point_kinetics_w_input_steady_state_follows_analytic_formula
    test_hydraulic_steady_state_is_a_root_for_a_single_loop_case
    test_hydraulic_steady_state_is_a_root_for_a_simple_parallel_case
    test_hydraulic_steady_state_assumes_0_pressure_drop_for_unsupported_calculations
    test_hydraulic_steady_state_uses_strategy_when_provided

.. automodule:: test_composition.test_subsystems
   :members:
   :undoc-members:

Addenda Tests
-------------

.. currentmodule:: test_composition.test_constructors

.. autosummary::
    :nosignatures:

    test_parabolic_FlowResistor_intersects_set_point
    test_constant_mdot_FlowResistor_intersects_set_point
    test_constant_dp_FlowResistor_intersects_set_point
    test_linear_FlowResistor_intersects_set_point
    test_constant_FlowResistor_errors_when_superimposed
    test_constant_FlowResistor_errors_when_under_imposed
    test_Calculation_factory_given_a_simple_example_is_set_up_and_calculates_correctly

.. automodule:: test_composition.test_constructors
   :members:
   :undoc-members:
