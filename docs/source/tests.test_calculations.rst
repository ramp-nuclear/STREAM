Calculations Tests
==================

.. automodule:: test_calculations
   :members:
   :undoc-members:
   :show-inheritance:


Channel Tests
-------------

.. currentmodule:: test_calculations.test_channel

.. autosummary::
   :nosignatures:

    test_channel_w_no_heat_flux_maintains_temperature_profile
    test_specific_initialized_Channel_for_precalculated_values
    test_specific_initialized_ChannelAndContacts_for_precalculated_values
    test_first_order_upwind_dTdt_is_zero_for_zero_heat_flux
    test_first_order_upwind_is_constant_for_linear_temperature_no_heat_flux
    test_channel_with_zero_flow_constant_h_reaches_wall_temperature
    test_first_order_upwind_inverts_correctly_on_flow_reversal_no_heat_flux
    test_no_exceptions_raised_through_external_flow_inversion_on_Channel

.. automodule:: test_calculations.test_channel
   :members:
   :undoc-members:

Heat Diffusion Test
-------------------

.. currentmodule:: test_calculations.test_heat

.. autosummary::
   :nosignatures:

    test_Fuel_at_constant_temperature_has_derivative_0
    test_derivative_of_one_cell_follows_the_x_diffusion_kernel
    test_not_equispaced
    test_specific_multi_cell_has_the_right_dimensions
    test_specific_multi_cell_has_the_right_meat
    test_specific_multi_cell_has_the_right_contacts
    test_specific_multi_cell_has_the_right_dTdt
    test_steady_state_with_zero_power
    test_steady_state_for_a_configuration_with_uniform_power
    test_initialization_of_Fuel_with_one_known_example
    test_diffusion_gives_0_for_uniform_temperatures
    test_derivative_of_one_cell_follows_the_r_diffusion_kernel
    test_annulus_given_wall_temperatures
    test_cylinder_given_heat_production_and_wall_temperature
    test_annulus_given_heat_production_and_wall_temperatures
    test_derivative_of_one_cell_follows_the_rz_diffusion_kernel


.. automodule:: test_calculations.test_heat
   :members:
   :undoc-members:


Ideal Calculations Tests
------------------------

.. currentmodule:: test_calculations.test_ideal

.. autosummary::
   :nosignatures:

    test_pump_as_ideal_dp_source
    test_pump_as_ideal_current_source
    test_pump_errors_on_impossibly_imposed_dp_and_mdot
    test_pump_errors_on_when_no_source_type_was_imposed
    test_resistor
    test_hx
    test_local_pressure_drop_is_always_non_positive
    test_resistor_sum_calculates_additions_of_different_resistors
    test_resistor_sum_from_a_sum_of_resistor_sums
    test_arbitrary_resistors_in_resistor_sum


.. automodule:: test_calculations.test_ideal
   :members:
   :undoc-members:

Flapper Tests
-------------

.. currentmodule:: test_calculations.test_flapper

.. autosummary::
   :nosignatures:

    test_fully_open_flapper_acts_as_resistor
    test_closed_flapper_zero_flow_residue_is_zero
    test_flapper_should_not_continue_at_opening_condition


.. automodule:: test_calculations.test_flapper
   :members:
   :undoc-members:


Kirchhoff Tests
---------------

.. currentmodule:: test_calculations.test_kirchhoff

.. autosummary::
   :nosignatures:

    test_a_multigraph_to_graph_for_cycles
    test_build_kvl_matrix_from_a_multigraph
    test_junction_mixing_a_given_set_of_currents
    test_kvl_matrix_works_on_mock_graph
    test_kcl_matrix_works_on_mock_graph
    test_kirchhoff_indexing_works_on_mock_graph
    test_kirchhoff_calculate_works_for_mock_graph
    test_kirchoff_w_mdot2_accepts_a_graph_and_has_correct_length
    test_Kirchoff_kcl_matrix_fits_known_example_with_weights
    test_Kirchoff_supplies_correct_absolute_pressures_for_one_example

.. automodule:: test_calculations.test_kirchhoff
   :members:
   :undoc-members:


Point Kinetics Tests
--------------------

.. currentmodule:: test_calculations.test_point_kinetics

.. autosummary::
   :nosignatures:

    test_pkc
    test_precursor_death
    test_pk_save_follows_known_pattern_for_mock
    test_pk_load
    test_reactivity_for_linear_temperature_in_relation_to_reference
    test_pk_with_decay
    test_pk_change_state_sets_SCRAM_time
    test_pk_should_continue_stops_at_SCRAM_time

.. automodule:: test_calculations.test_point_kinetics
   :members:
   :undoc-members:
