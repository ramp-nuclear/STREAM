Libraries Tests
===============

.. automodule:: test_libraries
   :members:
   :undoc-members:
   :show-inheritance:

Decay Heat Tests
----------------

.. currentmodule:: test_libraries.test_decay_heat

.. autosummary::
   :nosignatures:

    test_decay_heat_contributions_are_monotonous_in_time
    test_fission_products_timed_contributions_against_known_values

.. automodule:: test_libraries.test_decay_heat
   :members:
   :undoc-members:


Heat Transfer Coefficient Tests
-------------------------------

.. automodule:: test_libraries.test_htc
    :members:

.. autosummary::
   :nosignatures:

    test_BR_T_wall_inc_for_precalculated_case
    test_BR_partial_scb_for_precalculated_case
    test_htc_is_geq_to_Dittus_Boelter
    test_wall_temperature_for_precalculated_case
    test_regime_dependent_q_scb_is_zero_for_Twall_equals_Tsat
    test_regime_dependent_h_spl_assigns_regimes_correctly

Pressure Drop Tests
-------------------

.. currentmodule:: test_libraries.test_pressure_drop

.. autosummary::
   :nosignatures:


    test_turbulent_friction_for_precalculated_case
    test_viscosity_correction_for_precalculated_case
    test_rectangular_laminar_correction_for_precalculated_case
    test_Darcy_Weisbach_pressure_by_mdot_for_precalculated_case
    test_local_pressure_inverse_methods_are_inverse
    test_pressure_difference_for_a_few_simple_cases
    test_friction_factor_identifies_the_required_names
    test_friction_factor_raises_value_error_for_unknown_names
    test_friction_factor_keyword_passing
    test_regime_dependent_friction_assigns_regimes_correctly
    test_idelchik_expansion_is_analytic_for_large_re
    test_idelchik_local_pd_computes_nonpositive_dp
    test_idelchik_contraction_at_high_reynolds_follows_known_form


.. automodule:: test_libraries.test_pressure_drop
   :members:
   :undoc-members:


Thresholds Tests
----------------

.. currentmodule:: test_libraries.test_thresholds

.. autosummary::
   :nosignatures:

    test_SK_CHF_is_non_negative_for_specific_cases
    test_WF_OFI_for_precalculated_case
    test_Saha_Zuber_OSV_precalculated_case
    test_Mirshak_CHF_precalculated_case
    test_Fabrega_CHF_precalculated_case

.. automodule:: test_libraries.test_thresholds
   :members:
   :undoc-members:
