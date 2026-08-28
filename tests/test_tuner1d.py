import numpy as np
from ectuner.libs.tuner import Tuner1D

def test_objective_function_math(dummy_logger, mock_1d_data):
    """
    Test the mathematical correctness of the objective function in Tuner1D.
    Scenario where the model has a bias of +5 W/m2 on the global flux,
    and the sensitivity of paramA is 2.0 and paramB is -1.0. 
    """

    tuner = Tuner1D(inc=0.2, penalty=0.0, logger=dummy_logger)
    
    # tuner initialization and data preparation
    tuner.setup_parameters(
        mock_1d_data['current_values'], 
        mock_1d_data['ref_params'], 
        frozen_config={}
    )
    
    tuner.prepare_data(
        mock_1d_data['sensitivity'],
        mock_1d_data['difference'],
        mock_1d_data['reference'],
        mock_1d_data['weights_flux'],
        mock_1d_data['weights_season'],
        mock_1d_data['weights_region']
    )
    
    # no changes, the cost should reflect the initial bias
    changes_zero = np.array([0.0, 0.0])
    cost_at_zero = tuner._objective_function(changes_zero)
    
    # Changes that perfectly correct the bias: paramA should decrease by 2.5 (to reduce flux by 5), paramB remains unchanged.
    changes_perfect = np.array([-2.5, 0.0])
    cost_perfect = tuner._objective_function(changes_perfect)
    
    # initial bias is 5, and the weights are all 1, so the cost should be 5^2 = 25
    assert cost_at_zero == 25.0, f"I expected 25.0, got {cost_at_zero}"
    
    # The perfect cost should cancel out the error
    assert cost_perfect == 0.0, f"I expected 0.0, got {cost_perfect}"


def test_frozen_parameters_shift(dummy_logger, mock_1d_data):
    """
    Verify that when a parameter is frozen, the objective function correctly accounts for its contribution to the bias.
    In this test, paramA is frozen at 1.5 (instead of its default 1.0), and we check that the cost reflects the shift 
    in bias due to this frozen parameter. The expected cost is calculated based on the sensitivity of paramA and 
    the frozen value.
    The sensitivity of paramA is 2.0, so the shift in bias due to freezing paramA at 1.5 (a change of +0.5) 
    should be 0.5 * 2.0 = +1.0, leading to a new bias of 5.0 + 1.0 = 6.0. 
    Since paramB is not changed, the cost should be 6.0^2 = 36.0.
    """
    tuner = Tuner1D(inc=0.2, penalty=0.0, logger=dummy_logger)
    
    frozen_config = {'paramA': 1.5}
    
    tuner.setup_parameters(mock_1d_data['current_values'], mock_1d_data['ref_params'], frozen_config)
    tuner.prepare_data(
        mock_1d_data['sensitivity'], mock_1d_data['difference'],
        mock_1d_data['reference'], mock_1d_data['weights_flux'],
        mock_1d_data['weights_season'], mock_1d_data['weights_region']
    )
    
    changes = np.array([0.0]) 
    cost = tuner._objective_function(changes)
    
    # paramA starts at 1.0 and is forced to 1.5 (delta = +0.5).
    # The shift in bias is: delta * sensitivity = 0.5 * 2.0 = +1.0.
    # The new "starting" bias for paramB is 5.0 + 1.0 = 6.0.
    # Since paramB does not move (changes=0.0), the error is 6.0^2 = 36.0
    assert cost == 36.0, f"Expected 36.0, got {cost}"

def test_optimization_convergence(dummy_logger, mock_1d_data):
    """
    SciPy solver should converge to the correct solution for this simple 1D problem.
    The optimizer should find that paramA should decrease by 2.5 to correct the bias of +5 W/m2, 
    while paramB should remain unchanged at 0.0.
    The final cost should be 0.0, indicating that the bias has been fully corrected.
    """
    # ARRANGE: increase the increment to 3.0 to speed up convergence for this test
    tuner = Tuner1D(inc=3.0, penalty=0.0, logger=dummy_logger)
    frozen_config = {'paramB': "default"}
    tuner.setup_parameters(mock_1d_data['current_values'], mock_1d_data['ref_params'], frozen_config)
    tuner.prepare_data(
        mock_1d_data['sensitivity'], mock_1d_data['difference'],
        mock_1d_data['reference'], mock_1d_data['weights_flux'],
        mock_1d_data['weights_season'], mock_1d_data['weights_region']
    )
    
    result = tuner.optimize(method='L-BFGS-B')
    
    # the optimizer should have found that paramA should decrease by 2.5 to correct the bias of +5 W/m2, 
    # while paramB should remain unchanged at 0.0.
    assert np.isclose(result.optimal_changes['paramA'], -2.5, atol=1e-4)
    
    # The optimizer should have found that paramB should remain unchanged at 0.0
    assert np.isclose(result.optimal_changes['paramB'], 0.0, atol=1e-4)
    
    # The final cost found by the algorithm should be 0.0 (bias cancelled)
    assert np.isclose(result.metrics['score_final'], 0.0, atol=1e-4)