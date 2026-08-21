import numpy as np
from ectuner.libs.tuner import Tuner2D

def test_objective_function_2d_math(dummy_logger, mock_2d_data):
    """
    Test the mathematical correctness of the objective function in Tuner2D.
    This test checks that the cost is computed correctly when using a hybrid cost function
    """
    # alpha=0.5 (50% spatial cost, 50% global cost). 
    # Metric L2 (quadratic error)
    tuner = Tuner2D(inc=0.2, penalty=0.0, alpha=0.5, metric='l2', logger=dummy_logger)
    
    tuner.setup_parameters(mock_2d_data['current_values'], mock_2d_data['ref_params'], {})
    tuner.prepare_data(
        mock_2d_data['bias_maps'], mock_2d_data['ref_maps'],
        mock_2d_data['ds_sens'], mock_2d_data['mask_2d'],
        mock_2d_data['weights_flux'], mock_2d_data['weights_region']
    )

    # each pixel has a bias of 5.0, and the weights are all 1.0, 
    # so the spatial cost is:
    # Costo Spaziale L2 (MSE) = (5.0^2 + 5.0^2 + 5.0^2 + 5.0^2) / 4 pesi = 25.0
    # Costo Globale L2 (Bias medio)^2 = (5.0)^2 = 25.0
    # Costo Totale (alpha 0.5) = (0.5 * 25.0) + (0.5 * 25.0) = 25.0
    changes_zero = np.array([0.0, 0.0])
    cost_zero = tuner._objective_function(changes_zero)

    # try to solve the bias (paramA go down by 2.5)
    changes_perfect = np.array([-2.5, 0.0])
    cost_perfect = tuner._objective_function(changes_perfect)

    assert np.isclose(cost_zero, 25.0), f"expected 25.0, got {cost_zero}"
    assert np.isclose(cost_perfect, 0.0), f"expected 0.0, got {cost_perfect}"


def test_optimization_convergence_2d(dummy_logger, mock_2d_data):
    """
    Verify that the vectorized SciPy solver in 2D converges to the exact solution 
    and correctly computes diagnostic costs (RMSE and MAE).
    """
    
    tuner = Tuner2D(inc=3.0, penalty=0.0, alpha=0.0, metric='l1', logger=dummy_logger)
    
    # ParamB frozen to default, so the optimizer should only adjust paramA to fix the bias.
    frozen_config = {'paramB': "default"}
    
    tuner.setup_parameters(mock_2d_data['current_values'], mock_2d_data['ref_params'], frozen_config)
    tuner.prepare_data(
        mock_2d_data['bias_maps'], mock_2d_data['ref_maps'],
        mock_2d_data['ds_sens'], mock_2d_data['mask_2d'],
        mock_2d_data['weights_flux'], mock_2d_data['weights_region']
    )

    result = tuner.optimize(method='differential_evolution')

    assert np.isclose(result.optimal_changes['paramA'], -2.5, atol=1e-4)
    assert np.isclose(result.optimal_changes['paramB'], 0.0, atol=1e-4)
    
    assert np.isclose(result.metrics['total_spatial_cost'], 0.0, atol=1e-4)

import logging

def test_frozen_parameter_warning_and_shift(caplog, mock_2d_data):
    """
    Verify that freezing a parameter beyond the linear threshold (inc) generates 
    a warning and correctly applies the shift on the 2D map.
    """
    # local logger for the test
    logger = logging.getLogger("test_frozen")
    logger.setLevel(logging.WARNING)
    
    tuner = Tuner2D(inc=0.2, penalty=0.0, alpha=0.5, metric='l2', logger=logger)
    
    # paramA is frozen to 1.5, which is a shift of +0.5 from the reference (1.0).
    frozen_config = {'paramA': 1.5}
    tuner.setup_parameters(mock_2d_data['current_values'], mock_2d_data['ref_params'], frozen_config)
    
    with caplog.at_level(logging.WARNING):
        tuner.prepare_data(
            mock_2d_data['bias_maps'], mock_2d_data['ref_maps'],
            mock_2d_data['ds_sens'], mock_2d_data['mask_2d'],
            mock_2d_data['weights_flux'], mock_2d_data['weights_region']
        )
    
    assert "CRITICAL: Frozen parameter 'paramA' is shifted" in caplog.text, "Il warning non è stato emesso"
    
    # Check that the bias map has been shifted by +1.0 (the difference between frozen value 1.5 and reference 1.0)
    shifted_bias = tuner.bias_maps_2d['net_toa'].values
    assert np.allclose(shifted_bias, 6.0), f"Atteso bias 6.0, ma trovato: {shifted_bias}"

def test_nan_pixel_masking_2d(dummy_logger, mock_2d_data):
    """
    Verify that NaN pixels (e.g., ocean or land in masked maps) 
    are discarded without crashing the flattening or optimizer.
    """
    
    mock_2d_data['bias_maps']['net_toa'].values[0, 0] = np.nan
    
    tuner = Tuner2D(inc=0.2, penalty=0.0, alpha=0.5, metric='l2', logger=dummy_logger)
    tuner.setup_parameters(mock_2d_data['current_values'], mock_2d_data['ref_params'], {})
    
    tuner.prepare_data(
        mock_2d_data['bias_maps'], mock_2d_data['ref_maps'],
        mock_2d_data['ds_sens'], mock_2d_data['mask_2d'],
        mock_2d_data['weights_flux'], mock_2d_data['weights_region']
    )
    
    assert len(tuner.bias_flat['net_toa']) == 3, "Flattened bias should have 3 valid pixels after masking NaN"
    assert len(tuner.weights_vector_var['net_toa']) == 3
    assert tuner.sens_matrices_spatial['net_toa'].shape[0] == 3
    
    cost = tuner._objective_function(np.array([0.0, 0.0]))
    assert not np.isnan(cost), "The cost function returned NaN due to unhandled masks"