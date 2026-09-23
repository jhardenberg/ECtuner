import pytest

# le funzioni con @pytest.fixture sono fixture che possono 
# essere utilizzate nei test automaticamente, scrivendo il nome 
# della fixture come argomento della funzione di test

# mock object : replica semplificata di un oggetto reale 

@pytest.fixture
def dummy_logger():
    """Base logger fixture that can be used in tests to avoid cluttering the output with log messages."""
    import logging
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.CRITICAL) 
    return logger

# funzione che simula il lavoro che farebbe DataLoader1D
@pytest.fixture
def mock_1d_data():
    """Simple mock data dictionary for testing the Tuner1D."""
    
    current_values = {'paramA': 1.0, 'paramB': 2.0}
    ref_params = {'paramA': 1.0, 'paramB': 2.0}
    
    difference = {'net_toa': {'ALL': {'Global': 5.0}}}
    
    # Sensitivity: if we increase paramA by 1, the flux increases by 2. If we increase paramB, it decreases by 1.
    sensitivity = {
        'paramA': {'net_toa': {'ALL': {'Global': [2.0, 0.0]}}},
        'paramB': {'net_toa': {'ALL': {'Global': [-1.0, 0.0]}}}
    }
    
    reference = {'net_toa': {'ALL': {'Global': 100.0}}}
    weights_flux = {'net_toa': 1.0}
    weights_season = {'ALL': 1.0}
    weights_region = {'Global': 1.0}
    
    return {
        'current_values': current_values,
        'ref_params': ref_params,
        'difference': difference,
        'sensitivity': sensitivity,
        'reference': reference,
        'weights_flux': weights_flux,
        'weights_season': weights_season,
        'weights_region': weights_region
    }

import xarray as xr
import numpy as np

@pytest.fixture
def mock_2d_data():
    """Create a mock 2D spatialdataset on a small grid 2x2."""
    
    lat = [-45.0, 45.0]
    lon = [0.0, 180.0]
    
    # Bias maps: all pixels have a bias of 5.0 W/m2
    bias_val = np.array([[5.0, 5.0], [5.0, 5.0]])
    da_bias = xr.DataArray(bias_val, coords=[lat, lon], dims=["lat", "lon"])
    bias_maps = {'net_toa': da_bias}
    
    # reference map: to compute diagnostics
    ref_val = np.array([[100.0, 100.0], [100.0, 100.0]])
    da_ref = xr.DataArray(ref_val, coords=[lat, lon], dims=["lat", "lon"])
    ref_maps = {'net_toa': da_ref}

    # Weight mask: all pixels have equal weight (1.0)
    da_mask = xr.DataArray(np.array([[1.0, 1.0], [1.0, 1.0]]), coords=[lat, lon], dims=["lat", "lon"])

    # Sensitivity dataset (dimensions: variable, parameter, lat, lon)
    # slope paramA = 2.0, slope paramB = -1.0 (in all pixels)
    slope_data = np.array([
        [ [[2.0, 2.0], [2.0, 2.0]], [[-1.0, -1.0], [-1.0, -1.0]] ]
    ])
    r2_data = np.ones_like(slope_data) 

    ds_sens = xr.Dataset(
        {
            "slope": (("variable", "parameter", "lat", "lon"), slope_data),
            "r2": (("variable", "parameter", "lat", "lon"), r2_data)
        },
        coords={
            "variable": ["net_toa"],
            "parameter": ["paramA", "paramB"],
            "lat": lat,
            "lon": lon
        }
    )

    return {
        'bias_maps': bias_maps,
        'ref_maps': ref_maps,
        'ds_sens': ds_sens,
        'mask_2d': da_mask,
        'current_values': {'paramA': 1.0, 'paramB': 2.0},
        'ref_params': {'paramA': 1.0, 'paramB': 2.0},
        'weights_flux': {'net_toa': 1.0},
        'weights_region': {'Global': 1.0}
    }