import pytest
import numpy as np
import xarray as xr
from ectuner.libs.utils import (
    compute_difference,
    apply_imbalance_correction,
    apply_temperature_correction,
    compute_derived_flux
)

def test_compute_difference():
    """Verify that nested dictionaries are correctly subtracted (Model - Ref)."""
    base = {'net_toa': {'ALL': {'Global': 10.0, 'NH': 5.0}}}
    reference = {'net_toa': {'ALL': {'Global': 6.0, 'NH': 8.0}}}
    
    diff = compute_difference(base, reference)
    
    assert diff['net_toa']['ALL']['Global'] == 4.0
    assert diff['net_toa']['ALL']['NH'] == -3.0

def test_apply_imbalance_correction():
    """Verify the distribution of the imbalance correction on TOA fluxes."""
    reference = {
        'net_toa': {'ALL': {'Global': 10.0}},
        'rsnt': {'ALL': {'Global': 10.0}},
        'rlnt': {'ALL': {'Global': 10.0}}
    }
    
    # Imbalance of 2.0 W/m2. 
    # With sw_fraction=0.5, both SW (rsnt) and LW (rlnt) should be reduced by 1.0 W/m2.
    corrected = apply_imbalance_correction(
        reference, 
        imbalance=2.0, 
        adjust_individual_fluxes=True, 
        sw_fraction=0.5
    )
    
    assert corrected['net_toa']['ALL']['Global'] == 8.0
    assert corrected['rsnt']['ALL']['Global'] == 9.0
    assert corrected['rlnt']['ALL']['Global'] == 9.0

def test_temperature_correction_success():
    """Verify the linear shift of the reference based on temperature drift."""
    reference = {'net_toa': {'ALL': {'Global': 5.0}}}
    slopes = {'net_toa': {'ALL': {'Global': 2.0}}}
    
    weights_flux = {'net_toa': 1.0}
    weights_season = {'ALL': 1.0}
    weights_region = {'Global': 1.0}
    
    # Delta T = 1.0 K. Correction = -(delta_t * slope) = -(1.0 * 2.0) = -2.0.
    # Expected new reference: 5.0 - 2.0 = 3.0 W/m2.
    corrected, warnings = apply_temperature_correction(
        reference, slopes, delta_t=1.0,
        weights_flux=weights_flux, weights_season=weights_season, weights_region=weights_region
    )
    
    assert corrected['net_toa']['ALL']['Global'] == 3.0
    assert len(warnings) == 0

def test_temperature_correction_fail_fast():
    """Verify that the tool crashes intentionally if a required slope is missing."""
    reference = {'net_toa': {'ALL': {'Global': 5.0}}}
    slopes = {}  
    
    weights_flux = {'net_toa': 1.0}
    weights_season = {'ALL': 1.0}
    weights_region = {'Global': 1.0}
    
    # The weight is > 0, but the slope is missing. It MUST raise a ValueError.
    with pytest.raises(ValueError, match="Slope missing or NaN"):
        apply_temperature_correction(
            reference, slopes, delta_t=1.0,
            weights_flux=weights_flux, weights_season=weights_season, weights_region=weights_region
        )

def test_compute_derived_flux():
    """Verify the physical formulas for computing derived radiative fluxes."""
    # Create a dummy Dataset with mock solar and terrestrial radiation
    ds = xr.Dataset({
        'rsdt': xr.DataArray([340.0]), # Incoming solar
        'rsut': xr.DataArray([100.0]), # Outgoing solar
        'rlut': xr.DataArray([235.0])  # Outgoing longwave
    })
    
    # TOA Net = rsdt - rsut - rlut = 340 - 100 - 235 = 5.0
    net_toa = compute_derived_flux(ds, 'net_toa')
    assert net_toa.values[0] == 5.0
    
    # Unrecognized flux should raise ValueError
    with pytest.raises(ValueError, match="no physical formula defined"):
        compute_derived_flux(ds, 'unknown_magic_flux')