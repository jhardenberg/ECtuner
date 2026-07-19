"""
Mathematical, physical, and geospatial utilities for ECtuner.

This module provides a centralized toolset for data processing. It handles:
- 1D Corrections: Temperature drifts and intrinsic model imbalances.
- Physical Derivations: Computing complex atmospheric fluxes from raw OIFS variables.
- Spatial Operations: Safely regridding multidimensional NetCDF data using CDO/smmregrid.
"""
import copy
import math
import logging
from typing import Dict, Any, TYPE_CHECKING

FluxDict = Dict[str, Dict[str, Dict[str, float]]]

if TYPE_CHECKING:
    import xarray as xr


def compute_difference(base: FluxDict, reference: FluxDict) -> FluxDict:
    """
    Computes the difference (bias) between model fluxes and reference fluxes.

    Navigates the nested dictionaries comparing available data at the 
    Variable -> Season -> Region level.

    Args:
        base (FluxDict): Nested dictionary containing the model fluxes.
            Expected format: ``{'var': {'season': {'region': value}}}``.
        reference (FluxDict): Nested dictionary containing the target observation fluxes.
            Expected format matching the ``base`` dictionary.

    Returns:
        FluxDict: A new nested dictionary containing the calculated biases 
        (model - reference). Only populates keys where both base and 
        reference data exist.
    """
    difference: FluxDict = {}
    for var, season_data in base.items():
        difference[var] = {}
        for season, region_data in season_data.items():
            difference[var][season] = {}
            for region, value in region_data.items():
                
                # Check existence in reference to avoid KeyError
                if var in reference and season in reference[var] and region in reference[var][season]:
                    difference[var][season][region] = value - reference[var][season][region]
                    
    return difference


def apply_imbalance_correction(
    reference: FluxDict, 
    imbalance: float = 0.0, 
    adjust_individual_fluxes: bool = True, 
    sw_fraction: float = 0.5
) -> FluxDict:
    """
    Corrects target fluxes for intrinsic model energy imbalances.

    If the model creates or destroys energy artificially, the ``net_toa`` reference 
    is shifted to equilibrate the tuning target.

    Args:
        reference (FluxDict): The original observation reference dictionary.
        imbalance (float, default 0.0): The intrinsic imbalance (W/m2) to be subtracted.
        adjust_individual_fluxes (bool, default True): If True, propagates the 
            correction to Shortwave (rsnt) and Longwave (rlnt) fluxes.
        sw_fraction (float, default 0.5): The fraction of the imbalance attributed 
            to the Shortwave flux. The remainder (1 - sw_fraction) is attributed 
            to the Longwave flux.

    Returns:
        FluxDict: A deep copy of the reference dictionary with the applied 
        imbalance shifts.
    """
    corrected = copy.deepcopy(reference)

    if 'net_toa' in corrected:
        corrected['net_toa']['ALL']['Global'] -= imbalance
    
    if adjust_individual_fluxes:
        if 'rsnt' in corrected:
            corrected['rsnt']['ALL']['Global'] -= sw_fraction * imbalance
        
        if 'rlnt' in corrected:
            corrected['rlnt']['ALL']['Global'] -= (1.0 - sw_fraction) * imbalance

    return corrected


def apply_temperature_correction(
    reference: FluxDict, 
    slopes: Dict[str, Any], 
    delta_t: float, 
    weights_flux: Dict[str, float], 
    weights_season: Dict[str, float], 
    weights_region: Dict[str, float]
) -> FluxDict:
    """
    Modifies reference fluxes by subtracting the temperature drift.

    The correction applied is ``-(delta_t * slope)``. The function implements 
    a Fail-Fast mechanism: it raises an error if a slope is missing for a 
    specific region/variable that has an active tuning weight.

    Args:
        reference (FluxDict): The original observation reference dictionary.
        slopes (dict): Nested dictionary containing the temperature regression 
            slopes (T_slope).
        delta_t (float): The temperature adjustment step (K).
        weights_flux (dict): Weights assigned to each variable.
        weights_season (dict): Weights assigned to each season.
        weights_region (dict): Weights assigned to each spatial region.

    Raises:
        ValueError: If a slope is missing or NaN for a variable/season/region 
            combination that possesses a combined weight strictly greater than 0.

    Returns:
        FluxDict: A deep copy of the reference dictionary corrected for 
        temperature drift.
    """
    warnings_list = []
    corrected = copy.deepcopy(reference)

    for var in corrected:
        var_weight = weights_flux.get(var, 0.0)

        for season in corrected[var]:
            season_weight = weights_season.get(season, 0.0)

            for region in corrected[var][season]:
                region_weight = weights_region.get(region, 0.0)
                combined_weight = var_weight * season_weight * region_weight

                # Safe extraction of the slope value
                slope = slopes.get(var, {}).get(season, {}).get(region)

                if slope is None or (isinstance(slope, float) and math.isnan(slope)):
                    if combined_weight > 0.0:
                        raise ValueError(
                            f"Slope missing or NaN for '{var}', season '{season}', "
                            f"region '{region}', but its combined weight is "
                            f"{combined_weight} (> 0)."
                        )
                    else:
                        slope = 0.0  # Safe fallback for ignored regions
                        warnings_list.append(f"Missing slope for diagnostic '{var}' ({season}, {region}). Not temperature-corrected.")

                corrected[var][season][region] -= (delta_t * slope)
            
    return corrected, warnings_list

def compute_derived_flux(ds: 'xr.Dataset', var_name: str) -> 'xr.DataArray':
    """
    Computes derived atmospheric fluxes from raw OIFS variables.
    Single Source of Truth for physical definitions in ECtuner.
    """
    if var_name in ds:
        return ds[var_name]
        
    formulas = {
        'net_toa': lambda d: d['rsdt'] - d['rsut'] - d['rlut'],
        'rsnt':    lambda d: d['rsdt'] - d['rsut'],
        'rlnt':    lambda d: -1.0 * d['rlut'],  # OIFS rlut is UP (positive). We want DOWN (negative).
        'swcf':    lambda d: d['rsnt'] - d['rsntcs'] if 'rsntcs' in d else d['rsutcs'] - d['rsut'],
        'lwcf':    lambda d: d['rlnt'] - d['rlntcs'] if 'rlntcs' in d else d['rlutcs'] - d['rlut'],
        'net_sfc': lambda d: d['rsns'] + d['rlns'] - d['hfls'] - d['hfss'] - d.get('prsn', 0)*334000,
        'toamsfc': lambda d: (d['rsdt'] - d['rsut'] - d['rlut']) - (d['rsns'] + d['rlns'] - d['hfls'] - d['hfss'] - d.get('prsn', 0)*334000)
    }
    
    if var_name in formulas:
        return formulas[var_name](ds)
    else:
        raise ValueError(f"Variable '{var_name}' not found and no physical formula defined.")

def standardize_reference_signs(ref_maps: Dict[str, 'xr.DataArray']) -> Dict[str, 'xr.DataArray']:
    """
    Standardizes the signs of observational references to match the model conventions.
    For example, CERES RLNT is positive UP, but the model convention is positive DOWN.
    """
    corrected = {k: v.copy() for k, v in ref_maps.items()}
    
    if 'rlnt' in corrected:
        if float(corrected['rlnt'].mean()) > 0:
            corrected['rlnt'] = -1.0 * corrected['rlnt']
            
    return corrected


def regrid_to_regular_smm_safe(
    ds_averaged: 'xr.Dataset', 
    target_grid: str, 
    method: str, 
    raw_file_path: str, 
    varname: str = None
) -> 'xr.Dataset':
    """
    Safely regrids unstructured/model grids to a regular lat/lon grid.
    
    Implements a fallback mechanism: if the specified CDO method fails 
    (e.g., 'ycon' fails on certain mask topologies), it automatically 
    falls back to bilinear interpolation ('bil').
    
    Note: Imports xarray and smmregrid locally to prevent heavy dependencies 
    from breaking pure 1D tuning environments.
    """
    import logging
    import xarray as xr
    try:
        from smmregrid import cdo_generate_weights, Regridder
    except ImportError:
        raise ImportError("smmregrid is required for 2D spatial tuning. Please install it.")

    if varname and ds_averaged.name is None:
        ds_averaged.name = varname

    # Se è già su una griglia regolare (lat, lon) e non ha coordinate 'cell', salta.
    if 'lat' in ds_averaged.coords and 'lon' in ds_averaged.coords and 'cell' not in ds_averaged.dims:
        return ds_averaged

    ds_raw = xr.open_dataset(raw_file_path)
    t_dim = 'time_counter' if 'time_counter' in ds_raw.dims else 'time'
    source_grid = ds_raw.isel({t_dim: 0}) 

    if 'lat' in source_grid.coords: source_grid.lat.attrs['units'] = 'degrees_north'
    if 'lon' in source_grid.coords: source_grid.lon.attrs['units'] = 'degrees_east'

    try:
        weights = cdo_generate_weights(source_grid, target_grid=target_grid, method=method)
    except Exception as e:
        logging.warning(f"Regridding method '{method}' failed ({e}). Falling back to 'bil'...")
        weights = cdo_generate_weights(source_grid, target_grid=target_grid, method='bil')
    
    regridder = Regridder(weights=weights)
    return regridder.regrid(ds_averaged)


def save_diagnostic_maps(
    output_path: str, 
    target_vars: list, 
    bias_maps: dict, 
    ds_sens: 'xr.Dataset', 
    params: list, 
    optimal_changes: dict, 
    r2_threshold: float
) -> None:
    """
    Generates and saves a NetCDF file containing the initial and predicted final 2D biases.
    """
    import xarray as xr
    import logging
    
    diagnostics = []
    for var in target_vars:
        bias_init = bias_maps[var]
        delta_pred = xr.zeros_like(bias_init)
        
        for p in params:
            slope = ds_sens.sel(variable=var, parameter=p).slope.drop_vars(['variable', 'parameter'], errors='ignore')
            r2 = ds_sens.sel(variable=var, parameter=p).r2.drop_vars(['variable', 'parameter'], errors='ignore')
            
            clean_slope = slope.where(r2 > r2_threshold, 0)
            delta_pred += clean_slope * optimal_changes[p]
            
        bias_final = bias_init + delta_pred
        
        var_ds = xr.Dataset({
            f'{var}_bias_init': bias_init.drop_vars(['variable'], errors='ignore'),
            f'{var}_bias_final': bias_final.drop_vars(['variable'], errors='ignore'),
            f'{var}_improvement': (abs(bias_init) - abs(bias_final)).drop_vars(['variable'], errors='ignore')
        })
        diagnostics.append(var_ds)

    ds_diag = xr.merge(diagnostics, compat='override')
    ds_diag.to_netcdf(output_path)
    logging.getLogger('ectuner').info(f"Diagnostic maps saved to: {output_path}")

def get_region_mask(ds_sens: 'xr.Dataset', weights_region: Dict[str, float]) -> 'xr.DataArray':
        """
        Creates a 2D mask based on spatial regions defined in the config.
        
        The final weight of each grid cell is calculated as:
        Area_Weight (cosine of latitude) * Region_Weight.

        Args:
            ds_sens (xr.Dataset): Reference dataset to extract lat/lon coordinates.
            weights_region (Dict[str, float]): Weights mapped to region names.

        Returns:
            xr.DataArray: The computed 2D mask.
        """
        import numpy as np
        import xarray as xr
        
        lat = ds_sens.lat
        lon = ds_sens.lon
        
        mask = np.cos(np.deg2rad(lat))
        
        region_bounds = {
            'Global': (-90, 90),
            'Tropical': (-30, 30),
            'North Midlat': (30.0, 90.0),
            'South Midlat': (-90.0, -30.0),
            'North Pole': (60.0, 90.0),
            'South Pole': (-90.0, -60.0),
            'Equatorial': (-20.0, 20.0),
            'NH': (20.0, 90.0),
            'SH': (-90.0, -20.0),
        }

        regional_weight_map = xr.DataArray(np.zeros_like(lat), coords={'lat': lat}, dims='lat')

        for region, weight in weights_region.items():
            if weight <= 0: 
                continue
            if region in region_bounds:
                low, high = region_bounds[region]
                regional_weight_map = regional_weight_map.where(
                    (lat < low) | (lat > high), 
                    regional_weight_map + weight
                )
        
        if regional_weight_map.max() == 0:
            regional_weight_map += 1.0

        mask_2d = (mask * regional_weight_map).expand_dims(lon=len(lon)).assign_coords(lon=lon)
        return mask_2d