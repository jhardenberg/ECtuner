import xarray as xr
import numpy as np
import os
import glob
import yaml
import argparse
import sys
from smmregrid import cdo_generate_weights, Regridder
from sensitivity import (read_yaml_files, compare_with_reference, 
                        extract_tag_by_position, find_files_from_template, dicts_equal)

#to consider if we want to add 3d variables (e.g. ta, hus), by reducing them to 2D fields 
# (e.g. choose a pressure-level or compute a weighted average over a layer): 

# UTILITIES

def get_var_2d(ds, varname):
    if varname in ds: return ds[varname]
    formulas = {
        'net_toa': lambda d: d['rsnt'] + d['rlnt'],
        'swcf':    lambda d: d['rsnt'] - d['rsntcs'],
        'lwcf':    lambda d: d['rlnt'] - d['rlntcs'],
        'net_sfc': lambda d: d['rsns'] + d['rlns'] - d['hfls'] - d['hfss'] - d['prsn']*334000,
        'toamsfc': lambda d: d['rsnt'] + d['rlnt'] - d['rsns'] - d['rlns'] + d['hfls'] + d['hfss'] + d['prsn']*334000
    }
    if varname in formulas:
        return formulas[varname](ds)
    raise ValueError(f"Variable {varname} not found and no formula defined.")

def regrid_to_regular_smm_safe(ds_averaged, target_grid, method, raw_file_path, varname=None):
    """Regrid unstructured/model grid to regular grid safely"""
    # Assegna un nome se manca, per evitare il crash di smmregrid
    if varname and ds_averaged.name is None:
        ds_averaged.name = varname

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
        print(f"      {method} failed: {e}. Trying 'bil'...")
        weights = cdo_generate_weights(source_grid, target_grid=target_grid, method='bil')
    
    regridder = Regridder(weights=weights)
    return regridder.regrid(ds_averaged)

def compute_slope_and_linearity_2d(y_m, y_ref, y_p, x_vals):
    """Linear regression on 3 points for every pixel"""
    y_stack = xr.concat([y_m, y_ref, y_p], dim='param_change').assign_coords(param_change=x_vals)
    y_stack = y_stack.chunk({'param_change': -1})
    def linfit(x, y):
        if np.all(np.isnan(y)) or np.all(y == y[0]):
            return np.nan, np.nan
        try:
            p = np.polyfit(x, y, 1)
            r_matrix = np.corrcoef(x, y)
            r_squared = r_matrix[0, 1]**2
            return p[0], r_squared
        except:
            return np.nan, np.nan
    
    return xr.apply_ufunc(linfit, y_stack.param_change, y_stack,
                          input_core_dims=[["param_change"], ["param_change"]],
                          output_core_dims=[[], []], vectorize=True, 
                          dask="parallelized", output_dtypes=[float, float])

def parse_args():
    parser = argparse.ArgumentParser(description='Calc 2D sensitivity maps')
    parser.add_argument('-c', '--config', type=str, required=True, help='YAML config file')
    parser.add_argument('ref_tag', type=str, nargs='?', help='Reference experiment tag')
    parser.add_argument('exp_temp', type=str, nargs='?', help='Tag template (e.g. "s???")')
    parser.add_argument('year1', type=int, nargs='?', help='Start year')
    parser.add_argument('year2', type=int, nargs='?', help='End year')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get values from CLI or config
    y1 = args.year1 or config['args']['year1']
    y2 = args.year2 or config['args']['year2']
    exp_temp = args.exp_temp or config['files']['exp_temp']
    ref_tag = args.ref_tag or config['args'].get('ref_tag')
    
    target_grid = config.get('spatial_tuning', {}).get('target_grid', 'r180x90')
    target_vars = config.get('spatial_tuning', {}).get('target_vars', 
                  ['tas', 'pr', 'net_toa', 'rsnt', 'rlnt', 'swcf', 'lwcf', 'net_sfc'])
    
    methods_map = config.get('spatial_tuning', {}).get('methods_map', {})

    # 1. Discover experiments and perturbed parameters
    yaml_dir = config['files']['exps']
    params_template = config['files']['params']
    yaml_glob = params_template.format(exp=exp_temp)
    
    tuning_files = [os.path.join(yaml_dir, f) for f in find_files_from_template(yaml_glob, yaml_dir)]
    tunsets = read_yaml_files(tuning_files, params_template)
    
    # Identify reference tag
    reference_dict = config['reference_parameters']
    if not ref_tag:
        for tag, vals in tunsets.items():
            if dicts_equal(vals, reference_dict): # Più robusto dell'uguaglianza secca
                ref_tag = tag
                break
    
    if not ref_tag: 
        raise ValueError("Could not detect reference tag. Check if reference_parameters in config matches any YAML.")
    
    print(f"Reference experiment: {ref_tag}")

    # 2. Map parameters to experiments (Auto-discovery)
    # We need a 'min' and a 'max' for each parameter
    pardict = compare_with_reference(tunsets, reference_dict)
    
    param_map = {}
    for p in reference_dict.keys():
        if p in pardict:
            param_map[p] = [pardict[p]['min_tag'], pardict[p]['max_tag']]
            if not pardict[p]['changed_from_reference']:
                print(f"   ! Warning: Parameter {p} has no perturbations. Sensitivity will be zero.")
        else:
            print(f"   ! Warning: Parameter {p} defined in config but not found in YAML files.")
    
    print(f"Parameters to process: {list(param_map.keys())}")

    # 3. Load 2D Data (Scratch)
    scratch_dir = config['files']['raw_dir'] # Assicurati che sia nel config!
    all_tags = sorted(list(set([ref_tag] + [t for pair in param_map.values() for t in pair])))
    
    annual_cache = {}
    print("--- Loading experiments from scratch ---")
    for tag in all_tags:
        # Template per cercare i file NetCDF (adatta al tuo file system)
        path_pattern = os.path.join(scratch_dir, tag, "output/oifs/", f"{tag}_atm_cmip6_1m_*.nc")
        files = sorted(glob.glob(path_pattern))
        if not files:
            print(f"Warning: No files found for {tag} in {path_pattern}")
            continue
        
        ds = xr.open_mfdataset(files, chunks={'time_counter': 12}, combine='nested', 
                               concat_dim='time_counter', compat='override', 
                               coords='minimal', data_vars='minimal')
        if 'time_counter' in ds.dims: ds = ds.rename({'time_counter': 'time'})
        
        # Mean over the period and skip first year if possible
        ds_period = ds.sel(time=slice(str(y1), str(y2)))
        annual_cache[tag] = ds_period.mean('time').compute()
        print(f"   > Loaded {tag}")

    # 4. Compute Sensitivities
    all_var_datasets = []
    sample_raw = sorted(glob.glob(os.path.join(scratch_dir, ref_tag, "output/oifs/", "*.nc")))[0]

    for var in target_vars:
        method = methods_map.get(var, 'ycon')
        print(f"\nProcessing Variable: {var}")
        
        y0 = regrid_to_regular_smm_safe(get_var_2d(annual_cache[ref_tag], var), 
                                       target_grid, method, sample_raw, varname=var)

        param_entries = []
        for p, (tag_m, tag_p) in param_map.items():
            print(f"   -> Sensitivity to {p}")
            ym = regrid_to_regular_smm_safe(get_var_2d(annual_cache[tag_m], var), 
                                           target_grid, method, sample_raw, varname=var)
            yp = regrid_to_regular_smm_safe(get_var_2d(annual_cache[tag_p], var), 
                                           target_grid, method, sample_raw, varname=var)
            
            x_vals = [tunsets[tag_m][p], tunsets[ref_tag][p], tunsets[tag_p][p]]
            slope, r2 = compute_slope_and_linearity_2d(ym, y0, yp, x_vals)
            
            entry = xr.Dataset({'slope': slope, 'r2': r2}).expand_dims(parameter=[p])
            param_entries.append(entry)
        
        var_ds = xr.concat(param_entries, dim='parameter').expand_dims(variable=[var])
        all_var_datasets.append(var_ds)

    # 5. Save
    ds_master = xr.concat(all_var_datasets, dim='variable')
    output_file = config['files']['sensitivity_nc'].format(year1=y1, year2=y2)
    
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_master.data_vars}
    ds_master.to_netcdf(output_file, encoding=encoding)
    print(f"\nSuccess! 2D Sensitivities saved to: {output_file}")