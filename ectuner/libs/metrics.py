"""
Metrics Module.

Computes, from 2D maps, the quantities used to analyse tuning experiments and to
validate the linear emulator. Every function follows the same convention as the tuner
(``run_2d_tuning``): the bias is *model minus reference* on the reference grid, the area
weights are the cosine of latitude over the global domain, and the spatial cost is the
area-weighted MAE (``l1``) or RMSE (``l2``).

Main contents:

* **Data access**: :func:`get_realized_model_map`, :func:`load_reference_map`.
* **Bias metrics**: :func:`compute_bias_map` and :func:`bias_metrics`, which return the
  global-mean bias and the spatial cost of a bias map.
* **Emulator validation**: :func:`pattern_correlation`, :func:`drift_from_reference`,
  :func:`validation_table`.

The same functions are applied to the baseline, predicted and realized states, so no
definition offset can appear between them.
"""

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from .utils import normalize_metric, standardize_reference_signs, regrid_to_regular_smm_safe, compute_derived_flux, find_reference_files

__all__ = ['get_realized_model_map', 'load_reference_map', 'compute_bias_map', 'bias_metrics',
           'area_weights', 'zonal_mean', 'pattern_correlation', 'drift_from_reference',
           'validation_table']


def _file_year(path: str) -> Optional[int]:
    """Returns the first year in a raw OIFS file name (``..._1m_YYYY...``), or None."""
    m = re.search(r'_1m_(\d{4})', os.path.basename(path))
    return int(m.group(1)) if m else None


# ==============================================================================
# DATA ACCESS
# ==============================================================================
def get_realized_model_map(tag: str, var_name: str, base_2d_dir: str, scratch_dir: str, year1: int, year2: int, target_grid: str = "r180x90", methods_map = None) -> xr.DataArray:
    """
    Retrieves the 2D model map from cache or generates it from raw scratch files.

    The map is read from the cache written by the tuner
    (``base_2d_<tag>_<year1>_<year2>_<grid>.nc``) when it exists and contains the variable;
    otherwise it is computed from the raw OIFS monthly files (one file per year is assumed)
    and regridded. The result is not written to the cache.

    Args:
        tag: The experiment tag identifier.
        var_name: The target atmospheric variable to extract.
        base_2d_dir: Directory containing cached 2D base NetCDF files.
        scratch_dir: Directory containing the raw model scratch output files.
        year1: The starting year of the evaluation period.
        year2: The ending year of the evaluation period.
        target_grid: Target grid resolution for regridding. Defaults to "r180x90".
        methods_map: Optional dictionary mapping variable names to specific regridding methods (default ``"ycon"`` for every variable).

    Returns:
        An xarray.DataArray containing the time-averaged, regridded model map.

    Raises:
        FileNotFoundError: If neither the cache nor the raw scratch files exist.
        ImportError: If required regridding dependencies are not installed.
    """
    cache_file = os.path.join(base_2d_dir, f"base_2d_{tag}_{year1}_{year2}_{target_grid}.nc")
    if os.path.exists(cache_file):
        ds_base = xr.open_dataset(cache_file)
        if var_name in ds_base.data_vars:
            print(f"-> Using cache: {cache_file}")
            return ds_base[var_name]
        print(f"-> '{var_name}' not in {os.path.basename(cache_file)}, extracting from scratch...")
    else:
        others = glob.glob(os.path.join(base_2d_dir, f"base_2d_{tag}_*_{target_grid}.nc"))
        print(f"-> No cache for {tag} {year1}-{year2}"
              + (f" (other windows found: {[os.path.basename(o) for o in others]})" if others else "")
              + ". Extracting from scratch...")
    pattern = os.path.join(scratch_dir, tag, "output", "oifs", f"{tag}_atm_cmip6_1m_*.nc")
    valid_files = []
    for f in sorted(glob.glob(pattern)):
        year = _file_year(f)
        if year is not None and year1 <= year <= year2:
            valid_files.append(f)
    if not valid_files:
        raise FileNotFoundError(f"Missing scratch files for {tag}")
        
    ds = xr.open_mfdataset(valid_files, combine='nested', concat_dim='time_counter', chunks={'time_counter': 12}, compat='override', coords='minimal', data_vars='minimal')
    ds = ds.rename({'time_counter': 'time'}) if 'time_counter' in ds.dims else ds
    
    model_flux = compute_derived_flux(ds, var_name).sel(time=slice(str(year1), str(year2))).mean('time').compute()
    regrid_method = (methods_map or {}).get(var_name, 'ycon')
    return regrid_to_regular_smm_safe(model_flux, target_grid, method=regrid_method, raw_file_path=valid_files[0], varname=var_name)


def load_reference_map(var: str, ref_2d_dir: str, target_grid: str = "r180x90", period: Optional[str] = None) -> xr.DataArray:
    """
    Loads a reference climatology with the same file choice and sign convention as the tuner.

    Args:
        var: Variable name.
        ref_2d_dir: Directory of the reference climatologies.
        target_grid: Grid tag contained in the file name.
        period: Optional period string (e.g. ``"2000-2024"``); should match
            ``files.ref_2d_period`` of the tuning configuration.

    Returns:
        The time-averaged reference map, with the sign standardized to the model convention.

    Raises:
        FileNotFoundError: If no reference map is found.
    """
    files = find_reference_files(ref_2d_dir, var, target_grid, period)
    if not files:
        raise FileNotFoundError(f"No reference map for {var} in {ref_2d_dir}")
    if period and period not in os.path.basename(files[0]):
        print(f"Warning: no reference map for '{var}' with period {period}; using any available period.")
    if len(files) > 1:
        print(f"Warning: {len(files)} reference maps match '{var}'; using the first one.")
    print(f"-> Reference for {var}: {os.path.basename(files[0])}")
    ds = xr.open_dataset(files[0])
    ref = ds[var].mean('time') if 'time' in ds[var].dims else ds[var]
    return standardize_reference_signs({var: ref})[var]

# ==============================================================================
# BIAS METRICS
# ==============================================================================
def area_weights(da: xr.DataArray) -> xr.DataArray:
    """Returns the cosine-of-latitude area weights of a map (global region)."""
    return np.cos(np.deg2rad(da.lat))

def compute_bias_map(model_map: xr.DataArray, ref_map: xr.DataArray) -> xr.DataArray:
    """
    Computes the bias (model minus reference) on the reference grid, as in ``run_2d_tuning``.

    Args:
        model_map: Model map.
        ref_map: Reference map.

    Returns:
        The bias map on the grid of ``ref_map``.
    """
    return model_map.reindex_like(ref_map, method='nearest') - ref_map

def bias_metrics(bias_map: xr.DataArray, metric: str = 'l1') -> Dict[str, float]:
    """
    Computes the global-mean bias and the spatial cost of a bias map.

    Args:
        bias_map: Bias map (model minus reference).
        metric: ``"l1"``/``"mae"`` (cost = area-weighted MAE) or ``"l2"``/``"rmse"``
            (cost = area-weighted RMSE).

    Returns:
        Dictionary with the keys ``bias`` (signed, area-weighted global mean) and
        ``cost`` (spatial cost).
    """
    m = normalize_metric(metric)
    w = area_weights(bias_map)
    glob_bias = float(bias_map.weighted(w).mean(skipna=True))
    if m == 'l1':
        cost = float(abs(bias_map).weighted(w).mean(skipna=True))
    else:
        cost = float(np.sqrt((bias_map ** 2).weighted(w).mean(skipna=True)))
    return {'bias': glob_bias, 'cost': cost}


def zonal_mean(da: xr.DataArray) -> xr.DataArray:
    """Returns the zonal mean of a map, sorted by latitude."""
    return da.mean('lon').sortby('lat')


# ==============================================================================
# EMULATOR VALIDATION
# ==============================================================================
def pattern_correlation(pred_delta: xr.DataArray, real_delta: xr.DataArray) -> Dict[str, float]:
    """
    Compares a predicted and a realized change map.

    Args:
        pred_delta: Change predicted by the emulator (final minus initial bias).
        real_delta: Change realized by the model (child minus parent).

    Returns:
        The centred, area-weighted correlation coefficient.
    """
    pred = pred_delta.reindex_like(real_delta, method='nearest')
    w = area_weights(real_delta)
    wm = lambda x: float(x.weighted(w).mean(skipna=True))
    a, b = pred - wm(pred), real_delta - wm(real_delta)
    return wm(a * b) / np.sqrt(wm(a * a) * wm(b * b))


def drift_from_reference(diag_yaml_path: str) -> Tuple[float, float]:
    """
    Returns the RMS and the maximum relative departure of the parameters from their defaults.

    Uses the values recorded by the tuner (``drift_rms``, ``drift_max``) when present, and
    recomputes them from the parameter table for older diagnostics files.

    Args:
        diag_yaml_path: Path to a ``diagnostics_*.yaml`` file.

    Returns:
        Tuple ``(drift_rms, drift_max)``.
    """
    with open(diag_yaml_path) as f:
        d = yaml.safe_load(f)
    m = d.get('metrics', {})
    if m.get('drift_rms') is not None:
        return float(m['drift_rms']), float(m['drift_max'])
    rel = [(float(p['new_value']) - float(p['ref_value'])) / float(p['ref_value'])
           for p in d['parameters'] if float(p['ref_value']) != 0]
    return float(np.sqrt(np.mean(np.square(rel)))), float(np.max(np.abs(rel)))


def validation_table(manifest: Dict[str, Dict[str, Any]], var: str, metric: str, dir_diag: str,
                     base_2d_dir: str, scratch_dir: str, year1: int, year2: int,
                     ref_map: xr.DataArray, methods_map: Optional[Dict[str, str]] = None,
                     target_grid: str = "r180x90", return_maps: bool = False):
    """
    Builds the table of predicted versus realized states of each tuned run.
 
    For each run, the change predicted by the emulator (final minus initial bias in
    ``diagnostics_2d_<parent>_<tag>.nc``) is applied to the bias realized by the parent over
    ``year1``-``year2``. Predicted and realized states therefore share the same starting point,
    even if the tuner was run on a different window; ``init_check_*`` reports the difference
    between the starting state seen by the tuner and the realized parent.
 
    Args:
        manifest: Mapping ``tag -> {"parent", "alpha", "iteration"}``; runs without a parent are skipped.
        var: Variable to analyse.
        metric: ``"l1"`` or ``"l2"``.
        dir_diag: Directory of the diagnostics files.
        base_2d_dir: Directory of the cached 2D maps.
        scratch_dir: Directory of the raw model output.
        year1: First year of the evaluation period.
        year2: Last year of the evaluation period.
        ref_map: Reference map.
        methods_map: Optional regridding methods per variable.
        target_grid: Target grid.
        return_maps: If True, also return the bias maps of parent, prediction and model.
 
    Returns:
        DataFrame indexed by tag with the drift, ``init_check_*``, the global-mean bias and
        spatial cost of parent, prediction and model (``bias_*``, ``cost_*``), their changes
        with respect to the parent (``dbias_*``, ``dcost_*``), the errors (``err_*``,
        predicted minus realized change) and the pattern correlation of the change maps.
        With ``return_maps=True``, a tuple ``(table, maps)`` where
        ``maps[tag] = {"parent", "pred", "real"}``.
    """
    cache: Dict[str, xr.DataArray] = {}
 
    def bias_of(tag: str) -> xr.DataArray:
        if tag not in cache:
            mod = get_realized_model_map(tag, var, base_2d_dir, scratch_dir, year1, year2,
                                         target_grid, methods_map=methods_map)
            cache[tag] = compute_bias_map(mod, ref_map)
        return cache[tag]
 
    rows, maps = [], {}
    for tag, info in manifest.items():
        parent = info.get('parent')
        if parent is None:
            continue
        nc = os.path.join(dir_diag, f"diagnostics_2d_{parent}_{tag}.nc")
        if not os.path.exists(nc):
            print(f"Skipping {tag}: {nc} not found")
            continue
        ds = xr.open_dataset(nc)
        b_init, b_fin = ds[f'{var}_bias_init'], ds[f'{var}_bias_final']
        b_par, b_chi = bias_of(parent), bias_of(tag)
        d_pred = (b_fin - b_init).reindex_like(b_par, method='nearest')   # predicted change
        b_hat = b_par + d_pred                                              # predicted state
        m_init, m_par = bias_metrics(b_init, metric), bias_metrics(b_par, metric)
        m_hat, m_chi = bias_metrics(b_hat, metric), bias_metrics(b_chi, metric)
        d_rms, d_max = drift_from_reference(os.path.join(dir_diag, f"diagnostics_{tag}.yaml"))
        rows.append({
            'tag': tag, 'parent': parent, 'alpha': info.get('alpha'), 'iteration': info.get('iteration'),
            'drift_rms': d_rms, 'drift_max': d_max,
            'init_check_bias': m_init['bias'] - m_par['bias'],
            'init_check_cost': m_init['cost'] - m_par['cost'],
            'bias_parent': m_par['bias'], 'bias_pred': m_hat['bias'], 'bias_real': m_chi['bias'],
            'cost_parent': m_par['cost'], 'cost_pred': m_hat['cost'], 'cost_real': m_chi['cost'],
            'pattern_corr': pattern_correlation(d_pred, b_chi - b_par)})
        maps[tag] = {'parent': b_par, 'pred': b_hat, 'real': b_chi}
    if not rows:
        return (pd.DataFrame(), maps) if return_maps else pd.DataFrame()
    df = pd.DataFrame(rows).set_index('tag')
    for q in ('bias', 'cost'):
        df[f'd{q}_pred'] = df[f'{q}_pred'] - df[f'{q}_parent']
        df[f'd{q}_real'] = df[f'{q}_real'] - df[f'{q}_parent']
        df[f'err_d{q}'] = df[f'd{q}_pred'] - df[f'd{q}_real']
    return (df, maps) if return_maps else df