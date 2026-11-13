"""
Diagnostic Visualization Module.

Reads the diagnostic files written by ECtuner (``diagnostics_*.yaml`` and
``diagnostics_2d_*.nc``) and produces figures and tables for:

* parameter changes (scatter plot and heatmap);
* bias maps before and after tuning, and Pareto fronts between spatial and global error;
* validation of the linear emulator against the realized simulations (1:1 scatter of
  global bias or spatial error, zonal-mean profiles of all strategies on the same axes);
* decomposition of the predicted global bias into per-parameter contributions, and
  sensitivity maps.

The definitions of bias, spatial cost, drift and validation tables live in
:mod:`ectuner.libs.metrics`; its main functions are re-exported here so that
existing notebooks keep working. The validation figures take the output of
``metrics.validation_table(..., return_maps=True)``.
"""
import os
import glob
import yaml
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import xarray as xr
from pathlib import Path
from typing import Dict, Any, List
from .utils import normalize_metric, regrid_to_regular_smm_safe
from .metrics import (get_realized_model_map, load_reference_map, compute_bias_map, bias_metrics,
                      area_weights, zonal_mean, pattern_correlation, drift_from_reference, validation_table)

def _handle_show_or_save(output_path=None):
    """
    Internal helper to handle plot saving or showing cleanly.
    
    Args:
        output_path: If provided, saves the plot to this destination. 
            Otherwise, displays it interactively.
    """
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {output_path}")
    else:
        plt.show()
    plt.close()

def load_diagnostic_yamls(folder_path: str) -> Dict[str, Any]:
    """
    Scans a folder for ECtuner diagnostic files and loads them.

    Args:
        folder_path: Path to the directory containing diagnostics YAMLs.

    Returns:
        A dictionary mapping experiment tags to their parsed diagnostic data.
        
    Raises:
        FileNotFoundError: If the specified folder does not exist.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    diag_data = {}
    files = sorted(folder.glob("diagnostics_*.yaml"))

    for filepath in files:
        tag = filepath.stem.replace("diagnostics_", "")
        with open(filepath, 'r') as f:
            diag_data[tag] = yaml.safe_load(f)
            
    print(f"Loaded {len(diag_data)} diagnostic files from {folder_path}")
    return diag_data


def build_metrics_dataframe(diag_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the nested YAML dictionary into a Pandas DataFrame optimized 
    for Pareto front and Tradeoff plotting.

    Args:
        diag_data: Data loaded via `load_diagnostic_yamls`.

    Returns:
        A DataFrame containing sorted metrics for all experiments, structured 
        to isolate Spatial vs Global costs.
    """
    records = []
    
    for exp_tag, data in diag_data.items():
        metrics = data.get('metrics', {})
        raw_metric = metrics.get('metric_name', 'l1')
        try:
            metric = normalize_metric(raw_metric)
        except ValueError:                      # es. '1D_scalars'
            metric = str(raw_metric)
        record = {
            'Experiment': exp_tag,
            'Alpha': metrics.get('alpha', np.nan),
            'Metric': metric,
            'Raw_Spatial_Cost': metrics.get('total_spatial_cost', np.nan),
            'Raw_Global_Cost': metrics.get('total_global_cost', np.nan),
        }
        record['Metric_Type'] = {'l2': 'L2 (Quadratic)', 'l1': 'L1 (Linear)'}.get(metric, f'Other ({metric})')
        record['Phys_Spatial_Total'] = record['Raw_Spatial_Cost']
        record['Phys_Global_Total'] = record['Raw_Global_Cost']
            
        # Extract Variable-Specific Biases from the 'biases' block if needed
        # Assuming the structure is: biases -> targets -> list of dicts
        targets = data.get('biases', {}).get('targets', [])
        for t in targets:
            var = t['variable']
            if t['region'] == 'Global':
                record[f'{var}_AbsBias'] = abs(t['bias_final'])

        var_metrics = data.get('var_metrics', {})
        for var_name, stats in var_metrics.items():
            record[f'{var_name}_Spatial_Cost'] = stats.get('spatial_cost', np.nan)
                
        records.append(record)
        
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('Experiment', inplace=True)
        df.sort_values(by='Alpha', inplace=True)
        
    return df

def extract_1d_validation_data(experiment_tags, ecmean_dir, diagnostics_dir, reference_file, require_diagnostics=True,
                               base_template=None, year1=None, year2=None):
    """
    Extracts the emulator's predictions and actual model results for the 1:1 validation plot.
    
    Args:
        experiment_tags: List of tags identifying the experiments.
        ecmean_dir: Path to the directory containing ECmean output files.
        diagnostics_dir: Path to the directory containing ECtuner diagnostic files.
        reference_file: Path to the YAML file containing reference observations.
        require_diagnostics: If True, raises a warning when diagnostics files are missing.
        base_template: Optional ECmean file name pattern (``files.base`` of the config, e.g.
            ``GM_EC26-PDAY_{exp}_EC-Earth4_r1i1p1f1_{year1}_{year2}.yml``). With ``year1`` and
            ``year2`` it selects exactly one file per run; otherwise any ``*<tag>*.yml`` is used.
        year1, year2: Evaluation window, used with ``base_template``.
        
    Returns:
        A tuple of three dictionaries:
        - Reference observations (keyed by variable).
        - Realized model outputs from ECmean (keyed by tag, then variable).
        - Predicted biases from ECtuner (keyed by tag, then variable).
    """
    ref_obs = {}
    data_ecmean = {}
    data_predicted = {}

    # 1. Extract Reference Observations
    with open(reference_file, 'r') as f:
        ref_raw = yaml.safe_load(f)
        for var, data in ref_raw.items():
            if isinstance(data, dict) and isinstance(data.get('obs'), dict):
                obs = data['obs']
                if isinstance(obs.get('ALL'), dict) and 'Global' in obs['ALL']:
                    ref_obs[var] = obs['ALL']['Global']['mean']

    # 2. Extract Data for each Experiment
    for tag in experiment_tags:
        data_ecmean[tag] = {}
        data_predicted[tag] = {}

        # Read Actual Results (ECmean)
        if base_template and year1 is not None and year2 is not None:
            ecmean_pattern = os.path.join(ecmean_dir, base_template.format(exp=tag, year1=year1, year2=year2))
        else:
            ecmean_pattern = os.path.join(ecmean_dir, f"*_{tag}_*.yml")
        ecmean_files = sorted(glob.glob(ecmean_pattern))
        
        if ecmean_files:
            if len(ecmean_files) > 1:
                print(f"Warning: {len(ecmean_files)} ECmean files match {tag}; using {os.path.basename(ecmean_files[0])}")
            with open(ecmean_files[0], 'r') as f:
                ecm_raw = yaml.safe_load(f)
                for var, data in ecm_raw.items():
                    if isinstance(data, dict) and isinstance(data.get('ALL'), dict):
                        if 'Global' in data['ALL']:
                            data_ecmean[tag][var] = data['ALL']['Global']
        else:
            print(f"Warning: Missing ECmean file for tag {tag}")

        # Read Emulator Predictions (ECtuner Diagnostics)
        diag_file = os.path.join(diagnostics_dir, f"diagnostics_{tag}.yaml")
        if os.path.exists(diag_file):
            with open(diag_file, 'r') as f:
                diag_raw = yaml.safe_load(f)
                
                # Metodo 1: Prova a leggere il formato 2D (var_metrics)
                var_metrics = diag_raw.get('var_metrics', {})
                for var, stats in var_metrics.items():
                    if 'predicted_global_bias' in stats:
                        data_predicted[tag][var] = stats['predicted_global_bias']
                
                # Metodo 2: Se var_metrics è vuoto, è un file 1D. Cerca in biases
                if not var_metrics:
                    biases = diag_raw.get('biases', {})
                    for category in ['targets', 'diagnostics']:
                        for row in biases.get(category, []):
                            # Filtriamo per la media globale annuale per il plot 1:1
                            if row.get('region') == 'Global' and row.get('season') == 'ALL':
                                var = row['variable']
                                data_predicted[tag][var] = row['bias_final']
        else:
            if require_diagnostics:
                print(f"Warning: Missing Diagnostics file for tag {tag}")
            
    return ref_obs, data_ecmean, data_predicted

# ==============================================================================
# PARAMETER VISUALIZATIONS
# ==============================================================================

def plot_parameter_scatter(diag_data: Dict[str, Any], output_path: str = None) -> None:
    """
    Creates a scatter plot with parameters on the x-axis and their relative change.

    Visualizes how much each parameter drifted from its default reference value.
    A value of 0.0 means no change from the reference.

    Args:
        diag_data: Dictionary containing loaded diagnostic data.
        output_path: Optional file path to save the plot.
    """
    if not diag_data:
        print("No data to plot")
        return

    params_set = set()
    for data in diag_data.values():
        for p in data['parameters']:
            params_set.add(p['name'])
    all_params = sorted(list(params_set))

    fig, ax = plt.subplots(figsize=(max(12, len(all_params) * 0.8), 8))
    
    exp_names = list(diag_data.keys())
    okabe_ito = ['#D55E00', '#0072B2', '#009E73', '#E69F00', '#CC79A7', '#F0E442', '#56B4E9']
    colors = [okabe_ito[i % len(okabe_ito)] for i in range(len(exp_names))]
    
    for exp_idx, exp_name in enumerate(exp_names):
        x_positions = []
        y_values = []
        
        for param_idx, param_name in enumerate(all_params):
            param_data = next((p for p in diag_data[exp_name]['parameters'] if p['name'] == param_name), None)
            
            if param_data and float(param_data['ref_value']) != 0:
                x_positions.append(param_idx)
                rel_change = (float(param_data['new_value']) - float(param_data['ref_value'])) / float(param_data['ref_value'])
                y_values.append(rel_change)
        
        ax.scatter(x_positions, y_values, s=100, alpha=0.7, 
                   color=colors[exp_idx], label=exp_name, zorder=3)
    
    # Modifica qui: La linea di base ora è a y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xticks(range(len(all_params)))
    ax.set_xticklabels(all_params, rotation=45, ha='right')
    # Modifica qui: Aggiornata la label dell'asse Y
    ax.set_ylabel('Relative change (w.r.t default)', fontsize=12)
    ax.set_xlabel('Parameters', fontsize=12)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_parameter_heatmap(diag_data: Dict[str, Any], output_path: str = None) -> None:
    """
    Creates a heatmap showing the normalized relative change of parameters.

    Provides a quick visual indication of which parameters were pushed 
    towards their upper or lower boundaries during tuning.

    Args:
        diag_data: Dictionary containing loaded diagnostic data.
        output_path: Optional file path to save the plot.
    """
    if not diag_data:
        return

    data_matrix = {}
    for exp_tag, data in diag_data.items():
        data_matrix[exp_tag] = {}
        for p in data['parameters']:
            if float(p['ref_value']) != 0:
                rel_change = (float(p['new_value']) - float(p['ref_value'])) / float(p['ref_value'])
                data_matrix[exp_tag][p['name']] = rel_change
    df = pd.DataFrame(data_matrix).T

    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.8), max(4, len(df) * 0.6)))
    
    im = ax.imshow(df.values, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    plt.colorbar(im, ax=ax, label='Relative change')

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.25 else "black"
                ax.text(j, i, f'{val:.2e}', ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    _handle_show_or_save(output_path)

# ==============================================================================
# SPATIAL DIAGNOSTICS & PARETO FRONTS
# ==============================================================================

def compute_diagnostic_metrics(ds: xr.Dataset, var: str):
    """
    Computes zonal metrics and plot limits for a given variable from the diagnostic NetCDF.
    
    Args:
        ds: The diagnostic NetCDF dataset containing initial and final bias maps.
        var: The name of the atmospheric variable.
        
    Returns:
        A tuple containing the initial bias, final bias, improvement map, 
        min/max limits for plotting, and zonal mean profiles.
    """
    b_init = ds[f'{var}_bias_init']
    b_final = ds[f'{var}_bias_final']
    improv = ds[f'{var}_improvement']

    vmax = float(max(abs(b_init).max().item(), abs(b_final).max().item()))
    vmin = -vmax
    
    zonal_init = b_init.mean(dim='lon')
    zonal_final = b_final.mean(dim='lon')
    
    return b_init, b_final, improv, vmin, vmax, zonal_init, zonal_final

def plot_tuning_results_maps(diag_file: str, output_dir: str, run_tag: str, bias_limit: float = None, improv_limit: float = None) -> None:
    """
    Generates 2D maps comparing Initial Bias vs Final Predicted Bias.

    Reads the diagnostic NetCDF map produced by ECtuner's 2D engine and 
    plots a four-panel figure: Initial Bias, Final Bias, Absolute Improvement,
    and the Zonal Bias profile.

    Args:
        diag_file: Path to the diagnostic NetCDF file.
        output_dir: Directory to save the output plots.
        run_tag: Identifier for the current experiment run.
        bias_limit: Fixed symmetric limit for the Bias colorbar (e.g., 60 for -60 to 60).
        improv_limit: Fixed symmetric limit for the Improvement colorbar (e.g., 10 for -10 to 10).
    """
    ds = xr.open_dataset(diag_file)
    variables = [v.replace('_bias_init', '') for v in ds.data_vars if '_bias_init' in v]
    
    for var in variables:

        b_init, b_final, improv, dyn_vmin, dyn_vmax, zonal_init, zonal_final = compute_diagnostic_metrics(ds, var)

        vmin = -bias_limit if bias_limit else dyn_vmin
        vmax = bias_limit if bias_limit else dyn_vmax

        if improv_limit:
            vmin_imp, vmax_imp = -improv_limit, improv_limit
        else:
            max_imp = float(abs(improv).max().item())
            vmin_imp, vmax_imp = -max_imp, max_imp

        fig = plt.figure(figsize=(24, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.6])

        ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax1 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
        ax_zonal = fig.add_subplot(gs[3])
        
        im0 = b_init.plot(ax=ax0, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax0.set_title(f'{var} Initial Bias')
        
        im1 = b_final.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax1.set_title(f'{var} Final Bias (Predicted)')
        
        im2 = improv.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='PiYG', vmin=vmin_imp, vmax=vmax_imp, add_colorbar=False)
        ax2.set_title(f'{var} Absolute Improvement')
        
        for ax in [ax0, ax1, ax2]:
            ax.coastlines()
            
        fig.colorbar(im1, ax=[ax0, ax1], location='bottom', label='Bias Units', fraction=0.05, pad=0.08)
        fig.colorbar(im2, ax=[ax2], location='bottom', label='Improvement', fraction=0.05, pad=0.08)

        ax_zonal.plot(zonal_init, ds.lat, label='Init Bias', color='#D55E00', linestyle='--', lw=1.5)
        ax_zonal.plot(zonal_final, ds.lat, label='Final Bias', color='#0072B2', lw=2) 
        ax_zonal.axvline(0, color='black', lw=0.8, alpha=0.5)
        ax_zonal.set_title(f'{var} Zonal Bias')
        ax_zonal.legend(fontsize='small', loc='upper right')
        ax_zonal.grid(True, alpha=0.3)
        
        plot_name = f"comparison_2d_{var}_{run_tag}.png"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight', dpi=150)
        plt.close()

# ==============================================================================
def plot_tradeoff_comparison(df: pd.DataFrame, group_by: str = 'Metric_Type', output_path: str = None) -> None:
    """
    Plots the global Pareto front mapping total spatial error vs global bias.

    Allows dynamic grouping to visualize the tradeoffs encountered by the optimizer
    under different spatial weighting alphas.

    Args:
        df: DataFrame built from diagnostic data.
        group_by: Column name to group markers by.
        output_path: Optional file path to save the plot.
    """
    plt.figure(figsize=(9, 6.5))
    
    markers = ['o', 's', '^', 'D', 'v', 'P']
    last_sc = None
    legend_elements = []

    if group_by not in df.columns:
        df[group_by] = 'All Data'

    metrics_present = df['Metric'].unique() if 'Metric' in df.columns else []
    title_suffix = "(RMSE/MAE & Abs Bias)" if len(metrics_present) > 1 else ("(RMSE & Abs Bias)" if 'l2' in metrics_present else "(MAE & Abs Bias)")

    for i, group_name in enumerate(df[group_by].unique()):
        subset = df[df[group_by] == group_name].dropna(subset=['Phys_Spatial_Total', 'Phys_Global_Total']).sort_values('Alpha')
        if subset.empty: continue
        
        mkr = markers[i % len(markers)]
        last_sc = plt.scatter(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], 
                              c=subset['Alpha'], cmap='viridis', marker=mkr, s=140, edgecolors='black', zorder=3, vmin=0, vmax=1)
        plt.plot(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], linestyle='--', alpha=0.4, zorder=2)
        
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', (row['Phys_Spatial_Total'], row['Phys_Global_Total']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        legend_elements.append(
            Line2D([0], [0], marker=mkr, color='w', label=str(group_name), 
                   markerfacecolor='gray', markersize=10, markeredgecolor='black')
        )
    
    legend_title = group_by.replace('_', ' ')
    plt.legend(handles=legend_elements, loc='upper right', title=legend_title)
    plt.title(f"Pareto Front: Total Spatial Error vs Total Global Bias\nUnits: {title_suffix}", fontsize=12, fontweight='bold')
    plt.xlabel("Total Spatial Error (Physical Units)", fontsize=11)
    plt.ylabel("Total Global Bias (Physical Units)", fontsize=11)
    
    if last_sc: plt.colorbar(last_sc).set_label(r'Alpha Value ($\alpha$)', fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_variable_pareto(df: pd.DataFrame, var: str, group_by: str = 'Metric_Type', output_path: str = None) -> None:
    """
    Pareto Front specific to a single atmospheric variable.

    Args:
        df: DataFrame built from diagnostic data.
        var: The specific atmospheric variable to plot (e.g., 'net_toa').
        group_by: Column name to group markers by.
        output_path: Optional file path to save the plot.
    """
    plt.figure(figsize=(9, 6.5))
    
    col_bias = f'{var}_AbsBias'
    col_spat = f'{var}_Spatial_Cost'

    if col_bias not in df.columns:
        print(f"Variable {var} not found in DataFrame for Pareto plot.")
        return
    
    markers = ['o', 's', '^', 'D', 'v', 'P']
    last_sc = None
    legend_elements = []

    if group_by not in df.columns:
        df[group_by] = 'All Data'

    for i, group_name in enumerate(df[group_by].unique()):
        subset = df[df[group_by] == group_name].dropna(subset=[col_spat, col_bias]).sort_values('Alpha')
        if subset.empty: continue
        
        mkr = markers[i % len(markers)]
        last_sc = plt.scatter(subset[col_spat], subset[col_bias], 
                              c=subset['Alpha'], cmap='viridis', marker=mkr, s=140, edgecolors='black', zorder=3, vmin=0, vmax=1)
        plt.plot(subset[col_spat], subset[col_bias], linestyle='--', alpha=0.4, zorder=2)
        
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', (row[col_spat], row[col_bias]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        legend_elements.append(
            Line2D([0], [0], marker=mkr, color='w', label=str(group_name), 
                   markerfacecolor='gray', markersize=10, markeredgecolor='black')
        )

    legend_title = group_by.replace('_', ' ')
    plt.legend(handles=legend_elements, loc='upper right', title=legend_title)
        
    plt.title(f"Pareto Front: Spatial Error vs Global Bias ({var.upper()})", fontsize=12, fontweight='bold')
    plt.xlabel("Spatial Error [W/m²]", fontsize=11)
    plt.ylabel("Absolute Global Bias [W/m²]", fontsize=11)
    if last_sc: plt.colorbar(last_sc).set_label(r'Alpha Value ($\alpha$)', fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

# ==============================================================================
# EMULATOR VALIDATION
# ==============================================================================
# Both figures read the output of ``metrics.validation_table(..., return_maps=True)``:
#   tab[var]  -> DataFrame indexed by run tag (columns 'parent', 'alpha', 'bias_*', 'cost_*', ...)
#   maps[var] -> {tag: {'parent': map, 'pred': map, 'real': map}}

VAR_LONG_NAMES = {
    'rsnt': 'TOA net shortwave radiation',
    'rlnt': 'TOA net longwave radiation',
    'net_toa': 'TOA net radiation',
}

# Colours fixed by alpha, so the same strategy has the same colour in every figure.
TUNING_COLORS = {0.0: '#D55E00', 0.4: '#0072B2', 1.0: '#009E73'}
_FALLBACK_COLORS = ['#E69F00', '#CC79A7', '#56B4E9', '#F0E442']
# Marker shape fixed by alpha too (redundant with colour: readable in grey scale and on projectors)
TUNING_MARKERS = {0.0: 'o', 0.4: 's', 1.0: 'D'}
_FALLBACK_MARKERS = ['^', 'v', 'P', 'X']


def var_long_name(var: str, with_symbol: bool = True) -> str:
    """'rsnt' -> 'TOA net shortwave radiation (rsnt)'."""
    name = VAR_LONG_NAMES.get(var, var.upper())
    return f'{name} ({var})' if with_symbol and var in VAR_LONG_NAMES else name


def tuning_label(alpha, show_alpha: bool = False) -> str:
    """Readable name of the tuning strategy: 0 -> spatial, 1 -> global, otherwise hybrid."""
    a = float(alpha)
    name = 'Spatial tuning' if a == 0 else 'Global tuning' if a == 1 else 'Hybrid tuning'
    return f'{name} (α = {a:g})' if show_alpha else name


def tuning_color(alpha, i: int = 0) -> str:
    """Colour of a strategy; runs without alpha (e.g. 1D tuning) get the fallback palette."""
    try:
        a = round(float(alpha), 2)
    except (TypeError, ValueError):
        a = None
    return TUNING_COLORS.get(a, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def _run_label(tab_v: pd.DataFrame, tag: str, show_alpha: bool, show_tag: bool) -> str:
    """Legend label: the 'label' column if given, else the strategy name from alpha, else the tag."""
    if 'label' in tab_v.columns and pd.notna(tab_v.loc[tag, 'label']):
        lab = str(tab_v.loc[tag, 'label'])
    elif 'alpha' in tab_v.columns and pd.notna(tab_v.loc[tag, 'alpha']):
        lab = tuning_label(tab_v.loc[tag, 'alpha'], show_alpha)
    else:
        return tag
    return f'{lab} [{tag}]' if show_tag else lab


def tuning_marker(alpha, i: int = 0) -> str:
    """Marker of a strategy; runs without alpha get the fallback markers."""
    try:
        a = round(float(alpha), 2)
    except (TypeError, ValueError):
        a = None
    return TUNING_MARKERS.get(a, _FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)])


def _strategy_handles(alphas, show_alpha=False):
    """Legend handles: one marker (shape and colour) per strategy."""
    return [Line2D([0], [0], marker=tuning_marker(a, i), ls='', markersize=10, markerfacecolor=tuning_color(a, i),
                   markeredgecolor='black', label=tuning_label(a, show_alpha))
            for i, a in enumerate(alphas)]


def plot_validation_scatter(tab: Dict[str, pd.DataFrame], variables: List[str] = None, quantity: str = 'bias',
                            runs: List[str] = None, show_alpha: bool = False, connect: bool = False,
                            parent_label: str = None, output_path: str = None) -> None:
    """
    1:1 comparison between emulator prediction and realized model, one panel per variable.

    Colour = tuning strategy; the number inside each circle = iteration. With ``connect`` the
    successive iterations of a branch are joined by a thin line (off by default).
    Shape and colour both identify the strategy. With 'dbias'/'dcost' the quadrants where the
    predicted change has the wrong sign are shaded. The legend (strategies, parent, 1:1) is placed once, below the panels.

    Args:
        tab: Output of ``validation_table`` (dict variable -> DataFrame).
        variables: Variables to plot (default: all keys of ``tab``).
        quantity: 'bias' (global-mean bias), 'cost' (MAE / RMSE), or 'dbias' / 'dcost' (change
            with respect to the parent: predicted vs realized change).
        runs: Run tags to include (default: all rows of the table).
        show_alpha: Add the alpha value to the strategy names.
        connect: Join the iterations of each branch with a thin line.
        parent_label: Legend label for the parent star (default 'Parent (<tag>)').
        output_path: Optional file path to save the figure.
    """
    variables = variables or list(tab)
    is_change = quantity.startswith('d')
    base = quantity[1:] if is_change else quantity
    qname = {'bias': 'global-mean bias', 'cost': 'spatial error'}[base]
    qname = f'Change of {qname}' if is_change else qname[0].upper() + qname[1:]

    fig, axes = plt.subplots(1, len(variables), figsize=(5.2 * len(variables), 5.6), squeeze=False)
    star_label = None
    for ax, v in zip(axes[0], variables):
        t = tab[v] if runs is None else tab[v].loc[runs]
        t = t.copy()
        if 'iteration' not in t:
            t['iteration'] = 1
        x, y = t[f'{quantity}_pred'], t[f'{quantity}_real']
        alphas = list(dict.fromkeys(t['alpha'])) if 'alpha' in t else [None]

        for i, a in enumerate(alphas):
            g = t if a is None else t[t['alpha'] == a]
            g = g.sort_values('iteration')
            col = tuning_color(a, i)
            if connect and len(g) > 1:
                ax.plot(g[f'{quantity}_pred'], g[f'{quantity}_real'], color=col, lw=1, alpha=0.5, zorder=3)
            ax.scatter(g[f'{quantity}_pred'], g[f'{quantity}_real'], s=190, color=col, edgecolors='black',
                       marker=tuning_marker(a, i),
                       linewidths=0.8, zorder=5)
            for tag, r in g.iterrows():
                ax.annotate(str(int(r['iteration'])), (r[f'{quantity}_pred'], r[f'{quantity}_real']),
                            ha='center', va='center', fontsize=7.5, fontweight='bold', color='white', zorder=6)

        if is_change:
            vals = np.r_[x, y, 0.0]
            ax.axhline(0, color='grey', lw=0.8); ax.axvline(0, color='grey', lw=0.8)
        else:
            # Star only for parents that are not runs of the figure (e.g. only phis)
            par_vals = t.groupby('parent')[f'{quantity}_parent'].first()
            par_vals = par_vals[~par_vals.index.isin(t.index)]
            for par, val in par_vals.items():
                ax.scatter(val, val, marker='*', s=320, color='gold', edgecolors='black', zorder=6)
            if len(par_vals):
                star_label = parent_label or (f'Parent ({par_vals.index[0]})' if len(par_vals) == 1 else 'Parent runs')
            vals = np.r_[x, y, t[f'{quantity}_parent']]

        pad = 0.1 * (vals.max() - vals.min() or 1.0)
        lims = [vals.min() - pad, vals.max() + pad]
        ax.plot(lims, lims, color='grey', ls='--', lw=1, zorder=1)
        if is_change:                                    # quadrants where the predicted change has the wrong sign
            lo, hi = lims
            ax.fill_between([0, hi], lo, 0, color='grey', alpha=0.12, lw=0, zorder=0)
            ax.fill_between([lo, 0], 0, hi, color='grey', alpha=0.12, lw=0, zorder=0)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal', adjustable='box')
        ax.set_title(var_long_name(v), fontsize=11)
        ax.set_xlabel(f'{qname}, emulator [W m$^{{-2}}$]')
        ax.set_ylabel(f'{qname}, model [W m$^{{-2}}$]')
        ax.grid(alpha=0.3)

    handles = _strategy_handles([a for a in alphas if a is not None], show_alpha)
    if star_label:
        handles.append(Line2D([0], [0], marker='*', ls='', markersize=15, markerfacecolor='gold',
                              markeredgecolor='black', label=star_label))
    handles.append(Line2D([0], [0], color='grey', ls='--', lw=1, label='1:1'))
    if is_change:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='grey', alpha=0.25, label='wrong sign'))
    handles.append(Line2D([0], [0], marker='o', ls='', markersize=9, markerfacecolor='grey',
                          markeredgecolor='black', label='number = iteration'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    _handle_show_or_save(output_path)


def plot_validation_zonal(tab: Dict[str, pd.DataFrame], maps: Dict[str, Dict[str, Dict[str, xr.DataArray]]],
                          variables: List[str] = None, runs: List[str] = None, show_change: bool = True,
                          show_emulator: bool = True, show_alpha: bool = False, show_tag: bool = False,
                          parent_label: str = None, output_path: str = None) -> None:
    """
    Zonal-mean bias of the parent and of all tuned runs on the same axes, latitude on the x axis.

    Rows are variables. The left column shows the zonal-mean bias with respect to the reference;
    with ``show_change`` a right column shows the change from the parent (run minus parent), which
    is where the differences between strategies are visible. Solid lines are the model, dotted
    lines the emulator prediction, in the same colour.

    Args:
        tab, maps: Output of ``validation_table(..., return_maps=True)``.
        variables: Variables to plot (default: all keys of ``tab``).
        runs: Run tags to include (default: all rows of the table).
        show_change: Add the column with the change from the parent.
        show_emulator: Draw the emulator prediction (dotted).
        show_alpha, show_tag: Add alpha / run tag to the legend labels.
        parent_label: Legend label for the parent (default 'Parent (<tag>)').
        output_path: Optional file path to save the figure.
    """
    from matplotlib.lines import Line2D
    variables = variables or list(tab)
    ncol = 2 if show_change else 1
    fig, axes = plt.subplots(len(variables), ncol, figsize=(7 * ncol, 3.6 * len(variables)),
                             sharex=True, squeeze=False)

    for r, v in enumerate(variables):
        t = tab[v] if runs is None else tab[v].loc[runs]
        parents = list(dict.fromkeys(t['parent']))
        one_parent = len(parents) == 1
        ax_b = axes[r, 0]

        # Parent(s): black if there is only one, otherwise dashed in the colour of its branch
        for par in parents:
            first = t.index[t['parent'] == par][0]
            z = zonal_mean(maps[v][first]['parent'])
            col = 'black' if one_parent else tuning_color(t.loc[first, 'alpha'])
            ax_b.plot(z.lat, z, color=col, lw=2.5 if one_parent else 1.5, ls='-' if one_parent else '--',
                      zorder=4, label=(parent_label or f'Parent ({par})') if one_parent else None)

        for i, tag in enumerate(t.index):
            col = tuning_color(t.loc[tag, 'alpha'], i)
            lab = _run_label(t, tag, show_alpha, show_tag)
            z_par = zonal_mean(maps[v][tag]['parent'])
            z_real = zonal_mean(maps[v][tag]['real'])
            z_pred = zonal_mean(maps[v][tag]['pred'])
            ax_b.plot(z_real.lat, z_real, color=col, lw=2, zorder=5, label=lab)
            if show_emulator:
                ax_b.plot(z_pred.lat, z_pred, color=col, lw=1.6, ls=':', zorder=5)
            if show_change:
                ax_d = axes[r, 1]
                ax_d.plot(z_real.lat, z_real - z_par, color=col, lw=2, zorder=5)
                if show_emulator:
                    ax_d.plot(z_pred.lat, z_pred - z_par, color=col, lw=1.6, ls=':', zorder=5)

        ax_b.set_ylabel(f'{var_long_name(v)}\nbias [W m$^{{-2}}$]', fontsize=10)
        if show_change:
            axes[r, 1].set_ylabel(f'Change from parent [W m$^{{-2}}$]', fontsize=10)
        for ax in axes[r]:
            ax.axhline(0, color='grey', lw=1, ls='--', zorder=1)
            ax.grid(True, ls=':', alpha=0.6)

    for ax in axes[-1]:
        ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_xlabel('Latitude [°]')
    axes[0, 0].set_title('Zonal-mean bias w.r.t. CERES-EBAF', fontsize=11)
    if show_change:
        axes[0, 1].set_title('Zonal-mean change from the parent run (previous iteration)', fontsize=11)

    # Legend 1: runs (colours). Legend 2: line styles.
    axes[0, 0].legend(fontsize=8, loc='best')
    if show_emulator:
        styles = [Line2D([0], [0], color='grey', lw=2, label='Model'),
                  Line2D([0], [0], color='grey', lw=1.6, ls=':', label='Emulator')]
        if len(set(tab[variables[0]]['parent'] if runs is None else tab[variables[0]].loc[runs, 'parent'])) > 1:
            styles.append(Line2D([0], [0], color='grey', lw=1.5, ls='--', label='Parent (previous iteration)'))
        axes[0, -1].legend(handles=styles, fontsize=8, loc='best')
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_emulator_error(tab: Dict[str, pd.DataFrame], variables: List[str] = None, quantity: str = 'bias',
                        x: str = 'iteration', mark_after: int = None, mark_label: str = None,
                        show_alpha: bool = False, output_path: str = None) -> None:
    """
    Error of the emulator on the change of each run (predicted minus realized), by iteration.

    Args:
        tab: Output of ``validation_table``.
        variables: Variables to plot (default: all keys of ``tab``).
        quantity: 'bias' (global-mean bias) or 'cost' (spatial error).
        x: 'iteration' (x axis = iteration) or 'drift' (x axis = RMS drift from the default
            parameters, with the iteration number written next to each point).
        mark_after: Draw a vertical line after this iteration (e.g. 5, when the drift cap changes).
        mark_label: Text of that line (e.g. 'drift cap 0.5 -> 0.8').
        show_alpha: Add the alpha value to the strategy names.
        output_path: Optional file path to save the figure.
    """
    variables = variables or list(tab)
    qname = {'bias': 'global-mean bias', 'cost': 'spatial error'}[quantity]
    fig, axes = plt.subplots(1, len(variables), figsize=(5.4 * len(variables), 4.3), squeeze=False)
    for ax, v in zip(axes[0], variables):
        t = tab[v]
        for i, (a, g) in enumerate(t.groupby('alpha', sort=True)):
            g = g.sort_values('iteration')
            xs = g['iteration'] if x == 'iteration' else 100 * g['drift_rms']
            col = tuning_color(a, i)
            ax.plot(xs, g[f'err_d{quantity}'], '-o', color=col, lw=1.6, ms=6, label=tuning_label(a, show_alpha))
            if x == 'drift':
                for xi, yi, it in zip(xs, g[f'err_d{quantity}'], g['iteration']):
                    ax.annotate(str(int(it)), (xi, yi), xytext=(4, 4), textcoords='offset points',
                                fontsize=8, fontweight='bold', color=col)
        ax.axhline(0, color='grey', lw=0.8)
        if x == 'iteration':
            ax.set_xticks(sorted(t['iteration'].unique()))
            ax.set_xlabel('Iteration')
            if mark_after is not None:
                ax.axvline(mark_after + 0.5, color='black', ls=':', lw=1)
                if mark_label:
                    ax.text(mark_after + 0.55, 0.97, mark_label, transform=ax.get_xaxis_transform(),
                            va='top', fontsize=8)
        else:
            ax.set_xlabel('RMS drift from the default parameters [%]')
        ax.set_ylabel(f'Error on {qname} change\n(emulator - model) [W m$^{{-2}}$]')
        ax.set_title(var_long_name(v), fontsize=11)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    _handle_show_or_save(output_path)


def plot_zonal_trajectories(tab: Dict[str, pd.DataFrame], maps: Dict[str, Dict[str, Dict[str, xr.DataArray]]],
                            variables: List[str] = None, alphas: List[float] = None, change: bool = False,
                            show_alpha: bool = False, output_path: str = None) -> None:
    """
    Zonal means of the realized runs of each branch, iteration after iteration.

    Rows are variables, columns are tuning strategies. Each panel shows the default run (black)
    and the model runs of all iterations of that branch, coloured from light (iteration 1) to
    dark (last iteration). With ``change=True`` the curves are differences from the default run,
    which makes the cumulative effect of the iterations easier to see.

    Args:
        tab, maps: Output of ``validation_table(..., return_maps=True)``.
        variables: Variables to plot (default: all keys of ``tab``).
        alphas: Branches to plot (default: all alphas in the table).
        change: Plot differences from the default run instead of the bias.
        show_alpha: Add the alpha value to the column titles.
        output_path: Optional file path to save the figure.
    """
    variables = variables or list(tab)
    t0 = tab[variables[0]]
    alphas = alphas if alphas is not None else sorted(t0['alpha'].dropna().unique())
    n_it = int(t0['iteration'].max())
    colors = plt.get_cmap('viridis_r')(np.linspace(0.1, 1.0, n_it))      # iteration 1 light -> last dark

    fig, axes = plt.subplots(len(variables), len(alphas), figsize=(5.2 * len(alphas), 3.4 * len(variables)),
                             sharex=True, sharey='row', squeeze=False)
    root_name = None
    for r, v in enumerate(variables):
        for c, a in enumerate(alphas):
            ax = axes[r, c]
            g = tab[v][tab[v]['alpha'] == a].sort_values('iteration')
            if g.empty:
                ax.set_visible(False)
                continue
            first = g.index[0]
            root_name = g.loc[first, 'parent']
            z0 = zonal_mean(maps[v][first]['parent'])
            ax.plot(z0.lat, z0 * 0 if change else z0, color='black', lw=2.2, zorder=4)
            for tag, row in g.iterrows():
                z = zonal_mean(maps[v][tag]['real'])
                ax.plot(z.lat, z - z0 if change else z, color=colors[int(row['iteration']) - 1], lw=1.5, zorder=5)
            if not change:
                ax.axhline(0, color='grey', lw=0.8, ls='--')
            ax.grid(True, ls=':', alpha=0.6)
            if r == 0:
                ax.set_title(tuning_label(a, show_alpha), fontsize=11, color=tuning_color(a, c))
            if c == 0:
                what = f'change from {root_name}' if change else 'bias'
                ax.set_ylabel(f'{var_long_name(v)}\n{what} [W m$^{{-2}}$]', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30)); ax.set_xlabel('Latitude [°]')

    handles = [Line2D([0], [0], color='black', lw=2.2, label=f'{root_name} (default parameters)')]
    handles += [Line2D([0], [0], color=colors[k], lw=2, label=f'Iteration {k + 1}') for k in range(n_it)]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 9), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _handle_show_or_save(output_path)


# ==============================================================================
# BIAS DECOMPOSITION AND SENSITIVITY MAPS
# ==============================================================================

def plot_bias_decomposition(diag_yaml_path: str, diag_nc_path: str, sens_nc_path: str, var_name: str, exp_tag: str = "", output_path: str = None) -> dict:
    """
    Decomposes the predicted global bias into individual parameter contributions.
    
    Args:
        diag_yaml_path: Path to the diagnostics_*.yaml file.
        diag_nc_path: Path to the diagnostics_2d_*.nc file.
        sens_nc_path: Path to the 2D sensitivity NetCDF file.
        var_name: The name of the variable (e.g., 'rsnt' or 'rlnt').
        exp_tag: Experiment tag for the title (e.g., 'C200').
        output_path: Optional path to save the generated plot.
        
    Returns:
        A dictionary mapping parameters to their predicted global flux contribution.
    """
    # 1. Load data
    with open(diag_yaml_path, 'r') as f:
        diag_data = yaml.safe_load(f)
        
    ds_diag = xr.open_dataset(diag_nc_path)
    ds_sens = xr.open_dataset(sens_nc_path)
    
    bias_init_name = f'{var_name}_bias_init'
    if bias_init_name not in ds_diag.data_vars:
        print(f"Variable {bias_init_name} not found in {diag_nc_path}")
        return {}

    bias_init_map = ds_diag[bias_init_name]
    
    # 2. Extract optimal changes (Delta p)
    optimal_changes = {p['name']: float(p['change_abs']) for p in diag_data['parameters']}
        
    # 3. Calculate spatial weights (cosine of latitude)
    weights = np.cos(np.deg2rad(bias_init_map.lat))
    global_bias_init = float(bias_init_map.weighted(weights).mean())
    
    contributions = {}
    
    # 4. Reconstruct individual global contributions
    for p, delta_p in optimal_changes.items():
        if p in ds_sens.parameter.values and var_name in ds_sens.variable.values:
            slope_map = ds_sens.sel(variable=var_name, parameter=p).slope
            delta_flux_map = slope_map * delta_p
            global_contrib = float(delta_flux_map.weighted(weights).mean(skipna=True))
            contributions[p] = global_contrib
            
    sorted_contribs = dict(sorted(contributions.items(), key=lambda item: abs(item[1])))
    
    params = list(sorted_contribs.keys())
    vals = list(sorted_contribs.values())
    
    # 5. Plotting
    plt.figure(figsize=(10, 8))
    bars = plt.barh(params, vals, color=['#D55E00' if v > 0 else '#0072B2' for v in vals])
    plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
    
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        offset = 0.05 if width > 0 else -0.05
        if abs(width) > 0.01:
            plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}', va='center', ha=ha, fontsize=9)
    
    predicted_total = global_bias_init + sum(vals)
    
    plt.title(f"[{exp_tag}] Bias Decomposition: {var_name.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Contribution to Global Flux [W/m²]", fontsize=12)
    
    info_text = (f"Initial Bias: {global_bias_init:+.2f} W/m²\n"
                 f"Sum of Contributions: {sum(vals):+.2f} W/m²\n"
                 f"Predicted Final Bias: {predicted_total:+.2f} W/m²")
    
    plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    _handle_show_or_save(output_path)
    
    return sorted_contribs


def plot_sensitivity_maps(sens_nc_path: str, var_name: str, ref_values: dict, inc: float = 0.3, vmin: float = -3.0, vmax: float = 3.0, output_path: str = None) -> None:
    """
    Creates a composite map of all spatial sensitivities for a specific variable.
    
    Args:
        sens_nc_path: Path to the 2D sensitivity NetCDF file.
        var_name: The target atmospheric variable (e.g., 'rsnt').
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        output_path: Optional path to save the generated plot.
    """
    ds_sens = xr.open_dataset(sens_nc_path)
    
    if var_name not in ds_sens.variable.values:
        print(f"Variable {var_name} not found in {sens_nc_path}")
        return
        
    ds_var = ds_sens.sel(variable=var_name)
    parameters = ds_var.parameter.values
    
    n = len(parameters)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    if nrows * ncols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
        
    for i, param in enumerate(parameters):
        ax = axs[i]
        field = ds_var.sel(parameter=param).slope
        
        # Handle extra dimensions if present
        extra_dims = [d for d in field.dims if d not in ['lat', 'lon']]
        if extra_dims:
            field = field.mean(extra_dims)
            
        p_ref = ref_values.get(param, 1.0)
        delta_p = abs(p_ref) * inc if p_ref != 0 else inc 
        data = field.values * delta_p
        lon2d, lat2d = np.meshgrid(field['lon'], field['lat'])
        
        im = ax.pcolormesh(
            lon2d, lat2d, data,
            vmin=vmin, vmax=vmax, cmap='RdBu_r',
            transform=ccrs.PlateCarree(),
            shading="auto"
        )
        
        ax.coastlines(linewidth=0.5)
        ax.set_title(param, fontsize=12)
        
    for ax in axs[len(parameters):]:
        ax.remove()
        
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Impact on {var_name.upper()} for {inc*100:.0f}% param change (W/m²)')
    
    _handle_show_or_save(output_path)


def plot_sensitivity_maps_with_snr(sens_nc_path: str, path_ctrl: str, var_name: str, ref_values: dict, inc: float = 0.3, snr_threshold: float = 1.0, vmin: float = -3.0, vmax: float = 3.0, output_path: str = None) -> None:
    """
    Creates a composite map of all spatial sensitivities for a specific variable, applying a filter based on the Signal-to-Noise Ratio (SNR).
    """
    # 1. compute the noise map from the control run
    print("Computing the noise map from the control run...")
    ds_ctrl = xr.open_mfdataset(path_ctrl, chunks={'time_counter': 12})
    
    if var_name == 'net_toa' and 'net_toa' not in ds_ctrl:
        ds_ctrl['net_toa'] = ds_ctrl['rsdt'] - ds_ctrl['rsut'] - ds_ctrl['rlut']
        
    noise_map_raw = ds_ctrl[var_name].groupby('time_counter.year').mean().std(dim='year').compute()
    noise_map_raw.name = var_name 
    
    sample_raw_file = glob.glob(path_ctrl)[0]
    noise_map = regrid_to_regular_smm_safe(
        ds_averaged=noise_map_raw,
        target_grid='r180x90',
        method='ycon',
        raw_file_path=sample_raw_file,
        varname=var_name
    )

    ds_sens = xr.open_dataset(sens_nc_path)
    if var_name not in ds_sens.variable.values:
        print(f"Variable {var_name} not found in {sens_nc_path}")
        return
        
    ds_var = ds_sens.sel(variable=var_name)
    parameters = ds_var.parameter.values
    
    n = len(parameters)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    if nrows * ncols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
        
    for i, param in enumerate(parameters):
        ax = axs[i]
        field = ds_var.sel(parameter=param).slope
        
        extra_dims = [d for d in field.dims if d not in ['lat', 'lon']]
        if extra_dims:
            field = field.mean(extra_dims)
            
        p_ref = ref_values.get(param, 1.0)
        delta_p = abs(p_ref) * inc if p_ref != 0 else inc 
        
        signal = np.abs(field * delta_p)
        snr = signal / noise_map
        
        data_filtered = xr.where(snr > snr_threshold, field * delta_p, 0.0)
        data = data_filtered.values
        
        lon2d, lat2d = np.meshgrid(field['lon'], field['lat'])
        
        im = ax.pcolormesh(
            lon2d, lat2d, data,
            vmin=vmin, vmax=vmax, cmap='RdBu_r',
            transform=ccrs.PlateCarree(),
            shading="auto"
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(param, fontsize=12)
        
    for ax in axs[len(parameters):]:
        ax.remove()
        
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Robust Impact on {var_name.upper()} (SNR > {snr_threshold}) (W/m²)')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved in: {output_path}")
    else:
        plt.show()
    plt.close()

# ==============================================================================
# FIGURES FROM MAPS
# ==============================================================================

def plot_bias_maps_zonal(bias_maps, vlim=None, output_path=None):
    """Rows: variables. Columns: bias map and zonal-mean profile."""
    n = len(bias_maps)
    fig = plt.figure(figsize=(11, 3.6 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.2, 1])
    for r, (var, b) in enumerate(bias_maps.items()):
        lim = vlim or float(abs(b).quantile(0.99))
        ax = fig.add_subplot(gs[r, 0], projection=ccrs.PlateCarree())
        im = b.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-lim, vmax=lim, add_colorbar=False)
        ax.coastlines()
        ax.set_title(f'{var.upper()} bias')
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.06, label='W/m²')
        axz = fig.add_subplot(gs[r, 1])
        z = zonal_mean(b)
        axz.plot(z, z.lat, color='#0072B2', lw=2)
        axz.axvline(0, color='black', lw=0.8, alpha=0.5)
        axz.set_ylim(-90, 90)
        axz.grid(True, alpha=0.3)
        axz.set_xlabel('Zonal-mean bias [W/m²]')
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_bias_map_comparison(tab: Dict[str, pd.DataFrame], maps: Dict[str, Dict[str, Dict[str, xr.DataArray]]],
                             runs: List[str], variables: List[str] = None, metric: str = 'l1', vlim: float = None,
                             show_parent: bool = False, show_diff: bool = False, show_alpha: bool = False,
                             output_path: str = None) -> None:
    """
    Realized bias maps of a few runs side by side (rows: variables, columns: runs).

    All maps of a row share the same colour scale, so the runs can be compared by eye. The
    title of each map gives the strategy, the iteration and the run tag, plus global-mean bias
    and spatial error (``metric``) of the map. Typical use: spatial vs global tuning at the
    same iteration, e.g. ``runs=['v400', 'v410']``.

    Args:
        tab, maps: Output of ``validation_table(..., return_maps=True)``.
        runs: Run tags to show, in column order.
        variables: Variables to plot (default: all keys of ``tab``).
        metric: Spatial error in the titles ('l1' -> MAE, 'rmse').
        vlim: Colour limit [W m-2] (default: 99th percentile of |bias| over the maps of the row).
        show_parent: Add a first column with the parent of the first run.
        show_diff: Add a last column with the difference between the last and the first run.
        show_alpha: Add alpha to the strategy names.
        output_path: Optional file path to save the figure.
    """
    variables = variables or list(tab)
    cname = 'MAE' if normalize_metric(metric) == 'l1' else 'RMSE'

    # Columns: (title, function var -> map)
    cols = []
    if show_parent:
        par = tab[variables[0]].loc[runs[0], 'parent']
        cols.append((f'Parent [{par}]', lambda v: maps[v][runs[0]]['parent']))
    for tag in runs:
        t0 = tab[variables[0]].loc[tag]
        it = t0.get('iteration', np.nan)
        it_txt = f', iteration {int(it)}' if pd.notna(it) else ''
        cols.append((f'{tuning_label(t0["alpha"], show_alpha)}{it_txt} [{tag}]',
                     lambda v, tag=tag: maps[v][tag]['real']))

    ncol = len(cols) + int(show_diff)
    fig, axes = plt.subplots(len(variables), ncol, figsize=(5.0 * ncol, 3.1 * len(variables) + 0.4),
                             subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    for r, v in enumerate(variables):
        fields = [f(v) for _, f in cols]
        lim = vlim or max(float(abs(b).quantile(0.99)) for b in fields)
        for c, ((title, _), b) in enumerate(zip(cols, fields)):
            ax = axes[r, c]
            im = b.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-lim, vmax=lim,
                        add_colorbar=False)
            ax.coastlines(linewidth=0.5)
            m = bias_metrics(b, metric)
            ax.set_title(f'{title}\nglobal mean {m["bias"]:+.2f}, {cname} {m["cost"]:.2f} W m$^{{-2}}$', fontsize=9)
        cb = fig.colorbar(im, ax=list(axes[r, :len(cols)]), orientation='vertical', fraction=0.02, pad=0.01,
                          extend='both')
        cb.set_label(f'{var_long_name(v)}\nbias [W m$^{{-2}}$]', fontsize=8)
        if show_diff:
            d = fields[-1] - fields[len(cols) - len(runs)]
            dlim = float(abs(d).quantile(0.99)) or 1.0
            ax = axes[r, -1]
            imd = d.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='PuOr_r', vmin=-dlim, vmax=dlim,
                         add_colorbar=False)
            ax.coastlines(linewidth=0.5)
            ax.set_title(f'{runs[-1]} minus {runs[0]}\nglobal mean {bias_metrics(d, metric)["bias"]:+.2f} W m$^{{-2}}$',
                         fontsize=9)
            cbd = fig.colorbar(imd, ax=ax, orientation='vertical', fraction=0.04, pad=0.02, extend='both')
            cbd.set_label('difference [W m$^{-2}$]', fontsize=8)
    _handle_show_or_save(output_path)