#!/usr/bin/env python3
"""
Integrated Tuning Comparison and Validation Suite for EC-Earth4.
Supports CLI execution for automated plotting and interactive Notebook imports.
"""
#Argomenti da terminale:
#Per vedere solo la tabella: python compare_tuning.py -c config_tuner_2d.yaml
#Per vedere tabella + grafico: python compare_tuning.py -c config_tuner_2d.yaml --plot

#python compare_tuning.py -c ../../../config_tuner_2d.yaml -d yaml_files --plot -v (variables)
#python compare_tuning.py -c config.yaml -i a000 a050 a100 --plot -v (variables)
#python compare_tuning.py -c config.yaml -i tuned_phis_a030.yml tuned_phis_a095.yml --plot -v (variables)

import os
import yaml
import re
import argparse
import sys 
import glob
import pandas as pd
import numpy as np
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

path_to_utils = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner/utils/'
if path_to_utils not in sys.path:
    sys.path.append(path_to_utils)

try:
    from sensitivity_2D import regrid_to_regular_smm_safe
except ImportError:
    regrid_to_regular_smm_safe = None

# ==============================================================================
# 1. CORE PHYSICS LOGIC & CORE DATA LOADING
# ==============================================================================
def compute_model_flux(ds, var_name):
    """to compute descending fluxes or cloud forcing from raw OIFS fields."""
    if var_name in ds:
        return ds[var_name]
    
    formulas = {
        'net_toa': lambda d: d['rsdt'] - d['rsut'] - d['rlut'],
        'rsnt':    lambda d: d['rsdt'] - d['rsut'],
        'rlnt':    lambda d: -1.0 * d['rlut'],
        'swcf':    lambda d: d['rsutcs'] - d['rsut'] if 'rsutcs' in d else (d['rsdt'] - d['rsut']) - (d['rsdtcs'] - d['rsutcs']),
        'lwcf':    lambda d: d['rlutcs'] - d['rlut']
    }
    
    if var_name in formulas:
        return formulas[var_name](ds)
    else:
        raise ValueError(f"Variabile '{var_name}' non trovata e nessuna formula fisica definita.")

def parse_args():
    parser = argparse.ArgumentParser(description='Comparison between different tuning strategies.')
    parser.add_argument('-d', '--dir', type=str, default='net_TOA/new_yaml_files', help='Results directory.')
    parser.add_argument('-c', '--config', type=str, help='Path to config file for OIFS reference.')
    parser.add_argument('--plot', action='store_true', help='Activate the plots.')
    parser.add_argument('-i', '--include', nargs='+', help='List of filenames to include OR keywords (e.g. a000 a100).')
    parser.add_argument('-v', '--vars', nargs='+', default=['net_toa'], help='Variables for tradeoff plot.')
    return parser.parse_args()

def extract_alpha(filename):
    """
    extract alpha name: a0 -> 0.0, a05 -> 0.5, a09 -> 0.9, a1 -> 1.0
    """
    match = re.search(r'_a(\d+)', filename)
    if not match: return None
    val_str = match.group(1)
    if val_str == "0": return 0.0
    if val_str == "1": return 1.0
    # Gestione formati tipo a05 (0.5) o a095 (0.95)
    if val_str.startswith('0'):
        return float(f"0.{val_str[1:]}")
    return float(val_str) / 10.0 if len(val_str) == 1 else float(val_str) / 100.0

def load_results(results_dir, config_path=None, include_list=None):
    all_data = []
    param_names = []
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
            ref = conf.get('reference_parameters', {})
            ref_row = {p: float(v) for p, v in ref.items()}
            ref_row['Experiment'] = 'REFERENCE_OIFS'
            ref_row['Alpha'] = -0.1 
            all_data.append(ref_row)
            param_names = list(ref.keys())

    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found.")
        return None, []

    files = [f for f in os.listdir(results_dir) if f.endswith((".yml", ".yaml"))]

    if include_list:
        files = [f for f in files if any(key in f for key in include_list)]

    for filename in files:
        path = os.path.join(results_dir, filename)
        with open(path, 'r') as f:
            raw_text = f.read()
            f.seek(0)
            try:
                content = yaml.safe_load(f)
                params = {
                    'Experiment': filename.replace('.yml', '').replace('.yaml', ''),
                    'Alpha': extract_alpha(filename)
                }
                tuning_block = content[0]['base.context']['model_config']['oifs']['tuning']
                for group in tuning_block.values():
                    params.update({p: float(v) for p, v in group.items()})
                
                m_spat = re.search(r'# total_spatial_cost: ([\d\.-]+)', raw_text)
                m_glob = re.search(r'# total_global_cost: ([\d\.-]+)', raw_text)
                m_metr = re.search(r'# metric_used: (\w+)', raw_text)
                
                metric = m_metr.group(1).lower() if m_metr else 'l2'
                params['Metric'] = metric

                spat_tot = float(m_spat.group(1)) if m_spat else 0
                glob_tot = float(m_glob.group(1)) if m_glob else 0
                params['Raw_Spatial_Cost'] = spat_tot
                params['Raw_Global_Cost'] = glob_tot
                # weighted sum to have a single score for ranking (optional, can be used in future analyses)
                params['Total_Objective_Score'] = spat_tot + glob_tot
                
                if metric == 'l2':
                    params['Phys_Spatial_Total'] = np.sqrt(spat_tot)
                    params['Phys_Global_Total'] = np.sqrt(glob_tot)
                else:
                    params['Phys_Spatial_Total'] = spat_tot
                    params['Phys_Global_Total'] = glob_tot

                for var in ['net_toa', 'rsnt', 'rlnt', 'swcf', 'lwcf']:
                    bias_m = re.search(fr'# {var}_global_bias_final: ([\d\.-]+)', raw_text)
                    cost_m = re.search(fr'# {var}_spatial_cost_(?:\w+): ([\d\.-]+)', raw_text)
                    if bias_m and cost_m:
                        b_val = float(bias_m.group(1))
                        c_val = float(cost_m.group(1))
                        params[f'{var}_AbsBias'] = abs(b_val)
                        params[f'{var}_PhysSpatial'] = np.sqrt(c_val) if metric == 'l2' else c_val
                
                all_data.append(params)
            except Exception: continue
    
    df = pd.DataFrame(all_data).set_index('Experiment')
    if 'Metric_Type' not in df.columns and not df.empty:
        df['Metric_Type'] = df['Metric'].map({'l1': 'L1 (Linear)', 'l2': 'L2 (Quadratic)'})
    return df, param_names

# ==============================================================================
# 2. COMPUTATIONAL VALIDATION METRICS
# ==============================================================================
def calculate_realized_mae_flux(scratch_dir, exp_tag, year1, year2, ref_dir, var_name):
    """Compute spatially weighted MAE between model and reference for a specific variable."""
    if regrid_to_regular_smm_safe is None:
        raise ImportError("Utility smmregrid not available. Check your environment and path settings.")

    pattern = os.path.join(scratch_dir, f"{exp_tag}_atm_cmip6_1m_*.nc")
    files = sorted(glob.glob(pattern))
    valid_files = [f for f in files if year1 <= int(os.path.basename(f).split('_')[-1].split('-')[0]) <= year2]
        
    if not valid_files:
        raise FileNotFoundError(f"No files found in scratch for {exp_tag} in years {year1}-{year2}")
            
    ds = xr.open_mfdataset(valid_files, combine='nested', concat_dim='time_counter', chunks={'time_counter': 12})
    if 'time_counter' in ds.dims: ds = ds.rename({'time_counter': 'time'})

    model_flux = compute_model_flux(ds, var_name).mean('time').compute()
    model_reg = regrid_to_regular_smm_safe(model_flux, 'r180x90', method='ycon', raw_file_path=valid_files[0], varname=var_name)

    ref_files = sorted(glob.glob(os.path.join(ref_dir, f"climate_average_{var_name}_*.nc")))
    if not ref_files:
        raise FileNotFoundError(f"No reference files found for variable: {var_name}")
    
    ref_map = xr.open_dataset(ref_files[0])[var_name]
    if var_name == 'rlnt' and float(ref_map.mean()) > 0:
        ref_map = -1.0 * ref_map

    model_final = model_reg.sortby('lat')
    ref_final = ref_map.sortby('lat').reindex_like(model_final, method='nearest')

    abs_bias = np.abs(model_final - ref_final)
    weights = np.cos(np.deg2rad(ref_final.lat))
    return float(abs_bias.weighted(weights).mean(skipna=True).values)

# ==============================================================================
# 3. PLOTTING 
# ==============================================================================

def _handle_show_or_save(output_path=None):
    """Handle the logic for showing or saving a plot based on the presence of an output path."""
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_parameter_evolution(df, param_names, metric_filter=None, output_path=None):
    """Plot percentage parameter evolution to respect to OIFS."""
    if metric_filter and 'Metric_Type' in df.columns:
        print(f"--- Filtering parameter evolution for: {metric_filter} ---")
        df_filtered = df[(df['Metric_Type'] == metric_filter) | (df.index == 'REFERENCE_OIFS')]
    else:
        df_filtered = df

    plt.figure(figsize=(11, 6))
    exp_df = df_filtered[df_filtered.index != 'REFERENCE_OIFS'].dropna(subset=['Alpha']).sort_values('Alpha')
    ref_rows = df_filtered[df_filtered.index == 'REFERENCE_OIFS']
    if ref_rows.empty:
        raise ValueError("Error: REFERENCE_OIFS row missing from DataFrame. Cannot compute relative changes.")
    ref_values = ref_rows.iloc[0]

    colormap = plt.get_cmap('tab20')
    for i, p in enumerate(param_names):
        if p in exp_df.columns:
            denom = ref_values[p] if ref_values[p] != 0 else 1e-15
            rel_change = ((exp_df[p] - ref_values[p]) / denom) * 100
            plt.plot(exp_df['Alpha'], rel_change, label=p, marker='o', lw=1.5, color=colormap(i/len(param_names)))

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Parameter Evolution (%) relative to OIFS", fontsize=12, fontweight='bold')
    plt.xlabel("Alpha (Weight on Global Bias)")
    plt.ylabel("Variation (%)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize='small', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_tradeoff_comparison(df, output_path=None):
    """global pareto front plot: spatial error vs global bias, with alpha as color and different symbols for L1/L2"""
    plt.figure(figsize=(9, 6.5))
    marker_map = {'L1 (Linear)': 'o', 'L2 (Quadratic)': 's'}
    last_sc = None

    metrics_present = df['Metric'].unique() if 'Metric' in df.columns else []
    if 'l2' in metrics_present and 'l1' in metrics_present:
        title_suffix = "(RMSE/MAE & Abs Bias)"
    elif 'l2' in metrics_present:
        title_suffix = "(RMSE & Abs Bias)"
    else:
        title_suffix = "(MAE & Abs Bias)"

    for m_type in df['Metric_Type'].unique():
        subset = df[df['Metric_Type'] == m_type].dropna(subset=['Phys_Spatial_Total', 'Phys_Global_Total']).sort_values('Alpha')
        if subset.empty: continue
        mkr = marker_map.get(m_type, 'p')
        last_sc = plt.scatter(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], 
                              c=subset['Alpha'], cmap='viridis', marker=mkr, s=140, edgecolors='black', zorder=3, vmin=0, vmax=1)
        plt.plot(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], linestyle='--', alpha=0.4, zorder=2)
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', (row['Phys_Spatial_Total'], row['Phys_Global_Total']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='L1 (Linear)', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='L2 (Quadratic)', markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Metrica Obiettivo")
    plt.title(f"Pareto Front: Total Spatial Error vs Total Global Bias\nUnits: {title_suffix}", fontsize=12, fontweight='bold')
    # ... resto del codice del plot ...
    plt.xlabel("Total Spatial Error (Physical Units)", fontsize=11)
    plt.ylabel("Total Global Bias (Physical Units)", fontsize=11)
    if last_sc:
        plt.colorbar(last_sc).set_label(r'Alpha Value ($\alpha$)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_tradeoff_comparison_r2(df, output_path=None):
    """Grafico delle performance del fronte di Pareto specfico per il test del filtro R2."""
    plt.figure(figsize=(10, 7))
    marker_map = {'0.3': 'o', '0.0': 'X'}
    label_map = {'0.3': r'Con Filtro Robustezza ($R^2 \geq 0.3$)', '0.0': r'Senza Filtro ($R^2 = 0.0$)'}
    last_sc = None

    for r2_val in df['R2'].unique():
        subset = df[df['R2'] == r2_val].dropna(subset=['Phys_Spatial_Total', 'Phys_Global_Total']).sort_values('Alpha')
        if subset.empty: continue
        mkr = marker_map.get(r2_val, 'p')
        last_sc = plt.scatter(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], 
                              c=subset['Alpha'], cmap='viridis', marker=mkr, s=150, edgecolors='black', zorder=3, vmin=0, vmax=1)
        plt.plot(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], linestyle='--', alpha=0.4, zorder=2)
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', (row['Phys_Spatial_Total'], row['Phys_Global_Total']),
                         xytext=(6, 6), textcoords='offset points', fontsize=8, alpha=0.7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label_map['0.3'], markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='X', color='w', label=label_map['0.0'], markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Configurazione Filtro")
    plt.title("Pareto Front Comparison: Effect of $R^2$ Noise Filter on Tuning", fontsize=12, fontweight='bold')
    plt.xlabel("Total Spatial Error (Physical Units)", fontsize=11)
    plt.ylabel("Total Global Bias (Physical Units)", fontsize=11)
    if last_sc:
        plt.colorbar(last_sc).set_label(r'Alpha Value ($\alpha$)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_variable_pareto(df, var, output_path=None):
    """Pareto Front for a specific variable"""
    plt.figure(figsize=(9, 6.5))
    marker_map = {'L1 (Linear)': 'o', 'L2 (Quadratic)': 's'}
    col_spat, col_bias = f'{var}_PhysSpatial', f'{var}_AbsBias'
    last_sc = None

    for m_type in df['Metric_Type'].unique():
        subset = df[df['Metric_Type'] == m_type].dropna(subset=[col_spat, col_bias]).sort_values('Alpha')
        if subset.empty: continue
        mkr = marker_map.get(m_type, 'p')
        last_sc = plt.scatter(subset[col_spat], subset[col_bias], 
                              c=subset['Alpha'], cmap='viridis', marker=mkr, s=140, edgecolors='black', zorder=3, vmin=0, vmax=1)
        plt.plot(subset[col_spat], subset[col_bias], linestyle='--', alpha=0.4, zorder=2)
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', (row[col_spat], row[col_bias]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='L1 (Linear)', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='L2 (Quadratic)', markerfacecolor='gray', markersize=10, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Metrica Obiettivo")
    plt.title(f"Pareto Front: Spatial Error vs Global Bias ({var.upper()})", fontsize=12, fontweight='bold')
    plt.xlabel("Spatial Error [W/m²]", fontsize=11)
    plt.ylabel("Absolute Global Bias [W/m²]", fontsize=11)
    if last_sc:
        plt.colorbar(last_sc).set_label(r'Alpha Value ($\alpha$)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_emulator_performance_global(var_name, ref_obs, data_ecmean, data_predicted_bias, output_path=None):
    """Test 1:1 della performance dell'Emulatore sul Bias Globale."""
    alphas = list(data_ecmean.keys())
    realized_biases = [data_ecmean[a][var_name] - ref_obs[var_name] for a in alphas]
    predicted_biases = [data_predicted_bias[a][var_name] for a in alphas]

    plt.figure(figsize=(6.5, 6.5))
    colors = ['#e41a1c', '#4daf4a', '#377eb8', '#984ea3', '#ff7f00', '#ffff33'][:len(alphas)]
    
    for i, a in enumerate(alphas):
        lab = a.replace('_', ' = ').title().replace('Alpha', r'$\alpha$')
        plt.scatter(predicted_biases[i], realized_biases[i], color=colors[i], s=130, label=lab, edgecolors='black', zorder=5)

    lims = [min(predicted_biases + realized_biases) - 0.5, max(predicted_biases + realized_biases) + 0.5]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='1:1 Ideal Line')
    plt.xlabel(f'Predicted {var_name.upper()} Global Bias [W/m²]', fontsize=11)
    plt.ylabel(f'Realized {var_name.upper()} Global Bias (Model - CERES) [W/m²]', fontsize=11)
    plt.title(f'Global Bias Validation: Emulator vs Model\nVariable: {var_name.upper()}', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_emulator_vs_model_spatial(var_name, preds_dict, reals_dict, output_path=None):
    """Test 1:1 della performance dell'Emulatore sul Costo Spaziale (MAE)."""
    alphas = list(preds_dict.keys())
    x_vals = [preds_dict[a] for a in alphas]
    y_vals = [reals_dict[a] for a in alphas]

    plt.figure(figsize=(6.5, 6.5))
    colors = ['#e41a1c', '#4daf4a', '#377eb8', '#984ea3', '#ff7f00', '#ffff33'][:len(alphas)]

    for i, a in enumerate(alphas):
        lab = a.replace('_', ' = ').title().replace('Alpha', r'$\alpha$')
        plt.scatter(x_vals[i], y_vals[i], color=colors[i], s=130, label=lab, edgecolors='black', zorder=5)

    all_vals = x_vals + y_vals
    margin = (max(all_vals) - min(all_vals)) * 0.2 if len(all_vals) > 1 else 1.0
    lims = [min(all_vals) - margin, max(all_vals) + margin]
    plt.plot(lims, lims, color='gray', linestyle='--', alpha=0.6, label='1:1 Perfect Line')

    plt.xlabel('Spatial MAE Predicted (Emulator) [W/m²]', fontsize=11)
    plt.ylabel('Spatial MAE Realized (Model) [W/m²]', fontsize=11)
    plt.title(f'Spatial MAE Validation: Emulator vs Model\nVariable: {var_name.upper()}', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    _handle_show_or_save(output_path)

def plot_zonal_validation_from_scratch(tuner_diag_file, scratch_dir, exp_tag, year1, year2, ceres_file, var_tuner, output_path=None):
    """Profilo Zonale Cross-Validation (Initial vs Predicted vs Realized)."""
    ds_tuner = xr.open_dataset(tuner_diag_file)
    tuner_init = ds_tuner[f'{var_tuner}_bias_init'].mean(dim='lon').sortby('lat')
    tuner_pred = ds_tuner[f'{var_tuner}_bias_final'].mean(dim='lon').sortby('lat')
    lat_tuner = ds_tuner.lat.sortby('lat')

    files = sorted(glob.glob(os.path.join(scratch_dir, f"{exp_tag}_atm_cmip6_1m_*.nc")))
    valid_files = [f for f in files if year1 <= int(os.path.basename(f).split('_')[-1].split('-')[0]) <= year2]
    
    ds_model = xr.open_mfdataset(valid_files, combine='nested', concat_dim='time_counter', chunks={'time_counter': 12})
    if 'time_counter' in ds_model.dims: ds_model = ds_model.rename({'time_counter': 'time'})

    model_flux = compute_model_flux(ds_model, var_tuner).mean('time').compute()
    model_reg = regrid_to_regular_smm_safe(model_flux, 'r180x90', method='ycon', raw_file_path=valid_files[0], varname=var_tuner)

    ceres = xr.open_dataset(ceres_file).interp(lat=model_reg.lat)
    ceres_flux = ceres['toa_net_all_mon'] if var_tuner == 'net_toa' else (ceres['solar_mon'] - ceres['toa_sw_all_mon'] if var_tuner == 'rsnt' else -1.0 * ceres['toa_lw_all_mon'])

    model_zonal_realized = (model_reg - ceres_flux).mean(dim='lon').sortby('lat')

    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.axvline(0, color='black', lw=1, alpha=0.4)
    ax.plot(tuner_init, lat_tuner, label='Initial Bias (Base/Phis)', color='orange', linestyle='--', lw=1.8)
    ax.plot(tuner_pred, lat_tuner, label='Predicted Final (Emulator)', color='blue', linestyle=':', lw=2.2)
    ax.plot(model_zonal_realized, model_reg.lat.sortby('lat'), label=f'Realized Final ({exp_tag})', color='green', lw=2.5)
    
    ax.set_title(f'{var_tuner.upper()} Zonal Bias Validation\nRun: {exp_tag}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bias wrt CERES [W/m²]', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    _handle_show_or_save(output_path)

# ==============================================================================
# 4. SOBOL ENSEMBLE CONFIGURATION CHECK & VISUALIZATION
# ==============================================================================

from pathlib import Path
PARAM_STRUCTURE = { 
    'namcumf': {
        'RPRCON': 0.0014, 'ENTRORG': 0.00175, 'DETRPEN': 7.5E-05, 'ENTRDD': 0.0003, 'RMFDEPS': 0.3
    },
    'namcldp': {
        'RVICE': 0.13, 'RLCRITSNOW': 2.0E-05, 'RSNOWLIN2': 0.03, 'RCLDIFF': 0.6E-05,
        'RCLDIFF_CONVI': 10.0, 'RDEPLIQREFRATE': 0.5, 'RDEPLIQREFDEPTH': 500.0,
        'RCL_OVERLAPLIQICE': 0.65, 'RCL_INHOMOGAUT': 1.5, 'RCL_INHOMOGACC': 3.0
    },
    'naerad': {
        'RMINICE': 60.0
    }
}

DEFAULT_VALUES = {p: float(v) for net in PARAM_STRUCTURE.values() for p, v in net.items()}

def flatten_tuning_params(data):
    """Estrae i parametri eliminando i prefissi per avere le etichette pulite."""
    params = {}
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if 'base.context' in item:
            model_config = item['base.context'].get('model_config', {})
            for comp in ['oifs', 'nemo', 'naerad']:
                if comp in model_config and 'tuning' in model_config[comp]:
                    for section, section_params in model_config[comp]['tuning'].items():
                        for param, value in section_params.items():
                            params[param] = float(value)
    return params


def read_tuning_files(folder_path=None, exps=None, scratch_template='/ec/res4/scratch/ecme3038/ece4/{}/templates/tuning-sobol.yml'):
    """Legge i file e restituisce un DataFrame Pandas: Righe=Run, Colonne=Parametri."""
    tuning_data = {}
    
    if folder_path is not None:
        files = sorted(Path(folder_path).glob("*.yml")) + sorted(Path(folder_path).glob("*.yaml"))
        files = list(set(files))  
        for filepath in files:
            name = filepath.stem.replace("tuning_", "").replace("sobol_", "")
            with open(filepath, 'r') as f:
                content = yaml.safe_load(f)
            flat = flatten_tuning_params(content)
            if flat: tuning_data[name] = flat
    elif exps is not None:
        for exp in exps:
            filepath = Path(scratch_template.format(exp))
            if filepath.exists():
                with open(filepath, 'r') as f:
                    content = yaml.safe_load(f)
                flat = flatten_tuning_params(content)
                if flat: tuning_data[exp] = flat
    
    df = pd.DataFrame.from_dict(tuning_data, orient='index')
    return df.sort_index()


def plot_scatter(tuning_data, defaults=DEFAULT_VALUES, colors=None, output_path=None):
    """
    Scatter plot con i parametri sull'asse X e il cambio relativo sull'asse Y.
    Se len(exp) > 15, attiva automaticamente la Colorbar progressiva per Sobol.
    """
    if tuning_data.empty:
        print("No data to plot")
        return
    
    if defaults is None:
        print("No defaults provided")
        return
    
    all_params = sorted(tuning_data.columns.tolist())
    if not all_params:
        print("No parameters to plot")
        return
    
    tuning_data_rel = tuning_data.copy()
    for param in all_params:
        if param in defaults and defaults[param] != 0:
            tuning_data_rel[param] = tuning_data[param] / defaults[param]
        else:
            tuning_data_rel[param] = np.nan
    
    fig, ax = plt.subplots(figsize=(max(14, len(all_params) * 0.8), 8))
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.6, linewidth=1.2, zorder=1)
    
    exp_names = tuning_data_rel.index.tolist()
    is_large_sobol = len(exp_names) > 15
    
    if is_large_sobol:
        run_indices = []
        for name in exp_names:
            match = re.search(r'(\d+)', str(name))
            run_indices.append(int(match.group(1)) if match else 0)
            
        for param_idx, param_name in enumerate(all_params):
            y_vals = tuning_data_rel[param_name].values
            x_coords = np.full(len(y_vals), param_idx)
            sc = ax.scatter(x_coords, y_vals, c=run_indices, cmap='plasma', 
                            s=60, alpha=0.6, edgecolors='none', zorder=3)
            
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('Sobol Run Sequence Index (0 - 127)', fontsize=11)
    else:
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
            
        for exp_idx, exp_name in enumerate(exp_names):
            x_positions = []
            y_values = []
            for param_idx, param_name in enumerate(all_params):
                value = tuning_data_rel.loc[exp_name, param_name]
                if pd.notna(value) and np.isfinite(value):
                    x_positions.append(param_idx)
                    y_values.append(value)
            
            ax.scatter(x_positions, y_values, s=100, alpha=0.7, 
                      color=colors[exp_idx % len(colors)], label=exp_name, zorder=3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.set_xticks(range(len(all_params)))
    ax.set_xticklabels(all_params, rotation=45, ha='right')
    ax.set_ylabel('Relative value (with respect to default)', fontsize=12)
    ax.set_xlabel('Parameters', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _handle_show_or_save(output_path)


def plot_heatmap(df, default_values=DEFAULT_VALUES, output_path=None):
    """Genera la heatmap delle variazioni relative coerente con lo scatter plot."""
    if df.empty:
        return
        
    valid_cols = [c for c in df.columns if c in default_values]
    df_filtered = df[valid_cols]
    defaults_series = pd.Series({c: default_values[c] for c in valid_cols})
    df_normalized = (df_filtered - defaults_series) / defaults_series

    fig_height = max(5, len(df) * 0.25)
    fig, ax = plt.subplots(figsize=(max(14, len(df.columns) * 0.8), fig_height))
    
    im = ax.imshow(df_normalized.values, aspect='auto', cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    
    ax.set_xticks(range(len(df_filtered.columns)))
    ax.set_xticklabels(df_filtered.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=9 if len(df) > 20 else 10)
    
    plt.colorbar(im, ax=ax, label='Relative change (wrt Default)', pad=0.01)
    
    if len(df) <= 64:
        for i in range(len(df)):
            for j in range(len(df_filtered.columns)):
                value = df_filtered.iloc[i, j]
                color_text = "white" if abs(df_normalized.iloc[i, j]) > 0.25 else "black"
                ax.text(j, i, f'{value:.2e}', ha="center", va="center", color=color_text, fontsize=7)
    
    ax.set_ylabel('Experiment / Run ID', fontsize=12)
    plt.tight_layout()
    _handle_show_or_save(output_path)

# ==============================================================================
# 5. TERMINAL MAIN FLOW EXECUTION (CLI)
# ==============================================================================
def main():
    matplotlib.use('Agg')
    args = parse_args()
    df, param_names = load_results(args.dir, args.config, args.include)

    if df is None or df.empty: 
        print("No data. Check your directory and config file.")
        return

    df = df.sort_values(by='Alpha')

    if args.plot:
        # Salvataggio automatico file PNG
        plot_parameter_evolution(df, param_names, output_path=os.path.join(args.dir, 'parameter_evolution_sweep.png'))
        plot_tradeoff_comparison(df, output_path=os.path.join(args.dir, 'performance_tradeoff_sweep.png'))
        
        if 'R2' in df.columns and len(df['R2'].unique()) > 1:
            plot_tradeoff_comparison_r2(df, output_path=os.path.join(args.dir, 'tradeoff_comparison_r2.png'))
            
        for v in args.vars:
            plot_variable_pareto(df, v, output_path=os.path.join(args.dir, f'physical_tradeoff_{v}.png'))
        
        print(f"Plots saved in: {args.dir}")

if __name__ == "__main__":
    main()