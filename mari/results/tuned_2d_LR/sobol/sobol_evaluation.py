"""
1: Validation of ECtuner emulator on Sobol runs
-------------------------------------------------------------------------
For each Sobol run i:
    predicted = Net_TOA_ref + sum_p( slope_p * (param_sobol_i_p - param_default_p) )
    real    = Global Net TOA measured from run i (from ECmean)

Then scatter plot predicted [x] vs real [y] with a 1:1 diagonal line.
"""

import numpy as np
import pickle
import xarray as xr
import yaml
import glob
import os
import matplotlib.pyplot as plt

# ===========================================================================
# USER CONFIGURATION (SWITCHES)
# ===========================================================================
# Choose mode: '1D' (reads YAML, supports regions/seasons) or '2D' (reads NetCDF global)
SENSITIVITY_MODE = '2D'  

# Target variable to analyze (available in both modes)
# Options: 'net_toa', 'rsnt', 'rlnt', 'swcf', 'lwcf', 'rsns', 'rlns', 'hfss', 'hfls', 'net_sfc', 'toamsfc'
TARGET_VAR = 'hfss'   

# 1D Specific options (Ignored if SENSITIVITY_MODE = '2D')
TARGET_SEASON = 'ALL'     # e.g., 'ALL', 'DJF', 'JJA'
TARGET_REGION = 'Global'  # e.g., 'Global', 'Equatorial', 'NH', 'North Midlat', 'North Pole', 'Tropical', 

# ===========================================================================
# PATHS
# ===========================================================================
SOBOL_PICKLE   = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/results/tuned_2d_LR/sobol_params/sobolset_16d.p'
SOBOL_ECMEAN   = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/ectuner/ecmean/sobol_no1990/'   
REF_ECMEAN     = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner/ecmean/gm_TL63_v2/GM_EC26-PDAY_p000_EC-Earth4_r1i1p1f1_1991_1992.yml' 
OUTPUT_DIR     = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/results/tuned_2d_LR/sobol_output/'

# Sensitivity file paths (Update with your actual filenames)
SENSITIVITY_NC  = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner/sensitivities/2D/sensitivity_TL63_415_1991-2000_2D.nc'
SENSITIVITY_YML = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner/sensitivities/sensitivity_TL63_415_1991-2000.yaml'

# ===========================================================================
# 1. Default parameters
# ===========================================================================
param_structure = {
    'namcumf': {
        'RPRCON':   0.14E-02,
        'ENTRORG':  0.175E-02,
        'DETRPEN':  0.75E-04,
        'ENTRDD':   0.3E-03,
        'RMFDEPS':  0.3
    },
    'namcldp': {
        'RVICE':              0.13,
        'RLCRITSNOW':         2.0E-05,
        'RSNOWLIN2':          0.03,
        'RCLDIFF':            0.6E-05,
        'RCLDIFF_CONVI':      10.0,
        'RDEPLIQREFRATE':     0.5,
        'RDEPLIQREFDEPTH':    500.0,
        'RCL_OVERLAPLIQICE':  0.65,
        'RCL_INHOMOGAUT':     1.5,
        'RCL_INHOMOGACC':     3.0
    },
    'naerad': {
        'RMINICE': 60.0
    }
}

# Flat list of (namelist, param, default_value) per costruire array di default e nomi
flat_param_meta = [
    (nl, p, v)
    for nl, params in param_structure.items()
    for p, v in params.items()
]
param_names   = [p for _, p, _ in flat_param_meta]
param_defaults = np.array([v for _, _, v in flat_param_meta])   # shape (16,)

# ===========================================================================
# 2. Sobol sequence
# ===========================================================================
print("Upload pickle Sobol...")
_, gino = pickle.load(open(SOBOL_PICKLE, 'rb'))   # gino shape (128, 16)
 
# Reconstruct the actual parameter values used in the runs
max_pert = 0.50
# gino in [0,1] → perturbation in [-50%, +50%]
pert_matrix = (gino * 2.0 - 1.0) * max_pert               # shape (128, 16)
param_matrix = param_defaults[np.newaxis, :] * (1.0 + pert_matrix)   # shape (128, 16)
delta_matrix = param_matrix - param_defaults[np.newaxis, :]           # shape (128, 16)
 
print(f"  Sobol sequence: {gino.shape[0]} runs, {gino.shape[1]} parameters")
 
# ===========================================================================
# 3. Load real Net TOA of the 128 Sobol runs from ECmean YAMLs
# ===========================================================================
SIGN_FLIP_VARS = {'hfls', 'hfss'}
def load_flux_from_ecmean_yaml(ecmean_dir, varname, n_runs=128, exp_prefix='EC26-PDAY', 
                               period='1991_1992', season='ALL', region='Global'):
    """
    Load specific diagnostic variable from ECmean files. Supports combined formula fields if missing.
    """
    values_real = np.full(n_runs, np.nan)
    for i in range(n_runs):
        fname = f"GM_{exp_prefix}_i{i:03d}_EC-Earth4_r1i1p1f1_{period}.yml"
        fpath = os.path.join(ecmean_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = yaml.safe_load(f)
        try:
            # Derived diagnostic formulas matching your 2D script logic
            if varname == 'net_toa':
                values_real[i] = float(data['rsnt'][season][region]) + float(data['rlnt'][season][region])
            elif varname == 'swcf':
                values_real[i] = float(data['rsnt'][season][region]) - float(data['rsntcs'][season][region])
            elif varname == 'lwcf':
                values_real[i] = float(data['rlnt'][season][region]) - float(data['rlntcs'][season][region])
            else:
                # net_sfc, toamsfc, tas, pr, hfls, hfss, rsnt, rlnt, etc. — all read directly
                val = float(data[varname][season][region])
                # flip sign if needed for convention consistency with 2D sensitivity
                if varname in SIGN_FLIP_VARS:
                    val = -val
                values_real[i] = val
        except (KeyError, TypeError):
            pass
    return values_real
 
# Enforce 'Global' region if processing 2D NetCDF
chosen_region = 'Global' if SENSITIVITY_MODE == '2D' else TARGET_REGION
chosen_season = 'ALL' if SENSITIVITY_MODE == '2D' else TARGET_SEASON

print(f"Loading real {TARGET_VAR} ({chosen_season}, {chosen_region}) from Sobol runs...")
flux_real = load_flux_from_ecmean_yaml(SOBOL_ECMEAN, TARGET_VAR, season=chosen_season, region=chosen_region)
print(f"  Found {np.sum(~np.isnan(flux_real))}/128 valid runs")
 
# ===========================================================================
# 4. Load Net TOA of the reference run p000
# ===========================================================================
print("Loading reference run flux (p000)...")
with open(REF_ECMEAN) as f:
    ref_data = yaml.safe_load(f)
if TARGET_VAR == 'net_toa':
    flux_ref = float(ref_data['rsnt'][chosen_season][chosen_region]) + float(ref_data['rlnt'][chosen_season][chosen_region])
elif TARGET_VAR == 'swcf':
    flux_ref = float(ref_data['rsnt'][chosen_season][chosen_region]) - float(ref_data['rsntcs'][chosen_season][chosen_region])
elif TARGET_VAR == 'lwcf':
    flux_ref = float(ref_data['rlnt'][chosen_season][chosen_region]) - float(ref_data['rlntcs'][chosen_season][chosen_region])
else:
    flux_ref = float(ref_data[TARGET_VAR][chosen_season][chosen_region])
    if TARGET_VAR in SIGN_FLIP_VARS:
        flux_ref = -flux_ref
print(f"  Reference value = {flux_ref:.4f} W/m²")
 
# ===========================================================================
# 5. Load Sensitivities (Supports both 1D YAML and 2D NetCDF modes)
# ===========================================================================
slope_global = {}

if SENSITIVITY_MODE == '2D':
    print(f"Computing slopes using 2D NetCDF maps for {TARGET_VAR} (Global)...")
    ds_sens = xr.open_dataset(SENSITIVITY_NC)
    weights = np.cos(np.deg2rad(ds_sens.lat))
    
    # Handle derived variables inside the NC maps if not directly available
    if TARGET_VAR not in ds_sens.variable.values:
        if TARGET_VAR == 'net_toa' and 'rsnt' in ds_sens.variable.values and 'rlnt' in ds_sens.variable.values:
            slope_field = ds_sens.sel(variable='rsnt').slope + ds_sens.sel(variable='rlnt').slope
        elif TARGET_VAR == 'swcf' and 'rsnt' in ds_sens.variable.values and 'rsntcs' in ds_sens.variable.values:
            slope_field = ds_sens.sel(variable='rsnt').slope - ds_sens.sel(variable='rsntcs').slope
        elif TARGET_VAR == 'lwcf' and 'rlnt' in ds_sens.variable.values and 'rlntcs' in ds_sens.variable.values:
            slope_field = ds_sens.sel(variable='rlnt').slope - ds_sens.sel(variable='rlntcs').slope
        else:
            raise ValueError(f"Variable '{TARGET_VAR}' not found or formula cannot be computed from 2D file.")
    else:
        slope_field = ds_sens.sel(variable=TARGET_VAR).slope

    for p in param_names:
        if p in ds_sens.parameter.values:
            sl = slope_field.sel(parameter=p)
            slope_global[p] = float(sl.weighted(weights).mean(['lat', 'lon']).values)
        else:
            slope_global[p] = 0.0

elif SENSITIVITY_MODE == '1D':
    print(f"Extracting slopes from 1D YAML for {TARGET_VAR} ({chosen_season}, {chosen_region})...")
    with open(SENSITIVITY_YML, 'r') as f:
        sens_1d = yaml.safe_load(f)
        
    for p in param_names:
        try:
            # sensitivity[parameter][variable][season][region][0 -> slope]
            slope_global[p] = float(sens_1d[p][TARGET_VAR][chosen_season][chosen_region][0])
        except KeyError:
            print(f"  Warning: Parameter '{p}' or targeting missing in 1D file -> set to 0.0")
            slope_global[p] = 0.0

slope_vec = np.array([slope_global[p] for p in param_names])   # shape (16,)
 
# ===========================================================================
# 6. Compute emulator-predicted Net TOA for the 128 runs
# ===========================================================================
# predicted_i = ref + sum_p( slope_p * delta_p_i )
flux_pred = flux_ref + delta_matrix @ slope_vec  # shape (128,)
 
print(f"\nPredicted stats: min={flux_pred.min():.3f}, max={flux_pred.max():.3f}, "
      f'mean={flux_pred.mean():.3f}')
print(f"Real stats:      min={np.nanmin(flux_real):.3f}, max={np.nanmax(flux_real):.3f}, "
      f'mean={np.nanmean(flux_real):.3f}')
 
# ===========================================================================
# 7. Scatter plot 1:1 
# ===========================================================================
# mask valid runs
valid = ~np.isnan(flux_real)
x = flux_pred[valid] - flux_ref
y = flux_real[valid] - flux_ref

dist_origin = np.sqrt(x**2 + y**2)                    
dist_origin_pct = dist_origin / dist_origin.max() * 100
 
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(x, y, c=dist_origin_pct, cmap='plasma_r', s=40, alpha=0.85, edgecolors='k', linewidths=0.3, zorder=5)
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Distance from reference [% of max]', fontsize=10)
 
all_vals = np.concatenate([x, y])
lim_max = np.nanmax(np.abs(all_vals)) * 1.16 
lim = [-lim_max, lim_max]
ax.plot(lim, lim, color='gray', lw=1.2, ls='--', label='1:1 ideal', zorder=3)
ax.set_xlim(lim)
ax.set_ylim(lim)
 
rmse = np.sqrt(np.mean((y - x)**2))
corr = np.corrcoef(x, y)[0, 1]
bias = np.mean(y - x)
ax.text(0.05, 0.95, f'N = {valid.sum()}\nRMSE = {rmse:.3f} W/m²\nr = {corr:.3f}\nBias = {bias:.3f} W/m²',
        transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))
 
ax.set_xlabel(fr'$\Delta${TARGET_VAR.upper()} Emulator (predicted) [W/m²]', fontsize=11)
ax.set_ylabel(fr'$\Delta${TARGET_VAR.upper()} Model (realized) [W/m²]', fontsize=11)
ax.set_title(f'Linear Emulator Validation ({SENSITIVITY_MODE} Mode)\n'
             fr'$\Delta${TARGET_VAR.upper()} — distance from reference', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_aspect('equal')
 
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, f'emulator_validation_{TARGET_VAR}_{SENSITIVITY_MODE}_{chosen_region}_distance.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"1:1 plot saved to: {out_path}")

# ===========================================================================
# 8. Additional diagnostic plot: residuals vs total perturbation magnitude
# =========================================================================== 
# residuals = y - x
# total_pert = np.sqrt(np.sum(pert_matrix[valid]**2, axis=1))   
 
# fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
 
# # PANEL 1: Residuals vs perturbation magnitude (Cleaned, neutral styling)
# axes[0].scatter(total_pert, residuals, color='steelblue', edgecolor='k', linewidths=0.5, s=40, alpha=0.8)
# axes[0].axhline(0, color='gray', lw=1, ls='--')
# axes[0].set_xlabel('Total perturbation L2 norm [dimensionless]', fontsize=11)
# axes[0].set_ylabel('Residual (Realized − Predicted) [W/m²]', fontsize=11)
# axes[0].set_title('Linear Emulator Error vs Perturbation Size', fontsize=11)
# axes[0].grid(True, linestyle=':', alpha=0.5)
 
# # PANEL 2: Mean Directional Contribution 
# mean_contribution = slope_vec * np.mean(np.abs(delta_matrix[valid]), axis=0)
# sorted_idx = np.argsort(np.abs(mean_contribution))[::-1]
# bar_colors = ['crimson' if val > 0 else 'royalblue' for val in mean_contribution[sorted_idx]]
 
# axes[1].barh(range(16), mean_contribution[sorted_idx], color=bar_colors, edgecolor='k', linewidth=0.5)
# axes[1].axvline(0, color='black', lw=0.8, ls='-')
# axes[1].set_yticks(range(16))
# axes[1].set_yticklabels([param_names[i] for i in sorted_idx], fontsize=9)
# axes[1].set_xlabel('slope × mean|Δparam| [W/m²]', fontsize=11)
# axes[1].set_title('Mean directional emulator contribution\nper parameter', fontsize=11)
# axes[1].grid(True, axis='x', linestyle=':', alpha=0.5)
 
# plt.tight_layout()
# out_path2 = os.path.join(OUTPUT_DIR, f'emulator_diagnostics_{TARGET_VAR}_{SENSITIVITY_MODE}_{chosen_region}.png')
# plt.savefig(out_path2, dpi=150, bbox_inches='tight')
# plt.close()
# print(f"Diagnostic plot saved to: {out_path2}")

# ===========================================================================
# 9. Outlier Investigation (Where linearity breaks down completely)
# ===========================================================================
outlier_threshold = 1.0
outliers = np.where(np.abs(residuals) > outlier_threshold)[0]

if len(outliers) > 0:
    print(f"\n--- ANALYSIS OF THE {len(outliers)} RUNS WITH MAXIMUM LINEAR ERROR (|Residual| > {outlier_threshold} W/m²) ---")
    for idx in outliers:
        print(f"Sobol Run i={idx:03d} -> Realized-Predicted Residual: {residuals[idx]:+.3f} W/m²")
        print("  Status of the top 3 dominant parameters in this run:")
        for p_idx in sorted_idx[:3]: 
            p_name = param_names[p_idx]
            p_pert = pert_matrix[idx, p_idx] * 100 
            print(f"    - {p_name:<18}: {p_pert:+.1f}%")
else:
    print(f"\nNo runs exceeded the linear error threshold of {outlier_threshold} W/m² for {TARGET_VAR}.")
 