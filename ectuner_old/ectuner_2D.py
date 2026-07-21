import xarray as xr
import numpy as np
import sys
from logger import setup_logger
path_to_utils = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner/utils/'
if path_to_utils not in sys.path:
    sys.path.append(path_to_utils)
import os
import glob
import yaml
from scipy import optimize
from tabulate import tabulate
from ectuner import load_config, load_params, setup_logger, parse_arguments, get_arg
from sensitivity_2D import get_var_2d, regrid_to_regular_smm_safe


# 2D DATA LOADING & BIAS CALCULATION
def load_sensitivity_2d(sens_file):
    """Sensitivity file load"""
    return xr.open_dataset(sens_file)

def load_reference_2d(config, variables):
    """Reference maps load (ECmean4 climatology) con ricerca pattern"""
    ref_maps = {}
    ref_dir = config['files']['ref_2d_dir']
    grid = config.get('spatial_tuning', {}).get('target_grid', 'r180x90')
    
    for var in variables:
        pattern = os.path.join(ref_dir, f"climate_average_{var}_*{grid}*.nc")
        matching_files = sorted(glob.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(
                f"Missing observation map for {var} in {ref_dir}. "
                f"Tried pattern: {os.path.basename(pattern)}"
            )
        
        f_obs = matching_files[0]
        print(f"   > Loading reference for {var} from: {os.path.basename(f_obs)}")
        
        ds = xr.open_dataset(f_obs)
        data = ds[var].mean('time') if 'time' in ds.dims else ds[var]

        # rlnt correction
        # obs is upward flux (positive = TOA outgoing), model is downward (positive = TOA incoming)
        if var == 'rlnt':
            print(f"   > Flipping sign for {var} (Obs is UP, Model is DOWN)")
            data = -1.0 * data
        # -----------------------------

        ref_maps[var] = data
        
    return ref_maps

def load_base_2d(config, exp, year1, year2, variables):
    """exp 2d maps load (with caching)"""
    # use extract_2d_base 
    return extract_2d_base(config, exp, year1, year2, variables)

METHODS_MAP = {
    'tas': 'ycon', 'psl': 'ycon', 'pr': 'ycon', 'net_toa': 'ycon', 
    'rsnt': 'ycon', 'rlnt': 'ycon', 'swcf': 'ycon', 'lwcf': 'ycon', 
    'net_sfc': 'ycon', 'clt': 'ycon'
}

def extract_2d_base(config, exp, year1, year2, variables):
    """
    2d mean extraction from raw OIFS files, with caching mechanism. If the cache file exists, it loads it directly; otherwise, it performs the extraction and saves the cache for future use.
    """
    out_dir = config['files']['base_2d_dir']
    os.makedirs(out_dir, exist_ok=True)
    cache_file = os.path.join(out_dir, f"base_2d_{exp}_{year1}_{year2}.nc")

    if os.path.exists(cache_file):
        print(f"--- Loading cached base maps for {exp} ---")
        return xr.open_dataset(cache_file)

    print(f"--- Cache not found. Extracting 2D maps for {exp} (this may take a while...) ---")
    
    raw_pattern = os.path.join(config['files']['raw_dir'], exp, f"output/oifs/{exp}_atm_cmip6_1m_*.nc")
    filz = sorted(glob.glob(raw_pattern))
    if not filz:
        raise FileNotFoundError(f"No raw OIFS files found in {raw_pattern}")

    ds = xr.open_mfdataset(filz, chunks={'time_counter': 12}, combine='nested', 
                           concat_dim='time_counter', compat='override', 
                           coords='minimal', data_vars='minimal')
    
    if 'time_counter' in ds.dims: ds = ds.rename({'time_counter': 'time'})

    extracted_vars = []
    sample_file = filz[0] # grid reference 

    for var in variables:
        
        print(f"   > Processing {var}...")
        y_raw = get_var_2d(ds, var).sel(time=slice(str(year1), str(year2))).mean('time').compute()
        
        # Corretto: il secondo argomento deve essere la griglia target
        target_grid = config.get('spatial_tuning', {}).get('target_grid', 'r180x90')

        y_reg = regrid_to_regular_smm_safe(y_raw.to_dataset(name=var), target_grid, 
                                        method=METHODS_MAP.get(var, 'ycon'),
                                        raw_file_path=sample_file)

        extracted_vars.append(y_reg)

    ds_base = xr.merge(extracted_vars)
    ds_base.to_netcdf(cache_file)
    print(f"--- Cache saved: {cache_file} ---")
    
    return ds_base


# 2D OBJECTIVE FUNCTION
# def objective_function_2d(changes, params, target_vars, ds_sens, bias_maps, 
#                           weights_flux, reference_pars, values, penalty, r2_threshold=0.2, mask_2d=None):
#     """
#     Optimizator 
#     """
#     total_spatial_error = 0
    
#     weights_combined = mask_2d if mask_2d is not None else np.cos(np.deg2rad(ds_sens.lat))

#     for var in target_vars:
#         w_v = weights_flux.get(var, 0)
#         if w_v <= 0: continue
        
#         bias_init = bias_maps[var]
#         delta_bias_pred = xr.zeros_like(bias_init)
#         for i, p in enumerate(params):
#             slope = ds_sens.sel(variable=var, parameter=p).slope
#             r2 = ds_sens.sel(variable=var, parameter=p).r2
            
#             # R2 filter: sensitivity only where linear relationship is strong enough
#             clean_slope = slope.where(r2 > r2_threshold, 0)
#             delta_bias_pred += clean_slope * changes[i]
            
#         # Final residual error (map)
#         residual_bias = bias_init + delta_bias_pred
        
#         # spatial mse weighted for area
#         spatial_mse = (residual_bias**2).weighted(weights_combined).mean().values
#         total_spatial_error += w_v * spatial_mse

#     # penalty as for 1d
#     param_diff = sum([((reference_pars[p] - (values[p] + changes[i])) / reference_pars[p]) ** 2 
#                          for i, p in enumerate(params)])

#     return total_spatial_error + param_diff * penalty


# def objective_function_2d(changes, opt_params, target_vars, bias_flat, sens_matrices, 
#                                weights_vector, weights_flux, ref_params, current_values, penalty):
#     """
#     numpy version
#     """
#     total_error = 0
#     for var in target_vars:
#         # changes smultiplicati per la matrice di sensibilità (solo pixel validi) danno il delta predetto in ogni pixel
#         delta_pred = np.dot(sens_matrices[var], changes)
#         residual = bias_flat[var] + delta_pred
        
#         # MSE spatially weighted 
#         weighted_mse = np.average(residual**2, weights=weights_vector)
#         total_error += weights_flux[var] * weighted_mse

#     # Penalty term  (distance from reference parameters)
#     param_penalty = 0
#     for i, p in enumerate(opt_params):
#         param_penalty += ((ref_params[p] - (current_values[p] + changes[i])) / ref_params[p]) ** 2
    
#     return total_error + param_penalty * penalty

def objective_function_2d_hybrid(changes, opt_params, target_vars, bias_flat, sens_matrices_spatial, sens_matrices_global, 
                                weights_vector_var, weights_flux, ref_params, current_values, 
                                penalty, alpha=1, metric ='l2'):
    """
    alpha = 0: Ottimizzazione puramente spaziale (Pattern matching).
    alpha = 1: Ottimizzazione puramente globale (come il tuner 1D).
    metric = 'l1': Usa i valori assoluti (MAE / Absolute Bias).
    metric = 'l2': Usa i quadrati (MSE / Squared Bias).
    """

    total_error = 0
    
    for var in target_vars:
        if weights_flux.get(var, 0) <= 0:
            continue
        w_v = weights_vector_var[var]
        # --- TERMINE SPAZIALE (Pattern) ---
        delta_pred_spatial = np.dot(sens_matrices_spatial[var], changes)
        residual_spatial = bias_flat[var] + delta_pred_spatial
        
        # --- TERMINE GLOBALE (Accordo 1D) ---
        sens_global_mean = np.average(sens_matrices_global[var], axis=0, weights=w_v)
        bias_global_init = np.average(bias_flat[var], weights=w_v)
        bias_global_final = bias_global_init + np.dot(sens_global_mean, changes)

        if metric.lower() == 'l2':
            # Metrica quadrata
            cost_spatial = np.average(residual_spatial**2, weights=w_v)
            cost_global = bias_global_final**2
        else: 
            # Metrica lineare
            cost_spatial = np.average(np.abs(residual_spatial), weights=w_v)
            cost_global = np.abs(bias_global_final)
        
        # Mix ibrido
        var_error = (1 - alpha) * cost_spatial + alpha * cost_global
        total_error += weights_flux[var] * var_error

    # Penalty term (distanza dai parametri di riferimento)
    param_penalty = sum([((ref_params[p] - (current_values[p] + changes[i])) / ref_params[p])**2 
                         for i, p in enumerate(opt_params)])

    return total_error + param_penalty * penalty


def get_region_mask(ds_sens, weights_region):
    """
    Create a 2D mask based on the regional weights defined in the config.
    The final weight of each pixel is: Area_Weight (cosine of latitude) * Region_Weight
    """
    lat = ds_sens.lat
    lon = ds_sens.lon
    
    mask = np.cos(np.deg2rad(lat))
    
    # region bounds in latitude
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

    # regional weight map (1D in lat, to be expanded to 2D)
    regional_weight_map = xr.DataArray(np.zeros_like(lat), coords={'lat': lat}, dims='lat')

    for region, weight in weights_region.items():
        if weight <= 0: continue
        if region in region_bounds:
            low, high = region_bounds[region]
            regional_weight_map = regional_weight_map.where((lat < low) | (lat > high), 
                                                            regional_weight_map + weight)
    
    if regional_weight_map.max() == 0:
        regional_weight_map += 1.0

    mask_2d = (mask * regional_weight_map).expand_dims(lon=len(lon)).assign_coords(lon=lon)
    return mask_2d

def save_diagnostic_maps(output_path, target_vars, bias_maps, ds_sens, params, optimal_changes, r2_threshold):
    """Genera un NetCDF con i bias prima e dopo l'ottimizzazione"""
    diagnostics = []
    for var in target_vars:
        bias_init = bias_maps[var]
        delta_pred = xr.zeros_like(bias_init)
        for i, p in enumerate(params):
            # .drop_vars(['variable', 'parameter']) to avoid conflicts in merge 
            slope = ds_sens.sel(variable=var, parameter=p).slope.drop_vars(['variable', 'parameter'], errors='ignore')
            r2 = ds_sens.sel(variable=var, parameter=p).r2.drop_vars(['variable', 'parameter'], errors='ignore')
            delta_pred += slope.where(r2 > r2_threshold, 0) * optimal_changes[i]
        
        bias_final = bias_init + delta_pred
        var_ds = xr.Dataset({
            f'{var}_bias_init': bias_init.drop_vars(['variable'], errors='ignore'),
            f'{var}_bias_final': bias_final.drop_vars(['variable'], errors='ignore'),
            f'{var}_improvement': (abs(bias_init) - abs(bias_final)).drop_vars(['variable'], errors='ignore')
        })
        diagnostics.append(var_ds)

    ds_diag = xr.merge(diagnostics, compat='override')
    ds_diag.to_netcdf(output_path)

# def print_global_summary_from_2d(logger, target_vars, bias_maps, ds_sens, params, optimal_changes, mask_2d, r2_threshold, alpha, metric='l2'):
    
#     logger.info("\n" + f" GLOBAL & SPATIAL COSTS ({metric.upper()}) ".center(90, "="))
#     header = f"{'Variable':<12} | {'Metric':<16} | {'Initial':>18} -> {'Final':>18} | {'Status'}"
#     logger.info(header)
#     logger.info("-" * 90)
    
#     summary_results = {}
#     for var in target_vars:
#         b_init_map = bias_maps[var]
        
#         delta_pred_raw = xr.zeros_like(b_init_map)    # Per il Global Mean (alpha=1)
#         delta_pred_filt = xr.zeros_like(b_init_map)   # Per lo Spatial RMSE (alpha=0)
        
#         for i, p in enumerate(params):
#             slope = ds_sens.sel(variable=var, parameter=p).slope
#             r2 = ds_sens.sel(variable=var, parameter=p).r2
#             # 1. Versione senza filtro R2
#             delta_pred_raw += slope * optimal_changes[i]
#             # 2. Versione con filtro R2
#             delta_pred_filt += slope.where(r2 > r2_threshold, 0) * optimal_changes[i]

#         # --- 1. GLOBAL COST ---
#         # Calcolo bias medio (lineare)
#         glob_bias_init = b_init_map.weighted(mask_2d).mean().values
#         glob_bias_final = (b_init_map + delta_pred_raw).weighted(mask_2d).mean().values
        
#         # Calcolo del costo globale in base alla metrica scelta
#         if metric.lower() == 'l2':
#             g_cost_init, g_cost_final = glob_bias_init**2, glob_bias_final**2
#         else:
#             g_cost_init, g_cost_final = np.abs(glob_bias_init), np.abs(glob_bias_final)

#         status_glob = "IMPROVED" if g_cost_final < g_cost_init else "WORSENED"
#         logger.info(f"{var:<12} | Global Cost| {g_cost_init:>18.4f} -> {g_cost_final:>18.4f} | {status_glob}")

#         # --- 2. SPATIAL COST ---
#         b_final_map = b_init_map + delta_pred_filt
#         if metric.lower() == 'l2':
#             s_cost_init = (b_init_map**2).weighted(mask_2d).mean().values
#             s_cost_final = (b_final_map**2).weighted(mask_2d).mean().values
#         else:
#             s_cost_init = np.abs(b_init_map).weighted(mask_2d).mean().values
#             s_cost_final = np.abs(b_final_map).weighted(mask_2d).mean().values
        
#         status_spat = "IMPROVED" if s_cost_final < s_cost_init else "WORSENED"
#         logger.info(f"{'':<12} | Spatial Cost  | {s_cost_init:>18.4f} -> {s_cost_final:>18.4f} | {status_spat}")

#         logger.info("-" * 90)
#         summary_results[var] = {'bias': glob_bias_final,'global_cost': g_cost_final, 'spatial_cost': s_cost_final}
    
#     return summary_results

def print_global_summary_from_2d(logger, all_vars, bias_maps, ds_sens, params, optimal_changes, mask_2d, weights_flux, weights_region, r2_threshold, alpha, metric='l2'):
    
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

    summary_results = {}
    targets_data = []
    diagnostics_data = []

    for var in all_vars:
        b_init_map = bias_maps[var]
        delta_pred_raw = xr.zeros_like(b_init_map)
        delta_pred_filt = xr.zeros_like(b_init_map)
        
        # Calcolo dei delta preteddi su tutta la mappa
        for i, p in enumerate(params):
            slope = ds_sens.sel(variable=var, parameter=p).slope
            r2 = ds_sens.sel(variable=var, parameter=p).r2
            delta_pred_raw += slope * optimal_changes[i]
            delta_pred_filt += slope.where(r2 > r2_threshold, 0) * optimal_changes[i]
            
        b_final_map_raw = b_init_map + delta_pred_raw
        b_final_map_filt = b_init_map + delta_pred_filt

        # --- 1. Calcoli per l'output YAML (Mantiene la tua metrica spaziale complessiva) ---
        glob_bias_init = b_init_map.weighted(mask_2d).mean().values
        glob_bias_final = b_final_map_raw.weighted(mask_2d).mean().values
        
        if metric.lower() == 'l2':
            g_cost_final = glob_bias_final**2
            s_cost_final = (b_final_map_filt**2).weighted(mask_2d).mean().values
        else:
            g_cost_final = np.abs(glob_bias_final)
            s_cost_final = np.abs(b_final_map_filt).weighted(mask_2d).mean().values
            
        summary_results[var] = {'bias': glob_bias_final, 'global_cost': g_cost_final, 'spatial_cost': s_cost_final}

        # --- 2. Estrazione Regionale per il LOG (stile 1D) ---
        var_w = weights_flux.get(var, 0)
        
        for region, bounds in region_bounds.items():
            reg_w = weights_region.get(region, 0)
            combined_w = var_w * reg_w
            
            low, high = bounds
            lat = b_init_map.lat
            
            # Creiamo una maschera per la regione specifica pesata per l'area
            cos_lat = np.cos(np.deg2rad(lat))
            reg_mask = cos_lat.where((lat >= low) & (lat <= high), 0.0)
            
            # Calcolo del bias medio regionale (lineare, come in 1D)
            init_val = b_init_map.weighted(reg_mask).mean().values
            final_val = b_final_map_raw.weighted(reg_mask).mean().values
            
            row = [var, region, combined_w, init_val, final_val]
            
            if combined_w > 0:
                targets_data.append(row)
            else:
                diagnostics_data.append(row)

    # --- 3. Funzione di formattazione della tabella ---
    def print_table(data_rows):
        header = f"{'Variable':<12} | {'Region':<14} | {'Weight':<6} | {'Bias Init':>10} -> {'Bias Final':>10} | {'Status'}"
        logger.info(header)
        logger.info("-" * len(header))
        current_var = None
        for r in data_rows:
            var_name, region, weight, b_init, b_final = r

            if current_var is not None and current_var != var_name:
                print_cost_summary(current_var)
                logger.info("-" * len(header)) # Separatore tra variabili
            current_var = var_name
            
            # Logica: è "Migliorato" se il valore assoluto si avvicina a 0
            is_improved = abs(b_final) < abs(b_init)
            status = "IMPROVED" if is_improved else "WORSENED"
            color = "\033[92m" if is_improved else "\033[91m"
            reset = "\033[0m"
            
            w_str = f"{weight:.1f}"
            init_str = f"{float(b_init):10.3f}"
            final_str = f"{float(b_final):10.3f}"
            
            logger.info(f"{var_name:<12} | {region:<14} | {w_str:<6} | {init_str} -> {color}{final_str}{reset} | {status}")

        if current_var is not None:
            print_cost_summary(current_var)
    
    def print_cost_summary(var_name):
        """Funzione helper per stampare i Costi L1/L2 calcolati per una specifica variabile"""
        data = summary_results[var_name]
        g_init = abs(data['bias']) if metric.lower() == 'l1' else data['bias']**2 # Approssimazione per l'Init log (usa il final se vuoi, o ricalcola l'init vero)
        
        # Recuperiamo il vero costo iniziale (ricalcolato per il log)
        b_init_map = bias_maps[var_name]
        g_bias_init_val = b_init_map.weighted(mask_2d).mean().values
        if metric.lower() == 'l2':
            g_cost_init = g_bias_init_val**2
            s_cost_init = (b_init_map**2).weighted(mask_2d).mean().values
        else:
            g_cost_init = np.abs(g_bias_init_val)
            s_cost_init = np.abs(b_init_map).weighted(mask_2d).mean().values

        g_cost_final = data['global_cost']
        s_cost_final = data['spatial_cost']
        
        # Colori e Status per Global Cost
        g_imp = g_cost_final < g_cost_init
        g_stat = "IMPROVED" if g_imp else "WORSENED"
        g_col = "\033[92m" if g_imp else "\033[91m"
        
        # Colori e Status per Spatial Cost
        s_imp = s_cost_final < s_cost_init
        s_stat = "IMPROVED" if s_imp else "WORSENED"
        s_col = "\033[92m" if s_imp else "\033[91m"
        reset = "\033[0m"

        logger.info(f"{'':<12} | {'[Global Cost]':<16} | {'-':<6} | {g_cost_init:12.4f} -> {g_col}{g_cost_final:12.4f}{reset} | {g_stat}")
        logger.info(f"{'':<12} | {'[Spatial Cost]':<16} | {'-':<6} | {s_cost_init:12.4f} -> {s_col}{s_cost_final:12.4f}{reset} | {s_stat}")

    logger.info("\n" + f" OPTIMIZATION SUMMARY 2D (Biases & Costs: {metric.upper()}) ".center(90, "="))
    logger.info("Goal: Bring Biases to 0.0")
    
    if targets_data:
        logger.info("\n" + " PRIMARY TUNING TARGETS ".center(90, "-"))
        print_table(targets_data)
        
    if diagnostics_data:
        logger.info("\n" + " DIAGNOSTIC SIDE-EFFECTS ".center(90, "-"))
        print_table(diagnostics_data)
        
    logger.info("=" * 90 + "\n")
    
    return summary_results


#plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_tuning_results(diag_file, output_dir, run_tag):
    """
    Genera mappe di confronto Bias Iniziale vs Bias Finale
    """
    ds = xr.open_dataset(diag_file)
    variables = [v.replace('_bias_init', '') for v in ds.data_vars if '_bias_init' in v]
    
    for var in variables:
        fig = plt.figure(figsize=(24, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.6])

        ax0 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax1 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
        ax_zonal = fig.add_subplot(gs[3])

        axes_maps = [ax0, ax1, ax2]
        
        b_init = ds[f'{var}_bias_init']
        b_final = ds[f'{var}_bias_final']
        improv = ds[f'{var}_improvement']
        
        vmax = max(float(abs(b_init).max()), float(abs(b_final).max()))
        vmin = -vmax
        
        # Plot 1: initial bias
        im0 = b_init.plot(ax=ax0, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax0.set_title(f'{var} - Initial Bias')
        
        # Plot 2: final bias
        im1 = b_final.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax1.set_title(f'{var} - Final Bias (Predicted)')
        
        # Plot 3: Improvement (Green = Improved, Red = worsened)
        # If improvement > 0, the absolute bias has decreased
        im2 = improv.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='PiYG', add_colorbar=False)
        ax2.set_title(f'{var} - Absolute Improvement')
        
        for ax in axes_maps:
            ax.coastlines()
            
        fig.colorbar(im1, ax=[ax0, ax1], location='bottom', label='Bias Units', fraction=0.05, pad=0.08)
        fig.colorbar(im2, ax=[ax2], location='bottom', label='Improvement', fraction=0.05, pad=0.08)

        # Pesiamo i dati originali e poi facciamo la media su ENTRAMBE le dimensioni
        zonal_init = b_init.mean(dim='lon')
        zonal_final = b_final.mean(dim='lon')

        ax_zonal.plot(zonal_init, ds.lat, label='Init (Red)', color='red', linestyle='--', lw=1.5)
        ax_zonal.plot(zonal_final, ds.lat, label='Final (Blue)', color='blue', lw=2)
        
        ax_zonal.axvline(0, color='black', lw=0.8, alpha=0.5)
        ax_zonal.set_title(f'{var} Zonal Bias')
        ax_zonal.set_xlabel('Bias Units')
        ax_zonal.set_ylabel('Latitude')
        ax_zonal.legend(fontsize='small', loc='upper right')
        ax_zonal.grid(True, alpha=0.3)
        
        # Salvataggio dinamico
        plot_name = f"comparison_2d_{var}_{run_tag}.png"
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight', dpi=150)
        plt.close()

# MAIN EXECUTION
def main():
    
    args = parse_arguments(sys.argv[1:])
    config = load_config(get_arg(args, 'config', 'config-tuner-2d.yaml'))
    logger = setup_logger(level=get_arg(args, 'loglevel', 'INFO'))
    
    exp = get_arg(args, 'exp', None)
    y1, y2 = get_arg(args, 'year1', config['args']['year1']), get_arg(args, 'year2', config['args']['year2'])
    penalty = get_arg(args, 'penalty', config['args']['penalty'])
    inc = get_arg(args, 'inc', config['args']['inc'])
    out = get_arg(args, 'output', None)
    if not out:
        config_files = config.get('files', {})
        out_dir = config_files.get('output_dir')
        out_temp = config_files.get('output_template', 'tuned_{exp}_{year1}_{year2}_2D.yml')
        if out_dir:
            filename = out_temp.format(exp=exp, year1=y1, year2=y2)
            out = os.path.join(out_dir, filename)
            os.makedirs(out_dir, exist_ok=True)
            logger.info(f"No output specified, using automatic path: {out}")
        else:
            logger.warning("Output path not specified and 'output_dir' missing in config. YAML will NOT be saved.")

    weights_flux = {v: float(w) for v, w in config['weights'].items()}
    weights_region = {r: float(w) for r, w in config.get('weights_region', {}).items()}
    all_vars = list(weights_flux.keys())
   
    param_file = os.path.join(config['files']['exps'], config['files']['params'].format(exp=exp))
    params_names, vals = load_params(param_file)
    current_values = {p: vals[i] for i, p in enumerate(params_names)}
    ref_params = config['reference_parameters']
    frozen_params_list = config.get('frozen_parameters') or []
    
    # divide parameters into free and frozen, and log the frozen ones
    opt_params = [p for p in params_names if p not in frozen_params_list]
    frozen_params = {p: current_values[p] for p in frozen_params_list if p in current_values}

    if frozen_params:
        logger.info(f"Frozen parameters (keeping manual tuning): {', '.join(frozen_params.keys())}")
    
    # bounds only for free parameters, calculated as percentage change from reference
    bounds = []
    for p in opt_params:
        v_ref = ref_params[p]
        v_curr = current_values[p]
        min_change = v_ref * (1 - inc) - v_curr
        max_change = v_ref * (1 + inc) - v_curr
        bounds.append((min_change, max_change))

    # 3. Data loading and spatial mask generation
    logger.info("Loading 2D Data (Sensitivity, Reference, Base)...")
    ds_sens = load_sensitivity_2d(config['files']['sensitivity_nc'])
    ref_maps = load_reference_2d(config, all_vars)
    base_maps = load_base_2d(config, exp, y1, y2, all_vars)

    logger.info("Generating spatial weights (Area * Region)...")
    mask_2d = get_region_mask(ds_sens, weights_region)
    
    # 4. Initial bias calculation
    metric_choice = config['spatial_tuning'].get('metric', 'l2')
    bias_maps = {}
    for var in all_vars:
        mod_resampled = base_maps[var].reindex_like(ref_maps[var], method='nearest')
        bias_maps[var] = mod_resampled - ref_maps[var]
        # initial bias calculation based on the chosen metric
        if metric_choice == 'l2':
            init_cost = (bias_maps[var]**2).weighted(mask_2d).mean().values
            label = "Squared Cost (L2)"
        else:
            init_cost = np.abs(bias_maps[var]).weighted(mask_2d).mean().values
            label = "Absolute Cost (L1)"
        logger.info(f"Variable {var}: Initial Spatial {label} (Weighted) = {init_cost:.4f}")
   
    logger.info("Flattening data for fast optimization...")
    
    # 5. mask for valid pixels (where at least one variable has valid sensitivity and bias)
    
    alpha = args.alpha if args.alpha is not None else config.get('spatial_tuning', {}).get('alpha', 0)
    r2_thr = config['spatial_tuning']['r2_threshold']

    active_regions = [r for r, w in weights_region.items() if w > 0]
    region_tag = "-".join(active_regions) if active_regions else "NoRegion"
    run_tag = f"{region_tag}_a{alpha}".replace('.', '')

    #allineamento per ordinare i dati come la maschera (lat, lon ordinati) e poi flatten perchè diventi un vettore
    # mask_ordered = mask_2d.transpose('lat', 'lon').sortby(['lat', 'lon'])
    # mask_flat_raw = mask_ordered.values.flatten()
    # valid_pix = ~np.isnan(mask_flat_raw)
    # weights_vector = mask_flat_raw[valid_pix]
    
    bias_flat = {}
    sens_matrices_spatial = {}
    sens_matrices_global = {} 
    weights_vector_var = {}

    for var in all_vars:
        # ALLINEA E ORDINA il bias e la sensibilità come la maschera
        b_ordered = bias_maps[var].transpose('lat', 'lon').sortby(['lat', 'lon'])
        mask_ordered = mask_2d.transpose('lat', 'lon').sortby(['lat', 'lon'])
        actual_valid_mask = (~np.isnan(mask_ordered)) & (~np.isnan(b_ordered))
        valid_pix = actual_valid_mask.values.flatten()
        weights_vector_var[var] = mask_ordered.values.flatten()[valid_pix]
        
        bias_flat[var] = b_ordered.values.flatten()[valid_pix]
        
        slopes_raw = []
        slopes_filtered = []
        for p in opt_params:
            s_ds = ds_sens.sel(variable=var, parameter=p).sortby(['lat', 'lon'])
            s_val = s_ds.slope.values.flatten()[valid_pix]
            r2_val = s_ds.r2.values.flatten()[valid_pix]
            
            slopes_raw.append(s_val.copy())
            
            s_filt = s_val.copy()
            s_filt[r2_val < r2_thr] = 0.0
            slopes_filtered.append(s_filt)
        
        sens_matrices_global[var] = np.column_stack(slopes_raw)
        sens_matrices_spatial[var] = np.column_stack(slopes_filtered)

        check_bias = np.average(bias_flat[var], weights=weights_vector_var[var])
        logger.info(f"DEBUG: Global Bias for {var} in optimizer: {check_bias:.4f}")
    
    # === CHECK: CONFRONTO DOMINIO E REGRIDDING ===
    # logger.info("=== DEBUG: DOMAIN & REGRIDDING CHECK ===")
    
    # # Inserisci i valori di osservazione (CERES) che usi nel tuo script di plot per il calcolo del bias realizzato
    # obs_global_means = {
    #     'rsnt': 241.5,   
    #     'rlnt': -240.54,  
    #     'net_toa': 1.02   
    # }
    
    # # Inserisci i valori di ECmean per la simulazione "phis" (la tua baseline)
    # ecmean_global_means = {
    #     'rsnt': 240.31,
    #     'rlnt': -233.63,
    #     'net_toa': 6.68
    # }

    # for var in ['rsnt', 'rlnt', 'net_toa']:
    #     if var in target_vars:
    #         # 1. Calcolo del bias aggregato 2D del tuner (su griglia r180x90 con mask)
    #         tuner_bias = np.average(bias_flat[var], weights=weights_vector_var[var])
            
    #         # 2. Calcolo del bias reale (Nativo - CERES)
    #         realized_bias = ecmean_global_means[var] - obs_global_means[var]
            
    #         diff = abs(tuner_bias - realized_bias)
    #         logger.info(f"[{var.upper()}] Tuner 2D Bias: {tuner_bias:.4f} | Native 1D Bias: {realized_bias:.4f} | Delta: {diff:.4f}")
    # logger.info("=========================================")

    # 5. Optimization (FAST)
    logger.info(f"Starting 2D Optimization ({get_arg(args, 'method', 'dual_annealing')})...")
    
    m_kwargs = {
        "method": "L-BFGS-B", 
        "options": {
            "ftol": 1e-12,    # function tol 
            "gtol": 1e-12,    # grad tol
            "maxls": 50       # more research tentatives in line search to handle potential non-smoothness from R2 filtering
        }
    }
    result = optimize.dual_annealing(
        objective_function_2d_hybrid, 
        bounds=bounds,
        minimizer_kwargs=m_kwargs,
        maxiter=1000,
        args=(opt_params, all_vars, bias_flat, sens_matrices_spatial, sens_matrices_global,
              weights_vector_var, weights_flux, ref_params, current_values, penalty, alpha, metric_choice), 
    )
    
    # 5. Output 
    free_changes = result.x

    total_spat_cost = 0
    total_glob_cost = 0
    metric_name = config.get('spatial_tuning', {}).get('metric', 'l2').lower()

    for var in all_vars:
        w_v = weights_vector_var[var]
        # Delta spaziale predetto (filtrato R2)
        d_spat = np.dot(sens_matrices_spatial[var], free_changes)
        res_spat = bias_flat[var] + d_spat
        
        # Delta globale predetto (senza filtro R2)
        sens_glob_v = np.average(sens_matrices_global[var], axis=0, weights=w_v)
        res_glob = np.average(bias_flat[var], weights=w_v) + np.dot(sens_glob_v, free_changes)

        # Calcolo del costo finale con stessa logica della funzione obiettivo
        if metric_name == 'l2':
            v_spat = np.average(res_spat**2, weights=w_v)
            v_glob = res_glob**2
        else:
            v_spat = np.average(np.abs(res_spat), weights=w_v)
            v_glob = np.abs(res_glob)
        
        total_spat_cost += weights_flux[var] * v_spat
        total_glob_cost += weights_flux[var] * v_glob

    logger.info(f"Optimization finished. Final Weighted Cost ({metric_name.upper()}): "
                f"Spatial={total_spat_cost:.4f}, Global={total_glob_cost:.4f}")
    
    opt_changes_dict = {p: 0.0 for p in params_names} # Default a zero
    
    for i, p in enumerate(opt_params):
        opt_changes_dict[p] = free_changes[i]

    all_optimal_changes = [opt_changes_dict[p] for p in params_names]
    results_meta =print_global_summary_from_2d(logger, all_vars, bias_maps, ds_sens, 
                               params_names, all_optimal_changes, mask_2d, weights_flux, weights_region, config['spatial_tuning']['r2_threshold'], alpha, metric=metric_choice)

    if out:
        from ruamel.yaml import YAML
        yaml_ru = YAML(typ="rt")
        yaml_ru.indent(mapping=2, sequence=2, offset=0)
        
        tuning_block = {}
        for pg, p_list in config['parameter_group'].items():
            # opt_changes_dict to include all parameters, but only those in p_list will be added to the tuning block
            current_group = {p: float(f"{current_values[p] + opt_changes_dict[p]:.4e}") 
                            for p in p_list if p in current_values}
            if current_group:
                tuning_block[pg] = current_group

        full_structure = [{"base.context": {"model_config": {"oifs": {"tuning": tuning_block}}}}]
        
        with open(out, "w") as f:
            yaml_ru.dump(full_structure, f)
            f.write("\n# --- ECtuner 2D meta-parameters ---\n")
            f.write(f"# metric_used: {metric_name}\n") # <--- Aggiunto
            f.write(f"# total_spatial_cost: {total_spat_cost:.8f}\n") # <--- Aggiunto
            f.write(f"# total_global_cost: {total_glob_cost:.8f}\n") # <--- Aggiunto
            f.write(f"# alpha: {alpha} | penalty: {penalty} | inc: {inc} | r2: {r2_thr} \n")
            f.write("# weights (flux):\n")
            for k, v in weights_flux.items():
                f.write(f"#   {k}: {v}\n")
            f.write("# weights (region):\n")
            for k, v in weights_region.items(): 
                f.write(f"#   {k}: {v}\n")
            for var, data in results_meta.items():
                f.write(f"# {var}_global_bias_final: {data['bias']:.6f}\n")
                f.write(f"# {var}_global_cost_{metric_name}: {data['global_cost']:.6f}\n")
                f.write(f"# {var}_spatial_cost_{metric_name}: {data['spatial_cost']:.6f}\n")
        
        logger.info(f"Structured tuning YAML written to {out}")

    # 7. Table Output
    outtable = []
    for i, p in enumerate(params_names):
        change = all_optimal_changes[i]
        new_val = current_values[p] + change
        rel_change = change / current_values[p] if current_values[p] != 0 else 0
        dist_ref = (new_val - ref_params[p]) / ref_params[p]
        outtable.append([p, new_val, current_values[p], change, rel_change, dist_ref])

    head = ['Parameter', 'New Value', 'Old Value', 'Change', 'Rel. Change', 'Dist. from Ref.']
    print("\n" + tabulate(outtable, headers=head, floatfmt=".4e", tablefmt='orgtbl'))

    # 8. Diagnostic maps output
    diag_file = os.path.join(config['files']['output_dir'], f"diagnostics_2d_{exp}_{run_tag}.nc")
    save_diagnostic_maps(diag_file, all_vars, bias_maps, ds_sens, params_names, 
                         all_optimal_changes, r2_thr)
    
    logger.info(f"Diagnostic maps saved to {diag_file}")
    plot_tuning_results(diag_file, config['files']['output_dir'], run_tag)

if __name__ == "__main__":
    main()