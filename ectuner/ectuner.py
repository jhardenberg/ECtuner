"""
EC-Earth Tuning Executor (CLI & API).

This script serves as the main entry point for both 1D and 2D tuning.
It provides functions that can be imported directly into Jupyter Notebooks,
as well as a robust Command Line Interface.
"""
import sys
import os
import argparse
import copy
import logging

from .libs.config import Config
from .libs.logger import setup_logger
from .libs.result import TuningResult
from .libs import exporter

# 1D modules
from .libs.loader import DataLoader1D
from .libs.utils import compute_difference, apply_imbalance_correction, apply_temperature_correction
from .libs.tuner import Tuner1D

# 2D Modules
from .libs.loader import DataLoader2D
from .libs.tuner import Tuner2D
from .libs.utils import save_diagnostic_maps, get_region_mask


def run_1d_tuning(config: Config, logger: logging.Logger) -> TuningResult:
    """
    Executes the full 1D scalar tuning workflow. 

    Args:
        config: The initialized ECtuner configuration object.
        logger: The configured logger instance.

    Returns:
        The TuningResult object containing metrics and optimal parameters.
    """
    logger.info("==== Starting ECtuner 1D Workflow ====")
    
    loader = DataLoader1D(config, logger)
    sensitivity = loader.load_sensitivity()
    reference = loader.load_reference()
    original_reference = copy.deepcopy(reference)

    weights_flux = config.get('weights', {})
    weights_season = config.get('weights_season', {})
    weights_region = config.get('weights_region', {})

    if len(weights_region) > 1:
        logger.info("Note: Region weights are additive. Overlapping regions (e.g., Global and NH) will stack their weights.")

    model_imbalance = config.get('args.model_imbalance')
    if model_imbalance is not None:
        logger.info(f"[PRE-PROC] Applying model imbalance correction: {model_imbalance} W/m2")
        reference = apply_imbalance_correction(reference, imbalance=model_imbalance)
        
    delta_t = config.get('args.deltaT')
    slope_file = config.get('files.slope_file')
    if delta_t is not None and slope_file is not None:
        temp_config_kwargs = {'files.sensitivity': slope_file, 'args.exp': 'temp', 'args.year1': 0, 'args.year2': 0}
        slopes_yaml = DataLoader1D(Config(config_path=None, **temp_config_kwargs)).load_sensitivity()
        reference, temp_warnings = apply_temperature_correction(reference, slopes_yaml.get('T_slope', {}), delta_t, weights_flux, weights_season, weights_region)
        logger.info(f"[PRE-PROC] Applied Temperature Correction: {delta_t} K")
        
        for w in temp_warnings:
            logger.warning(f"[Drift Correction] {w}")
            
    logger.info("\n" + " PHYSICS-BASED REFERENCE ADJUSTMENTS (Target Shifts) ".center(85, "-"))
    for v in ['net_toa', 'rsnt', 'rlnt']:
        if v in reference:
            v_orig = original_reference[v]['ALL']['Global']
            v_corr = reference[v]['ALL']['Global']
            logger.info(f"{v:<12} | Global | {v_orig:>10.4f} | {v_corr:>10.4f} | {v_corr-v_orig:>+12.4f} W/m2")
    logger.info("-" * 85 + "\n")

    base = loader.load_base()
    difference = compute_difference(base, reference)
    param_names, current_values = loader.load_params()
    ref_params = config.get('reference_parameters') or {}
    frozen_config = config.get('frozen_parameters') or {}
    if frozen_config:
        logger.info(f"Frozen parameters (keeping manual tuning): {', '.join(frozen_config.keys())}")
        
    inc_val = config.get('args.inc', 0.2)
    penalty_val = config.get('args.penalty', 0.0)
    method = config.get('args.method', 'dual_annealing')
    
    tuner = Tuner1D(inc=inc_val, penalty=penalty_val, logger=logger)
    tuner.setup_parameters(current_values, ref_params, frozen_config)
    tuner.prepare_data(sensitivity, difference, reference, weights_flux, weights_season, weights_region)
    
    result = tuner.optimize(method=method)

    return result

def run_2d_tuning(config: Config, logger: logging.Logger) -> TuningResult:
    """
    Executes the full 2D spatial tuning workflow.

    Args:
        config: The initialized ECtuner configuration object.
        logger: The configured logger instance.

    Returns:
        The TuningResult object containing global/spatial metrics and optimal parameters.
    """
    logger.info("==== Starting ECtuner 2D Spatial Workflow ====")
    
    loader = DataLoader2D(config, logger)
    weights_flux = config.get('weights', {})
    weights_region = config.get('weights_region', {})
    target_vars = list(weights_flux.keys())

    if len(weights_region) > 1:
        logger.info("Note: Region weights are additive. Overlapping regions (e.g., Global and NH) will stack their weights.")
    
    # 1. Load Data
    logger.info("Loading 2D Data (Sensitivity, Reference, Base)...")
    ds_sens = loader.load_sensitivity()
    ref_maps = loader.load_reference(target_vars)
    target_vars = list(ref_maps.keys())
    base_maps = loader.load_base(target_vars)
    
    # 2. Mask & Biases
    logger.info("Generating spatial weights and calculating initial biases...")
    mask_2d = get_region_mask(ds_sens, weights_region)
    
    bias_maps = {}
    for var in target_vars:
        mod_resampled = base_maps[var].reindex_like(ref_maps[var], method='nearest')
        bias_maps[var] = mod_resampled - ref_maps[var]
        
    # 3. Setup Tuner
    inc_val = config.get('args.inc', 0.2)
    penalty_val = config.get('args.penalty', 0.0)
    alpha_val = config.get('spatial_tuning.alpha', 0.0)
    metric_val = config.get('spatial_tuning.metric', 'l2').lower()
    method = config.get('args.method', 'dual_annealing')
    tuner = Tuner2D(inc=inc_val, penalty=penalty_val, alpha=alpha_val, metric=metric_val, logger=logger)
    param_names, current_values = loader.load_params()
    ref_params = config.get('reference_parameters') or {}
    frozen_config = config.get('frozen_parameters') or {}
    if frozen_config:
        logger.info(f"Frozen parameters (keeping manual tuning): {', '.join(frozen_config.keys())}")
        
    tuner.setup_parameters(current_values, ref_params, frozen_config)
    tuner.prepare_data(bias_maps, ref_maps, ds_sens, mask_2d, weights_flux, weights_region)
    
    # 4. Optimize
    result = tuner.optimize(method=method)
    
    # 5. Diagnostic NetCDF Export
    out_dir = config.get('files.output_dir', './')
    exp = config.get('args.exp', 'unknown')
    alpha_str = str(alpha_val).replace('.', '')
    
    output_tag = config.get('args.output_tag') or config.get('args.tag') or ''
    if output_tag:
        run_tag = output_tag
    else:
        active_regions = [r for r, w in weights_region.items() if w > 0]
        region_tag = "-".join(active_regions) if active_regions else "NoRegion"
        run_tag = f"{region_tag}_a{alpha_str}"

    diag_nc_path = os.path.join(out_dir, f"diagnostics_2d_{exp}_{run_tag}.nc")
    
    save_diagnostic_maps(
        output_path=diag_nc_path,
        target_vars=target_vars,
        bias_maps=bias_maps,
        ds_sens=ds_sens,
        params=result.param_names,
        optimal_changes=result.optimal_changes,
        r2_threshold=config.get('spatial_tuning.r2_threshold', 0.0)
    )
    
    return result

def parse_arguments():
    """Parses CLI arguments using subparsers for 1D and 2D modes."""
    parser = argparse.ArgumentParser(description='EC-Earth Tuning Suite (1D and 2D)')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Tuning mode')

    # Common arguments for both 1D and 2D
    for mode in ['1d', '2d']:
        sp = subparsers.add_parser(mode, help=f'Run {mode.upper()} tuning')
        sp.add_argument('-c', '--config', type=str, required=True, help='YAML config file')
        sp.add_argument('-o', '--output', type=str, help='Output YAML for Script Engine')
        sp.add_argument('-l', '--loglevel', type=str, default='INFO')
        sp.add_argument('-m', '--method', type=str, default='dual_annealing')
        sp.add_argument('-p', '--penalty', type=float, help='Penalty weight')
        sp.add_argument('-i', '--inc', type=float, help='Fractional max parameter change')
        sp.add_argument('-t', '--output_tag', type=str, default='')
        sp.add_argument('--logfile', type=str, help='Explicit path for the structured log file (overrides auto-generated name)')
        
        # Positional
        sp.add_argument('exp', type=str, help='Experiment to tune')
        sp.add_argument('year1', type=int, help='Start year', nargs='?', default=None)
        sp.add_argument('year2', type=int, help='End year', nargs='?', default=None)

        if mode == '1d':
            sp.add_argument('-dT', '--deltaT', type=float, help='Temperature adjustment')
            sp.add_argument('-mi', '--model_imbalance', type=float, help='Intrinsic model imbalance')
            
    return parser.parse_args()


def main():
    """Command Line Interface Entry Point."""
    args = parse_arguments()

    config = Config(args.config, exp=args.exp, year1=args.year1, year2=args.year2)
    
    # Override from CLI
    if args.penalty is not None: config.set('args.penalty', args.penalty)
    if args.inc is not None: config.set('args.inc', args.inc)
    if args.method is not None: config.set('args.method', args.method)
    if args.output_tag is not None: config.set('args.output_tag', args.output_tag)
    
    if args.mode == '1d':
        if args.deltaT is not None: config.set('args.deltaT', args.deltaT)
        if args.model_imbalance is not None: config.set('args.model_imbalance', args.model_imbalance)

    # Output Path 
    out = args.output
    out_dir = config.get('files.output_dir', './')
    if not out:
        tag = f"_{args.output_tag}" if args.output_tag else ""
        filename = f"tuned_{args.exp}_{config.get('args.year1')}-{config.get('args.year2')}_{args.mode.upper()}{tag}.yml"
        out = os.path.join(out_dir, filename)
        
    out_dir_actual = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir_actual, exist_ok=True)
    if args.logfile:
        logname = os.path.abspath(args.logfile)
        os.makedirs(os.path.dirname(logname), exist_ok=True)
    else:
        out_filename = os.path.basename(out)
        log_filename = out_filename.replace('tuned_', 'log_tuned_').replace('.yml', '.log')
        logname = os.path.join(out_dir_actual, log_filename)

    logger = setup_logger(level=args.loglevel, log_file=logname)
    
    if args.mode == '1d':
        result = run_1d_tuning(config, logger)
    elif args.mode == '2d':
        result = run_2d_tuning(config, logger)
    else:
        logger.error(f"Unknown tuning mode: {args.mode}")
        sys.exit(1)
    
    exporter.print_summary(result,logger)
    exporter.save_model_yaml(result, out, config.get('parameter_group', {}), config.get('weights', {}), config.get('weights_region', {}))
        
    diag_yaml = out.replace('tuned_', 'diagnostics_').replace('.yml', '.yaml')
    exporter.save_diagnostics_yaml(result, diag_yaml)

if __name__ == '__main__':
    main()