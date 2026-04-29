"""
ECtuner: A tuning tool for EC-Earth
----------------------------------------------
This atmospheric tool computes optimal parameter suggestions for the EC-Earth OIFS component
by minimizing a cost function based on model biases and parameter deviations.

How it works:
    1. It reads model diagnostics (Global Means) via ECmean4.
    2. It compares model output against a reference dataset (Observations).
    3. It uses a pre-computed Sensitivity Matrix to estimate how parameter
       changes affect model biases.
    4. It identifies the optimal parameter set using global optimization 
       algorithms (Scipy-based) and uses the dual_annealing optimization method by default.

Usage:
    python ectuner.py [options] <experiment> <year1> <year2>

Arguments:
    experiment          Experiment tag to be tuned.
    year1, year2        Start and end years for the tuning period.
    
Options:
    -c, --config        Path to the YAML configuration file.
    -o, --output        Output YAML file with tuned parameters (Script Engine format).
    -l, --loglevel      Logging level (DEBUG, INFO, WARNING, ERROR).
    -m, --method        Optimization method: 'dual_annealing' (recommended), 
                        'differential_evolution', or 'shgo'.
    -p, --penalty       Weight for the penalty term (distance from reference params).
    -i, --inc           Maximum allowed fractional change for parameters (e.g., 0.2 = 20%).
    -dT, --deltaT       Global temperature adjustment (K) for reference correction.
    -imb, --imbalance   Target NetTOA imbalance (W/m2) to correct.
    --freeze            List of parameter names to keep fixed during optimization.

Examples:
    python ectuner.py -c config-tuner.yaml -l INFO -p 0.5 -i 0.1 -o tuned_parameters.yml -m dual_annealing s000 1990 1997
    python ectuner.py lr00 1990 2000 -c config_tuner_TL63.yaml -p 0.5 -i 0.1 -o tuned_params_TL63.yml -m dual_annealing

Author:  Jost von Hardenberg    
Updated: 2026-02-24
"""

import sys
import os
import yaml
import argparse
import numpy as np
from scipy import optimize
import math
from tabulate import tabulate
import copy

from logger import setup_logger

def load_config(config_file='config-tuner.yaml'):
    """
    Load configuration file
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_sensitivity(sens_file='sensitivity_1990-1997.yaml'):
    """
    Load sensitivity file (computed externally)
    """
    with open(sens_file, 'r') as file:
        sensitivity = yaml.safe_load(file)
    return sensitivity

def load_reference(ref_file='gm_reference_EC23.yml'):
    """
    Load reference file with reference fluxes/targets
    """

    with open(ref_file, 'r') as file:
        ref = yaml.safe_load(file)

    reference = {}
    # Organize reference data in structure of nesteed dics with
    # variable, season, region as keys and fluxes as values

    for t in ref.keys():
        reft = ref[t]['obs']
        if isinstance(reft, dict):
            for key1 in reft:
                for key2 in reft[key1]:
                    reft[key1][key2] = reft[key1][key2]['mean']
        else:
            reft={'ALL': {'Global': reft}}

        reference[t] = reft
    
    return reference


def apply_imbalance_correction(reference, imbalance = 0., adjust_individual_fluxes = True, sw_fraction = 0.5):
    """
    Correct net_toa target for intrinsic model imbalances.

    If the imbalances are > 0, the model creates energy. If < 0, the model destroys energy.
    So the net_toa reference is corrected in the opposite direction: if the model destroys energy, as it is for ece4 lowres, the net_toa equilibrates at a value > 0.

    If adjust_individual_fluxes is set, the imbalance is propagated to rsnt and rlnt (annual, global). sw_fraction is the part attributed to rsnt (by default it is 0.5), (1-sw_fraction) is attributed to rlnt.
    """
    corrected_reference = copy.deepcopy(reference)

    if 'net_toa' in corrected_reference:
        corrected_reference['net_toa']['ALL']['Global'] -= imbalance
        print(f'net toa reference: old {reference['net_toa']['ALL']['Global']} -> new {corrected_reference['net_toa']['ALL']['Global']}')
    
    if adjust_individual_fluxes:
        print('Propagating imbalance correction to rsnt and rlnt')
        if 'rsnt' in corrected_reference:
            corrected_reference['rsnt']['ALL']['Global'] -= sw_fraction*imbalance
        
        if 'rlnt' in corrected_reference:
            corrected_reference['rlnt']['ALL']['Global'] -= (1-sw_fraction)*imbalance

    return corrected_reference


def apply_temperature_correction(reference, slopes, delta_t, weights, weights_season, weights_region):
    """
    Modify reference fluxes by subtracting delta_t * slope, only if slope exists.
    Raise error only if combined weight > 0 and slope is missing.
    """
    corrected_reference = copy.deepcopy(reference)

    for var in corrected_reference:
        var_weight = weights.get(var, 0.0)

        for season in corrected_reference[var]:
            season_weight = weights_season.get(season, 0.0)

            for region in corrected_reference[var][season]:
                region_weight = weights_region.get(region, 0.0)

                combined_weight = var_weight * season_weight * region_weight

                # Try to safely get the slope, return None if any level is missing
                slope = (
                    slopes.get(var, {})
                          .get(season, {})
                          .get(region)
                )

                # If missing or invalid, handle based on weight
                if slope is None or (isinstance(slope, float) and math.isnan(slope)):
                    if combined_weight > 0.0:
                        raise ValueError(
                            f"Slope missing or NaN for variable '{var}', season '{season}', region '{region}', "
                            f"but its combined weight is {combined_weight} (> 0)."
                        )
                    else:
                        slope = 0.0  # Safe fallback if weight is zero

                corrected_reference[var][season][region] += -(delta_t * slope)
            
    return corrected_reference


def load_base(base_file='ecmean/global_mean_s000_EC-Earth4_r1i1p1f1_1990_1997.yml'):
    """
    Load base file with fluxes of configuration to tune
    """

    with open(base_file, 'r') as file:
        base = yaml.safe_load(file)
    return base

def load_params(param_file):
    """
    Load parameter file with parameters of configuration to tune
    """

    with open(param_file, 'r') as file:
        coso = yaml.safe_load(file)
        if isinstance(coso, list):
            # new tuning file in se format
            params = coso[0]['base.context']['model_config']['oifs']
        else:
            # old tuning file
            params = coso

    if 'tuning' in params:
        # the tuning file is in SE format
        old_par = params.copy()
        params = {}

        for ke1 in old_par['tuning']:
            for ke2 in old_par['tuning'][ke1]:
                params[ke2] = old_par['tuning'][ke1][ke2]
        
    # Cast all values to float
    for p in params:
        params[p] = float(params[p])

    return list(params.keys()), list(params.values())

def compute_difference(base, reference):
    """
    Compute the difference between base and reference fluxes
    """

    difference = {}
    for key, value in base.items():
        difference[key] = {}
        for subkey, subvalue in value.items():
            if key in reference and subkey in reference[key]:
                difference[key][subkey] = {}
                for subsubkey, subsubvalue in subvalue.items():
                    if subsubkey in reference[key][subkey]:
                        difference[key][subkey][subsubkey] = subsubvalue - reference[key][subkey][subsubkey]
            #         else:
            #             difference[key][subkey][subsubkey] = np.nan
            # else:
            #     difference[key][subkey] = np.nan
    return difference

#Objective function to minimize: sum of squared differences + penalty for exceeding maximum parameter changes
def objective_function(changes, params, values, reference_pars, penalty, sensitivity,
                       difference, weights_flux, weights_season, weights_region,
                       frozen_params=None):
    """
    Objective function with frozen parameters support.
    
    Parameters:
        changes (list): list of changes for free parameters only
        params (list): all parameters (free + frozen)
        frozen_params (dict): dictionary of frozen parameters {param_name: value}
    Returns:
        float: score to minimize
    """

    if frozen_params is None:
        frozen_params = {}

    # Reconstrtuction of all changes including frozen parameters
    all_changes = []
    free_idx = 0
    for p in params:
        if p in frozen_params:
            all_changes.append(0.0)
        else:
            all_changes.append(changes[free_idx])
            free_idx += 1

    total_difference = 0
    param_difference = 0

    for fluxname in sensitivity[params[0]].keys():
        for season in sensitivity[params[0]][fluxname].keys():
            for region in sensitivity[params[0]][fluxname][season].keys():
                if not math.isnan(difference.get(fluxname,{}).get(season, {}).get(region, np.nan)):
                    flux_change = sum(sensitivity[param][fluxname][season][region][0] * all_changes[i]
                                      for i, param in enumerate(params))
                    total_difference += (weights_flux[fluxname] *
                                         weights_region[region] *
                                         weights_season[season] *
                                         (difference[fluxname][season][region] + flux_change) ** 2)

    param_difference += sum([((reference_pars[param] - (values[param] + all_changes[i])) / reference_pars[param]) ** 2
                             for i, param in enumerate(params)])

    return total_difference + param_difference * penalty

def print_table(logger, data):
    """Prints formatted rows for a professional look"""
    # Header della tabella
    header = f"{'Variable':<12} | {'Season':<6} | {'Region':<12} | {'Weight':<6} | {'Bias Init':>10} -> {'Bias Final':>10} | {'Status'}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in data:
        is_improved = abs(r[5]) < abs(r[4])
        status = "IMPROVED" if is_improved else "WORSENED"
        color = "\033[92m" if is_improved else "\033[91m"
        reset = "\033[0m"
        
        # r[0]:var, r[1]:season, r[2]:region, r[3]:weight, r[4]:bias_init, r[5]:bias_final
        logger.info(f"{r[0]:<12} | {r[1]:<6} | {r[2]:<12} | {r[3]:<6} | {r[4]:>10.3f} -> {color}{r[5]:>10.3f}{reset} | {status}")

def log_optimization_results(logger, params, optimal_changes_list, sensitivity, difference, 
                             weights_flux, weights_season, weights_region):
    targets = []
    diagnostics = []

    for fluxname in difference:
        if fluxname not in sensitivity[params[0]]: continue

        for season in difference[fluxname]:
            for region in difference[fluxname][season]:
                bias_init = difference[fluxname][season][region]
                if math.isnan(bias_init): continue

                flux_change = sum(sensitivity[p][fluxname][season][region][0] * optimal_changes_list[i] 
                                  for i, p in enumerate(params))
                bias_final = bias_init + flux_change
                
                w_flux = weights_flux.get(fluxname, 0)
                w_season = weights_season.get(season, 0)
                w_region = weights_region.get(region, 0)
                combined_weight = w_flux * w_season * w_region
                
                row = [fluxname, season, region, combined_weight, bias_init, bias_final]
                
                if combined_weight > 0:
                    targets.append(row)
                else:
                    diagnostics.append(row)

    logger.info("\n" + " OPTIMIZATION SUMMARY (Biases: Model - Target) ".center(90, "="))
    logger.info("Goal: Bring Biases to 0.0")
    
    logger.info("\n" + " PRIMARY TUNING TARGETS ".center(90, "-"))
    print_table(logger, targets)
    
    logger.info("\n" + " DIAGNOSTIC SIDE-EFFECTS ".center(90, "-"))
    print_table(logger, diagnostics)
    logger.info("=" * 90 + "\n")

def parse_arguments(arguments):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='EC-Earth tuning tool')

    parser.add_argument('-c', '--config', type=str,
                        help='yaml configuration file')
    parser.add_argument('-o', '--output', type=str,
                        help='output yaml for Script Engine')
    parser.add_argument('-l', '--loglevel', type=str,
                        help='logging level')
    parser.add_argument('-m', '--method', type=str,
                        help='optimization method (shgo (not recommended), dual_annealing (default), differential_evolution)')
    # parser.add_argument('-m', '--maxiter', type=int,
    #                     help='the maximumum number of iterations')
    parser.add_argument('-p', '--penalty', type=float,
                        help='penalty for distance from reference parameters')
    parser.add_argument('-i', '--inc', type=float,
                        help='fractional maximum parameter change wrt reference')
    parser.add_argument('-dT', '--deltaT', type=float, 
                        help='Temperature adjustment for reference correction')
    parser.add_argument('-mi', '--model_imbalance', type=float, 
                        help='Intrinsic model imbalance to correct net_toa')
    # positional
    parser.add_argument('exp', type=str, help='experiment to tune')
    parser.add_argument('year1', type=int, help='start year', nargs='?', default=None)
    parser.add_argument('year2', type=int, help='end year', nargs='?', default=None)
    parser.add_argument('-a', '--alpha', type=float, help='Hybrid weight (0: spatial, 1: global)')
    
    return parser.parse_args(arguments)

def get_arg(args, arg, default):
    """
    Support function to get arguments

    Args:
        args: the arguments
        arg: the argument to get
        default: the default value

    Returns:
        The argument value or the default value
    """

    res = getattr(args, arg)
    if not res:
        res = default
    return res

if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    config_file = get_arg(args, 'config', 'config-tuner.yaml')
    year1 = get_arg(args, 'year1', None)
    year2 = get_arg(args, 'year2', None)
    exp = get_arg(args, 'exp', None)
    loglevel = get_arg(args, 'loglevel', 'INFO')
    #maxiter = get_arg(args, 'maxiter', 10000)
    penalty = get_arg(args, 'penalty', None)
    inc = get_arg(args, 'inc', None)
    out = get_arg(args, 'output', None)
    method = get_arg(args, 'method', None)

    logger = setup_logger(level=loglevel)

    if not exp:
        print("Error:  experiment not specified")
        sys.exit(1)

    config = load_config(config_file)
    logger.info("==== ECtuner configuration ====")
    logger.info("\n" + yaml.safe_dump(config, sort_keys=False))

    if not year1:
        year1 = config['args']['year1']
    if not year2:
        year2 = config['args']['year2']
    if not penalty:
        penalty = config['args']['penalty']
    if not inc:
        inc = config['args']['inc']
    if not method:
        method = config['args']['method']

    # logger.debug("year1: %s", year1)
    # logger.debug("year2: %s", year2)
    # logger.debug("experiment: %s", exp)
    # logger.debug("loglevel: %s", loglevel)
    # # logger.debug("maxiter: %s", maxiter)
    # logger.debug("penalty: %s", penalty)
    # logger.debug("inc: %s", inc)
    # logger.debug("output: %s", out)
    # logger.debug("method: %s", method)

    reference_pars = config['reference_parameters']
    for par in reference_pars:
        reference_pars[par] = float(reference_pars[par])
    weights_flux=config['weights']
    for we in weights_flux:
        weights_flux[we] = float(weights_flux[we])
    weights_region=config['weights_region']
    for we in weights_region:
        weights_region[we] = float(weights_region[we])
    weights_season=config['weights_season']
    for we in weights_season:
        weights_season[we] = float(weights_season[we])
    targets=list(weights_flux.keys())

    # Load sensitivities
    sens_file = config['files']['sensitivity'].format(year1=year1, year2=year2)
    sensitivity = load_sensitivity(sens_file)

    # Load reference fluxes
    ref_file = config['files']['reference']
    reference = load_reference(ref_file)
    original_reference = copy.deepcopy(reference)

    # Save in results directory
    if not out:
        config_files = config.get('files', {})
        out_dir = config_files.get('output_dir')
        out_temp = config_files.get('output_template')
        if out_dir and out_temp:
            filename = out_temp.format(exp=exp, year1=year1, year2=year2)
            out = os.path.join(out_dir, filename)
            os.makedirs(out_dir, exist_ok=True)
    
    logger.debug("year1: %s", year1)
    logger.debug("year2: %s", year2)
    logger.debug("experiment: %s", exp)
    logger.debug("loglevel: %s", loglevel)
    # logger.debug("maxiter: %s", maxiter)
    logger.debug("penalty: %s", penalty)
    logger.debug("inc: %s", inc)
    logger.debug("output: %s", out)
    logger.debug("method: %s", method)

    if out:
        logger.info(f"[CONFIG] Output will be saved to: {out}")
    else:
        logger.warning("[CONFIG] No output path specified. Results will only be printed to screen.")


    # model_imbalance from command line or config if needed
    model_imbalance = args.model_imbalance if args.model_imbalance is not None else config.get('args', {}).get('model_imbalance')
    if model_imbalance is not None:
        # LOG PRE-CORRECTION
        old_val = reference.get('net_toa', {}).get('ALL', {}).get('Global', 0.0)
        
        logger.info(f"[PRE-PROC] Applying model imbalance correction: {model_imbalance} W/m2")
        reference = apply_imbalance_correction(reference, model_imbalance)
        
        # LOG POST-CORRECTION
        new_val = reference.get('net_toa', {}).get('ALL', {}).get('Global', 0.0)
        logger.info(f"           net_toa Global reference: {old_val} -> {new_val}")

    # Modify reference file if there is delta t in config file and the slope file
    # Check if delta_t and sensitivity (slopes) file exist in config
    delta_t = args.deltaT if args.deltaT is not None else config.get('args', {}).get('deltaT')
    slope_file = config['files'].get('slope_file')
    if delta_t is not None and slope_file is not None:
        slopes_yaml = load_sensitivity(slope_file)
        slopes = slopes_yaml.get('T_slope', {})
        weights = config.get('weights', {})
        weights_season = config.get('weights_season', {})
        weights_region = config.get('weights_region', {})
        corrected_reference = apply_temperature_correction(reference, slopes, delta_t, weights_flux, weights_season, weights_region)
    else:
        corrected_reference = reference

    logger.info("\n" + " PHYSICS-BASED REFERENCE ADJUSTMENTS (Target Shifts) ".center(85, "-"))
    logger.info(f"{'Variable':<12} | {'Region':<12} | {'Original':>10} | {'Corrected':>10} | {'Total Shift':>12}")
    logger.info("-" * 85)

    for v in ['net_toa', 'rsnt', 'rlnt']: # key variables
        if v in corrected_reference:
            v_orig = original_reference[v]['ALL']['Global']
            v_corr = corrected_reference[v]['ALL']['Global']
            shift = v_corr - v_orig
            logger.info(f"{v:<12} | {'Global':<12} | {v_orig:>10.4f} | {v_corr:>10.4f} | {shift:>+12.4f} W/m2")
    
    logger.info("-" * 85)
    if delta_t: logger.info(f" * Applied Temperature Correction: {delta_t} K")
    if model_imbalance: logger.info(f" * Applied Model Imbalance: {model_imbalance} W/m2")
    logger.info("-" * 85 + "\n")
        
    # Load fluxes of configuration to tune
    base_file = config['files']['base'].format(exp=exp, year1=year1, year2=year2)
    base_file = os.path.join(config['files']['ecmean'], base_file)
    base = load_base(base_file)

    # Load parameters of configuration to tune
    param_file = config['files']['params'].format(exp=exp)
    param_file = os.path.join(config['files']['exps'], param_file)
    params, vals = load_params(param_file)

    values = {p: vals[i] for i, p in enumerate(params)}
    difference = compute_difference(base, corrected_reference)

    # Frozen parameters
    frozen_params_list = config.get('frozen_parameters', [])
    frozen_params = {p: values[p] for p in frozen_params_list if p in values}  

    if frozen_params:
        logger.info("Frozen parameters detected: %s", ", ".join(f"{p}={v}" for p, v in frozen_params.items()))
    else:
        logger.info("No frozen parameters specified.")

    opt_params = [p for p in params if p not in frozen_params]

    # Minval and maxval 
    epsilon = 1e-12
    minval = {}
    maxval = {}
    for p in params:
        if p in frozen_params:
            minval[p] = values[p] - epsilon
            maxval[p] = values[p] + epsilon
        else:
            minval[p] = reference_pars[p] * (1 - inc) - values[p]
            maxval[p] = reference_pars[p] * (1 + inc) - values[p]

    bounds = [(minval[p], maxval[p]) for p in opt_params]

    logger.debug("Parameter bounds:")
    logger.debug("-----------------")
    for p in opt_params:
        logger.debug("%s: %s - %s", p, minval[p], maxval[p])

    logger.info(f"Optimizing parameters using {method} ...")

    # shgo, dual_annealing o differential_evolution
    if method == 'shgo':
        result = optimize.shgo(objective_function, bounds,args=(params, values, reference_pars, penalty, sensitivity, difference,
                               weights_flux, weights_season, weights_region, frozen_params))
    elif method == 'dual_annealing':
        result = optimize.dual_annealing(objective_function, bounds, args=(params, values, reference_pars, penalty, sensitivity, difference,
                                         weights_flux, weights_season, weights_region,frozen_params))
    elif method == 'differential_evolution':
        result = optimize.differential_evolution(objective_function, bounds, args=(params, values, reference_pars, penalty, sensitivity, difference,
                                                 weights_flux, weights_season, weights_region,frozen_params))
    else:
        logger.error("Method not supported")
        sys.exit(1)
    
    # Print the optimal parameter changes
    optimal_changes = {}
    free_idx = 0
    for p in params:
        if p in frozen_params:
            optimal_changes[p] = 0.0
        else:
            optimal_changes[p] = result.x[free_idx]
            free_idx += 1

    logger.debug("Optimization result:")
    logger.debug("--------------------")
    logger.debug(result)

    logger.info("")

    log_optimization_results(logger, params, [optimal_changes[p] for p in params], 
                         sensitivity, difference, weights_flux, weights_season, weights_region)

    initial_guess_free = np.zeros(len(opt_params))
    logger.info("Total score before optimization: %s", 
            objective_function(initial_guess_free, params, values, reference_pars, penalty, 
                               sensitivity, difference, weights_flux, weights_season, weights_region, frozen_params))

    logger.info("Total score after optimization: %s", 
            objective_function(result.x, params, values, reference_pars, penalty, 
                               sensitivity, difference, weights_flux, weights_season, weights_region, frozen_params))
    
    if out:
        from ruamel.yaml import YAML
        yaml_ru = YAML(typ="rt")  
        yaml_ru.indent(mapping=2, sequence=2, offset=0)
        yaml_ru.preserve_quotes = True

        tuning_block = {}
        for pg in config['parameter_group']:
            current_group = {}
            for p in config['parameter_group'][pg]:
                if p not in values: continue # skip parameter if not in tuning_file of exp to be tuned
                
                val_to_write = values[p] if p in frozen_params else values[p] + optimal_changes[p]
                current_group[p] = float(f"{val_to_write:.4e}")
            
            # Only if the group has at least one parameter, we add it to the tuning block
            if current_group:
                tuning_block[pg] = current_group

        full_structure = [
            {
                "base.context": {
                    "model_config": {
                        "oifs": {
                            "tuning": tuning_block
                        }
                    }
                }
            }
        ]

        with open(out, "w") as f:
            yaml_ru.dump(full_structure, f)
        
            f.write("\n# --- ECtuner meta-parameters ---\n")
            f.write(f"# penalty: {penalty}\n")
            f.write(f"# inc (fractional max change): {inc}\n")
            
            for section, data in [("flux", weights_flux), ("region", weights_region), ("season", weights_season)]:
                f.write(f"# weights ({section}):\n")
                for k, v in data.items():
                    f.write(f"#   {k}: {v}\n")

        logger.info("Structured tuning YAML written to %s", out)

    print("\nParameters:")
    print("-----------")
    outtable = []
    for p in optimal_changes:
        outtable.append([p, values[p]+optimal_changes[p], values[p],
                         optimal_changes[p], optimal_changes[p]/values[p], minval[p], maxval[p], (values[p]+optimal_changes[p]-reference_pars[p])/reference_pars[p]])
        print(p,':', values[p]+optimal_changes[p])
    print("")
    head=['Parameter','New value','Old value', 'Change', 'Relative change','Min change', 'Max change', 'Rel. dist. from ref.']
    print(tabulate(outtable, headers=head, stralign='center', tablefmt='orgtbl'))