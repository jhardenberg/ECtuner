"""
ECtuner: A tuning tool for EC-Earth
This atmospheric tuning tool uses ECmean output files to compute new suggested values for EC-Earth OIFS parameters.
The tool uses a reference dataset and a sensitivity dataset to compute the optimal parameter changes.
The tool is based on the scipy.optimize module and uses the dual_annealing optimization method by default.

Usage:
    python ectuner.py [options] <experiment> <year1> <year2>
    
Options:    
    -c, --config <file>     yaml configuration file
    -o, --output <file>     output yaml for Script Engine
    -l, --loglevel <level>  logging level
    -m, --method <method>   optimization method (shgo (not recommended), dual_annealing (default), differential_evolution)
    -p, --penalty <value>   penalty for distance from reference parameters
    -i, --inc <value>       fractional maximum parameter change wrt reference

Arguments:
    experiment              experiment to tune
    year1                   start year
    year2                   end year

Example:
    python ectuner.py -c config-tuner.yaml -l INFO -p 0.5 -i 0.1 -o tuned_parameters.yml -m dual_annealing s000 1990 1997
    
    mio:
    python ectuner.py lr00 1990 2000 -c config_tuner_TL63.yaml -p 0.5 -i 0.1 -o tuned_params_TL63.yml -m dual_annealing

Author:  Jost von Hardenberg    
Date:    2024-10-15
"""

import sys
import os
import yaml
import argparse
import numpy as np
from scipy import optimize
import math
from tabulate import tabulate

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

# Objective function to minimize: sum of squared differences + penalty for exceeding maximum parameter changes
def objective_function(changes, params, values, reference_pars, penalty, sensitivity,
                       difference, weights_flux, weights_season, weights_region):
    """
    Objective function to minimize: sum of squared differences + penalty for exceeding maximum parameter changes
    
    Parameters:
        changes (list): list of parameter changes
    
    Returns:
        float: score to minimize
    """
    total_difference = 0
    param_difference = 0
    for fluxname in sensitivity[params[0]].keys():
        for season in  sensitivity[params[0]][fluxname].keys():
            for region in sensitivity[params[0]][fluxname][season].keys():
                if not math.isnan(difference.get(fluxname,{}).get(season, {}).get(region, np.nan)):  # Skip NaN values
                    # Calculate the sum of the product of sensitivity and parameter changes
                    flux_change = sum(sensitivity[param][fluxname][season][region][0] * changes[i] for i, param in enumerate(params))
                    # Difference between current and desired state minus the calculated flux change
                    total_difference += weights_flux[fluxname] * weights_region[region] * weights_season[season] * (difference[fluxname][season][region] + flux_change) ** 2

    param_difference += sum([((reference_pars[param] - (values[param] + changes[i]) ) / reference_pars[param]) ** 2 for i, param in enumerate(params)])



    # print(total_difference, param_difference)
    return total_difference + param_difference * penalty

def print_change(logger, changes):
    """
    Print the change in fluxes after optimization
    """
    for fluxname in targets:
        for region in difference[fluxname]['ALL']:
            if not math.isnan(difference[fluxname]['ALL'][region]):  # Skip NaN values
                flux_change = sum(sensitivity[param][fluxname]['ALL'][region][0] * changes[i] for i, param in enumerate(params))

                # Paint green if the change is in the right direction, red otherwise
                if abs(difference[fluxname]['ALL'][region] + flux_change) < abs(difference[fluxname]['ALL'][region]):
                    fmt = "\033[92m%s %s %s %s\033[0m"
                else:
                    fmt = "\033[91m%s %s %s %s\033[0m"

                logger.info(fmt, fluxname, region, difference[fluxname]['ALL'][region] + flux_change, difference[fluxname]['ALL'][region])

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
    # positional
    parser.add_argument('exp', type=str, help='experiment to tune')
    parser.add_argument('year1', type=int, help='start year', nargs='?', default=None)
    parser.add_argument('year2', type=int, help='end year', nargs='?', default=None)
    
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

    logger.debug("year1: %s", year1)
    logger.debug("year2: %s", year2)
    logger.debug("experiment: %s", exp)
    logger.debug("loglevel: %s", loglevel)
    # logger.debug("maxiter: %s", maxiter)
    logger.debug("penalty: %s", penalty)
    logger.debug("inc: %s", inc)
    logger.debug("output: %s", out)
    logger.debug("method: %s", method)

    reference_pars = config['reference_parameters']
    weights_flux=config['weights']
    weights_region=config['weights_region']
    weights_season=config['weights_season']
    targets=list(weights_flux.keys())

    # Load sensitivities
    sens_file = config['files']['sensitivity'].format(year1=year1, year2=year2)
    sensitivity = load_sensitivity(sens_file)

    # Load reference fluxes
    ref_file = config['files']['reference']
    reference = load_reference(ref_file)
    
    # Load fluxes of configuration to tune
    base_file = config['files']['base'].format(exp=exp, year1=year1, year2=year2)
    base_file = os.path.join(config['files']['ecmean'], base_file)
    base = load_base(base_file)

    # Load parameters of configuration to tune
    param_file = config['files']['params'].format(exp=exp)
    param_file = os.path.join(config['files']['exps'], param_file)
    params, vals = load_params(param_file)

    difference = compute_difference(base, reference)

    minval = {}
    maxval = {}
    values = {}

    for i in range(len(params)):
        p = params[i]
        minval[p] = reference_pars[p] * (1 - inc) - vals[i]  # Minimum value
        maxval[p] = reference_pars[p] * (1 + inc) - vals[i]  # Maximum value
        values[p] = vals[i]

    bounds = [(minval[params[i]], maxval[params[i]]) for i in range(len(params))]
    logger.debug("Parameter bounds:")
    logger.debug("-----------------")
    for i in range(len(params)):
        logger.debug("%s: %s - %s", params[i], minval[params[i]], maxval[params[i]])

    logger.info(f"Optimizing parameters using {method} ...")

    # shgo or dual_annealing
    if method == 'shgo':  # Not recommended
        result = optimize.shgo(objective_function, bounds,
                    args=(params, values, reference_pars, penalty, sensitivity, difference, weights_flux, weights_season, weights_region))
    elif method == 'dual_annealing':  # Recommended!
        result = optimize.dual_annealing(objective_function, bounds,
                    args=(params, values, reference_pars, penalty, sensitivity, difference, weights_flux, weights_season, weights_region))
    elif method == 'differential_evolution':
        result = optimize.differential_evolution(objective_function, bounds,
                    args=(params, values, reference_pars, penalty, sensitivity, difference, weights_flux, weights_season, weights_region))
    else:
        logger.error("Method not supported")
        sys.exit(1)
    
    # Print the optimal parameter changes
    optimal_changes = {params[i]: result.x[i] for i in range(len(params))}

    logger.debug("Optimization result:")
    logger.debug("--------------------")
    logger.debug(result)

    logger.info("")

    logger.info("Target offset after and before optimization:")
    logger.info("-------------------------------------------")
    print_change(logger, result.x)

    initial_guess = np.zeros(len(params))
    logger.info("")
    logger.info("Total score before optimization: %s", objective_function(initial_guess, params, values, reference_pars, penalty, sensitivity, difference, weights_flux, weights_season, weights_region))
    logger.info("Total score after optimization: %s", objective_function(result.x, params, values, reference_pars, penalty, sensitivity, difference, weights_flux, weights_season, weights_region))
    logger.info("")

    if out:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap, CommentedSeq

        yaml_ru = YAML(typ="rt")  
        yaml_ru.indent(mapping=2, sequence=2, offset=0)  

        tuning_block = CommentedMap()
        for pg in config['parameter_group']:
            tuning_block[pg] = CommentedMap()
            for p in config['parameter_group'][pg]:
                new_val = float(values[p] + optimal_changes[p])
                tuning_block[pg][p] = new_val

        oifs_block = CommentedMap()
        oifs_block["tuning"] = tuning_block
        model_config_block = CommentedMap()
        model_config_block["oifs"] = oifs_block
        base_context_block = CommentedMap()
        base_context_block["model_config"] = model_config_block

        output_list = CommentedSeq()
        output_list.append(CommentedMap({"base.context": base_context_block}))

        with open(out, "w") as f:
            yaml_ru.dump(output_list, f)
            f.write("\n")
            f.write("# --- ECtuner meta-parameters ---\n")
            f.write(f"# penalty: {penalty}\n")
            f.write(f"# inc (fractional max change): {inc}\n")
            f.write("# weights (flux):\n")
            for k, v in weights_flux.items():
                f.write(f"#   {k}: {v}\n")
            f.write("# weights (region):\n")
            for k, v in weights_region.items():
                f.write(f"#   {k}: {v}\n")
            f.write("# weights (season):\n")
            for k, v in weights_season.items():
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


        
 