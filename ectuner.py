# ECE Tuner 
# A tool to perform tuning of EC-Earth model parameters
# based on ECmean4 output

import sys
import os
import yaml
import argparse
import numpy as np
from scipy.optimize import minimize
import math
from tabulate import tabulate

from logger import setup_logger

def load_config(config_file='config-tuner.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_sensitivity(sens_file='sensitivity_1990-1997.yaml'):
    with open(sens_file, 'r') as file:
        sensitivity = yaml.safe_load(file)
    return sensitivity

def load_reference(ref_file='gm_reference_EC23.yml'):
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
    with open(base_file, 'r') as file:
        base = yaml.safe_load(file)
    return base

def load_params(param_file):
    with open(param_file, 'r') as file:
        params = yaml.safe_load(file)
    return list(params.keys()), list(params.values())

def compute_difference(base, reference):
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
def objective_function(changes, params, sensitivity, difference, weights_flux, weights_season, weights_region):
    """
    Objective function to minimize: sum of squared differences + penalty for exceeding maximum parameter changes
    
    Parameters:
        changes (list): list of parameter changes
    
    Returns:
        float: score to minimize
    """
    total_difference = 0
    for fluxname in sensitivity[params[0]].keys():
        for season in  sensitivity[params[0]][fluxname].keys():
            for region in sensitivity[params[0]][fluxname][season].keys():
                if not math.isnan(difference.get(fluxname,{}).get(season, {}).get(region, np.nan)):  # Skip NaN values
                    # Calculate the sum of the product of sensitivity and parameter changes
                    flux_change = sum(sensitivity[param][fluxname][season][region][0] * changes[i] for i, param in enumerate(params))
                    # Difference between current and desired state minus the calculated flux change
                    total_difference += weights_flux[fluxname] * weights_region[region] * weights_season[season] * (difference[fluxname][season][region] + flux_change) ** 2
    return total_difference

def print_change(logger, changes):
    for fluxname in targets:
        for region in difference[fluxname]['ALL']:
            if not math.isnan(difference[fluxname]['ALL'][region]):  # Skip NaN values
                flux_change = sum(sensitivity[param][fluxname]['ALL'][region][0] * changes[i] for i, param in enumerate(params))
                logger.info("%s %s %s", fluxname, region, difference[fluxname]['ALL'][region] + flux_change)    

def parse_arguments(arguments):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='EC-Earth tuning tool')

    parser.add_argument('-c', '--config', type=str,
                        help='yaml configuration file')
    parser.add_argument('-l', '--loglevel', type=str,
                        help='logging level')
    parser.add_argument('-i', '--maxiter', type=int,
                        help='the maximumum number of iterations')
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
    maxiter=get_arg(args, 'maxiter', 10000)

    logger = setup_logger(level=loglevel)

    if not exp:
        print("Error:  experiment not specified")
        sys.exit(1)

    config = load_config(config_file)

    if not year1:
        year1 = config['args']['year1']
    if not year2:
        year2 = config['args']['year2']

    weights_flux=config['weights']
    weights_region=config['weights_region']
    weights_season=config['weights_season']
    inc=config['inc']
    targets=list(weights_flux.keys())

    # params=list(config['pars'].keys())
    # vals=list(config['pars'].values())

    sens_file = config['files']['sensitivity'].format(year1=year1, year2=year2)
    ref_file = config['files']['reference']
    
    base_file = config['files']['base'].format(exp=exp, year1=year1, year2=year2)
    base_file = os.path.join(config['files']['ecmean'], base_file)

    param_file = config['files']['params'].format(exp=exp)
    param_file = os.path.join(config['files']['exps'], param_file)

    sensitivity = load_sensitivity(sens_file)    
    reference = load_reference(ref_file)
    base = load_base(base_file)
    params, vals = load_params(param_file)

    difference = compute_difference(base, reference)

    diffmax = {}
    values = {}
    for i in range(len(params)):
        diffmax[params[i]] = vals[i] * inc
        values[params[i]] = vals[i]

    # List of parameters

    # params = list(sensitivity.keys())
   
    # Constraints: parameter changes should be within the prescribed maximum differences
    # constraints = []
    # for i, param in enumerate(params):
    #     constraints.append({'type': 'ineq', 'fun': lambda x, i=i: diffmax[params[i]] - np.abs(x[i])})
    constraints = [{'type': 'ineq', 'fun': lambda x, i=i: diffmax[params[i]] - np.abs(x[i])} for i in range(len(params))]

    # Initial guess: no change
    initial_guess = np.zeros(len(params))

    # Minimize the objective function with constraints
    #result = minimize(objective_function, initial_guess, constraints=constraints)
    #result = minimize(objective_function, initial_guess, constraints=constraints, method='COBYLA', options={'disp': False, 'ftol': 1e-10, 'maxiter': 100})

    result = minimize(objective_function, initial_guess, constraints=constraints, 
                      method='trust-constr', options={'disp': True, 'maxiter': maxiter},
                      args=(params, sensitivity, difference, weights_flux, weights_season, weights_region))

    # Print the optimal parameter changes
    optimal_changes = {params[i]: result.x[i] for i in range(len(params))}

    logger.info("Total score before optimization: %s", objective_function(initial_guess, params, sensitivity, difference, weights_flux, weights_season, weights_region))

    logger.info("Target offset before optimization:")
    logger.info("-------------------------------")
    print_change(logger, initial_guess)

    logger.info("Total score after optimization: %s", objective_function(result.x, params, sensitivity, difference, weights_flux, weights_season, weights_region))

    logger.info("Target offset after optimization:")
    logger.info("-------------------------------")
    print_change(logger, result.x)

    print("\nParameters:")
    print("-----------")
    outtable = []
    for p in optimal_changes:
        outtable.append([p, values[p]+optimal_changes[p], optimal_changes[p], optimal_changes[p]/values[p], diffmax[p]])
        print(p,':', values[p]+optimal_changes[p])
    print("")
    head=['Parameter','New value','Change','Relative change','Max change']
    print(tabulate(outtable, headers=head, stralign='center', tablefmt='orgtbl'))
        
