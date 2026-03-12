"""
Sensitivities for ECtuner.
-------------------------
Script to compute sensitivity coefficients from an ensemble of tuning runs. 
It performs a linear regression between parameter perturbations and model diagnostic changes (via ECmean4).

Prerequisites:
    1. AMIP Experiments: A set of simulations perturbing one parameter at a time (OAT), free naming of experiments. 
    2. Control Run: One experiment must use the default parameter set. (e.g. s000 or k000)
    3. Tuning Files: YAML files for each experiment containing the parameter values, located in the directory specified by 'exps' in the config file.
    4. Diagnostics: Global mean files (.yml) produced by ECmean4. 
       Note: If these files are missing, the script will attempt to calculate them automatically (requires ECmean4 config).

Usage:
    python sensitivity.py -c <config.yaml> [ref_tag] [exp_temp] [year1] [year2]
    
Options:    
    -c, --config <file>     yaml configuration file

Arguments:
    -c, --config    Path to the YAML configuration file.
    ref_tag         (Optional) The tag of the control experiment (e.g., s000 or 00).
    exp_temp        (Optional) Glob pattern for tags (e.g., "s???" or "??").
    year1, year2    (Optional) Start and end years for the analysis.

Examples:
    # Example 1: Using 's' prefix and 4-digit tags
    python sensitivity.py -c config_sens.yaml s000 "s???" 1990 1997

    # Example 2: Using standard 2-digit tags
    python sensitivity.py -c config_sens.yaml 00 "??" 1991 2000

#########################################################################
## A selection of tuning parameters:
#           
#    namcumf:
#        RPRCON            # conversion from cloud water to rain
#        ENTRORG           # entrainment rate (positive buoyant convection)
#        DETRPEN           # detrainment rate for penetrative convection
#        ENTRDD            # entrainment rate for cumulus downdrafts
#        RMFDEPS           # fractional massflux for downdrafts at lfs
#    namcldp:
#        RVICE             # fixed ice fallspeed
#        RLCRITSNOW        # critical autoconversion threshold
#        RSNOWLIN2         # temp dependence of ice-to-snow autoconversion
#        RCLDIFF           # diffusion-coefficient for evaporation
#        RCLDIFF_CONVI     # enhancement factor of rcldiff for convection
#        RDEPLIQREFRATE    # deposition efficiency in top layer
#        RDEPLIQREFDEPTH   # recovery depth scale (m)
#        RCL_OVERLAPLIQICE # overlap assumption for liquid and ice
#        RCL_INHOMOGAUT    # inhomogeneity factor for autoconversion
#        RCL_INHOMOGACC    # inhomogeneity factor for accretion
#    naerad:
#        RMINICE           # min ice crystal radius for sedimentation (microns)


Authors: Federico Fabiano, Jost von Hardenberg
Date: 2025-06-04 (Updated: 2026-02-24)

"""

import yaml
import numpy as np
import argparse
import glob
import sys

from ecmean import global_mean

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

#################### Functions ############################################

def flatten_yaml_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, float]:
    """
    Flatten nested dictionary keeping only the last layer of keys.
    Returns a flat dictionary with parameter names as keys and values as floats.
    """
    flattened = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            flattened.update(flatten_yaml_dict(value, new_key, sep))
        else:
            try:
                flattened[key] = float(value)  # Keep only the last key name
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {key}={value} to float")
    
    return flattened

def read_yaml_files(yaml_files: List[str], params_template: str) -> Dict[str, Dict[str, float]]:
    """
    Read multiple YAML files and return flattened dictionaries.
    Uses tags extracted from filenames as keys.
    """
    all_data = {}
    
    for yaml_file in yaml_files:
        filename = os.path.basename(yaml_file)
        tag = extract_tag_by_position(params_template, filename)
        
        if tag is None:
            print(f"Warning: Could not extract tag from {filename} using template {params_template}")
            continue

        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                
                if isinstance(data, list) and len(data) > 0 and 'base.context' in data[0]:
                    # SE case (list with 'base.context' key)
                    raw_params = data[0]['base.context']['model_config']['oifs']
                elif isinstance(data, dict) and 'tuning' in data:
                    # Format 1 (Nested with 'tuning' key)
                    raw_params = data['tuning']
                else:
                    # Format 2 (Already flat or unknown structure)
                    raw_params = data
                
                flattened = flatten_yaml_dict(raw_params)
                all_data[tag] = flattened
                print(f"Loaded {len(flattened)} parameters from {filename} (tag: {tag})")
                
        except Exception as e:
            print(f"Error reading {yaml_file}: {e}")
    
    return all_data

def compare_with_reference(all_data: Dict[str, Dict[str, float]], 
                          ref_dict: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Compare parameters across files and with reference values.
    Returns info about min/max values and changes from reference.
    Uses tags instead of filenames.
    """
    all_params = set()
    for file_data in all_data.values():
        all_params.update(file_data.keys())
    
    results = {}
    
    for param in all_params:
        param_values = {}
        
        for tag, file_data in all_data.items():
            if param in file_data:
                param_values[tag] = file_data[param]
        
        if not param_values:
            continue
            
        min_tag = min(param_values.keys(), key=lambda x: param_values[x])
        max_tag = max(param_values.keys(), key=lambda x: param_values[x])
        min_value = param_values[min_tag]
        max_value = param_values[max_tag]
        
        ref_value = ref_dict.get(param)
        changed_from_ref = {}
        
        if ref_value is not None:
            for tag, value in param_values.items():
                if abs(value - ref_value) > 1e-10:  # Account for floating point precision
                    changed_from_ref[tag] = {
                        'current': value,
                        'reference': ref_value,
                        'difference': value - ref_value,
                        'relative_change': ((value - ref_value) / ref_value * 100) if ref_value != 0 else float('inf')
                    }
        
        results[param] = {
            'min_tag': min_tag,
            'min_value': min_value,
            'max_tag': max_tag,
            'max_value': max_value,
            'all_values': param_values,
            'reference_value': ref_value,
            'changed_from_reference': changed_from_ref
        }
    
    return results

def print_summary(results: Dict[str, Dict[str, Any]], 
                  show_unchanged: bool = False):
    """
    Print a summary of the comparison results.
    Uses tags instead of filenames.
    """
    print("\n" + "="*80)
    print("PARAMETER COMPARISON SUMMARY")
    print("="*80)
    
    for param, info in results.items():
        print(f"\n{param}:")
        print(f"  Min: {info['min_value']:.6e} (tag: {info['min_tag']})")
        print(f"  Max: {info['max_value']:.6e} (tag: {info['max_tag']})")
        
        if info['reference_value'] is not None:
            print(f"  Reference: {info['reference_value']:.6e}")
            
            if info['changed_from_reference']:
                print("  Changed from reference in:")
                for tag, change_info in info['changed_from_reference'].items():
                    print(f"    {tag}: {change_info['current']:.6e} "
                          f"(diff: {change_info['difference']:+.6e}, "
                          f"{change_info['relative_change']:+.2f}%)")
            elif show_unchanged:
                print("  ✓ No changes from reference")

def parse_namelist_to_yaml(log_content: str) -> Dict[str, Any]:
    """
    Parse FORTRAN namelist format and convert to nested dictionary for YAML output.
    """
    result = {"tuning": {}}
    
    # Pattern to match namelist blocks: &NAME ... /
    namelist_pattern = r'&([A-Z_]+)\s*(.*?)\s*/'
    namelists = re.findall(namelist_pattern, log_content, re.DOTALL | re.IGNORECASE)
    
    for namelist_name, content in namelists:
        clean_name = namelist_name.lower()
        result["tuning"][clean_name] = {}
        # Pattern to match parameter assignments: PARAM = value
        param_pattern = r'([A-Z0-9_]+)\s*=\s*([^\n,]+)'
        # Find all parameters in this namelist
        params = re.findall(param_pattern, content, re.IGNORECASE)
        
        for param_name, param_value in params:
            param_name = param_name.strip()
            param_value = param_value.strip()
        
            converted_value = float(param_value)
            
            result["tuning"][clean_name][param_name] = converted_value
    
    return result

def extract_and_convert_namelist(log_file_path: str, output_yaml_path: str = None):
    """
    Extract namelist from log file and convert to YAML format.
    """
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    
        yaml_data = parse_namelist_to_yaml(log_content)
        
        if output_yaml_path is None:
            base_name = log_file_path.replace('.log', '').replace('.txt', '').replace('.nam', '')
            output_yaml_path = f"{base_name}.yml"
        
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        
        print(f"Successfully converted namelist to: {output_yaml_path}")
        return yaml_data
        
    except FileNotFoundError:
        print(f"Error: Could not find file {log_file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def convert_string_to_yaml(namelist_string: str) -> str:
    """
    Convert a namelist string directly to YAML format.
    """
    yaml_data = parse_namelist_to_yaml(namelist_string)
    return yaml.dump(yaml_data, default_flow_style=False, indent=2)


def parse_arguments(arguments):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='Calc sensitivity to parameters')

    parser.add_argument('-c', '--config', type=str,
                        help='yaml configuration file')

    # positional
    parser.add_argument('ref_tag', type=str, help='name of reference experiment', nargs='?', default=None)
    parser.add_argument('exp_temp', type=str, help='template name of tuning experiments to use for sensitivity calc (usually something like "s???")', nargs='?', default=None)
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


def find_dirs_from_template(template, base_path):
    """
    Find directories matching a glob template.
    
    Args:
        template (str): Glob pattern like 'lr??', 'model_*', etc.
        base_path (str): Base directory path to search in
    
    Returns:
        list: List of matching exp names
    """
    full_pattern = os.path.join(base_path, template)
    
    matching_paths = glob.glob(full_pattern)
    matching_dirs = [os.path.basename(p) for p in matching_paths if os.path.isdir(p)]

    return matching_dirs


def find_files_from_template(template, base_path):
    """
    Find files matching a glob template.
    
    Args:
        template (str): Glob pattern like 'lr??', 'model_*', etc.
        base_path (str): Base directory path to search in
    
    Returns:
        list: List of matching exp names
    """
    full_pattern = os.path.join(base_path, template)
    
    matching_paths = glob.glob(full_pattern)
    matching_files = [os.path.basename(p) for p in matching_paths if os.path.isfile(p)]

    return matching_files
    

def extract_tag_by_position(template, formatted_string, tag='exp'):
    """
    Extract tag value from formatted string using position-based approach.
    
    Args:
        template (str): Template like "tuning_{exp}.yml"
        formatted_string (str): Actual string like "tuning_lr01.yml"
        tag (str): Tag name to extract (default: 'exp')
    
    Returns:
        str or None: Extracted tag value, or None if extraction fails
    """
    tag_placeholder = f"{{{tag}}}"
    
    start_index = template.find(tag_placeholder)
    if start_index == -1:
        return None  # Tag not found in template
    
    # Find what comes after {exp} in the template
    end_of_tag = start_index + len(tag_placeholder)
    suffix = template[end_of_tag:]
    
    # Remove the suffix from the formatted string to get the tag value
    if suffix:
        # If there's a suffix, remove it from the end
        if formatted_string.endswith(suffix):
            end_index = len(formatted_string) - len(suffix)
            tag_value = formatted_string[start_index:end_index]
        else:
            return None  # Suffix doesn't match
    else:
        # If no suffix, take from start_index to the end
        tag_value = formatted_string[start_index:]
    
    return tag_value


def dicts_equal(d1, d2, tol = 1e-7):
    if d1.keys() != d2.keys():
        print('keys differ!')
        return False
    
    for key in d1:
        v1, v2 = d1[key], d2[key]
        print(key, v1, v2)
        
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not dicts_equal(v1, v2, tol):
                return False
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v1 - v2) > tol:
                return False
        else:
            if v1 != v2:
                return False
    
    return True

######################################################################################


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    config_file = get_arg(args, 'config', 'config-tuner.yaml')
    ref_tag = get_arg(args, 'ref_tag', None)
    exp_temp = get_arg(args, 'exp_temp', None)
    year1 = get_arg(args, 'year1', None)
    year2 = get_arg(args, 'year2', None)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    if not year1:
        year1 = config['args']['year1']
    if not year2:
        year2 = config['args']['year2']
    if not exp_temp:
        exp_temp = config['files']['exp_temp']

    # Directory containing the tuning YAML files
    yaml_dir = config['files']['exps']
    # yaml_template = config['files']['params'].format(exp = exp_temp)
    params_template = config['files']['params'] # Es: tuning_{exp}.yml
    yaml_template_glob = params_template.format(exp = exp_temp)

    tuning_files_paths = [os.path.join(yaml_dir, f) for f in find_files_from_template(yaml_template_glob, yaml_dir)]
    
    if not tuning_files_paths:
        raise ValueError(f"Nessun file trovato in {yaml_dir} con pattern {yaml_template_glob}")

    print("Reading sets of tuning parameters...")
    tunsets = read_yaml_files(tuning_files_paths, params_template)
    allmems = list(tunsets.keys())
    print('All exps found: ', allmems)

    reference_dict = config['reference_parameters'] # could also read these from the control exp
    
    if ref_tag is None:
        for exp in tunsets:
            if dicts_equal(tunsets[exp], reference_dict):
                print(f'{exp} is the reference exp')
                ref_tag = exp
                break
    if ref_tag is None:
        raise ValueError('No reference exp found! Automatic recognition may fail if the reference exp was run with different parameters than specified in the config file. Set it manually running: python sensitivity.py ref_tag -c config.yml')

    if tunsets:
        print("Comparing parameters with reference...")
        pardict = compare_with_reference(tunsets, reference_dict)
        print_summary(pardict, show_unchanged=True)
    else:
        print("No YAML files were successfully loaded.")

    ### Load ecmean indices. Check if ecmean files are there. If not, run ecmean
    print("Reading sets of ecmean indices...")
    res_all = dict()
    for mem in allmems:
        print(f"Processing experiment: {mem}")
        base_file = os.path.join(config['files']['ecmean'], config['files']['base'].format(exp=mem, year1=year1, year2=year2))

        if os.path.isfile(base_file):
            print(f'ecmean yml already there: {base_file}')
        else:
            print(f'ecmean yml {base_file} not found, computing from exp...')
            ecmean_conf_file = config['ecmean']['ecmean_config']
            if not os.path.isfile(ecmean_conf_file):
                raise ValueError(f'Config file for ecmean {ecmean_conf_file} does not exist. Please provide one or change the path in the ectuner config.')
            global_mean(mem, year1=year1, year2=year2, config = ecmean_conf_file)

        with open(base_file, 'r') as file:
            res_all[mem] = yaml.safe_load(file)
    
    # Compute regressions and save sensitivities.
    targets = ['net_toa', 'rsnt', 'rlnt', 'swcf', 'lwcf', 'rsns', 'rlns', 'hfss', 'hfls', 'net_sfc', 'toamsfc'] # could be specified in the config?
    parnames = reference_dict.keys()
    sensitivity = {}

    for parnam in parnames:
        sensitivity[parnam] = {}
        st0 = res_all[ref_tag]
        st1 = res_all[pardict[parnam]['min_tag']]
        st2 = res_all[pardict[parnam]['max_tag']]

        for key1 in targets:
            sensitivity[parnam][key1] = {}

            for key2 in st0[key1]:
                sensitivity[parnam][key1][key2] = {}

                for key3 in st0[key1][key2]:
                    ## metric values
                    am = st1[key1][key2][key3]
                    a0 = st0[key1][key2][key3]
                    ap = st2[key1][key2][key3]

                    ## parameter values
                    vm = pardict[parnam]['min_value']
                    v0 = tunsets[ref_tag][parnam]
                    vp = pardict[parnam]['max_value']

                    x = np.array([vm, v0, vp])
                    y = np.array([am, a0, ap])
                    coefficients = np.polyfit(x, y, 1)
                    sensitivity[parnam][key1][key2][key3] = [float(coefficients[0]),
                                                            float(coefficients[0])*v0]
                
    output_file = config['files']['sensitivity'].format(year1 = year1, year2 = year2)

    # Save the sensitivity dictionary into the YAML file
    with open(output_file, 'w') as file:
        yaml.dump(sensitivity, file)
        print(f'Sentitivities computed! and saved to: {output_file}')
