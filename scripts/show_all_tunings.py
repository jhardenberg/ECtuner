import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import pandas as pd

def read_tuning_files_from_experiments(base_path, exp_list):
    """Read tuning files from experiment folders based on paths in {exp}.yml files."""
    tuning_data = {}
    
    for exp_name in exp_list:
        # Read the experiment yml file to get the tuning file path
        exp_yml_path = Path(base_path) / exp_name / f"{exp_name}.yml"
        
        if not exp_yml_path.exists():
            print(f"Warning: {exp_yml_path} not found, skipping {exp_name}")
            continue
        
        # Read the file as text to extract tuning_file path
        tuning_file_name = None
        with open(exp_yml_path, 'r') as f:
            for line in f:
                line = line.split('#')[0]
                if 'tuning_file:' in line:
                    # Extract filename from pattern like: tuning_file: !noparse "{{se.cli.cwd}}/tuning-TL63ORCA2_v09_efr.yml"
                    
                    match = re.search(r'tuning_file:.*["\']{{se\.cli\.cwd}}/(.*?\.yml)["\']', line)
                    #match = re.search(r'tuning_file:.*["\'].*/(.*?\.yml)["\']', line)
                    if match:
                        tuning_file_name = match.group(1)
                        break
        
        if not tuning_file_name:
            print(f"Warning: No tuning file found in {exp_yml_path}, using default")
            tuning_file_name = 'templates/tuning-example.yml'
        
        # Construct full path to tuning file (in the experiment folder)
        full_tuning_path = Path(base_path) / exp_name / tuning_file_name
        
        if not full_tuning_path.exists():
            print(f"Warning: {full_tuning_path} not found, skipping {exp_name}")
            continue
        
        # Read tuning file
        with open(full_tuning_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten and store
        params = flatten_tuning_params(data)
        tuning_data[exp_name] = params
    
    return tuning_data


def read_tuning_files(folder_path):
    """Read all tuning_*.yml files and extract name and parameters."""
    tuning_data = {}
    
    # Get all tuning_*.yml files
    pattern = "tuning_*.yml"
    files = sorted(Path(folder_path).glob(pattern))
    
    for filepath in files:
        # Extract name from filename: tuning_{name}.yml
        name = filepath.stem.replace("tuning_", "")
        
        # Read YAML file
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten the nested structure to get all parameter values
        params = flatten_tuning_params(data)
        tuning_data[name] = params
    
    return tuning_data

def flatten_tuning_params(data):
    """Flatten nested dictionary to get all tuning parameters using only last layer keys."""
    params = {}
    
    # Navigate through the structure
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if 'base.context' in item:
            model_config = item['base.context'].get('model_config', {})
            
            # Extract from oifs
            if 'oifs' in model_config and 'tuning' in model_config['oifs']:
                for section, section_params in model_config['oifs']['tuning'].items():
                    if isinstance(section_params, dict):
                        for param, value in section_params.items():
                            # Use only the last layer key
                            params[param] = value
            
            # Extract from nemo
            if 'nemo' in model_config and 'tuning' in model_config['nemo']:
                for domain in model_config['nemo']['tuning']:
                    for section, section_params in model_config['nemo']['tuning'][domain].items():
                        if isinstance(section_params, dict):
                            for param, value in section_params.items():
                                # Use only the last layer key
                                params[param] = value
    
    return params

def get_all_parameters(tuning_data):
    """Get all unique parameter keys across all experiments."""
    all_params = set()
    for params in tuning_data.values():
        all_params.update(params.keys())
    return all_params

def fill_missing_with_defaults(tuning_data, default_values):
    """Fill missing parameters with default values."""
    all_params = get_all_parameters(tuning_data)
    
    filled_data = {}
    for name, params in tuning_data.items():
        for par in default_values:
            if par not in params:
                #print(name, par)
                params[par] = default_values[par]

        filled_data[name] = params
    
    return filled_data

def remove_empty_keys(tuning_data):
    """Remove parameters that have no values in any experiment."""
    # Find parameters that have at least one value
    non_empty_params = set()
    for params in tuning_data.values():
        for key, value in params.items():
            if value is not None and value != '':
                non_empty_params.add(key)
    
    # Filter data to keep only non-empty parameters
    cleaned_data = {}
    for name, params in tuning_data.items():
        cleaned_params = {k: v for k, v in params.items() if k in non_empty_params}
        cleaned_data[name] = cleaned_params
    
    return cleaned_data

def remove_duplicates(tuning_data):
    """Remove duplicate configurations, keeping first alphabetically."""
    # Convert parameter dicts to hashable format for comparison
    unique_configs = {}
    
    for name in sorted(tuning_data.keys()):
        params = tuning_data[name]
        # Create a hashable representation
        params_tuple = tuple(sorted(params.items()))
        
        if params_tuple not in unique_configs:
            unique_configs[params_tuple] = name
    
    # Keep only unique configurations
    cleaned_data = {name: tuning_data[name] for name in unique_configs.values()}
    
    return cleaned_data

# def plot_scatter(tuning_data):
#     """Create scatter plot with parameters on x-axis and experiment names on y-axis."""
#     if not tuning_data:
#         print("No data to plot")
#         return
    
#     # Get all unique parameter names
#     all_params = set()
#     for params in tuning_data.values():
#         all_params.update(params.keys())
#     all_params = sorted(all_params)
    
#     if not all_params:
#         print("No parameters to plot")
#         return
    
#     # Get all experiment names
#     exp_names = sorted(tuning_data.keys())
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(max(12, len(all_params) * 0.8), max(8, len(exp_names) * 0.5)))
    
#     # Plot each parameter value
#     for exp_idx, exp_name in enumerate(exp_names):
#         params = tuning_data[exp_name]
#         for param_idx, param_name in enumerate(all_params):
#             if param_name in params and params[param_name] is not None:
#                 value = params[param_name]
#                 ax.scatter(param_idx, exp_idx, s=100, alpha=0.6)
#                 # Add value as text
#                 ax.text(param_idx, exp_idx, f'{value:.2e}', 
#                        fontsize=6, ha='center', va='center')
    
#     # Set labels and formatting
#     ax.set_xticks(range(len(all_params)))
#     ax.set_xticklabels(all_params, rotation=45, ha='right')
#     ax.set_yticks(range(len(exp_names)))
#     ax.set_yticklabels(exp_names)
    
#     ax.set_xlabel('Parameters', fontsize=12)
#     ax.set_ylabel('Experiment Names', fontsize=12)
#     ax.set_title('Tuning Parameters Scatter Plot', fontsize=14)
    
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

default_values = {
"RPRCON": 0.14E-02,
"ENTRORG": 0.175E-02,
"DETRPEN": 0.75E-04,
"ENTRDD": 0.3E-03,
"RMFDEPS": 0.3,
"RVICE": 0.13,
"RLCRITSNOW": 2E-05,
"RSNOWLIN2": 0.3E-01,
"RCLDIFF": 0.3E-05,
"RCLDIFF_CONVI": 10.0,
"RDEPLIQREFRATE": 0.5,
"RDEPLIQREFDEPTH": 500.0,
"RCL_OVERLAPLIQICE": 0.65,
"RCL_INHOMOGAUT": 1.5,
"RCL_INHOMOGACC": 3,
"RMINICE": 60,
"RMINCDNC": 20.0,
"RSIGMA_W": 0.8
}

default_values.update({'nn_etau': 0, 'rn_lc': 0.2})

def plot_scatter(tuning_data, defaults=default_values, colors = None):
    """
    Create scatter plot with parameters on x-axis and relative change from
    default on y-axis. Each experiment is a different color.
    
    Parameters:
    -----------
    tuning_data : pd.DataFrame
        DataFrame with parameters in columns and exp_names in index.
    defaults : dict
        Dictionary of default parameter values.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    if tuning_data.empty:
        print("No data to plot")
        return
    
    if defaults is None:
        print("No defaults provided")
        return
    
    # Get parameter names
    all_params = sorted(tuning_data.columns.tolist())
    
    if not all_params:
        print("No parameters to plot")
        return
    
    # Calculate relative change
    tuning_data_rel = tuning_data.copy()
    for param in all_params:
        if param in defaults and defaults[param] != 0:
            tuning_data_rel[param] = tuning_data[param] / defaults[param]
        else:
            tuning_data_rel[param] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_params) * 0.8), 8))
    
    # Plot each experiment with different color
    exp_names = tuning_data_rel.index.tolist()
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
                  color=colors[exp_idx], label=exp_name, zorder=3)
    
    # Add reference line at y=1 (default)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set labels and formatting
    ax.set_xticks(range(len(all_params)))
    ax.set_xticklabels(all_params, rotation=45, ha='right')
    ax.set_ylabel('Relative value (with respect to default)', fontsize=12)
    ax.set_xlabel('Parameters', fontsize=12)
    #ax.set_title('Parameter values relative to default', fontsize=14)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(df, default_values = default_values):
    """Create heatmap with normalized values."""
    # Create DataFrame with numeric values
    
    # Normalize each column (parameter) to [0, 1]
    #df_normalized = (df - df.min()) / (df.max() - df.min())
    df_normalized = (df-default_values)/default_values

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.8), max(4, len(df) * 0.5)))
    
    # Plot heatmap
    im = ax.imshow(df_normalized.values, aspect='auto', cmap='RdBu_r', vmin = -0.5, vmax = 0.5)
    
    # Set ticks and labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Relative change')
    
    # Add text annotations with original values
    for i in range(len(df)):
        for j in range(len(df.columns)):
            value = df.iloc[i, j]
            text = ax.text(j, i, f'{value:.2e}',
                          ha="center", va="center", color="white", fontsize=8)
    
    #ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    #ax.set_title('Tuning Parameters Heatmap (Normalized)', fontsize=14)
    
    plt.tight_layout()
    plt.show()


# default_nemo_tuning = {
#     'nn_etau': 1,
#     'nn_efr': 0.05,
# }


def main(folder_path, exp_list = None, remove_duplicates = False, output_df = True, default_values = default_values):    
    # Define default values (you can modify this or load from a default file)

    # Read all tuning files
    print("Reading tuning files...")
    if exp_list is not None:
        print('Reading from exp tuning set')
        tuning_data = read_tuning_files_from_experiments(folder_path, exp_list = exp_list)
    else:
        tuning_data = read_tuning_files(folder_path)
    print(f"Found {len(tuning_data)} tuning configurations")
    
    # Remove empty keys
    print("Removing empty keys...")
    tuning_data = remove_empty_keys(tuning_data)
    
    if remove_duplicates:
        # Remove duplicates
        print("\nRemoving duplicates...")
        tuning_data = remove_duplicates(tuning_data)
        print(f"After cleaning: {len(tuning_data)} unique configurations")

    # Fill missing parameters with defaults
    print("\nFilling missing parameters with defaults...")
    tuning_data = fill_missing_with_defaults(tuning_data, default_values)

    # Print summary
    # print("\nUnique configurations:")
    # for name, params in tuning_data.items():
    #     print(f"  {name}: {len(params)} parameters")

    # Create scatter plot
    #print("\nCreating scatter plot...")
    #plot_scatter(cleaned_data)
    if output_df:
        df = pd.DataFrame(tuning_data).T
        return df
    else:
        return tuning_data
    

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read and plot tuning files')
    parser.add_argument('folder_path', type=str, help='Path to folder containing tuning_*.yml files')
    args = parser.parse_args()
    
    folder_path = args.folder_path
    main(folder_path)