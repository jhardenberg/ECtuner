"""
Generates a lightweight environment to test ECtuner (1D and 2D)
without requiring real climate model output.
"""
import os
import numpy as np
import xarray as xr
from ruamel.yaml import YAML

def create_tutorial():
    base_dir = "tutorial"
    data_dir = os.path.join(base_dir, "dummy_data")
    nc_dir = os.path.join(data_dir, "nc_files")
    os.makedirs(nc_dir, exist_ok=True)
    yaml = YAML()

    # ==========================================
    # 1. 1D Dummy Data (YAML)
    # ==========================================
    reference = {
        'net_toa': {'obs': {'ALL': {'Global': {'mean': 0.8}}}},
        'rsnt': {'obs': {'ALL': {'Global': {'mean': 240.0}}}},
        'rlnt': {'obs': {'ALL': {'Global': {'mean': -239.2}}}}
    }
    with open(os.path.join(data_dir, "reference_dummy.yml"), "w") as f:
        yaml.dump(reference, f)

    base_exp = {
        'net_toa': {'ALL': {'Global': 1.5}},
        'rsnt': {'ALL': {'Global': 242.0}},
        'rlnt': {'ALL': {'Global': -240.5}}
    }
    with open(os.path.join(data_dir, "base_dummy_exp.yml"), "w") as f:
        yaml.dump(base_exp, f)

    sensitivity = {
        'ENTRORG': {
            'net_toa': {'ALL': {'Global': [-500.0, -0.8]}}, 
            'rsnt': {'ALL': {'Global': [-400.0, -0.6]}},
            'rlnt': {'ALL': {'Global': [100.0, 0.2]}}
        },
        'RVICE': {
            'net_toa': {'ALL': {'Global': [20.0, 0.5]}},
            'rsnt': {'ALL': {'Global': [15.0, 0.3]}},
            'rlnt': {'ALL': {'Global': [5.0, 0.2]}}
        }
    }
    with open(os.path.join(data_dir, "sensitivity_dummy.yaml"), "w") as f:
        yaml.dump(sensitivity, f)

    params = {
        "tuning": {
            "namcumf": {"ENTRORG": 0.00175},
            "namcldp": {"RVICE": 0.13}
        }
    }
    with open(os.path.join(data_dir, "tuning_dummy_exp.yml"), "w") as f:
        yaml.dump(params, f)

    # ==========================================
    # 2. 2D Dummy Data (NetCDF)
    # ==========================================
    # Create a tiny 10x10 grid
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 10)
    
    # Fake Reference Map
    ref_data = np.random.rand(10, 10) * 240.0
    ds_ref = xr.Dataset({'net_toa': (['lat', 'lon'], ref_data)}, coords={'lat': lat, 'lon': lon})
    ds_ref.to_netcdf(os.path.join(nc_dir, "climate_average_net_toa_r10x10.nc"))
    
    # Fake Base Cache Map (model output)
    base_data = ref_data + (np.random.rand(10, 10) * 5.0) # Adds some bias
    ds_base = xr.Dataset({'net_toa': (['lat', 'lon'], base_data)}, coords={'lat': lat, 'lon': lon})
    ds_base.to_netcdf(os.path.join(nc_dir, "base_2d_dummy_exp_2000_2010_r10x10.nc"))
    
    # Fake Sensitivity Map
    slope_data = np.random.randn(1, 2, 10, 10) * 100.0
    r2_data = np.random.rand(1, 2, 10, 10)
    ds_sens = xr.Dataset(
        {
            'slope': (['variable', 'parameter', 'lat', 'lon'], slope_data),
            'r2': (['variable', 'parameter', 'lat', 'lon'], r2_data)
        },
        coords={
            'variable': ['net_toa'],
            'parameter': ['ENTRORG', 'RVICE'],   # <--- Aggiunto RVICE
            'lat': lat,
            'lon': lon
        }
    )
    ds_sens.to_netcdf(os.path.join(nc_dir, "sensitivity_2d_dummy.nc"))
    
    # ==========================================
    # 3. Tutorial Config File
    # ==========================================
    config = {
        'files': {
            'reference': f"{data_dir}/reference_dummy.yml",
            'sensitivity': f"{data_dir}/sensitivity_dummy.yaml",
            'ecmean': data_dir,            # <--- AGGIUNTO PER L'1D
            'base': "base_{exp}.yml",
            'params': "tuning_{exp}.yml",
            'exps': data_dir,
            'output_dir': f"{base_dir}/output",
            
            # 2D specific paths
            'ref_2d_dir': nc_dir,
            'sensitivity_nc': f"{nc_dir}/sensitivity_2d_dummy.nc",
            'base_2d_dir': nc_dir,
            'raw_dir': nc_dir              # <--- AGGIUNTO PER IL 2D
        },
        'args': {
            'penalty': 0.1,
            'inc': 0.2,
            'method': 'dual_annealing'
        },
        'spatial_tuning': {
            'target_grid': 'r10x10',
            'alpha': 0.5,
            'metric': 'l2'
        },
        'reference_parameters': {
            'ENTRORG': 0.00175,
            'RVICE': 0.13
        },
       'parameter_group': {                 
            'namcumf': ['ENTRORG'],
            'namcldp': ['RVICE']
        },
        'weights': {
            'net_toa': 1.0,
            'rsnt': 0.0,
            'rlnt': 0.0
        },
        'weights_region': {'Global': 1.0},
        'weights_season': {'ALL': 1.0}
    }
    
    with open(os.path.join(base_dir, "config_tutorial.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"Tutorial created in './{base_dir}'!")
    print("Run the following commands to test ECtuner:")
    print(f"  [1D Mode]: ectuner 1d -c {base_dir}/config_tutorial.yaml -o {base_dir}/output/tuned_1d.yml dummy_exp 2000 2010")
    print(f"  [2D Mode]: ectuner 2d -c {base_dir}/config_tutorial.yaml -o {base_dir}/output/tuned_2d.yml dummy_exp 2000 2010")

if __name__ == "__main__":
    create_tutorial()