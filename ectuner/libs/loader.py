"""
Data Loading Operations for ECtuner (1D and 2D).

This module provides the data ingestion layer, handling the extraction of 
tuning parameters, sensitivity matrices, and reference observations from 
YAML and NetCDF files.
"""
import os
from ruamel.yaml import YAML
import glob
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import xarray as xr 

from .utils import compute_derived_flux, regrid_to_regular_smm_safe, standardize_reference_signs

FluxDict = Dict[str, Dict[str, Dict[str, float]]]


class BaseDataLoader(ABC):
    """
    Abstract Base Class for ECtuner Data Loaders.

    Establishes the common interface and shared state for reading model parameters 
    and target observations. 

    Attributes:
        config (Config): The ECtuner configuration object.
        exp (str): The name of the experiment being tuned.
        year1 (int): The start year of the analysis period.
        year2 (int): The end year of the analysis period.
    """
    def __init__(self, config: Any) -> None:
        """
        Initializes the shared state for the DataLoader.

        Raises:
            ValueError: If 'exp', 'year1', or 'year2' are missing from the configuration.
        """
        self.config = config
        self.exp = self.config.get('args.exp')
        self.year1 = self.config.get('args.year1')
        self.year2 = self.config.get('args.year2')

        if not self.exp or not self.year1 or not self.year2:
            raise ValueError("DataLoader requires 'exp', 'year1', and 'year2' in the configuration.")

    def load_params(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Loads the physical parameter file of the configuration to tune.

        Features robust extraction supporting both the Script Engine (SE) format 
        and legacy flattened dictionaries. It dynamically scans across multiple 
        model components (``oifs``, ``nemo``, etc.) to extract tuning variables.

        Returns:
            Tuple containing:
                - **list[str]**: A list of all parameter names.
                - **dict[str, float]**: A dictionary mapping parameter names to their current values.

        Raises:
            KeyError: If the experiments directory or params template are missing.
            FileNotFoundError: If the target parameter YAML file does not exist.
            ValueError: If one or more extracted parameter values cannot be cast to a float.
        """
        exps_dir = self.config.get('files.exps')
        params_template = self.config.get('files.params')
        
        if not exps_dir or not params_template:
            raise KeyError("Missing 'files.exps' or 'files.params' in configuration.")

        param_file = os.path.join(exps_dir, params_template.format(exp=self.exp))

        if not os.path.exists(param_file):
            raise FileNotFoundError(f"Tuning parameter file not found: {param_file}")

        with open(param_file, 'r') as file:
            raw_data = YAML().load(file)

        params: Dict[str, Any] = {}
        
        if isinstance(raw_data, list) and len(raw_data) > 0 and 'base.context' in raw_data[0]:
            model_config = raw_data[0]['base.context'].get('model_config', {})

            if 'oifs' in model_config and 'tuning' in model_config['oifs']:
                for namelist, namelist_params in model_config['oifs']['tuning'].items():
                    if isinstance(namelist_params, dict):
                        params.update(namelist_params)
                        
            if 'nemo' in model_config and 'tuning' in model_config['nemo']:
                for domain, domain_params in model_config['nemo']['tuning'].items():
                    if isinstance(domain_params, dict):
                        params.update(domain_params)
        else:
            if 'tuning' in raw_data:
                for section in raw_data['tuning'].values():
                    if isinstance(section, dict):
                        params.update(section)
            else:
                params = raw_data

        try:
            float_params = {k: float(v) for k, v in params.items()}
        except ValueError as e:
            raise ValueError(f"Could not cast all parameters to float in {param_file}. Details: {e}")

        return list(float_params.keys()), float_params
    
    @abstractmethod
    def load_sensitivity(self) -> Any:
        """
        Loads the pre-computed sensitivity data.
        
        Returns:
            The sensitivity data structure (format depends on the 1D/2D implementation).
        """
        pass

    @abstractmethod
    def load_reference(self, *args, **kwargs) -> Any:
        """
        Loads the reference target observations.
        
        Returns:
            The observation data structure (format depends on the 1D/2D implementation).
        """
        pass

    @abstractmethod
    def load_base(self, *args, **kwargs) -> Any:
        """
        Loads the base model fluxes.

        Returns:
            The base model flux data structure (format depends on the 1D/2D implementation).
        """
        pass

class DataLoader1D(BaseDataLoader):
    """
    DataLoader for 1D tuning operations.
    
    Reads scalar metrics, sensitivity maps, and parameter configurations 
    from YAML files, standardizing them into nested dictionaries.

    Attributes:
        logger (logging.Logger): The logger instance for tracking I/O operations.
    """

    def __init__(self, config: Any, logger: Any) -> None:
        super().__init__(config)
        self.logger = logger

    def load_sensitivity(self) -> Dict[str, Any]:
        """
        Loads the externally computed 1D sensitivity file.

        Returns:
            A nested dictionary containing the sensitivity regression coefficients.
        
        Raises:
            KeyError: If the 'files.sensitivity' template is missing in the config.
            FileNotFoundError: If the generated sensitivity file path does not exist.
        """
        sens_template = self.config.get('files.sensitivity')
        if not sens_template:
            raise KeyError("Missing 'files.sensitivity' in configuration.")
            
        sens_file = sens_template.format(year1=self.year1, year2=self.year2)
        
        if not os.path.exists(sens_file):
            raise FileNotFoundError(f"Sensitivity file not found: {sens_file}")

        with open(sens_file, 'r') as file:
            return YAML().load(file)

    def load_reference(self) -> FluxDict:
        """
        Loads and standardizes the 1D reference target observation fluxes.

        Transforms the raw observation YAML data into a perfectly symmetric 
        nested structure matching the model output format.

        Returns:
            A structured dictionary of the reference targets.
            Format: {'var': {'season': {'region': value}}}.

        Raises:
            FileNotFoundError: If the reference file is missing or its path is undefined.
        """
        ref_file = self.config.get('files.reference')
        if not ref_file or not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference file not found or path missing: {ref_file}")

        with open(ref_file, 'r') as file:
            ref_raw = YAML().load(file)

        reference: FluxDict = {}
        
        for var_name in ref_raw.keys():
            obs_data = ref_raw[var_name].get('obs')
            if isinstance(obs_data, dict):
                for season in obs_data:
                    for region in obs_data[season]:
                        # Extract the mean scalar value
                        obs_data[season][region] = obs_data[season][region]['mean']
            else:
                # Fallback for simple global scalars
                obs_data = {'ALL': {'Global': obs_data}}

            reference[var_name] = obs_data
            
        return reference

    def load_base(self) -> FluxDict:
        """
        Loads the 1D base fluxes of the configuration to be tuned.

        If the base YAML file is not found, attempts to compute the global means 
        on the fly invoking the ECmean library.

        Returns:
            A structured dictionary of the ECmean base fluxes.
            Format: {'var': {'season': {'region': value}}}.

        Raises:
            KeyError: If the ECmean directory or base file template are missing.
            ValueError: If on-the-fly computation is needed but 'ecmean_config' is missing.
            ImportError: If the 'ecmean' package is not installed in the environment.
            RuntimeError: If the ECmean external execution fails.
            FileNotFoundError: If the base file remains missing even after ECmean execution.
        """
        ecmean_dir = self.config.get('files.ecmean')
        base_template = self.config.get('files.base')
        
        if not ecmean_dir or not base_template:
            raise KeyError("Missing 'files.ecmean' or 'files.base' in configuration.")

        base_file = os.path.join(
            ecmean_dir, 
            base_template.format(exp=self.exp, year1=self.year1, year2=self.year2)
        )

        if not os.path.exists(base_file):
            self.logger.warning(f"Base ECmean file not found: {base_file}")
            self.logger.info("Attempting to compute Global Mean on the fly using ECmean...")
            
            # ecmean_config_path is required for on-the-fly computation
            ecmean_config_path = self.config.get('files.ecmean_config')
            if not ecmean_config_path:
                raise ValueError("To compute base data on the fly, 'files.ecmean_config' must be specified in the YAML.")
                
            try:
                # Import the ECmean wrapper function
                from ecmean.global_mean import global_mean
                
                # Esecuzione del wrapper
                global_mean(
                    exp=self.exp,
                    year1=self.year1,
                    year2=self.year2,
                    config=ecmean_config_path
                )
                self.logger.info("ECmean computation completed successfully.")
            except ImportError:
                raise ImportError("The 'ecmean' package is required to compute base fields on the fly but is not installed.")
            except Exception as e:
                raise RuntimeError(f"ECmean execution failed with error: {e}")

        # double-check if the base file was created after ECmean execution
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"ECmean executed, but expected file was not created: {base_file}")

        with open(base_file, 'r') as file:
            return YAML().load(file)


class DataLoader2D(BaseDataLoader):
    """
    DataLoader for 2D spatial tuning operations.
    
    Overrides flux loading mechanisms to handle multidimensional xarray Datasets 
    and implements a NetCDF caching mechanism for heavy I/O operations.

    Attributes:
        logger (logging.Logger): The logger instance.
        target_grid (str): The common spatial grid resolution used for regridding.
    """

    def __init__(self, config: Any, logger: Any) -> None:
        
        super().__init__(config)
        self.logger = logger
        self.target_grid = self.config.get('spatial_tuning.target_grid', 'r180x90')

    def load_sensitivity(self) -> xr.Dataset:
        """
        Loads the 2D spatial sensitivity NetCDF file.

        Returns:
            The xarray Dataset containing slope and R2 maps for all parameters.

        Raises:
            FileNotFoundError: If the sensitivity file does not exist.
        """
        sens_file = self.config.get('files.sensitivity_nc')
        if not sens_file or not os.path.exists(sens_file):
            raise FileNotFoundError(f"2D Sensitivity file not found: {sens_file}")

        self.logger.info(f"Loading 2D sensitivity from: {sens_file}")
        return xr.open_dataset(sens_file)

    def load_reference(self, variables: List[str]) -> Dict[str, xr.DataArray]:
        """
        Loads reference climatology maps (observations) for target variables.

        Standardizes the observation signs (e.g., UP vs DOWN) to match the 
        model's internal conventions.

        Args:
            variables: List of variable names to load.

        Returns:
            Dictionary mapping variable names to their standardized observation arrays.

        Raises:
            KeyError: If the reference 2D directory is missing in the config.
        """
        ref_maps = {}
        ref_dir = self.config.get('files.ref_2d_dir')
        
        if not ref_dir:
            raise KeyError("Missing 'files.ref_2d_dir' in configuration.")

        for var in variables:
            pattern = os.path.join(ref_dir, f"climate_average_{var}_*{self.target_grid}*.nc")
            matching_files = sorted(glob.glob(pattern))
            
            if not matching_files:
                self.logger.warning(
                    f"Missing observation map for '{var}'. Skipping it from 2D targets/diagnostics."
                )
                continue
            
            f_obs = matching_files[0]
            self.logger.info(f"Loading reference for {var} from: {os.path.basename(f_obs)}")
            
            ds = xr.open_dataset(f_obs)
            data = ds[var].mean('time') if 'time' in ds.dims else ds[var]

            ref_maps[var] = data
            
        ref_maps_standardized = standardize_reference_signs(ref_maps)

        return ref_maps_standardized

    def load_base(self, variables: List[str]) -> Dict[str, xr.DataArray]:
        """
        Extracts, time-averages, and regrids raw OIFS output maps.
        
        Implements a caching mechanism: if the processed NetCDF exists for the 
        given experiment and years, it loads it directly avoiding expensive 
        regridding operations.

        Args:
            variables: Variables to extract from the raw model output.

        Returns:
            Dictionary mapping variable names to the regridded model base arrays.

        Raises:
            KeyError: If the base 2D directory is missing in the config.
            FileNotFoundError: If cache is missed and raw OIFS files are not found.
        """
        out_dir = self.config.get('files.base_2d_dir')
        if not out_dir:
            raise KeyError("Missing 'files.base_2d_dir' in configuration.")
            
        os.makedirs(out_dir, exist_ok=True)
        cache_file = os.path.join(out_dir, f"base_2d_{self.exp}_{self.year1}_{self.year2}.nc")

        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached base maps for {self.exp} from {cache_file}")
            ds_base = xr.open_dataset(cache_file)
            return {var: ds_base[var] for var in variables if var in ds_base}

        self.logger.info(f"Cache not found. Extracting 2D maps for {self.exp} (this may take a while...)")
        raw_dir = self.config.get('files.raw_dir')
        raw_pattern = os.path.join(raw_dir, self.exp, f"output/oifs/{self.exp}_atm_cmip6_1m_*.nc")
        
        filz = sorted(glob.glob(raw_pattern))
        if not filz:
            raise FileNotFoundError(f"No raw OIFS files found in {raw_pattern}")

        ds = xr.open_mfdataset(
            filz, chunks={'time_counter': 12}, combine='nested', 
            concat_dim='time_counter', compat='override', 
            coords='minimal', data_vars='minimal'
        )
        
        if 'time_counter' in ds.dims: 
            ds = ds.rename({'time_counter': 'time'})

        extracted_vars = []
        sample_file = filz[0]
        methods_map = self.config.get('spatial_tuning.methods_map', {})

        for var in variables:
            self.logger.info(f"Processing and regridding {var}...")
            y_raw = compute_derived_flux(ds, var).sel(time=slice(str(self.year1), str(self.year2))).mean('time').compute()
            
            regrid_method = methods_map.get(var, 'ycon')
            y_reg = regrid_to_regular_smm_safe(
                y_raw.to_dataset(name=var), 
                self.target_grid, 
                method=regrid_method,
                raw_file_path=sample_file
            )
            extracted_vars.append(y_reg)

        ds_base = xr.merge(extracted_vars)
        ds_base.to_netcdf(cache_file)
        self.logger.info(f"Cache saved successfully: {cache_file}")
        
        return {var: ds_base[var] for var in variables}
    