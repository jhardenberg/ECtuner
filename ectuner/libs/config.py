"""
Configuration management for ECtuner.

This module provides the central classes for handling YAML configuration.
"""
import os
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from copy import deepcopy
from typing import Any, Dict, Optional


class Config:
    """
    Manages the ECtuner configuration state.
    
    Allows dynamic reading, modification, and saving of nested YAML files 

    Attributes:
        config (Dict[str, Any]): The internal dictionary holding the configuration.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initializes the configuration, optionally loading from a YAML file.

        Args:
            config_path (str | None): Path to the YAML configuration file.
            **kwargs: Additional key-value pairs to override or inject directly 
            into the 'args' block of the configuration (e.g., exp='phis').
            
        Raises:
            FileNotFoundError: If the provided ``config_path`` does not exist.
        """
        self.config: Dict[str, Any] = {}
        
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    loaded_config = YAML().load(f)
                    # Fail-safe: if the file is empty, yaml returns None
                    self.config = loaded_config if loaded_config is not None else {}
            except YAMLError as e:
                raise ValueError(f"YAML syntax error in {config_path}.\nDetails: {e}")
        

        # Initialize the args block if it does not exist
        if 'args' not in self.config:
            self.config['args'] = {}
            
        # Safely overwrite parameters with arguments passed directly
        for k, v in kwargs.items():
            if v is not None:  # Prevent overwriting valid configs with empty CLI args
                self.config['args'][k] = v

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a value from the configuration using dot-notation.

        Args:
            key_path (str): The dot-separated path to the key 
                (e.g., 'spatial_tuning.alpha').
            default (Any, optional): The value to return if the key is missing.

        Returns:
            Any: The requested value, or the default if the path is invalid.
        """
        keys = key_path.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key_path: str, value: Any) -> None:
        """
        Sets a value in the configuration using dot-notation.
        
        Automatically creates missing nested dictionaries along the path.

        Args:
            key_path (str): The dot-separated path to the key.
            value (Any): The value to assign.
        """
        keys = key_path.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        
    def save(self, filepath: str) -> None:
        """
        Saves the current internal configuration state to a YAML file.

        Args:
            filepath (str): The destination path for the YAML file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            YAML().dump(self.config, f)