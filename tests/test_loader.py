import os
import pytest
from unittest.mock import patch, MagicMock
from ruamel.yaml import YAML
from ectuner.libs.config import Config
from ectuner.libs.loader import DataLoader1D

@pytest.fixture
def dummy_config(tmp_path):
    config = Config()
    config.set('args.exp', 'test_exp')
    config.set('args.year1', 1990)
    config.set('args.year2', 2000)
    config.set('files.exps', str(tmp_path))
    config.set('files.params', 'tuning_{exp}.yml')
    config.set('files.reference', str(tmp_path / 'reference.yml'))
    config.set('files.ecmean', str(tmp_path))
    config.set('files.base', 'base_{exp}.yml')
    config.set('files.ecmean_config', 'dummy_ecmean_config.yml')
    return config

def test_load_params_script_engine_format(dummy_config, tmp_path, dummy_logger):
    """
    Check that the loader can correctly extract parameters 
    if the YAML file is in the nested Script Engine (SE) format.
    """
    
    param_file = tmp_path / "tuning_test_exp.yml"
    se_data = [{
        "base.context": {
            "model_config": {
                "oifs": {
                    "tuning": {
                        "namcumf": {"ENTRORG": 0.00175, "DETRPEN": 7.5e-05}
                    }
                }
            }
        }
    }]
    
    with open(param_file, 'w') as f:
        YAML().dump(se_data, f)
        
    loader = DataLoader1D(dummy_config, dummy_logger)
    param_names, current_values = loader.load_params()
    
    assert "ENTRORG" in param_names
    assert "DETRPEN" in param_names
    assert current_values["ENTRORG"] == 0.00175
    assert current_values["DETRPEN"] == 7.5e-05

def test_load_reference_standardization(dummy_config, tmp_path, dummy_logger):
    """
    Check that the loader standardizes observations by extracting the 'mean' key
    from the nested dictionary.
    """
    ref_file = tmp_path / "reference.yml"
    ref_data = {
        "net_toa": {
            "obs": {
                "ALL": {
                    "Global": {"mean": 5.0, "std": 0.1},
                    "NH": {"mean": 2.0, "std": 0.2}
                }
            }
        }
    }
    with open(ref_file, 'w') as f:
        YAML().dump(ref_data, f)
        
    loader = DataLoader1D(dummy_config, dummy_logger)
    reference = loader.load_reference()
    
    assert reference['net_toa']['ALL']['Global'] == 5.0
    assert reference['net_toa']['ALL']['NH'] == 2.0

from unittest.mock import patch, MagicMock

@patch('ectuner.libs.loader.os.path.exists')
@patch('builtins.open')
def test_load_base_ecmean_fallback(mock_open, mock_exists, dummy_config, dummy_logger):
    """
    Check the fallback mechanism: if the base file does not exist,
    the loader should attempt to invoke ecmean.global_mean.
    """
    call_state = {'count': 0}
    def smart_exists(path):
        if "base_test_exp.yml" in str(path):
            call_state['count'] += 1
            return call_state['count'] > 1 
        return True
        
    mock_exists.side_effect = smart_exists
    
    loader = DataLoader1D(dummy_config, dummy_logger)
    
    mock_global_mean_func = MagicMock()
    mock_ecmean_module = MagicMock()
    mock_ecmean_module.global_mean = mock_global_mean_func
    
    with patch.dict('sys.modules', {'ecmean.global_mean': mock_ecmean_module}):
        with patch('ruamel.yaml.YAML.load', return_value={'mocked': 'data'}):
            
            result = loader.load_base()
            
            mock_global_mean_func.assert_called_once_with(
                exp='test_exp',
                year1=1990,
                year2=2000,
                config='dummy_ecmean_config.yml'
            )
            assert result == {'mocked': 'data'}