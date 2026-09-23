import pytest
from ectuner.libs.config import Config

def test_config_get_and_set():
    """Verify that dot-notation correctly gets, sets, and creates nested dictionaries."""
    config = Config()
    
    # Test set() and get()
    config.set('args.inc', 0.5)
    assert config.get('args.inc') == 0.5
    
    # Test that set() creates missing intermediate dictionaries automatically
    config.set('spatial_tuning.deep.nested.key', 42)
    assert config.get('spatial_tuning.deep.nested.key') == 42
    
    # Test default fallback
    assert config.get('missing.key', 'fallback_value') == 'fallback_value'

def test_config_kwargs_override():
    """Verify that CLI kwargs are safely injected into the args block."""
    config = Config(exp='test_exp', year1=2000)
    
    assert config.get('args.exp') == 'test_exp'
    assert config.get('args.year1') == 2000