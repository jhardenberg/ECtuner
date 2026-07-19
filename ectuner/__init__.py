"""
ECtuner module.
(1D and 2D).
"""

__version__ = '1.0.0'

# 1. Core Data Structures
from .libs.config import Config
from .libs.result import TuningResult

from .ectuner import run_1d_tuning, run_2d_tuning

from .libs.tuner import Tuner1D, Tuner2D
from .libs.loader import DataLoader1D, DataLoader2D

# Define the Public API
# __all__ dictates exactly what gets imported when someone runs: `from ectuner import *`
__all__ = [
    '__version__',
    'Config',
    'TuningResult',
    'run_1d_tuning',
    'run_2d_tuning',
    'Tuner1D',
    'Tuner2D',
    'DataLoader1D',
    'DataLoader2D'
]