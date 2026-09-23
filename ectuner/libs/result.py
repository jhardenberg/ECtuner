"""
Tuning Results Management.

Provides the TuningResult class, responsible for storing optimization metrics in memory
"""
from typing import Dict, List, Any, Optional, Tuple


class TuningResult:
    """
    Collects, formats, and saves optimization results.

    Maintains all metrics in memory for programmatic access and provides
    methods to export data for the EC-Earth Script Engine and data analysis.

    Attributes:
        target_vars: Variables targeted during optimization.
        param_names: All parameter names evaluated.
        optimal_changes: Absolute changes applied to parameters.
        initial_values: Starting values of the parameters.
        ref_values: Default reference values.
        bounds: Min/max limits for parameters.
        frozen_params: Parameters locked out of optimization.
        metrics: Global metadata and final score metrics.
        var_metrics: Metrics broken down by variable.
        bias_evaluation: Initial vs Final bias comparisons.
    """

    def __init__(
        self, 
        target_vars: List[str], 
        param_names: List[str], 
        optimal_changes: Dict[str, float], 
        initial_values: Dict[str, float], 
        ref_values: Dict[str, float], 
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        frozen_params: Optional[Dict[str, float]] = None
    ) -> None:
        self.target_vars = target_vars
        self.param_names = param_names
        
        # Parameter data
        self.optimal_changes = optimal_changes 
        self.initial_values = initial_values   
        self.ref_values = ref_values           
        self.bounds = bounds if bounds else {} 
        self.frozen_params = frozen_params if frozen_params else {}

        # Post-optimization overall metrics
        self.metrics: Dict[str, Any] = {
            'metric_name': 'unknown',
            'score_init': 0.0,
            'total_global_cost': 0.0,
            'total_spatial_cost': 0.0,
            'alpha': None,
            'penalty': None,
            'inc': None
        }
        
        # Variable-specific metrics (populated by Tuner2D)
        self.var_metrics: Dict[str, Dict[str, float]] = {}
        
        # Detailed bias metrics (populated by Tuner1D)
        self.bias_evaluation: Dict[str, List[Dict[str, Any]]] = {}

    def set_var_metrics(self, var_name: str, predicted_global_bias: float, spatial_cost: float, global_cost: float) -> None:
        """
        Stores the predicted metrics for a specific variable.
        
        Args:
            var_name: The name of the target variable.
            predicted_global_bias: The expected new global bias.
            spatial_cost: The spatial cost component.
            global_cost: The global cost component.
        """
        self.var_metrics[var_name] = {
            'predicted_global_bias': predicted_global_bias,
            'spatial_cost': spatial_cost,
            'global_cost': global_cost
        }

    def get_predicted_global_bias(self, var_name: str) -> Optional[float]:
        """Programmatically retrieve the predicted bias for a variable."""
        return self.var_metrics.get(var_name, {}).get('predicted_global_bias')

    def get_spatial_cost(self, var_name: str) -> Optional[float]:
        """Programmatically retrieve the spatial cost for a variable."""
        return self.var_metrics.get(var_name, {}).get('spatial_cost')

    def get_new_parameters(self) -> Dict[str, float]:
        """
        Returns a dictionary with the new absolute parameter values.
        Safely falls back to initial values if an optimal change is missing.
        """
        return {
            p: self.frozen_params[p] if p in self.frozen_params
                else self.initial_values[p] + self.optimal_changes.get(p, 0.0) 
            for p in self.param_names
        }
    