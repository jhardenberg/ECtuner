"""
Export and Presentation Module for ECtuner.

Handles logging summaries and writing model-compatible configurations to disk.
"""
import os
import numpy as np
from tabulate import tabulate
from ruamel.yaml import YAML
from typing import Dict, List, Any

# Import only for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .result import TuningResult


def print_summary(result: 'TuningResult', logger: Any) -> None:
    """
    Prints the parameter summary table and bias evaluation in the log.

    Args:
        result: The TuningResult object containing optimization data.
        logger: The configured logger instance.
    """
    outtable = []
    new_params = result.get_new_parameters()
    
    for p in result.param_names:
        old_val = result.initial_values[p]
        new_val = new_params[p]
        change = result.optimal_changes[p]
        
        # Safe relative calculations
        rel_change = change / old_val if old_val != 0 else 0.0
        dist_ref = (new_val - result.ref_values[p]) / result.ref_values[p] if result.ref_values[p] != 0 else 0.0
        
        min_c, max_c = result.bounds.get(p, (None, None))
        outtable.append([p, new_val, old_val, change, rel_change, min_c, max_c, dist_ref])

    head = ['Parameter', 'New value', 'Old value', 'Change', 'Rel. change', 'Min change', 'Max change', 'Dist. from ref.']
    
    logger.info("\nParameters Optimization Result:")
    logger.info("-" * 80)
    table_str = tabulate(outtable, headers=head, floatfmt=".4e", stralign='center', tablefmt='orgtbl')
    for line in table_str.split('\n'):
        logger.info(line)
    logger.info("-" * 80 + "\n")

    if result.bias_evaluation:
        _print_biases(result, logger)


def _print_biases(result: 'TuningResult', logger: Any) -> None:
    """Internal helper to print the colorized bias tables."""
    logger.info("\n" + " OPTIMIZATION SUMMARY (Biases: Model - Target) ".center(90, "="))
    logger.info("Goal: Bring Biases to 0.0\n")

    targets = result.bias_evaluation.get('targets', [])
    diagnostics = result.bias_evaluation.get('diagnostics', [])

    if targets:
        logger.info(" PRIMARY TUNING TARGETS ".center(90, "-"))
        _print_bias_table(logger, targets)

    if diagnostics:
        logger.info("\n" + " DIAGNOSTIC SIDE-EFFECTS ".center(90, "-"))
        _print_bias_table(logger, diagnostics)

    logger.info("=" * 90 + "\n")

    
def _print_bias_table(logger: Any, data: List[Dict[str, Any]]) -> None:
    """Internal helper to format a single bias table."""
    header = f"{'Variable':<12} | {'Season':<6} | {'Region':<12} | {'Weight':<6} | {'Bias Init':>10} -> {'Bias Final':>10} | {'Status'}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in data:
        color = "\033[92m" if r['status'] == "IMPROVED" else "\033[91m"
        reset = "\033[0m"
        logger.info(
            f"{r['variable']:<12} | {r['season']:<6} | {r['region']:<12} | "
            f"{r['weight']:<6.2f} | {r['bias_init']:>10.3f} -> {color}{r['bias_final']:>10.3f}{reset} | {r['status']}"
        )


def save_model_yaml(result: 'TuningResult', filepath: str, parameter_group_config: Dict[str, List[str]], weights_flux: Dict[str, float], weights_region: Dict[str, float]) -> None:
    """
    Writes the YAML file in the format required by the OIFS Script Engine.

    Args:
        result: The TuningResult object containing optimization data.
        filepath: Destination path for the output YAML file.
        parameter_group_config: Mapping of group names (e.g., 'NAMCUMF') to parameter lists.
        weights_flux: Variables and their corresponding weights.
        weights_region: Regions and their corresponding weights.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    yaml_ru = YAML(typ="rt")  
    yaml_ru.indent(mapping=2, sequence=2, offset=0)
    yaml_ru.preserve_quotes = True

    new_params = result.get_new_parameters()
    tuning_block = {}
    
    for pg_name, param_list in parameter_group_config.items():
        current_group = {}
        for p in param_list:
            if p in new_params:
                current_group[p] = float(f"{new_params[p]:.4e}")
        
        if current_group:
            tuning_block[pg_name] = current_group

    full_structure = [{"base.context": {"model_config": {"oifs": {"tuning": tuning_block}}}}]

    with open(filepath, "w") as f:
        yaml_ru.dump(full_structure, f)
        
        # Metadata as comments
        f.write("\n# --- ECtuner Meta-Diagnostics ---\n")
        f.write(f"# metric_used: {result.metrics.get('metric_name')}\n")
        f.write(f"# score_init: {result.metrics.get('score_init'):.8f}\n")
        f.write(f"# total_spatial_cost: {result.metrics.get('total_spatial_cost'):.8f}\n")
        f.write(f"# total_global_cost: {result.metrics.get('total_global_cost'):.8f}\n")
        f.write(f"# alpha: {result.metrics.get('alpha')} | penalty: {result.metrics.get('penalty')} | inc: {result.metrics.get('inc')}\n")
        
        f.write("# weights (flux):\n")
        for k, v in weights_flux.items():
            f.write(f"#   {k}: {v}\n")
            
        if weights_region:
            f.write("# weights (region):\n")
            for k, v in weights_region.items():
                f.write(f"#   {k}: {v}\n")
                
        for var_name, stats in result.var_metrics.items():
            f.write(f"# {var_name}_predicted_global_bias: {stats.get('predicted_global_bias'):.6f}\n")
            f.write(f"# {var_name}_spatial_cost: {stats.get('spatial_cost'):.6f}\n")

    print(f"Structured tuning YAML written to: {filepath}")

    
def _adjustment_for_yaml(data: Any) -> Any:
    """Recursively converts numpy data types to native Python types for YAML serialization."""
    import numpy as np
    if isinstance(data, dict):
        return {k: _adjustment_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_adjustment_for_yaml(v) for v in data]
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return _adjustment_for_yaml(data.tolist())
    return data


def save_diagnostics_yaml(result: 'TuningResult', filepath: str) -> None:
    """
    Exports the detailed optimization results to a YAML file for data analysis.

    Args:
        result: The TuningResult object containing optimization data.
        filepath: Destination path for the diagnostic YAML file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    new_params = result.get_new_parameters()
    params_list = []
    for p in result.param_names:
        old_val = result.initial_values[p]
        new_val = new_params[p]
        change_abs = new_val - old_val
        change_rel = change_abs / old_val if old_val != 0 else 0.
        params_list.append({
            'name': p,
            'old_value': float(old_val),
            'new_value': float(new_val),
            'change_abs': float(change_abs),
            'change_rel': float(change_rel),
            'ref_value': float(result.ref_values[p])
        })

    raw_diagnostic_data = {
        'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in result.metrics.items()},
        'parameters': params_list,
        'biases': result.bias_evaluation
    }

    diagnostic_data = _adjustment_for_yaml(raw_diagnostic_data)

    yaml_ru = YAML(typ="rt")
    yaml_ru.default_flow_style = False
    with open(filepath, 'w') as f:
        yaml_ru.dump(diagnostic_data, f)
        
    print(f"Diagnostic YAML written to: {filepath}")