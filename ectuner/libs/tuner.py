"""
Optimization Engines for EC-Earth.

This module contains the BaseTuner interface and its concrete implementations 
for 1D (scalar) and 2D (spatial) tuning. It acts as the orchestration layer 
between the loaded data and the SciPy minimization algorithms.
"""
import math
import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union, Optional, TYPE_CHECKING

from .result import TuningResult
from .utils import REGION_BOUNDS, normalize_metric, build_targets_1d


if TYPE_CHECKING:
    import xarray as xr


class BaseTuner(ABC):
    """
    Base class for EC-Earth tuning. 
    
    Handles the initialization of parameters, the calculation of boundary 
    limits based on fractional increments, and the SciPy optimizer setup.

    Attributes:
        config: The ECtuner configuration object.
        logger: The initialized logging object.
        inc: Fractional maximum parameter change relative to the current value.
        penalty: Penalty weight for distance from reference parameters.
        params_names: Names of all loaded parameters.
        current_values: Current values of the parameters.
        ref_params: Reference baseline values for parameters.
        frozen_params: Parameters locked out of optimization.
        opt_params: Parameters actively undergoing optimization.
        bounds: Calculated min/max limits for SciPy.
    """

    def __init__(self, inc: float, penalty: float,logger: Any) -> None:
        """
        Initializes the base tuner with configuration and logger.

        Args:
            inc: Fractional maximum parameter change relative to the current value.
            penalty: Penalty weight for distance from reference parameters.
            logger: The configured logger instance.
        """
        self.logger = logger
        self.inc = float(inc)
        self.penalty = float(penalty)
        self.custom_inc: Dict[str, float] = {}

        # Parameter structures
        self.params_names: List[str] = []
        self.current_values: Dict[str, float] = {}
        self.ref_params: Dict[str, float] = {}
        self.frozen_params: Dict[str, float] = {}
        self.opt_params: List[str] = []
        self.bounds: List[Tuple[float, float]] = []
        self.bounds_around: str = 'current'  # Options: 'ref' or 'current'
        self.drift_cap: Optional[float] = None   
        self.seed: Optional[int] = None

    def setup_parameters(self, param_dict: Dict[str, float], ref_dict: Dict[str, float], frozen_config: Union[List[str], Dict[str, Any]]) -> None:
        """
        Organizes parameters into free and frozen sets.
        
        Handles both legacy lists and new dictionary formats for frozen parameters.

        Args:
            param_dict: Current values of parameters.
            ref_dict: Reference baseline values of parameters.
            frozen_config: List of frozen parameter names, or dictionary mapping names 
                to custom frozen values (or "default").
        """
        self.params_names = list(param_dict.keys())
        self.current_values = {k: float(v) for k, v in param_dict.items()}
        self.ref_params = {k: float(v) for k, v in ref_dict.items()}
        
        self.frozen_params = {}
        
        if isinstance(frozen_config, list):
            frozen_config = {p: "default" for p in frozen_config}
            
        for p, val in frozen_config.items():
            if p in self.current_values:
                if val == "default":
                    self.frozen_params[p] = self.ref_params[p]
                else:
                    self.frozen_params[p] = float(val)

        self.opt_params = [p for p in self.params_names if p not in self.frozen_params]
        self._setup_bounds()

    def _setup_bounds(self) -> None:
        if self.bounds_around not in ('ref', 'current'):
            raise ValueError(f"bounds_around must be 'ref' or 'current', got '{self.bounds_around}'")
        self.bounds = []
        for p in self.opt_params:
            v_ref = self.ref_params[p]
            v_curr = self.current_values[p]
            param_inc = float(self.custom_inc.get(p, self.inc))
            center = v_ref if self.bounds_around == 'ref' else v_curr
            lo, hi = sorted((center * (1.0 - param_inc), center * (1.0 + param_inc)))
    
            if self.drift_cap is not None and v_ref != 0:
                cap_lo, cap_hi = sorted((v_ref * (1.0 - self.drift_cap), v_ref * (1.0 + self.drift_cap)))
                tol = 1e-3 * abs(v_ref)  
                if v_curr < cap_lo - tol or v_curr > cap_hi + tol:
                    self.logger.warning(f"Parameter '{p}' ({v_curr:.4e}) is outside the drift cap "
                                        f"[{cap_lo:.4e}, {cap_hi:.4e}]: it can only move back towards it.")
                lo = max(lo, min(cap_lo, v_curr))
                hi = min(hi, max(cap_hi, v_curr))

            lo_change, hi_change = lo - v_curr, hi - v_curr
            if lo_change > 0 or hi_change < 0:
                self.logger.warning(
                    f"Parameter '{p}': current value {v_curr:.4e} is outside its bounds "
                    f"[{lo:.4e}, {hi:.4e}]; zero change is not feasible."
                )
            self.bounds.append((lo_change, hi_change))

    def _record_run_info(self, result, free_changes) -> None:
        new_params = result.get_new_parameters()
        rel = [(new_params[p] - self.ref_params[p]) / self.ref_params[p]
            for p in self.params_names if self.ref_params[p] != 0]
        n_bound = 0
        for (lo, hi), c in zip(self.bounds, free_changes):
            tol = 0.01 * (hi - lo)
            if (c - lo) <= tol or (hi - c) <= tol:
                n_bound += 1
        result.metrics['seed'] = self.seed
        result.metrics['drift_cap'] = self.drift_cap
        result.metrics['drift_rms'] = float(np.sqrt(np.mean(np.square(rel))))
        result.metrics['drift_max'] = float(np.max(np.abs(rel)))
        result.metrics['n_at_bound'] = n_bound
        self.logger.info(f"Drift from reference: rms {result.metrics['drift_rms']:.1%}, "
                        f"max {result.metrics['drift_max']:.1%}; parameters at bound: "
                        f"{n_bound}/{len(self.opt_params)}")

    @abstractmethod        
    def _objective_function(self, changes: np.ndarray) -> float:
        """
        The objective cost function to be minimized by SciPy.

        Args:
            changes: An array of numerical changes proposed by the optimizer.

        Returns:
            The calculated cost (error + penalty) to be minimized.
        """
        pass

    def run_optimizer(self, method: str = 'dual_annealing') -> np.ndarray:
        """
        Executes the SciPy minimization algorithm.

        Args:
            method: The optimization algorithm to use. Supported values: 'L-BFGS-B', 
                'dual_annealing', 'differential_evolution', 'shgo'.

        Returns:
            An array containing the optimal changes found for the free parameters.

        Raises:
            ValueError: If an unsupported optimization method is provided.
        """
        self.logger.info(f"Starting Optimization ({method})...")
        
        if method == 'L-BFGS-B':
            initial_guess = np.zeros(len(self.opt_params))
            result = optimize.minimize(
                self._objective_function,
                x0=initial_guess,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={"ftol": 1e-12, "gtol": 1e-12, "maxls": 50}
            )
            
            if not result.success:
                self.logger.warning(f"L-BFGS-B did not fully converge. Message: {result.message}")

        elif method == 'dual_annealing':
            m_kwargs = {
                "method": "L-BFGS-B", 
                "options": {"ftol": 1e-12, "gtol": 1e-12, "maxls": 50}
            }
            result = optimize.dual_annealing(
                self._objective_function, 
                bounds=self.bounds,
                minimizer_kwargs=m_kwargs,
                maxiter=1000, seed=self.seed
            )
        elif method == 'differential_evolution':
            result = optimize.differential_evolution(
                self._objective_function, 
                bounds=self.bounds, seed=self.seed
            )
        elif method == 'shgo':
            result = optimize.shgo(self._objective_function, self.bounds)
        else:
            raise ValueError(f"Method {method} not supported.")
            
        return result.x 


class Tuner1D(BaseTuner):
    """
    1D Tuner evaluating the cost function on global and regional mean scalars.
    """

    def __init__(self, inc: float, penalty: float, logger: Any) -> None:
        super().__init__(inc, penalty, logger)
        self.sensitivity = {}
        self.difference = {}
        self.reference = {}
        self.weights_flux = {}
        self.weights_season = {}
        self.weights_region = {}

    def _apply_frozen_shift(self) -> None:
        """
        Shifts the initial baseline biases if any frozen parameters 
        are forced to a custom value different from their reference.
        """
        for p, custom_val in self.frozen_params.items():
            curr_val = self.current_values[p]
            ref_val = self.ref_params[p]
            
            if custom_val == curr_val:
                continue
                
            delta_p = custom_val - curr_val

            center = ref_val if self.bounds_around == 'ref' else curr_val
            if center != 0:
                limit = float(self.custom_inc.get(p, self.inc))
                rel_shift = abs(custom_val - center) / abs(center)
                if rel_shift > limit:
                    self.logger.warning(
                        f"CRITICAL: Frozen parameter '{p}' is shifted by {rel_shift:.1%} "
                        f"from {self.bounds_around} (max linear bound is {limit:.1%}). "
                    )

            self.logger.info(f"[Pre-Shift 1D] Absorbing frozen parameter '{p}' (delta_p: {delta_p:e})")
            
            for fluxname in self.difference:
                for season in self.difference[fluxname]:
                    for region in self.difference[fluxname][season]:
                        slope = self.sensitivity.get(p, {}).get(fluxname, {}).get(season, {}).get(region, [0.0])[0]
                        
                        shift = slope * delta_p
                        self.difference[fluxname][season][region] += shift

    def prepare_data(
        self, 
        sensitivity: Dict[str, Any], 
        difference: Dict[str, Any], 
        reference: Dict[str, Any],
        weights_flux: Dict[str, float], 
        weights_season: Dict[str, float], 
        weights_region: Dict[str, float],
        targets: Optional[Dict[Tuple[str, str, str], float]] = None
    ) -> None:
        """
        Injects data and builds the target arrays used by the objective function.

        Args:
            sensitivity: Pre-calculated sensitivity coefficients.
            difference: Realized biases between model and observations.
            reference: Observational references for final output evaluations.
            weights_flux: Weights for the tuning target variables.
            weights_season: Weights for the seasonal intervals.
            weights_region: Weights for the spatial regions.
            targets: {(variable, season, region): weight}, from ``utils.build_targets_1d``.
                If None, built from the product of the three weight dictionaries.
        """
        self.sensitivity = sensitivity
        self.difference = difference
        self.reference = reference
        self.weights_flux = {k: float(v) for k, v in weights_flux.items()}
        self.weights_season = {k: float(v) for k, v in weights_season.items()}
        self.weights_region = {k: float(v) for k, v in weights_region.items()}
        if targets is None:
            targets, notes = build_targets_1d(sensitivity, None, weights_flux, weights_season, weights_region)
            for n in notes:
                self.logger.warning(n)
        self.targets = dict(targets)
        self._apply_frozen_shift()
        self._build_target_arrays()

    def _build_target_arrays(self) -> None:
        """
        One row per target: bias (after the frozen shift), weight and slopes of all parameters,
        so that the cost is a few vector operations.
        """
        keys, diffs, weights, rows = [], [], [], []
        for (v, s, r), w in self.targets.items():
            diff = self.difference.get(v, {}).get(s, {}).get(r, np.nan)
            if diff is None or math.isnan(diff):
                self.logger.warning(f"Target ({v}, {s}, {r}): no model-reference difference, skipped.")
                continue
            keys.append((v, s, r))
            diffs.append(diff)
            weights.append(w)
            rows.append([self.sensitivity[p][v][s][r][0] for p in self.params_names])
        if not keys:
            raise ValueError("No valid 1D target: check weights / targets in the config.")
        self._t_keys = keys
        self._t_diff = np.asarray(diffs, dtype=float)
        self._t_w = np.asarray(weights, dtype=float)
        self._t_sens = np.asarray(rows, dtype=float).reshape(len(keys), len(self.params_names))
        self._opt_idx = np.array([self.params_names.index(p) for p in self.opt_params], dtype=int)

    def _objective_function(self, changes: np.ndarray) -> float:
        """
        Calculates the scalar cost function: sum over the targets of
        weight * (bias + predicted change)**2, plus the parameter penalty.
        """
        c = np.zeros(len(self.params_names))          # frozen parameters: no change
        c[self._opt_idx] = changes
        residual = self._t_diff + self._t_sens @ c
        total_difference = float(np.sum(self._t_w * residual ** 2))
        # Note: a perfect paraboloid. A gradient solver like L-BFGS-B finds the same
        # global minimum as dual_annealing in a fraction of the time.

        param_penalty = 0.0
        for i, param in enumerate(self.params_names):
            ref_val = self.ref_params[param]
            if ref_val != 0:
                new_val = self.current_values[param] + c[i]
                param_penalty += ((ref_val - new_val) / ref_val) ** 2

        return total_difference + (param_penalty * self.penalty)

    def optimize(self, method: str = 'dual_annealing') -> TuningResult:
        """
        Executes the optimization process for 1D targets.

        Args:
            method: The chosen optimizer.

        Returns:
            An object containing metrics and the calculated optimal parameters.
        """
        free_changes = self.run_optimizer(method)
        
        optimal_changes_dict = {p: 0.0 for p in self.params_names}
        for i, p in enumerate(self.opt_params):
            optimal_changes_dict[p] = free_changes[i]

        bounds_dict = {p: b for p, b in zip(self.opt_params, self.bounds)}
        
        initial_guess_free = np.zeros(len(self.opt_params))
        score_init = self._objective_function(initial_guess_free)
        score_final = self._objective_function(free_changes)

        bias_evaluation = self.evaluate_biases(optimal_changes_dict)

        result = TuningResult(
            target_vars=sorted({v for v, _, _ in self.targets}),
            param_names=self.params_names,
            optimal_changes=optimal_changes_dict,
            initial_values=self.current_values,
            ref_values=self.ref_params,
            bounds=bounds_dict,
            frozen_params=self.frozen_params
        )

        self._record_run_info(result, free_changes)
        result.targets = dict(self.targets)
        result.metrics['metric_name'] = '1D_scalars'
        result.metrics['penalty'] = self.penalty
        result.metrics['inc'] = self.inc
        result.metrics['bounds_around'] = self.bounds_around
        result.metrics['score_init'] = score_init
        result.metrics['score_final'] = score_final
        result.bias_evaluation = bias_evaluation
        
        self.logger.info(f"Optimization finished. Final Global Score: {score_final:.4f}")
        return result
    
    def evaluate_biases(self, optimal_changes_dict: dict) -> dict:
        """
        Calculates initial and final biases for all targets and diagnostics.

        Args:
            optimal_changes_dict: Mapping of parameter names to their optimal changes.

        Returns:
            A structured dictionary containing 'targets' and 'diagnostics' evaluations.
        """
        evaluation = {
            'targets': [],
            'diagnostics': []
        }

        ref_param_name = self.params_names[0]

        for fluxname in self.difference:
            if fluxname not in self.sensitivity.get(ref_param_name, {}): continue

            for season in self.difference[fluxname]:
                for region in self.difference[fluxname][season]:
                    bias_init = self.difference[fluxname][season][region]
                    if math.isnan(bias_init): continue

                    # Calcolo bias finale
                    flux_change = sum(
                        self.sensitivity[p][fluxname][season][region][0] * optimal_changes_dict[p] 
                        for p in self.params_names
                    )
                    bias_final = bias_init + flux_change
                    
                    combined_weight = self.targets.get((fluxname, season, region), 0.0)

                    ref_val = self.reference.get(fluxname, {}).get(season, {}).get(region, None)
                    
                    row_data = {
                        'variable': fluxname,
                        'season': season,
                        'region': region,
                        'weight': combined_weight,
                        'ref_val': ref_val,
                        'model_init': ref_val + bias_init if ref_val is not None else None,
                        'model_final': ref_val + bias_final if ref_val is not None else None,
                        'bias_init': bias_init,
                        'bias_final': bias_final,
                        'status': 'IMPROVED' if abs(bias_final) < abs(bias_init) else 'WORSENED'
                    }
                    
                    if combined_weight > 0:
                        evaluation['targets'].append(row_data)
                    else:
                        evaluation['diagnostics'].append(row_data)
                        
        return evaluation


class Tuner2D(BaseTuner):
    """
    2D Spatial Tuner implementing a hybrid global/spatial objective function.
    
    This class leverages vectorized numpy operations to efficiently evaluate
    the cost function across thousands of grid points simultaneously.
    """

    def __init__(self, inc: float, penalty: float, alpha: float, metric: str, logger: Any) -> None:
        """
        Initializes the 2D Tuner.

        Args:
            inc: Fractional maximum parameter change relative to the current value.
            penalty: Penalty weight for distance from reference parameters.
            alpha: Blending weight between spatial error (0.0) and global error (1.0).
            metric: Error metric to compute ('l2'/'rmse' for RMSE, 'l1'/'mae' for MAE).
            logger: The configured logger instance.
        """
        super().__init__(inc, penalty, logger)
        self.alpha = float(alpha)
        self.metric = normalize_metric(metric)

        self.bias_flat = {}
        self.sens_matrices_spatial = {}
        self.sens_matrices_global = {}
        self.weights_vector_var = {}
        self.mask_2d: Any = None
    
    def _apply_frozen_shift(self) -> None:
        """
        Shifts the 2D bias maps for custom frozen parameters.
        """
        for p, custom_val in self.frozen_params.items():
            curr_val = self.current_values[p]
            ref_val = self.ref_params[p]
            
            if custom_val == curr_val:
                continue
                
            delta_p = custom_val - curr_val

            center = ref_val if self.bounds_around == 'ref' else curr_val
            if center != 0:
                limit = float(self.custom_inc.get(p, self.inc))
                rel_shift = abs(custom_val - center) / abs(center)
                if rel_shift > limit:
                    self.logger.warning(
                        f"CRITICAL: Frozen parameter '{p}' is shifted by {rel_shift:.1%} "
                        f"from {self.bounds_around} (max linear bound is {limit:.1%}). "
                    )

            self.logger.info(f"[Pre-Shift 2D] Absorbing frozen parameter '{p}' (delta_p: {delta_p:e})")
            
            for var in self.target_vars:
                if p in self.ds_sens.parameter.values and var in self.ds_sens.variable.values:
                    slope_map = self.ds_sens.sel(variable=var, parameter=p).slope
                    self.bias_maps_2d[var] += (slope_map.fillna(0) * delta_p)
        
    def prepare_data(
        self, 
        bias_maps: Dict[str, 'xr.DataArray'], 
        ref_maps: Dict[str, 'xr.DataArray'],
        ds_sens: 'xr.Dataset', 
        mask_2d: 'xr.DataArray',
        weights_flux: Dict[str, float], 
        weights_region: Dict[str, float]
    ) -> None: 
        """
        Flattens the 2D geospatial maps into 1D arrays for fast matrix operations.

        Args:
            bias_maps: Dictionary mapping variable names to their 2D bias arrays.
            ref_maps: Dictionary mapping variable names to their reference arrays.
            ds_sens: The 2D sensitivity dataset.
            mask_2d: The regional and area-weighted mask.
            weights_flux: Weights for the tuning target variables.
            weights_region: Weights for the spatial regions.
        """
        self.logger.info("Flattening data")
        self.bias_maps_2d = bias_maps
        self.ref_maps_2d = ref_maps
        self.ds_sens = ds_sens
        self.mask_2d = mask_2d

        self.weights_flux = {k: float(v) for k, v in weights_flux.items()}
        self.weights_region = {k: float(v) for k, v in weights_region.items()}
        self.target_vars = list(bias_maps.keys())

        self._apply_frozen_shift()

        mask_ordered = self.mask_2d.transpose('lat', 'lon').sortby(['lat', 'lon'])
        
        for var in self.target_vars:
            b_ordered = bias_maps[var].transpose('lat', 'lon').sortby(['lat', 'lon'])
            
            actual_valid_mask = (~np.isnan(mask_ordered)) & (~np.isnan(b_ordered))
            valid_pix = actual_valid_mask.values.flatten()
            
            self.weights_vector_var[var] = mask_ordered.values.flatten()[valid_pix]
            self.bias_flat[var] = b_ordered.values.flatten()[valid_pix]

            slopes = []
            for p in self.opt_params:
                s_ds = ds_sens.sel(variable=var, parameter=p).sortby(['lat', 'lon'])
                slopes.append(np.nan_to_num(s_ds.slope.values.flatten()[valid_pix], nan=0.0))

            self.sens_matrices_spatial[var] = np.column_stack(slopes)
            self.sens_matrices_global[var] = self.sens_matrices_spatial[var]  # stesso array, mai modificato in place

    def _objective_function(self, changes: np.ndarray) -> float:
        """
        Fast numpy-based hybrid cost function combining spatial and global errors.
        """
        total_error = 0.0
        
        for var in self.target_vars:
            w_flux = self.weights_flux.get(var, 0.0)
            if w_flux <= 0: continue
                
            w_v = self.weights_vector_var[var]
            
            delta_pred_spatial = np.dot(self.sens_matrices_spatial[var], changes)
            residual_spatial = self.bias_flat[var] + delta_pred_spatial
            
            sens_global_mean = np.average(self.sens_matrices_global[var], axis=0, weights=w_v)
            bias_global_init = np.average(self.bias_flat[var], weights=w_v)
            bias_global_final = bias_global_init + np.dot(sens_global_mean, changes)

            
            if self.metric == 'l2': #MSE (Mean Squared Error) 
                cost_spatial = np.average(residual_spatial**2, weights=w_v) #MSE
                cost_global = bias_global_final**2 
            else: #caso lineare (MAE), Entrambi i costi sono in W/m²: penalizza errori in modo proporzionale
                cost_spatial = np.average(np.abs(residual_spatial), weights=w_v)
                cost_global = np.abs(bias_global_final)
                
            var_error = (1.0 - self.alpha) * cost_spatial + self.alpha * cost_global
            total_error += w_flux * var_error
            
        param_penalty = 0.0
        for i, p in enumerate(self.opt_params):
            ref_val = self.ref_params[p]
            if ref_val != 0:
                new_val = self.current_values[p] + changes[i]
                param_penalty += ((ref_val - new_val) / ref_val) ** 2
                             
        return total_error + (param_penalty * self.penalty)

    def optimize(self, method: str = 'dual_annealing') -> TuningResult:
        """
        Executes the optimization process for 2D spatial targets.

        Args:
            method: The chosen optimizer.

        Returns:
            An object containing all spatial and global metrics, along with 
            the calculated optimal parameters.
        """
        free_changes = self.run_optimizer(method)
        
        initial_guess_free = np.zeros(len(self.opt_params))
        score_init = self._objective_function(initial_guess_free)

        optimal_changes_dict = {p: 0.0 for p in self.params_names}
        for i, p in enumerate(self.opt_params):
            optimal_changes_dict[p] = free_changes[i]
            
        bounds_dict = {p: b for p, b in zip(self.opt_params, self.bounds)}
        
        result = TuningResult(
            target_vars=self.target_vars,
            param_names=self.params_names,
            optimal_changes=optimal_changes_dict,
            initial_values=self.current_values,
            ref_values=self.ref_params,
            bounds=bounds_dict,
            frozen_params=self.frozen_params
        )

        self._record_run_info(result, free_changes)
        result.metrics['alpha'] = self.alpha
        result.metrics['metric_name'] = self.metric
        result.metrics['penalty'] = self.penalty
        result.metrics['inc'] = self.inc
        result.metrics['bounds_around'] = self.bounds_around
        result.metrics['score_init'] = score_init
        
        tot_spat = 0.0
        tot_glob = 0.0
        tot_spat_init = 0.0
        tot_glob_init = 0.0
        
        for var in self.target_vars:
            w_flux = self.weights_flux.get(var, 0.0)
            w_v = self.weights_vector_var[var]

            res_spat_init = self.bias_flat[var]
            res_glob_init = np.average(self.bias_flat[var], weights=w_v)
            
            d_spat = np.dot(self.sens_matrices_spatial[var], free_changes)
            res_spat = self.bias_flat[var] + d_spat
            
            sens_glob_v = np.average(self.sens_matrices_global[var], axis=0, weights=w_v)
            res_glob = np.average(self.bias_flat[var], weights=w_v) + np.dot(sens_glob_v, free_changes)
            
            if self.metric == 'l2':
                v_spat_init = np.sqrt(np.average(res_spat_init**2, weights=w_v))
                v_spat = np.sqrt(np.average(res_spat**2, weights=w_v))
            else:
                v_spat_init = np.average(np.abs(res_spat_init), weights=w_v)
                v_spat = np.average(np.abs(res_spat), weights=w_v)
            v_glob_init, v_glob = np.abs(res_glob_init), np.abs(res_glob)

            tot_spat += w_flux * v_spat;  tot_glob += w_flux * v_glob
            tot_spat_init += w_flux * v_spat_init;  tot_glob_init += w_flux * v_glob_init

            result.set_var_metrics(var, res_glob, v_spat, v_glob,
                                initial_global_bias=res_glob_init,
                                initial_spatial_cost=v_spat_init)
            self.logger.info(f"[{var}] INIT: global bias {res_glob_init:+.3f}, spatial {v_spat_init:.3f} "
                            f"-> FINAL: global bias {res_glob:+.3f}, spatial {v_spat:.3f}")

        result.metrics['total_spatial_cost'] = tot_spat
        result.metrics['total_global_cost'] = tot_glob
        result.metrics['total_spatial_cost_init'] = tot_spat_init
        result.metrics['total_global_cost_init'] = tot_glob_init

        result.bias_evaluation = self.evaluate_biases(optimal_changes_dict)
        
        self.logger.info(f"Optimization finished. Spatial Cost: {tot_spat:.4f} | Global Cost: {tot_glob:.4f}")
        return result

    def evaluate_biases(self, optimal_changes_dict: dict) -> dict:
        """
        Calculates initial and final regional biases from 2D maps.

        Args:
            optimal_changes_dict: Mapping of parameter names to optimal changes.

        Returns:
            A structured dictionary populating the unified TuningResult tables.
        """
        import xarray as xr
        import numpy as np
        
        evaluation = {'targets': [], 'diagnostics': []}

        for var in self.target_vars:
            b_init_map = self.bias_maps_2d[var]
            ref_map = self.ref_maps_2d.get(var)
            delta_pred = xr.zeros_like(b_init_map)

            # Reconstruct the full predicted 2D map
            for p in self.opt_params:
                slope = self.ds_sens.sel(variable=var, parameter=p).slope
                delta_pred += slope.fillna(0) * optimal_changes_dict[p]

            b_final_map = b_init_map + delta_pred
            var_w = self.weights_flux.get(var, 0.0)

            for region, bounds in REGION_BOUNDS.items():
                reg_w = self.weights_region.get(region, 0.0)
                combined_w = var_w * reg_w

                low, high = bounds
                lat = b_init_map.lat
                
                # Area-weighted regional mask
                cos_lat = np.cos(np.deg2rad(lat))
                reg_mask = cos_lat.where((lat >= low) & (lat <= high), 0.0)

                init_val = float(b_init_map.weighted(reg_mask).mean().values)
                final_val = float(b_final_map.weighted(reg_mask).mean().values)
                ref_val = float(ref_map.weighted(reg_mask).mean().values) if ref_map is not None else None

                row_data = {
                    'variable': var,
                    'season': 'ALL',
                    'region': region,
                    'weight': combined_w,
                    'ref_val': ref_val,
                    'model_init': ref_val + init_val if ref_val is not None else None,
                    'model_final': ref_val + final_val if ref_val is not None else None,
                    'bias_init': init_val,
                    'bias_final': final_val,
                    'status': 'IMPROVED' if abs(final_val) < abs(init_val) else 'WORSENED'
                }

                if combined_w > 0:
                    evaluation['targets'].append(row_data)
                else:
                    evaluation['diagnostics'].append(row_data)

        return evaluation