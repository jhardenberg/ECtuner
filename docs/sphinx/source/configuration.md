## Configuration & Data Catalog

ECtuner is driven by master YAML configuration files. You can find ready-to-use templates inside the `ectuner/templates/` directory of the repository. 

## 1. The Master Configuration File

A standard configuration file (e.g., `ectuner_master_config.yaml`) is divided into some logical blocks. Any value can be overridden from the command line (see the [Usage Guide](usage.md)) or programmatically.

### Files & Paths (`files`)
Defines where ECtuner should look for inputs and save outputs. It handles both 1D and 2D specific paths. 
**1D mode**
* `reference`: path to the 1D observational reference YAML (e.g., CERES-based global means).
* `sensitivity`: path to the 1D sensitivity regression coefficients YAML.
* `ecmean`: directory containing the ECmean global-mean files of the experiment being tuned. `base` is the file name template, formatted with `exp`, `year1` and `year2`.
* `ecmean_config`: ECmean configuration, used only when the base file is missing and must be computed on the fly. (The sensitivity script reads the same information from the `ecmean.* ecmean_config` key of its own configuration file.)
* `slope_file`: path to the YAML file containing the climate feedback parameters (λ), derived from the slopes of a Gregory plot. Required for temperature drift corrections in coupled simulations.
**2D mode**
* `sensitivity_nc`: path to the 2D NetCDF sensitivity map (for spatial mode).
* `ref_2d_dir`: directory of the 2D reference climatologies, named `climate_average_<var>_<dataset>_<grid>_<period>.nc`.
* `ref_2d_period` *(optional)*: period string, e.g. `"2000-2024"`, used to select the reference file when several periods are available. If no file matches, ECtuner warns and uses any available period. Without this key the first file in alphabetical order is used, and a warning lists the alternatives.
* `base_2d_dir`: cache of the regridded model maps, base_2d_<exp>_<year1>_<year2>_<grid>.nc. Maps missing from the cache are computed from raw_dir and added to it.
**Common**
* `raw_dir`: raw model output, read as `<raw_dir>/<exp>/output/oifs/<exp>_atm_cmip6_1m_*.nc`(2D maps, automatic detection of the years, checks of the iterative loop).
* `params`: file name template of the tuning (parameter) files, formatted with `exp` (e.g., `tuning_{exp}.yml`).
* `exps`: directory containing those parameter files.
* `output_dir`: directory where outputs are written.

### Tuning Arguments (`args`)
Controls the runtime parameters:
* `year1` & `year2`: the time window used for averaging model climatology.
* `inc`: the maximum fractional change allowed in **one tuning step**. The allowed range of a parameter is `center × (1 ± inc)`, where `center` is set by `bounds_around`.
* `bounds_around`: centre of the `inc` range.
  * `current` (default): the parameter value at the start of the step, i.e. the value in the tuning file of the experiment being tuned. Each iteration moves a parameter by at most ±`inc` from where it is.
  * `ref`: the `reference_parameters`. The total departure from the defaults never exceeds ±`inc`, however many iterations are run.

  At the first iteration from the default configuration the two options coincide.

* `drift_cap` *(optional)*: maximum **cumulative** departure of a parameter from its `reference_parameters` value, as a fraction (e.g., `0.3` for ±30%), applied on top of `inc`. A parameter already outside the cap can only move back towards it. Choose a value not larger than the range spanned by the sensitivity simulations, so that the linear emulator is not applied outside the range where the sensitivities were estimated. Empty means no cap.
* `custom_inc`: overrides `inc` for specific parameters (e.g., a narrower range for highly sensitive ones).
* `penalty`: weight of the penalty on the distance of the new parameters from the `reference_parameters`, always measured from the defaults, whatever `bounds_around` is. It prevents mathematically correct but physically unrealistic solutions. `0` switches it off.
* `seed`: random seed of the stochastic optimizers (`dual_annealing`, `differential_evolution`). It is recorded in the outputs, so a run can be reproduced exactly. `L-BFGS-B` and `shgo` are deterministic and ignore it.
* `method`: the optimization algorithm: dual_annealing (robust default), differential_evolution, L-BFGS-B, shgo. In 1D the cost function is quadratic, so L-BFGS-B reaches the same minimum much faster.
* `output_tag` *(optional)*: tag in the name of the 2D diagnostic map, diagnostics_2d_<exp>_<output_tag>.nc (default: active regions and alpha, e.g. Global_a00).

#### 1D Specific Physics Arguments:
* `deltaT`: the temperature adjustment step (K) applied to compute flux corrections. It represents the difference between the target and the simulated surface temperature (ΔT). ECtuner uses this along with the `slope_file` to adjust the reference fluxes prior to tuning (ΔF=ΔT⋅λ).
* `model_imbalance`: the intrinsic model energy imbalance to be subtracted from the net_toa target. Useful to equilibrate tuning targets for low-resolution configurations that inherently create or destroy energy.

### Spatial Tuning Options (`spatial_tuning`)
Ignored in 1D mode, these parameters dictate how the 2D spatial engine behaves:
* `alpha`: the blending weight between spatial error ($\alpha = 0$) and global error ($\alpha = 1$).
* `metric`: the cost error metric, typically `"l2"` (area weighted MSE) or `"l1"` (area-weighted MAE). With l2 the global term is the squared global-mean bias.
* `target_grid`: regular grid on which model and reference maps are compared (default `r180x90`, 2°×2°).
* `methods_map`: optional mapping from variable name to the CDO regridding method (default `ycon`, with a fallback to `bil` if the method fails).

### Parameters Setup (`frozen_parameters`, `reference_parameters`, `parameter_group`)
* `frozen_parameters`: allows locking specific parameters to their default values or forcing them to a custom value (e.g., `RPRCON: default`). A custom value shifts the initial bias linearly through the sensitivities; ECtuner warns if the shift is larger than `inc`.
* `reference_parameters`: the baseline (default) values, used by the penalty, by `bounds_around: ref` and by `drift_cap`.
* `parameter_group`: grouping of the parameters in the namelist blocks of the output file (e.g., `namcumf`, `namcldp`, `naerad`).


### Weights & Targets (`weights`, `weights_region`, `weights_season`)
Defines the relative importance of each atmospheric variable to tune (e.g., `net_toa`, `rsnt`), geographical region, and season in the cost function. Variables with weight `0` are not in the cost function but are reported as side effects. `weights_season` is used only in 1D.
> *Note on regions:* Region weights are **additive**. If you overlap regions (e.g., `Global` and `NH`), their weights will stack.

---

## 2. Sensitivity Configurations

Before running the tuner, sensitivities must be computed using an ensemble of perturbed runs via the built-in CLI commands `ectuner-sens-1d` or `ectuner-sens-2d`. These commands require a dedicated configuration file (you can find `config_sens.yaml` and `config_sens_2d.yaml` in the `templates/` directory) that dictates how the tool processes the ensemble members to extract the linear regression slopes. 

> *For a practical, step-by-step guide on how to prepare your data and compute sensitivities, please refer to the **[Usage Guide](usage.md)**.*

---

## 3. Data Catalog & Pre-computed Sensitivities

To save computational time, pre-calculated sensitivities for standard EC-Earth4 configurations are provided in the `data/sensitivities/` folder.

### File Naming Convention
* **1D (YAML):** `sensitivity_{Resolution}_{Version}_{Years}.yaml`
* **2D (NetCDF):** `2D/sensitivity_{Resolution}_{Version}_{Years}_2D.nc`

### Available Standard Configurations
1. **TL255 1991-2000 - Version 2 (Latest):** EC-Earth4 v4.1.5, Nested Namelist format, 16 parameters considered (`namcumf`, `namcldp`, `naerad`).
2. **TL255 (Legacy Versions):** Flat namelist format, 10 parameters.
3. **TL63 (Low Resolution):** Versions 4.1.3 and 4.1.5 available for faster developmental tuning loops.

> *Note:* For an extensive explanation give a look to the README.md in the `data/sensitivities/` folder.
---

## 4. Repository Structure Overview

* `ectuner/ectuner.py`: main Command Line Interface (CLI) and API entry point.
* `ectuner/libs/`: core library: `loader` (data input), `tuner` (1D/2D optimizers), `result` and `exporter` (outputs), `metrics` (bias metrics and emulator validation), `diagnostics` (figures), `automation` (SLURM), `config`, `logger` and `utils`.
* `integrations/`: infrastructure-specific automation tools (e.g., `ecearth4_loop.py` SLURM orchestrator).
* `data/sensitivities/`: pre-computed sensitivity matrices and catalog README.
* `ectuner/templates/`: master configuration templates.
* `generate_tutorial.py`: lightweight sandbox environment to generate a tutorial for quick testing.