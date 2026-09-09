## Configuration & Data Catalog

ECtuner is driven by master YAML configuration files. You can find ready-to-use templates inside the `ectuner/templates/` directory of the repository. 

## 1. The Master Configuration File

A standard configuration file (e.g., `ectuner_master_config.yaml`) is divided into some logical blocks:

### Files & Paths (`files`)
Defines where ECtuner should look for inputs and save outputs. It handles both 1D and 2D specific paths:
* `reference`: path to the 1D observational reference YAML (e.g., CERES-based global means).
* `sensitivity`: path to the 1D sensitivity regression coefficients YAML.
* `sensitivity_nc`: path to the 2D NetCDF sensitivity map (for spatial mode).
* `raw_dir` & `exps`: directories containing raw OIFS model outputs and parameter perturbation files.

### Tuning Arguments (`args`)
Controls the runtime parameters:
* `year1` & `year2`: the time window used for averaging model climatology.
* `inc`: the maximum fractional change allowed relative to reference values (e.g., `0.2` limits changes to $\pm 20\%$).
* `penalty`: weight for the distance penalty from OIFS default parameters. Prevents the optimizer from finding mathematically correct but physically unrealistic solutions.
* `method`: the optimization algorithm. `dual_annealing` is the robust default.

### Spatial Tuning Options (`spatial_tuning`)
Ignored in 1D mode, these parameters dictate how the 2D spatial engine behaves:
* `alpha`: the blending weight between spatial error ($\alpha = 0$) and global error ($\alpha = 1$).
* `metric`: the cost error metric, typically `"l2"` (MSE) or `"l1"` (MAE).

### Parameters Setup (`frozen_parameters` & `reference_parameters`)
* `frozen_parameters`: allows locking specific parameters to their default values or forcing them to a custom value (e.g., `RPRCON: default`).
* `reference_parameters`: the baseline values used by the penalty function.

### Weights & Targets (`weights`, `weights_region`, `weights_season`)
Defines the relative importance of each atmospheric variable to tune (e.g., `net_toa`, `rsnt`), geographical region, and season in the cost function. 
> *Note on regions:* Region weights are **additive**. If you overlap regions (e.g., `Global` and `NH`), their weights will stack.

---

## 2. Sensitivity Configurations

Before running the tuner, sensitivities must be computed using an ensemble of perturbed runs via the built-in CLI commands `ectuner-sens-1d` or `ectuner-sens-2d`. TThese commands require a dedicated configuration file (you can find `config_sens.yaml` and `config_sens_2d.yaml` in the `templates/` directory) that dictates how the tool processes the ensemble members to extract the linear regression slopes. 

> *For a practical, step-by-step guide on how to prepare your data and compute sensitivities, please refer to the **[Usage Guide](usage.md)**.*

---

## 3. Data Catalog & Pre-computed Sensitivities

To save computational time, pre-calculated sensitivities for standard EC-Earth4 configurations are provided in the `data/sensitivities/` folder.

### File Naming Convention
* **1D (YAML):** `sensitivity_{Resolution}_{Version}_{Years}.yaml`
* **2D (NetCDF):** `2D/sensitivity_{Resolution}_{Version}_{Years}_2D.yaml`

### Available Standard Configurations
1. **TL255 1991-2000 - Version 2 (Latest):** EC-Earth4 v4.1.5, Nested Namelist format, 16 parameters considered (`namcumf`, `namcldp`, `naerad`).
2. **TL255 (Legacy Versions):** Flat namelist format, 10 parameters.
3. **TL63 (Low Resolution):** Versions 4.1.3 and 4.1.5 available for faster developmental tuning loops.

> *Note:* For an extensive explanation give a look to the README.md in the `data/sensitivities/` folder.
---

## 4. Repository Structure Overview

* `ectuner/ectuner.py`: main Command Line Interface (CLI) and API entry point.
* `ectuner/libs/`: core containing Loaders, Tuners (1D/2D), and Exporters.
* `integrations/`: infrastructure-specific automation tools (e.g., `ecearth4_loop.py` SLURM orchestrator).
* `data/sensitivities/`: pre-computed sensitivity matrices and catalog README.
* `templates/`: master configuration templates.
* `tutorial/`: lightweight sandbox environment generated for quick testing.