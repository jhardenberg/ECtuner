## Usage
ECtuner operates through a structured workflow that spans from pre-computing sensitivities to executing the optimization via CLI, Python API, or automated SLURM loops.

---

### 1. Sensitivity Stage (Pre-requisite)
Optimization requires precomputed parameter sensitivities. You can compute these sensitivities using an ensemble of perturbed runs (One At a Time perturbations).

**Data Requirements:**
* **1D Mode:** A directory containing the YAML parameter files for each perturbed run, along with the corresponding global mean files computed by ECmean4.
* **2D Mode:** The raw NetCDF outputs (`*atm_cmip6_1m*.nc`) from your ensemble model runs.

The tool provides built-in CLI commands to compute sensitivities. It automatically recognizes the parameters changed in each run, extracts the changes, and builds the response file.

**For 1D (Global Scalars):**
```bash
# Basic usage (uses defaults from config_sens.yaml)
ectuner-sens-1d -c config_sens.yaml

# Explicitly setting the base experiment (e.g., s000), the ref tag, and years
ectuner-sens-1d -c config_sens.yaml s000 "s???" 1990 1997
```
**For 2D (Spatial Maps):**
```bash
ectuner-sens-2d -c config_sens_2d.yaml
```

> *Note: To save computational time, pre-calculated sensitivities for standard EC-Earth4 configurations are provided in the `data/sensitivities/` folder of this repository.*


### 2. The Tuning Stage (CLI)
`ectuner` operates via subcommands (`1d` or `2d`). You must provide the YAML config, the target experiment ID, and the time window (start/end years).

**Run 1D Global Tuning:**
```bash
ectuner 1d -c config.yaml -o output/tuned_{exp}.yml {exp} 1990 2000
```
**Advanced 1D Physics Options**
- `-dT, --deltaT`: Applies a reference correction based on temperature using slopes defined in `data/utils/slopes.yaml`. Important for tuning coupled model simulations o remove temperature drifts.
- `-imb, --model_imbalance`: Corrects the `net_toa` target to cope with intrinsic model energy imbalances (mainly for low-resolution configurations).

**Run 2D Spatial Tuning:**
```bash
ectuner 2d -c config.yaml -o output/tuned_{exp}.yml -t tag {exp} 1990 2000
```

**Key CLI Arguments:**
* `-c, --config`: Path to the master YAML configuration.
* `-p, --penalty`: (Default: 0) Sets the weight for the penalty term. Higher values keep the new parameters closer to the OIFS defaults to avoid physically unrealistic solutions.
* `-i, --inc`: (Default: 0.2) The maximum allowed fractional change (e.g., 0.1 limits changes to ±10% of the reference value).
* `-m, --method`: Choose the optimization algorithm. `dual_annealing` is recommended for most cases, but `differential_evolution` and `L-BFGS-B` are also supported.
* `-o, --output`: Specifies the path to save the suggested tuning as a YAML file.

### 3. Usage from Jupyter Notebooks (API)
Since ECtuner is packaged, you can import its core functions directly into Python scripts or Notebooks for interactive workflows:

```python
from ectuner.libs.config import Config
from ectuner.libs.logger import setup_logger
from ectuner.ectuner import run_1d_tuning

# Initialize
config = Config('config.yaml', exp='ie00', year1=1990, year2=2000)
logger = setup_logger(level='INFO')

# Run optimization programmatically
result = run_1d_tuning(config, logger)
print(result.get_new_parameters())
```

###  4. HPC Integration & Automation
ECtuner is designed to be easily wrapped in automated SLURM scripts for continuous tuning loops on HPC clusters, regardless of the underlying climate model. You can trigger the CLI commands within your own job submission scripts to create fully automated workflows.
**EC-Earth4 Integration (Included):**
For EC-Earth4 users, we provide a ready-to-use orchestrator out of the box. Instead of manually running the tuner and submitting jobs, you can use this wrapper to clone a finished experiment, optimize its parameters, and submit the new one automatically:
```bash
python integrations/ecearth4/ecearth4_loop.py exp_old exp_new -a duplicate -c config.yaml -m 1d
```
> *Note: This specific wrapper reads the outputs of `exp_old`, computes the new parameters, uses the external `ecearth-quests/ece4/duplicate-job.py` (https://github.com/asozza/ecearth-quests/tree/main) script to clone the environment into `exp_new`, injects the new YAML, and automatically submits the job via `launch.sh`*


### Optimization Output
At the end of an optimization run, ECtuner generates:
1. `tuned_<exp>.yml`: The model-compatible namelist block with the new parameters.
2. `diagnostics_<exp>.yaml`: A structured file containing final cost scores, relative parameter changes, and bias evaluations.
3. `diagnostics_2d_<exp>.nc` (2D mode only): A NetCDF file containing the spatial maps of initial vs. predicted final biases.
You can feed these diagnostic files directly into the `diagnostics.py` module to plot parameter scatter plot, parameter heatmaps and tuning validation profiles.

#### Example output table
Below is an example of a tuning run involving 16 parameters:

|     Parameter     |     New value |   Old value |       Change |   Relative change |   Min change |   Max change |   Rel. dist. from ref. |
|-------------------|---------------|-------------|--------------|-------------------|--------------|--------------|------------------------|
|      RPRCON       |   0.00138269  |     0.0014  | -1.73056e-05 |      -0.0123611   |     -0.00084 |      0.00084 |           -0.0123611   |
|      ENTRORG      |   0.00158176  |     0.00175 | -0.000168237 |      -0.0961354   |     -0.00105 |      0.00105 |           -0.0961354   |
|      DETRPEN      |   7.40646e-05 |     7.5e-05 | -9.35368e-07 |      -0.0124716   |     -4.5e-05 |      4.5e-05 |           -0.0124716   |
|      ENTRDD       |   0.000305958 |     0.0003  |  5.95828e-06 |       0.0198609   |     -0.00018 |      0.00018 |            0.0198609   |
|      RMFDEPS      |   0.296217    |     0.3     | -0.0037834   |      -0.0126113   |     -0.18    |      0.18    |           -0.0126113   |
|       RVICE       |   0.12499     |     0.13    | -0.00501046  |      -0.038542    |     -0.078   |      0.078   |           -0.038542    |
|    RLCRITSNOW     |   2.02131e-05 |     2e-05   |  2.13131e-07 |       0.0106565   |     -1.2e-05 |      1.2e-05 |            0.0106565   |
|     RSNOWLIN2     |   0.0312179   |     0.03    |  0.00121795  |       0.0405983   |     -0.018   |      0.018   |            0.0405983   |
|      RCLDIFF      |   3.03635e-06 |     3e-06   |  3.63473e-08 |       0.0121158   |     -1.8e-06 |      1.8e-06 |            0.0121158   |
|   RCLDIFF_CONVI   |  10.1805      |    10       |  0.180548    |       0.0180548   |     -6       |      6       |            0.0180548   |
|  RDEPLIQREFRATE   |   0.500985    |     0.5     |  0.000984559 |       0.00196912  |     -0.3     |      0.3     |            0.00196912  |
|  RDEPLIQREFDEPTH  | 499.541       |   500       | -0.458564    |      -0.000917128 |   -300       |    300       |           -0.000917128 |
| RCL_OVERLAPLIQICE |   0.1         |     0.1     |  0           |       0           |      0.1     |      0.1     |           -0.846154    |
|  RCL_INHOMOGAUT   |   1.52303     |     1.5     |  0.0230251   |       0.0153501   |     -0.9     |      0.9     |            0.0153501   |
|  RCL_INHOMOGACC   |   3.19116     |     3       |  0.191158    |       0.0637194   |     -1.8     |      1.8     |            0.0637194   |
|      RMINICE      |  58.4919      |    60       | -1.50815     |      -0.0251358   |    -36       |     36       |           -0.0251358   |