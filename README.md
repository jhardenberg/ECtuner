# ECtuner

## Atmospheric tuning tool for EC-Earth

ECtuner is an optimization framework based on [ECmean](https://github.com/oloapinivad/ECmean4) output files to compute new suggested values for EC-Earth OIFS parameters. It allowd to minimize model biases relative to obsevation by balancing radiative fluxes and state variables.


### Configuration and Architecture

The tool relies on a structured interaction between three configuration levels. ECmean4 is integrated into the workflow, specifically within the sensitivity analysis stage.
1. ECmean Config (`config_ecmean.tmpl`): template (to modify by the user) used by the tool to compute global means if they are missing during the sensitivity stage.
2. Sensitivity Config (`config_sens.yaml`): defines the perturbation ensemble, naming patterns, and paths for the sensitivity runs.
3. Tuner Config (`config_tuner.yaml`): the master file for the optimization process (weights, penalties, and target paths).


### Pre-computed sensitivities 

To save computational time, this repository includes several pre-calculated sensitivity matrices for standard EC-Earth4 configurations. You can find them in the ectuner/sensitivities/ directory where you can also find a dedicated README.md. 


### Required Data Structure

1. For Sensitivity Analysis (`sensitivity.py`)
To map the model's response, you need a full ensemble of simulations:
- Ensemble ECmean Outputs: one `.yml` file for the unperturbed (base) experiment and two (or more) for each parameter (one for positive and one for negative perturbations).
- Ensemble Parameter Files: corresponding `.yml` files listing the perturbed values for each run in the ensemble.
    Note: If the ECmean global means are not yet computed, `sensitivity.py` can trigger the calculation automatically using the provided template.

2. For Target Tuning (`ectuner.py`)
To find the optimal values for a specific experiment, you only need the data for that single target:
- Target ECmean Output: The `.yml` file containing the climate state of the experiment you wish to tune.
- Target Parameter File: The `.yml` file with the parameters used for that specific simulation.

Important: ectuner.py cannot run ECmean. You must import or compute the global mean for your target experiment externally before running the tuner.


### Workflow

1. Sensitivity Stage
The script `sensitivity.py` computes sensitivities of radiative fluxes and target variables to model parameters. It automatically recognizes the parameters changed in each run.
```
# Basic usage (uses defaults from config)
python ectuner/utils/sensitivity.py -c config_sens.yaml

# Explicitly setting the base experiment and years
python ectuner/utils/sensitivity.py s000 1990 1997 -c config_sens.yaml
```
The tool identifies the ensemble members, extracts the parameter changes, and builds the response file.

2. Tuning Stage

Once you have the sensitivity file, the global mean and the `.yml` file with the parameters used by your target experiment, you can use the script `ectuner.py` to compute the suggested parameter values.
```
# Example tuning a target experiment for a 10-year period
python ectuner/ectuner.py <exp_id> 1991 2000 -c config_tuner_<exp_id>.yaml -o tuned_parameters<exp_id>.yml -m dual_annealing > tuning<exp_id>.log 2>&1
```

#### Command Line Options
You can override configuration defaults using the following flags:
- -p, --penalty: (Default: 10) Sets the weight for the penalty term. Higher values keep the new parameters closer to the OIFS defaults to avoid physically unrealistic solutions.
- -i, --inc: (Default: 0.2) The maximum allowed fractional change (e.g., 0.1 limits changes to ±10% of the reference value).
- -m, --method: Choose the optimization algorithm. dual_annealing is recommended for most cases, but differential_evolution is also supported.
- -o, --output: Specifies the path to save the suggested tuning as a YAML file, formatted for the EC-Earth4 Script Engine (SE).
- --freeze: A list of parameters to keep fixed at their current values during the optimization.


#### Advanced Physics Options
- Delta T Adjustment (-dT): Applies a reference correction based on temperature using slopes defined in `slopes.yaml`. Crucial for tuning coupled model simulations.
- Model Imbalance (-imb): Corrects the net_toa target to cope with intrinsic model energy imbalances (mainly for low-resolution configurations).


### Optimization Output
When running the tuner, the tool identifies the optimal parameter set. Below is an example of a tuning run involving 16 parameters:

|     Parameter     |     New value |   Old value |       Change |   Relative change |   Min change |   Max change |   Rel. dist. from ref. |
|-------------------+---------------+-------------+--------------+-------------------+--------------+--------------+------------------------|
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