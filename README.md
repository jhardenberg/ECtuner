# ECtuner
#[![Documentation Status](https://readthedocs.org/projects/ectuner/badge/?version=latest)](https://ectuner.readthedocs.io/)

ECtuner is an advanced optimization framework designed to objectively tune EC-Earth4 OpenIFS parameters in both 1D (global scalars) and 2D (spatial maps).

## Features
* **1D & 2D Tuning**: Scalar optimization and pixel-by-pixel spatial tuning.
* **Automated Orchestration**: SLURM-integrated loops for EC-Earth4.
* **Diagnostic Suite**: Pareto fronts, spatial error maps, and parameter heatmaps.

## Documentation
For full installation instructions, YAML configuration details, and the Python API reference, read the Official Documentation.

## Quick installation
```bash
git clone [https://github.com/your-repo/ECtuner.git](https://github.com/your-repo/ECtuner.git)
cd ECtuner
conda env create -f environment.yml
pip install -e .
```

## Quick Start
Compute sensitivities and run the optimizer via CLI:
```bash
ectuner 1d -c config.yaml -o output/tuned_exp.yml exp 1990 2000
```

## Example Ouput
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