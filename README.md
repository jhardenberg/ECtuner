# ECtuner

## A tuning tool for EC-Earth

This atmospheric tuning tool uses [ECmean](https://github.com/oloapinivad/ECmean4) output files to compute new suggested values for EC-Earth OIFS parameters.

### Usage

The main configuration file is `config-tuner.yaml`.
You will need to specify a directory (e.g. `ecmean`) containing ECmean output yaml files, one file for each experiment.
Another directory (e.g. `exps`) should contain yaml files listing the parameter values used for each experiments

The notebook `sensitivity.ipynb` (to be substituted with a python tool) is used to compute sensitivities.
The python script `tuner.py` is used to compute new suggested parameter values starting from a `base` experiment.
The two `year` arguments control the range of years used to identify the ECmean output.

For example:
```
tuner.py s000 1990 1997
```

will produce this output

|   Parameter     |   New value  |       Change   |   Relative change  |   Max change  |
|-----------------|--------------|----------------|--------------------|---------------|
|    DETRPEN      | 8.56686e-05  |  1.06686e-05   |        0.142248     |      1.5e-05  |
|    ENTRDD       | 0.000359189  |  5.91894e-05   |        0.197298     |      6e-05    |
|    ENTRORG      | 0.0014007    | -0.000349304   |       -0.199602     |      0.00035  |
|    RCLDIFF      | 3.00461e-06  |  4.61308e-09   |        0.00153769   |      6e-07    |
| RCLDIFF_CONVI   | 6.96824      | -0.0317619     |       -0.00453741   |      1.4      |
|  RLCRITSNOW     | 2.40017e-05  | -5.99834e-06   |       -0.199945     |      6e-06    |
|    RMFDEPS      | 0.248534     | -0.051466      |       -0.171553     |      0.06     |
|    RPRCON       | 0.00139787   | -2.13319e-06   |       -0.0015237    |      0.00028  |
|   RSNOWLIN2     | 0.0241312    | -0.00586877    |       -0.195626     |      0.006    |
|     RVICE       | 0.147864     |  0.0178643     |        0.137418     |      0.026    |


This repository contains for now example `exps` and `ecmean` directories.