# ECtuner

## A tuning tool for EC-Earth

This atmospheric tuning tool uses [ECmean](https://github.com/oloapinivad/ECmean4) output files to compute new suggested values for EC-Earth OIFS parameters.

## Usage

The main configuration file is `config-tuner.yaml`.
You will need to specify a directory (e.g. `ecmean`) containing ECmean output yaml files, one file for each experiment.
Another directory (e.g. `exps`) should contain yaml files listing the parameter values used for each experiments

### Compute sensitivities to model parameters

The python script `sensitivity.py` is used to compute sensitivities of radiative fluxes and other target variables to model parameters. Before doing this, you will need an ensemble of simulations modifying one parameter at a time in both directions. 
In the config file, you need to specify your directories (where to find the experiments, their tuning files and the ecmean output files). If the ecmean calculations have not been computed, the code computes them automatically. 
A pattern for the name of the ensemble members is needed, e.g. "s???", and can be specified either in the config or in command line. The first and last years to consider are also taken from either the config or the command line. The experiment numbering can be arbitrary, the code will automatically recognize the parameters changed in each run. 
```
python sensitivity.py -c config_tuner.yaml
```
If the base experiment is not found, you can set it explicitly:
```
python sensitivity.py s000 -c config_tuner.yaml
```

### Find the optimal tuning of parameters

The python script `ectuner.py` is used to compute new suggested parameter values starting from a `base` experiment.
The two `year` arguments control the range of years used to identify the ECmean output. The tuner can only be run after the sensitivities have been computed (see above).

For example (s000 is the base unperturbed experiment, considering only 1990-1997):
```
python ectuner.py s000 1990 1997
```

will produce this output

|   Parameter   |   New value |   Old value |       Change |   Relative change |   Min change |   Max change |   Rel. dist. from ref. |
|---------------|-------------|-------------|--------------|-------------------|--------------|--------------|------------------------|
|    DETRPEN    | 7.61657e-05 |     7.5e-05 |  1.16569e-06 |       0.0155425   |     -1.5e-05 |      1.5e-05 |            0.0155425   |
|    ENTRDD     | 0.000321388 |     0.0003  |  2.13879e-05 |       0.0712931   |     -6e-05   |      6e-05   |            0.0712931   |
|    ENTRORG    | 0.00143017  |     0.00175 | -0.000319833 |      -0.182761    |     -0.00035 |      0.00035 |           -0.182761    |
|    RCLDIFF    | 2.97434e-06 |     3e-06   | -2.56633e-08 |      -0.00855442  |     -6e-07   |      6e-07   |           -0.00855442  |
| RCLDIFF_CONVI | 6.99846     |     7       | -0.00154054  |      -0.000220077 |     -1.4     |      1.4     |           -0.000220077 |
|  RLCRITSNOW   | 2.8315e-05  |     3e-05   | -1.68496e-06 |      -0.0561653   |     -6e-06   |      6e-06   |           -0.0561653   |
|    RMFDEPS    | 0.295002    |     0.3     | -0.00499813  |      -0.0166604   |     -0.06    |      0.06    |           -0.0166604   |
|    RPRCON     | 0.00141322  |     0.0014  |  1.32195e-05 |       0.00944252  |     -0.00028 |      0.00028 |            0.00944252  |
|   RSNOWLIN2   | 0.0291834   |     0.03    | -0.000816646 |      -0.0272215   |     -0.006   |      0.006   |           -0.0272215   |
|     RVICE     | 0.143017    |     0.13    |  0.0130167   |       0.100129    |     -0.026   |      0.026   |            0.100129    |

More options:
```
./ectuner.py s000 1990 1997  -o tuned-s000.yml -i 0.2 -p 30 -m differential_evolution 
```
writes output to a given file (already in SE compliant format), limits changes to 20% from default OIFS values, 
applies a penalty with weight 30 to being far from the original values, changes the global optimization method.
Actually the default optimization method, dual_annealing, seems to work best, so no need to change it.

This repository contains for now example `exps` and `ecmean` directories.
