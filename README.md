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

This repository contains for now example `exps` and `ecmean` directories.