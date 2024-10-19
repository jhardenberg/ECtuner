.. _ectuner:

Overview of ECtuner
-------------------

This atmospheric tuning tool uses `ECmean <https://github.com/oloapinivad/ECmean4>`_
output files to compute new suggested values for EC-Earth OIFS parameters.

Usage
-----

The main configuration file is `config-tuner.yaml`.
You will need to specify a directory (e.g. `ecmean`) containing ECmean output yaml files, one file for each experiment.
Another directory (e.g. `exps`) should contain yaml files listing the parameter values used for each experiments

The notebook `sensitivity.ipynb` (to be substituted with a python tool) is used to compute sensitivities.
The python script `tuner.py` is used to compute new suggested parameter values starting from a `base` experiment.
The two `year` arguments control the range of years used to identify the ECmean output.

For example:

.. code-block:: bash

   ./ectuner.py s000 1990 1997

will produce this output

.. list-table:: 
   :header-rows: 1

   * - Parameter
     - New value
     - Old value
     - Change
     - Relative change
     - Min change
     - Max change
     - Rel. dist. from ref.
   * - DETRPEN
     - 7.61657e-05
     - 7.5e-05
     - 1.16569e-06
     - 0.0155425
     - -1.5e-05
     - 1.5e-05
     - 0.0155425
   * - ENTRDD
     - 0.000321388
     - 0.0003
     - 2.13879e-05
     - 0.0712931
     - -6e-05
     - 6e-05
     - 0.0712931
   * - ENTRORG
     - 0.00143017
     - 0.00175
     - -0.000319833
     - -0.182761
     - -0.00035
     - 0.00035
     - -0.182761
   * - RCLDIFF
     - 2.97434e-06
     - 3e-06
     - -2.56633e-08
     - -0.00855442
     - -6e-07
     - 6e-07
     - -0.00855442
   * - RCLDIFF_CONVI
     - 6.99846
     - 7
     - -0.00154054
     - -0.000220077
     - -1.4
     - 1.4
     - -0.000220077
   * - RLCRITSNOW
     - 2.8315e-05
     - 3e-05
     - -1.68496e-06
     - -0.0561653
     - -6e-06
     - 6e-06
     - -0.0561653
   * - RMFDEPS
     - 0.295002
     - 0.3
     - -0.00499813
     - -0.0166604
     - -0.06
     - 0.06
     - -0.0166604
   * - RPRCON
     - 0.00141322
     - 0.0014
     - 1.32195e-05
     - 0.00944252
     - -0.00028
     - 0.00028
     - 0.00944252
   * - RSNOWLIN2
     - 0.0291834
     - 0.03
     - -0.000816646
     - -0.0272215
     - -0.006
     - 0.006
     - -0.0272215
   * - RVICE
     - 0.143017
     - 0.13
     - 0.0130167
     - 0.100129
     - -0.026
     - 0.026
     - 0.100129


An example with more options:

.. code-block:: bash
   
   ./ectuner.py s000 1990 1997  -o tuned-s000.yml -i 0.2 -p 30 -m differential_evolution 

writes output to a given file (already in SE compliant format), limits changes to 20% from default OIFS values, 
applies a penalty with weight 30 to being far from the original values, changes the global optimization method.
Actually the default optimization method, dual_annealing, seems to work best, so no need to change it.

This repository contains for now example `exps` and `ecmean` directories.
