===================================
ECtuner: Climate Model Optimization Tool
===================================

ECtuner is an advanced, automated optimization framework designed to objectively tune climate model parameters. 
Originally developed for EC-Earth4 OpenIFS, it allows for robust parameter estimation 
in both **1D (global/regional scalars)** and **2D (spatial maps)**.

By leveraging SciPy minimization algorithms and pre-computed parameter sensitivities, 
ECtuner reduces biases in the tuning process, balancing the trade-offs between 
different atmospheric variables, seasons, and geographical regions.

Core Features
-------------

* **Dual Optimization Modes:** 
  
  * *1D Tuning:* Optimizes based on regional and global scalar means.
  * *2D Spatial Tuning:* Performs pixel-by-pixel optimization using a hybrid spatial-global loss function.

* **Objective Cost Functions:** Highly customizable weighting for specific target fluxes (e.g., ``net_toa``, ``rsnt``), seasons, and geographical domains.
* **Penalty Constraints:** Built-in penalty mechanisms keep tuned parameters within physically realistic bounds relative to their defaults.
* **Diagnostic Suite:** Automatically evaluates initial vs. predicted biases, generates NetCDF spatial diagnostic maps, and exports ready-to-use YAML parameter blocks.
* **HPC Integration:** Includes utility scripts for integration into SLURM-based continuous tuning workflows.

How it works
------------

The optimization process relies on **Parameter Sensitivities**—linear approximations of how the model's output reacts to changes in specific parameters. 
By comparing a baseline model run to observational references (e.g., CERES data), ECtuner algebraically finds the optimal parameter shifts to minimize the overall model bias.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   configuration
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_cli
   api_core
   api_utils
   api_sensitivity