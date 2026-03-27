# Sensitivity Files Catalog

This directory contains pre-computed sensitivity matrices used by ECtuner to estimate model response to parameter perturbations.

## File Naming Convention
* **1D (YAML)**`sensitivity_{Resolution}_{Version}_{Years}.yaml`
* **2D (NetCDF)**`2D/sensitivity_{Resolution}_{Version}_{Years}_2D.yaml`

---

## Technical Details by Version

### 1. TL255 1991-2000 - Version 2 (Latest) 
* **EC-Earth Version:** EC-Earth4 v4.1.5
* **Reference Years:** 2000-2024
* **Format:** Nested Namelist
* **Parameters Considered (16 total):**
    * **namcumf:** `RPRCON`, `ENTRORG`, `DETRPEN`, `ENTRDD`, `RMFDEPS`
    * **namcldp:** `RVICE`, `RLCRITSNOW`, `RSNOWLIN2`, `RCLDIFF`, `RCLDIFF_CONVI`, `RDEPLIQREFRATE`, `RDEPLIQREFDEPTH`, `RCL_OVERLAPLIQICE`, `RCL_INHOMOGAUT`, `RCL_INHOMOGACC`
    * **naerad:** `RMINICE`

### 2. TL255 1991-2001 / 1990-1997 - Version 1 (Legacy)
* **EC-Earth Version:** Previous EC-Earth4
* **Reference Years:** 1991-2021
* **Format:** Flat
* **Parameters Considered (10 total):**
    `DETRPEN`, `ENTRDD`, `ENTRORG`, `RCLDIFF`, `RCLDIFF_CONVI`, `RLCRITSNOW`, `RMFDEPS`, `RPRCON`, `RSNOWLIN2`, `RVICE`

### 3. TL63 - Low Resolution
* **EC-Earth Version:** Ec-Earth4 v4.1.3
* **Reference Years:** 1990-2000
* **Format:** Nested Namelist
* **Parameters Considered (15 total):**
    * **namcumf:** `RPRCON`, `ENTRORG`, `DETRPEN`, `ENTRDD`, `RMFDEPS`
    * **namcldp:** `RVICE`, `RLCRITSNOW`, `RSNOWLIN2`, `RCLDIFF`, `RCLDIFF_CONVI`, `RDEPLIQREFRATE`, `RDEPLIQREFDEPTH`, `RCL_OVERLAPLIQICE`, `RCL_INHOMOGAUT`, `RCL_INHOMOGACC`

### 4. 2D Spatial Sensitivities TL255 1991-2000 - Version 2 
Located in the `2D/ subdirectory`. These files allow for spatial tuning by providing pixel-by-pixel response maps.
* **EC-Earth Version:** Ec-Earth4 v4.1.5
* **Reference Years:** 2000-2024
* **Format:** NetCDF
* **Dimensions:** `(variable, parameter, lat, lon)`
* **Grid:** `r180x90`
* **Parameters Considered (16 total):**
    * **namcumf:** `RPRCON`, `ENTRORG`, `DETRPEN`, `ENTRDD`, `RMFDEPS`
    * **namcldp:** `RVICE`, `RLCRITSNOW`, `RSNOWLIN2`, `RCLDIFF`, `RCLDIFF_CONVI`, `RDEPLIQREFRATE`, `RDEPLIQREFDEPTH`, `RCL_OVERLAPLIQICE`, `RCL_INHOMOGAUT`, `RCL_INHOMOGACC`
    * **naerad:** `RMINICE`
---

## Usage
When configuring your `config_sens.yaml`, ensure the `reference_parameters` section matches the keys available in the sensitivity file you have selected.