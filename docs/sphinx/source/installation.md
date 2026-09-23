# Installation
## Prerequistites
Ensure you have Conda installed (Miniconda/Mamba recommended): [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
## 1. Clone the Repository
Download the source code from the repository:
```bash
git clone [https://github.com/jhardenberg/ECtuner](https://github.com/jhardenberg/ECtuner)
cd ECtuner
```
## 2. Set up the Conda Environment
Create and activate the dedicated environment containing all required dependencies (like `xarray`, `scipy`, and `ecmean`)
```bash
conda env create -f environment.yml
conda activate ectuner
```
## 3. Install the ECtuner Package
Install the tool as a Python package. Using the `-e` (editable) flag is recommended if you plan to modify the source code or update it via `git pull` without needing to reinstall.
```bash
pip install -e .
```
> *(Note: This step reads the `pyproject.toml` file and creates the `ectuner` CLI command.)*
## 4. Verify Installation
To confirm that the installation was successful and the CLI is available, run:
```bash
ectuner --help
```
You should see the help menu displaying the available subcommands (`1d` and `2d`).
You can also verify that the sensitivity scripts have been successfully registered by running:
```bash
ectuner-sens-1d --help
ectuner-sens-2d --help
```