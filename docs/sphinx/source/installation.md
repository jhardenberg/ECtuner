# Getting started
## Install Conda
Install Conda (Miniconda/Mamba recommended): [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
## 1. Clone the Repository
Download the source code from the repository:
```bash
git clone https://github.com/jhardenberg/ECtuner
cd ECtuner
```
## 2. Set up the Conda Environment
```bash
conda env create -f environment.yml
conda activate ectuner
```
## 3. Install the ECtuner Package
Install the tool as a Python package. Using the `-e` (editable) flag is recommended if you plan to modify the source code or update it via `git pull` without reinstalling.
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