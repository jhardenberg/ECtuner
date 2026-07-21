"""
Experiment Automation Module.

Provides wrappers to interact with SLURM for batch submission 
of EC-Earth tuning experiments.
"""
import os
import subprocess
from copy import deepcopy
from typing import Any
from .config import Config


class ExperimentSLURM:
    """
    Prepares and submits a tuning experiment to the SLURM scheduler.

    Allows on-the-fly modification of configuration values before dispatching
    the batch job.
    """
    def __init__(self, base_config: Config, exp_name: str) -> None:
        """
        Initializes the SLURM experiment wrapper.

        Args:
            base_config: The base ECtuner configuration object.
            exp_name: The targeted experiment identifier (e.g., 'aa00').
        """
        self.base_config = deepcopy(base_config)
        self.base_config.set('args.exp', exp_name)
        
    def set(self, key_path: str, value: Any) -> None:
        """
        Modifies a configuration value dynamically before submission.
        
        Args:
            key_path: Dot-notated string indicating the config key (e.g., 'args.inc').
            value: The new value to set.
        """
        self.base_config.set(key_path, value)
        
    def submit(self, job_template: str, run_tag: str) -> None:
        """
        Saves the current configuration to disk and submits the job to SLURM.
        
        Args:
            job_template: Path to the SLURM bash script template.
            run_tag: Identifier tag for this specific run.
        """
        out_dir = self.base_config.get('files.output_dir', './output')
        os.makedirs(out_dir, exist_ok=True)
        temp_config_path = os.path.join(out_dir, f"temp_{run_tag}.yaml")
        self.base_config.save(temp_config_path)
       
        out_yml = os.path.join(out_dir, f"tuned_{run_tag}.yml")
        log_file = os.path.join(out_dir, f"log_{run_tag}.log")
        slurm_out = os.path.join(out_dir, f"slurm_{run_tag}.out")
        
        # Scelta del comando (1D o 2D a seconda del file config)
        mode = '2d' if self.base_config.get('spatial_tuning') else '1d'

        exp_name = self.base_config.get('args.exp')
        year1 = self.base_config.get('args.year1')
        year2 = self.base_config.get('args.year2')
        
        cmd = [
            "sbatch", f"--job-name={run_tag}", f"--output={slurm_out}", 
            job_template, 
            mode, temp_config_path, out_yml, log_file, exp_name, str(year1), str(year2), run_tag
        ]
        
        subprocess.run(cmd, check=True)
        print(f">>> Job {run_tag} sent. (Outputs in {out_dir}).")