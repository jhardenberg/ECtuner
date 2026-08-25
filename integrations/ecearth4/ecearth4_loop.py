"""
EC-Earth4 Orchestration Loop.
Coordinates the tuning workflow:
1. Checks if sensitivity files exist (if not, instructs on running ensemble generation).
2. Performs automated pull of the tuning file from ATOS if necessary.
3. Invokes ECtuner core mathematics (1D or 2D).
4. Prepares the next generation configuration for ecearth-quests.
"""

import os
import re
import shutil
import subprocess
from ectuner.libs.config import Config
from ectuner.ectuner import run_1d_tuning, run_2d_tuning
from ectuner.libs import exporter
from ectuner.libs.logger import setup_logger
from ruamel.yaml import YAML


def fetch_yaml_from_hpc(exp_name: str, job_dir: str, target_dest: str) -> bool:
    """
    Pull tuning file from HPC safely by parsing the YAML structure.
    """
    exp_job_dir = os.path.join(job_dir, exp_name)
    main_yml_path = os.path.join(exp_job_dir, f"{exp_name}.yml")

    if not os.path.exists(main_yml_path):
        return False

    tuning_filename = None
    try:
        yaml = YAML()
        with open(main_yml_path, 'r') as f:
            data = yaml.load(f)
            
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'base.context' in item:
                    ctx = item['base.context']
                    if 'model_config' in ctx and 'tuning_file' in ctx['model_config']:
                        # Estrae il percorso (es. '{{se.cli.cwd}}/tuned_phis_l1_a000.yml')
                        full_path = ctx['model_config']['tuning_file']
                        # Prende solo il nome del file finale, ignorando il path/variabili precedenti
                        tuning_filename = os.path.basename(full_path)
    except Exception as e:
        print(f"Error reading {main_yml_path}: {e}")
        return False

    if not tuning_filename:
        print(f"Tuning file parameter not found inside {main_yml_path}")
        return False

    source_path = os.path.join(exp_job_dir, tuning_filename)
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(target_dest), exist_ok=True)
        shutil.copy(source_path, target_dest)
        print(f"  -> [AUTO-PULL] file retrieved : {source_path} -> {target_dest}")
        return True
        
    return False

def check_sensitivities(config: Config, logger) -> bool:
    """
    Verifies the existence of the required sensitivity file (1D or 2D).
    If it doesn't exist, notifies the user that the ensemble must be run first.
    """
    mode = '2d' if config.get('spatial_tuning') else '1d'
    year1 = config.get('args.year1')
    year2 = config.get('args.year2')
    
    if mode == '2d':
        sens_path = config.get('files.sensitivity_nc')
    else:
        sens_path = config.get('files.sensitivity')
        
    if sens_path:
        # Se il path contiene i placeholder, li formattiamo
        sens_file = sens_path.format(year1=year1, year2=year2) if '{year1}' in sens_path else sens_path
        if os.path.exists(sens_file):
            logger.info(f"[CHECK] Sensitivity file found: {sens_file}")
            return True
        else:
            logger.error(f"[CHECK] Sensitivity file NOT found in: {sens_file}")
            logger.error("-> It is necessary to first generate the perturbed ensemble and calculate the sensitivities.")
            return False
    return False

def update_tuning_file_in_yaml(filepath: str, new_tuning_filename: str) -> bool:
    """Helper safely updating the tuning_file path inside user-config.yml or main exp.yml"""
    if not os.path.exists(filepath):
        return False
        
    yaml = YAML()
    yaml.preserve_quotes = True
    updated = False
    
    with open(filepath, 'r') as f:
        data = yaml.load(f)
        
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'base.context' in item:
                ctx = item['base.context']
                if 'model_config' in ctx and 'tuning_file' in ctx['model_config']:
                    # Extract the old path to keep prefixes like '{{se.cli.cwd}}/'
                    old_val = ctx['model_config']['tuning_file']
                    prefix = old_val.rsplit('/', 1)[0] + '/' if '/' in old_val else ''
                    ctx['model_config']['tuning_file'] = f"{prefix}{new_tuning_filename}"
                    updated = True
    elif isinstance(data, dict):
        if 'tuning' in data:
            data['tuning'] = new_tuning_filename
            updated = True
            
    if updated:
        with open(filepath, 'w') as f:
            yaml.dump(data, f)
        return True
    return False


def run_pipeline(
    exp_prev: str, 
    exp_next: str, 
    config_path: str, 
    job_dir: str, 
    quests_dir: str, 
    mode: str = '1d',
    action: str = 'duplicate',
    model_kind: str = 'CPLD',    # AMIP, CPLD, OMIP
    model_sub: str = 'FAST',     # FAST, PALEO
    quest_base_config: str = 'config.yml' # Template di base per quests
) -> None:
    """
    Tuning execution pipeline.
    """
    logger = setup_logger(level='INFO', log_file=f"log_pipeline_{exp_prev}_{exp_next}.log")
    logger.info(f"=== Start EC-Earth4 pipeline: {exp_prev} -> {exp_next} (Mode: {mode.upper()}) ===")
    
    # 1. Load the configuration
    config = Config(config_path, exp=exp_prev)
    
    # 2. Check sensitivities before proceeding with the mathematics
    if not check_sensitivities(config, logger):
        raise FileNotFoundError("No sensitivity files found. Impossible to proceed with tuning. Please run the ensemble generation first.")

    # 3. Ensure the parameter input file exists locally (optional pull from HPC)
    exps_dir = config.get('files.exps')
    params_template = config.get('files.params')
    local_param_file = os.path.join(exps_dir, params_template.format(exp=exp_prev))
    
    if not os.path.exists(local_param_file):
        logger.info(f"Local file {local_param_file} not found. Attempting to retrieve from HPC...")
        success = fetch_yaml_from_hpc(exp_prev, job_dir, local_param_file)
        if not success:
            raise FileNotFoundError(f"Unable to find tuning file for {exp_prev} locally or on HPC.")

    # 4. Execute the optimization via the ECtuner library
    if mode == '2d':
        result = run_2d_tuning(config, logger)
    else:
        result = run_1d_tuning(config, logger)

    # 5. Save the optimized YAML file
    output_dir = config.get('files.output_dir', './')
    os.makedirs(output_dir, exist_ok=True)
    final_tuning_name = f"tuned_{exp_next}.yml"
    result_path = os.path.join(output_dir, final_tuning_name)
    
    exporter.save_model_yaml(
        result, result_path, 
        config.get('parameter_group', {}), 
        config.get('weights', {}), 
        config.get('weights_region', {})
    )

    diag_name = f"diagnostics_{exp_next}.yaml"
    diag_path = os.path.join(output_dir, diag_name)
    exporter.save_diagnostics_yaml(result, diag_path)
    logger.info(f"Model YAML and Diagnostics successfully saved for {exp_next}")

    logger.info(f"=== Optimization completed! Ready to {action.upper()} the job for {exp_next} ===")
    exp_job_dir = os.path.join(job_dir, exp_next)

    # 6. generate or duplicate the job in quests
    if action == 'generate':
        # Strategy A: Use generate-job.py (Standard)
        quests_tuning_folder = os.path.join(quests_dir, "tuning")
        os.makedirs(quests_tuning_folder, exist_ok=True)
        final_tuning_path = os.path.join(quests_tuning_folder, final_tuning_name)
        shutil.copy(result_path, final_tuning_path)
        logger.info(f"Optimized file copied to quests: {final_tuning_path}")

        yaml_writer = YAML()
        yaml_writer.preserve_quotes = True
        base_quest_config_path = os.path.join(quests_dir, quest_base_config)
        temp_quest_config_path = os.path.join(quests_dir, f"config_{exp_next}_temp.yml")

        try:
            with open(base_quest_config_path, 'r') as f:
                quest_config = yaml_writer.load(f)
            quest_config['tuning'] = final_tuning_name
            with open(temp_quest_config_path, 'w') as f:
                yaml_writer.dump(quest_config, f)
                
            cmd_generate = ["python", "generate-job.py", "-k", model_kind, "-m", model_sub, "-e", exp_next, "-c", temp_quest_config_path]
            logger.info(f"Executing: {' '.join(cmd_generate)}")
            subprocess.run(cmd_generate, cwd=quests_dir, check=True)
        finally:
            if os.path.exists(temp_quest_config_path):
                os.remove(temp_quest_config_path)

    elif action == 'duplicate':
        # Strategy B: Use duplicate-job.py (Preserves local modifications)
        cmd_duplicate = [
            "python", "duplicate-job.py", 
            "--expname1", exp_prev, 
            "--expname2", exp_next, 
            "-c", quest_base_config
        ]
        logger.info(f"Executing: {' '.join(cmd_duplicate)}")
        subprocess.run(cmd_duplicate, cwd=quests_dir, check=True)

        # Move the new tuning file directly into the newly cloned job directory
        job_tuning_path = os.path.join(exp_job_dir, final_tuning_name)
        shutil.copy(result_path, job_tuning_path)
        logger.info(f"Injected new tuning file into: {job_tuning_path}")

        # Update the YAML configuration inside the job folder to point to the new tuning file
        user_config_path = os.path.join(exp_job_dir, "user-config.yml")
        main_config_path = os.path.join(exp_job_dir, f"{exp_next}.yml")
        
        if not update_tuning_file_in_yaml(user_config_path, final_tuning_name):
            update_tuning_file_in_yaml(main_config_path, final_tuning_name)

    #7. Launch the job
    cmd_launch = ["./launch.sh"]
    try:
        logger.info(f"Submitting job from: {exp_job_dir}")
        subprocess.run(cmd_launch, cwd=exp_job_dir, check=True)
        logger.info(f"=== Loop completed successfully! {exp_next} is running on SLURM ===")
    except subprocess.CalledProcessError as e:
        logger.error(f"Critical error during job submission: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    # Valori di default 
    BASE_PERM = os.environ.get("HPCPERM", "/ec/res4/hpcperm/ecme3038")
    WORKSPACE = os.path.join(BASE_PERM, "ecearth/ecearth4")
    DEFAULT_CONFIG = os.path.join(WORKSPACE, "ECtuner/mari/ectuner/myconfigs/myconfig_415_LR.yaml")
    DEFAULT_JOB_DIR = os.path.join(WORKSPACE, "jobs/v4.1.5/")
    DEFAULT_QUESTS_DIR = os.path.join(WORKSPACE, "ecearth-quests/ece4")

    parser = argparse.ArgumentParser(description="EC-Earth4 Orchestration Loop")
    parser.add_argument("exp_prev", type=str, help="Name of the terminated experiment")
    parser.add_argument("exp_next", type=str, help="Name of the new experiment to generate")

    parser.add_argument("-a", "--action", type=str, choices=['generate', 'duplicate'], default='duplicate', help="Action to create the new job")
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG, help="Path to ECtuner config")
    parser.add_argument("--job_dir", type=str, default=DEFAULT_JOB_DIR, help="Jobs directory")
    parser.add_argument("--quests_dir", type=str, default=DEFAULT_QUESTS_DIR, help="Quests directory")

    parser.add_argument("-m", "--mode", type=str, choices=['1d', '2d'], default='1d', help="Tuning mode")
    parser.add_argument("-k", "--kind", type=str, default='CPLD', help="Model type (CPLD, AMIP, OMIP)")
    parser.add_argument("--submodel", type=str, default='FAST', help="Sub-configuration (FAST, PALEO)")
    parser.add_argument("--quest_config", type=str, default='config_TL63.yml', help="Base configuration for quests")

    args = parser.parse_args()

    run_pipeline(
        exp_prev=args.exp_prev,
        exp_next=args.exp_next,
        config_path=args.config,
        job_dir=args.job_dir,
        quests_dir=args.quests_dir,
        mode=args.mode,
        action=args.action,
        model_kind=args.kind,
        model_sub=args.submodel,
        quest_base_config=args.quest_config
    )

# how to use: 
# python integrations/ecearth4/ecearth4_loop.py exp_old exp_new -a generate -c /path/to/config.yaml