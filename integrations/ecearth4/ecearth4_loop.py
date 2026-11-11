"""
EC-Earth4 Orchestration Loop.
Coordinates the tuning workflow:
1. Determines the evaluation window (command line, config, or auto-detected from the parent).
2. Checks that the sensitivity file exists and that the parent covers the evaluation window
   (model output; in 1D mode also the ECmean global-mean file).
3. Retrieves the parameter file of the parent (local exps folder or job folder).
4. Runs the ECtuner optimization (1D or 2D).
5. Saves the tuning and diagnostics files, copies the tuning file to the exps folder (where the
   next iteration reads it as the parameters of its parent) and optionally records the run in a
   manifest readable by ``metrics.validation_table``.
6. Prepares the job of the new run with ecearth-quests (duplicate or generate) and optionally
   submits it.
"""

import glob
import os
import re
import shutil
import subprocess
import time 
from ruamel.yaml import YAML

from ectuner.libs.config import Config
from ectuner.ectuner import run_1d_tuning, run_2d_tuning
from ectuner.libs import exporter
from ectuner.libs.logger import setup_logger
from ectuner.libs.utils import autodetect_experiment_years


def fetch_yaml_from_hpc(exp_name: str, job_dir: str, target_dest: str) -> bool:
    """Copies the tuning file referenced in the job configuration of ``exp_name`` to ``target_dest``."""
    exp_job_dir = os.path.join(job_dir, exp_name)
    main_yml_path = os.path.join(exp_job_dir, f"{exp_name}.yml")
    if not os.path.exists(main_yml_path):
        return False
 
    tuning_filename = None
    try:
        with open(main_yml_path, 'r') as f:
            data = YAML().load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'base.context' in item:
                    ctx = item['base.context']
                    if 'model_config' in ctx and 'tuning_file' in ctx['model_config']:
                        # e.g. '{{se.cli.cwd}}/tuning_v000.yml' -> 'tuning_v000.yml'
                        tuning_filename = os.path.basename(ctx['model_config']['tuning_file'])
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

def check_sensitivities(config: Config, logger, mode: str) -> bool:
    """
    Verifies the existence of the required sensitivity file (1D or 2D).
    If it doesn't exist, notifies the user that the ensemble must be run first.
    """
    
    if mode == '2d':
        sens_path = config.get('files.sensitivity_nc')
    else:
        sens_path = config.get('files.sensitivity')
        
    if sens_path:
        sens_file = sens_path
        if os.path.exists(sens_file):
            logger.info(f"[CHECK] Sensitivity file found: {sens_file}")
            return True
        else:
            logger.error(f"[CHECK] Sensitivity file NOT found in: {sens_file}")
            logger.error("-> It is necessary to first generate the perturbed ensemble and calculate the sensitivities.")
            return False
    return False

def check_parent_output(config: Config, exp_prev: str, logger) -> None:
    """
    Aborts if the parent run is missing years of the evaluation window.
 
    Looks for the monthly OpenIFS output ``<raw_dir>/<exp>/output/oifs/<exp>_atm_cmip6_1m_<YYYY>*.nc``.
    If no file matches this pattern (different output naming), only a warning is issued.
    """
    raw_dir = config.get('files.raw_dir')
    if not raw_dir:
        logger.warning("[CHECK] files.raw_dir not set: parent output not checked.")
        return
    y1, y2 = int(config.get('args.year1')), int(config.get('args.year2'))
    pattern = os.path.join(raw_dir, exp_prev, 'output', 'oifs', f"{exp_prev}_atm_cmip6_1m_*.nc")
    years = {int(m.group(1)) for f in glob.glob(pattern)
             if (m := re.search(r'_1m_(\d{4})', os.path.basename(f)))}
    if not years:
        logger.warning(f"[CHECK] No file matches {pattern}: parent output not checked.")
        return
    missing = sorted(set(range(y1, y2 + 1)) - years)
    if missing:
        raise RuntimeError(f"{exp_prev} is missing years {missing}: is the run finished?")
    logger.info(f"[CHECK] {exp_prev} covers the evaluation window {y1}-{y2}")
 
 
def check_ecmean_file(config: Config, exp_prev: str, logger) -> str:
    """
    1D mode: checks that ECmean has been run on the parent over the evaluation window.
 
    Looks in ``files.ecmean`` for the file named by ``files.base`` (formatted with exp, year1,
    year2), or for ``GM_*_<exp>_*_<year1>_<year2>.yml`` if ``files.base`` is not set.
    If the file is missing and ``files.ecmean_config`` is set, only a warning is issued, since the
    1D loader then computes the global means with ECmean. Returns the path of the file found (or None).
    """
    ecmean_dir = config.get('files.ecmean')
    y1, y2 = int(config.get('args.year1')), int(config.get('args.year2'))
    if not ecmean_dir or not os.path.isdir(ecmean_dir):
        raise FileNotFoundError(f"ECmean folder not found (files.ecmean = {ecmean_dir}).")
    base = config.get('files.base')          # e.g. GM_EC26-PDAY_{exp}_EC-Earth4_r1i1p1f1_{year1}_{year2}.yml
    pattern = (base.format(exp=exp_prev, year1=y1, year2=y2) if base
               else f"GM_*_{exp_prev}_*_{y1}_{y2}.yml")
    files = sorted(glob.glob(os.path.join(ecmean_dir, pattern)))
    if not files and config.get('files.ecmean_config'):
        # The 1D loader computes the missing global means with ECmean (files.ecmean_config)
        logger.warning(f"[CHECK] No ECmean file for {exp_prev} over {y1}-{y2}: "
                       f"it will be computed with ECmean ({config.get('files.ecmean_config')}).")
        return None
    if not files:
        other = sorted(glob.glob(os.path.join(ecmean_dir, f"GM_*_{exp_prev}_*.yml")))
        hint = (f" Files for {exp_prev} on other windows: {[os.path.basename(f) for f in other]}"
                if other else f" No ECmean file for {exp_prev}: run ECmean on the parent first.")
        raise FileNotFoundError(f"No ECmean file for {exp_prev} over {y1}-{y2} in {ecmean_dir}.{hint}")
    if len(files) > 1:
        logger.warning(f"[CHECK] {len(files)} ECmean files for {exp_prev} {y1}-{y2}; "
                       f"check that the reference is the intended one: {[os.path.basename(f) for f in files]}")
    logger.info(f"[CHECK] ECmean file of the parent: {os.path.basename(files[0])}")
    return files[0]

def update_manifest(manifest_path: str, exp_next: str, entry: dict) -> None:
    """
    Adds or updates the entry of ``exp_next`` in a runs manifest (read by ``validation_table``).
 
    ``iteration`` is the iteration of the parent plus one (a parent missing from the manifest,
    e.g. the default run, counts as iteration 0).
    """
    yaml = YAML()
    data = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            data = yaml.load(f) or {}
    parent = entry.get('parent')
    if parent not in data:
        data[parent] = {'parent': None}
    entry['iteration'] = int((data.get(parent) or {}).get('iteration', 0)) + 1
    data[exp_next] = entry
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, 'w') as f:
        yaml.dump(data, f)
        
def update_tuning_file_in_yaml(filepath: str, new_tuning_filename: str) -> bool:
    """Updates the ``tuning_file`` path inside user-config.yml or the main experiment yml."""
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
                    # Keep prefixes such as '{{se.cli.cwd}}/'
                    old_val = ctx['model_config']['tuning_file']
                    prefix = old_val.rsplit('/', 1)[0] + '/' if '/' in old_val else ''
                    ctx['model_config']['tuning_file'] = f"{prefix}{new_tuning_filename}"
                    updated = True
    elif isinstance(data, dict) and 'tuning' in data:
        data['tuning'] = new_tuning_filename
        updated = True
    if updated:
        with open(filepath, 'w') as f:
            yaml.dump(data, f)
    return updated
 
 
def submit_job(exp_job_dir: str, exp_next: str, logger) -> None:
    """
    Runs ``launch.sh`` of the new job.
 
    The SLURM_* variables are removed from the environment: when this script runs inside a SLURM
    job, ScriptEngine would otherwise run the model inside the current allocation instead of
    submitting a new job. Afterwards the queue is checked for ``ECE4_<exp>`` (the default job name
    in ecearth-quests); a missing job only triggers a warning, since the name can be customized.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
    logger.info(f"Submitting job from: {exp_job_dir}")
    subprocess.run(["./launch.sh"], cwd=exp_job_dir, check=True, env=env)
    time.sleep(10)
    queued = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', f'ECE4_{exp_next}'],
                            capture_output=True, text=True).stdout.strip()
    if queued:
        logger.info(f"=== {exp_next} submitted ===")
    else:
        logger.warning(f"launch.sh returned, but no job named ECE4_{exp_next} is in the queue: "
                       f"check the job log in {exp_job_dir}.")
 


def run_pipeline(
    exp_prev: str,
    exp_next: str,
    config_path: str,
    job_dir: str,
    quests_dir: str,
    mode: str = '1d',
    action: str = 'duplicate',
    model_kind: str = 'CPLD',               # AMIP, CPLD
    model_sub: str = 'FAST',                # FAST, PALEO
    quest_base_config: str = 'config_TL63.yml',
    year1: int = None,
    year2: int = None,
    bounds_around: str = None,
    alpha: float=None,
    inc: float = None,
    drift_cap: float = None,
    output_dir: str = None,
    manifest: str = None,
    submit: bool = False,
) -> None:
    """
    Runs one tuning iteration ``exp_prev -> exp_next`` and prepares the new job.
 
    Arguments left to None keep the value of the config (the evaluation window is auto-detected
    from the parent if it is set neither here nor in the config).
    """
    logger = setup_logger(level='INFO', log_file=f"log_pipeline_{exp_prev}_{exp_next}.log")
    logger.info(f"=== Start EC-Earth4 pipeline: {exp_prev} -> {exp_next} (mode {mode.upper()}) ===")
 
    # 1. Configuration
    config = Config(config_path, exp=exp_prev)
    if year1 is not None and year2 is not None:
        config.set('args.year1', int(year1))
        config.set('args.year2', int(year2))
    elif config.get('args.year1') is None or config.get('args.year2') is None:
        logger.info("Evaluation years not set: auto-detecting them from the parent...")
        y1, y2 = autodetect_experiment_years(config, exp_prev)
        config.set('args.year1', y1)
        config.set('args.year2', y2)
    for key, val in (('bounds_around', bounds_around), ('inc', inc), ('drift_cap', drift_cap)):
        if val is not None:
            config.set(f'args.{key}', val)
    if alpha is not None:
        config.set('spatial_tuning.alpha', float(alpha))
    # 2D: names the map file diagnostics_2d_<exp_prev>_<exp_next>.nc, as expected by validation_table
    config.set('args.output_tag', exp_next)
    output_dir = output_dir or config.get('files.output_dir', './')
    config.set('files.output_dir', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Evaluation window {config.get('args.year1')}-{config.get('args.year2')} | "
                f"output {output_dir}")
 
    # 2. Checks
    if not check_sensitivities(config, logger, mode):
        raise FileNotFoundError("No sensitivity file found: run the ensemble generation first.")
    check_parent_output(config, exp_prev, logger)
    if mode == '1d':
        check_ecmean_file(config, exp_prev, logger)
 
    # 3. Parameter file of the parent
    exps_dir = config.get('files.exps')
    params_template = config.get('files.params')
    local_param_file = os.path.join(exps_dir, params_template.format(exp=exp_prev))
    if not os.path.exists(local_param_file):
        logger.info(f"{local_param_file} not found locally, retrieving it from the job folder...")
        if not fetch_yaml_from_hpc(exp_prev, job_dir, local_param_file):
            raise FileNotFoundError(f"Unable to find the tuning file of {exp_prev}.")
 
    # 4. Optimization
    result = run_2d_tuning(config, logger) if mode == '2d' else run_1d_tuning(config, logger)
 
    # 5. Outputs; the tuning file also goes to exps, where the next iteration reads its parent
    final_tuning_name = f"tuning_{exp_next}.yml"
    result_path = os.path.join(output_dir, final_tuning_name)
    exporter.save_model_yaml(result, result_path, config.get('parameter_group', {}),
                             config.get('weights', {}), config.get('weights_region', {}))
    exporter.save_diagnostics_yaml(result, os.path.join(output_dir, f"diagnostics_{exp_next}.yaml"))
    shutil.copy(result_path, os.path.join(exps_dir, params_template.format(exp=exp_next)))
    if manifest:
        alpha = config.get('spatial_tuning.alpha') if mode == '2d' else None
        update_manifest(manifest, exp_next,
                        {'parent': exp_prev, 'mode': mode,
                         'alpha': None if alpha is None else float(alpha),
                         'eval_years': f"{config.get('args.year1')}-{config.get('args.year2')}",
                         'drift_rms': result.metrics.get('drift_rms'),
                         'n_at_bound': result.metrics.get('n_at_bound')})
        logger.info(f"Manifest updated: {manifest}")
    logger.info(f"Tuning file of {exp_next} saved in {output_dir} and copied to {exps_dir}")
 
    # 6. Job of the new run
    exp_job_dir = os.path.join(job_dir, exp_next)
    if action == 'generate':
        quests_tuning_folder = os.path.join(quests_dir, "tuning")
        os.makedirs(quests_tuning_folder, exist_ok=True)
        shutil.copy(result_path, os.path.join(quests_tuning_folder, final_tuning_name))
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
            cmd = ["python", "generate-job.py", "-k", model_kind, "-m", model_sub,
                   "-e", exp_next, "-c", temp_quest_config_path]
            logger.info(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=quests_dir, check=True)
        finally:
            if os.path.exists(temp_quest_config_path):
                os.remove(temp_quest_config_path)
 
    elif action == 'duplicate':
        cmd = ["python", "duplicate-job.py", "--expname1", exp_prev, "--expname2", exp_next,
               "-c", quest_base_config]
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=quests_dir, check=True)
        shutil.copy(result_path, os.path.join(exp_job_dir, final_tuning_name))
        if not update_tuning_file_in_yaml(os.path.join(exp_job_dir, "user-config.yml"), final_tuning_name):
            update_tuning_file_in_yaml(os.path.join(exp_job_dir, f"{exp_next}.yml"), final_tuning_name)
 
    # 7. Submission
    if submit:
        submit_job(exp_job_dir, exp_next, logger)
    else:
        logger.info(f"=== Job for {exp_next} prepared in {exp_job_dir} (not submitted) ===")
 
 
if __name__ == "__main__":
    import argparse
 
    # Defaults relative to $HPCPERM (or the current folder); adjust them or pass the options
    BASE_DIR = os.environ.get("HPCPERM", os.getcwd())
    DEFAULT_CONFIG = os.path.join(BASE_DIR, "path/to/your/config.yaml")
    DEFAULT_JOB_DIR = os.path.join(BASE_DIR, "path/to/your/jobs")
    DEFAULT_QUESTS_DIR = os.path.join(BASE_DIR, "ecearth-quests/ece4")
 
    parser = argparse.ArgumentParser(description="EC-Earth4 tuning loop: one iteration")
    parser.add_argument("exp_prev", help="Parent experiment (finished)")
    parser.add_argument("exp_next", help="New experiment to prepare")
    parser.add_argument("-a", "--action", choices=['generate', 'duplicate'], default='duplicate',
                        help="How to create the new job")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="ECtuner config")
    parser.add_argument("--job_dir", default=DEFAULT_JOB_DIR, help="Jobs directory")
    parser.add_argument("--quests_dir", default=DEFAULT_QUESTS_DIR, help="ecearth-quests directory")
    parser.add_argument("-m", "--mode", choices=['1d', '2d'], default='1d', help="Tuning mode")
    parser.add_argument("-k", "--kind", default='CPLD', help="Model type (CPLD, AMIP, OMIP)")
    parser.add_argument("--submodel", default='FAST', help="Sub-configuration (FAST, PALEO)")
    parser.add_argument("--quest_config", default='config_TL63.yml', help="Base configuration for quests")
    parser.add_argument("--year1", type=int, help="First evaluation year (default: config or auto-detect)")
    parser.add_argument("--year2", type=int, help="Last evaluation year (default: config or auto-detect)")
    parser.add_argument("--bounds_around",
                        help="Centre of the parameter bounds, e.g. 'current' (default: config)")
    parser.add_argument("--inc", type=float, help="Relative half-width of the bounds (default: config)")
    parser.add_argument("--drift_cap", type=float, help="Maximum relative drift from the default (default: config)")
    parser.add_argument("--alpha", type=float, help="2D only: weight of the global term (default: config)")
    parser.add_argument("--output_dir", help="Folder of the tuning outputs (default: files.output_dir)")
    parser.add_argument("--manifest", help="Runs manifest to update (for validation_table); not written if omitted")
    parser.add_argument("--submit", action='store_true', help="Run launch.sh after preparing the job")
    args = parser.parse_args()
 
    run_pipeline(exp_prev=args.exp_prev, exp_next=args.exp_next, config_path=args.config,
                 job_dir=args.job_dir, quests_dir=args.quests_dir, mode=args.mode,
                 action=args.action, model_kind=args.kind, model_sub=args.submodel,
                 quest_base_config=args.quest_config, year1=args.year1, year2=args.year2,
                 bounds_around=args.bounds_around, inc=args.inc, drift_cap=args.drift_cap,
                 alpha=args.alpha, output_dir=args.output_dir, manifest=args.manifest, submit=args.submit)
 
# Examples:
# python integrations/ecearth4/ecearth4_loop.py exp_old exp_new -c /path/to/config.yaml
# python integrations/ecearth4/ecearth4_loop.py r000 r100 -m 2d -k AMIP --year1 2000 --year2 2009 \
#        --bounds_around current --inc 0.1 --drift_cap 0.5 --alpha 0.0--manifest runs_manifest.yaml --submit
 