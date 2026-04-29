import yaml
import os
import subprocess
import shutil

# --- CONFIG ---
exp_name = "phis"
metric_to_test = "l2"
# alfa list
#alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0] 
alphas = [0.0]
base_config_path = "../ectuner/myconfigs/myconfig_415_2d_LR.yaml"
results_root = f"../results/tuned_2d_LR/{metric_to_test}/net_toa_tol/"
central_yaml_dir = os.path.join(results_root, "yaml_files")
log_dir = f"../results/logs/logs_2d_LR/{metric_to_test}/net_toa_tol/"

def run_experiment(alpha_val, metric):
    # 1. base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. name and folder setup
    # from 0.95 to "095" for files name
    alpha_str = f"{alpha_val:.2f}".replace('.', '')
    
    file_tag = f"{exp_name}_{metric}_a{alpha_str}"
    output_dir = os.path.join(results_root, f"alfa_{alpha_str}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(central_yaml_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # config update
    config['spatial_tuning']['alpha'] = alpha_val
    config['spatial_tuning']['metric'] = metric
    config['files']['output_dir'] = output_dir 

    # 3. temporary config
    temp_config = f"temp_config_{file_tag}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # 4. output paths
    output_yml_local = os.path.join(output_dir, f"tuned_{file_tag}.yml")
    log_file = os.path.join(log_dir, f"log_tuned_{file_tag}.log")
    
    # Comando (usando sbatch come avevamo ipotizzato)
    # Passiamo sia il percorso locale che quello centralizzato se vuoi che lo faccia lo script python
    job_script = "../jobs/auto_spatial_tuner.sh"
    cmd = f"sbatch --job-name=a{alpha_str} {job_script} {temp_config} {output_yml_local} {log_file}"
    
    print(f">>> Alpha {alpha_val} -> Job sent to SLURM (Config: {temp_config})")
    os.system(cmd)
    
    # Nota: La copia nel central_yaml_dir la facciamo fare direttamente al tuner 
    # o aggiungiamo una riga nel job_template.sh (vedi sotto)

# --- LOOP ---
for a in alphas:
    run_experiment(a, metric_to_test)