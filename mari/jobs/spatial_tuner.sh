#!/bin/bash
#SBATCH --job-name=tuner_2d_v9
#SBATCH --output=tuner_slurm_%j.out
#SBATCH --error=tuner_slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=128G             
#SBATCH --time=04:00:00        
#SBATCH --qos=np               

source /home/ecme3038/miniforge3/bin/activate
conda activate ectuner


echo "Spatial optimization phis..."

python -u ../../ectuner/ectuner_2D.py phis 2000 2017 \
    -c ../ectuner/myconfigs/myconfig_415_2d_LR.yaml \
    -m dual_annealing \
    -o ../results/tuned_2d_LR/net_TOA/new_alfa_05/tuned_phis_2000_2017_2D_a05.yml \
    > ../results/logs/log_tuned_phis_2D_a05.log 2>&1

echo "Job completato il $(date)"