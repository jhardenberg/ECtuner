#!/bin/bash
#SBATCH --job-name=spatial_sens
#SBATCH --output=sens_2d_%j.out
#SBATCH --error=sens_2d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=128G             
#SBATCH --time=04:00:00        
#SBATCH --qos=np               

source /home/ecme3038/miniforge3/bin/activate
conda activate ectuner

# DEFINISCI I PERCORSI (usa percorsi assoluti per sicurezza)
ECTUNER_DIR="/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/ectuner"
CONFIG_PATH="/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/ectuner/myconfigs/config_sens_2D.yaml"

# AGGIUNGI LA CARTELLA UTILS AL PYTHONPATH
# Questo serve perché sensitivity_2D.py possa importare sensitivity.py
export PYTHONPATH="${ECTUNER_DIR}/utils:${PYTHONPATH}"

# LANCIO DEL JOB
python ${ECTUNER_DIR}/utils/sensitivity_2D.py -c ${CONFIG_PATH}

echo "2D sensitivities computed: $(date)"
