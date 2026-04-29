#!/bin/bash
#SBATCH --job-name=tuner_2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --qos=np

CONFIG=$1   # Riceve temp_config_...
OUTPUT=$2   # Riceve il percorso nella cartella alfa_X
LOG=$3      # Riceve il percorso del log

# central dir 
CENTRAL_DIR="../results/tuned_2d_LR/l2/all_flux_tol/yaml_files"

source /home/ecme3038/miniforge3/etc/profile.d/conda.sh
conda activate ectuner

# 1. Esecuzione del Tuner
python -u ../../ectuner/ectuner_2D.py phis 2000 2017 \
    -c $CONFIG \
    -m dual_annealing \
    -o $OUTPUT > $LOG 2>&1

# 2. COPIA AUTOMATICA nel raccoglitore centralizzato
if [ -f "$OUTPUT" ]; then
    cp "$OUTPUT" "$CENTRAL_DIR/"
    echo "Copia del file YAML inviata a $CENTRAL_DIR" >> $LOG
fi

# 3. Pulizia config temporaneo
rm $CONFIG