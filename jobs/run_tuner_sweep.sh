#!/bin/bash
#SBATCH --job-name=tuner_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --qos=np

# args from ExperimentSLURM.submit() in test_tuner.ipynb
MODE=$1     # 1d or 2d
CONFIG=$2   # Receives temp_config_...
OUTPUT=$3   # Receives the output path of the yml file
LOG=$4      # Receives the log path
EXP=$5      # Receives the experiment name (e.g., phis)
YEAR1=$6    # Receives the first year of the experiment
YEAR2=$7    # Receives the last year of the experiment
RUNTAG=$8   # Receives the run tag (e.g., NoRegion_a0)

# central dir (optional) where to copy the generated YAML files
CENTRAL_DIR="/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/mari/results/tuned_2d_LR/refactor/2d/sweep_2/yaml_files"
mkdir -p "$CENTRAL_DIR"

source /home/ecme3038/miniforge3/etc/profile.d/conda.sh
conda activate ectuner

# 1. Tuner from CLI
echo "Launch ECtuner in mode: $MODE on $EXP ($YEAR1-$YEAR2)" > "$LOG"

ectuner $MODE -c "$CONFIG" -o "$OUTPUT" --logfile "$LOG" -t "$RUNTAG" $EXP $YEAR1 $YEAR2 
# 2. Automatic copy of the generated YAML to central dir (if exists)
if [ -f "$OUTPUT" ]; then
    cp "$OUTPUT" "$CENTRAL_DIR/"
    echo "Automatic copy of the generated YAML file to $CENTRAL_DIR" >> "$LOG"
else
    echo "ERROR: YAML file not generated. Check the logs above." >> "$LOG"
fi

# 3. Cleanup temporary config
rm -f "$CONFIG"