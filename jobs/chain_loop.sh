#!/bin/bash
#SBATCH --job-name=tuning_chain
#SBATCH --output=chain_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# Add the --qos / --account options of your platform.

# ==============================================================================
# Iterative tuning chain (1D or 2D), one link per iteration.
#
# Usage:   sbatch chain_loop.sh PARENT CHILD1 [CHILD2 ...]
# Example: sbatch chain_loop.sh phis r000 r100 r200
#
# Each link waits until PARENT has written all years YEAR1-YEAR2 and its model job has left
# the queue, then runs ecearth4_loop.py (tuning PARENT -> CHILD1, job creation and submission)
# and queues the next link (CHILD1 -> CHILD2 ...). All links share OUTPUT_DIR and its
# runs_manifest.yaml, which the validation tools read.
# ==============================================================================

# ---------------- settings ----------------
SELF=/abs/path/to/chain_loop.sh                  # absolute path of this script
ECTUNER_DIR=/abs/path/to/ECtuner
CONFIG=/abs/path/to/config.yaml
JOB_DIR=/abs/path/to/jobs
QUESTS_DIR=/abs/path/to/ecearth-quests/ece4
OUTPUT_DIR=/abs/path/to/results                  # tuning files, diagnostics, runs_manifest.yaml
RAW_DIR=/abs/path/to/raw_data                    # files.raw_dir of the config
YEAR1=2000                                       # evaluation window (same for all runs)
YEAR2=2009
JOB_PREFIX=ECE4_                                 # model job name = ${JOB_PREFIX}<exp>
# Options for ecearth4_loop.py, e.g. 1D:  "-m 1d -k CPLD --bounds_around current --inc 0.2"
#                                     2D:  "-m 2d -k AMIP --alpha 0.0 --bounds_around current --inc 0.1 --drift_cap 0.5"
LOOP_OPTS="-m 1d --bounds_around current"
POLL=2hours                                      # time between checks of the parent
MAX_POLLS=48                                     # give up after MAX_POLLS checks
CONDA_SH=/abs/path/to/miniforge3/etc/profile.d/conda.sh
CONDA_ENV=ectuner
# -------------------------------------------

POLLS=${POLLS:-0}
if [ $# -lt 2 ]; then
    echo "Usage: sbatch chain_loop.sh PARENT CHILD1 [CHILD2 ...]"; exit 1
fi
PARENT=$1
CHILD=$2
shift 2
REST=("$@")

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -eo pipefail

# 1. Has the parent finished? All evaluation years written and its model job out of the queue
missing=0
for y in $(seq "$YEAR1" "$YEAR2"); do
    compgen -G "${RAW_DIR}/${PARENT}/output/oifs/${PARENT}_atm_cmip6_1m_${y}*.nc" > /dev/null || missing=$((missing + 1))
done
queued=$(squeue -h -u "$USER" -n "${JOB_PREFIX}${PARENT}" | wc -l || true)
if [ "$missing" -gt 0 ] || [ "$queued" -gt 0 ]; then
    if [ "$POLLS" -ge "$MAX_POLLS" ]; then
        echo "Gave up waiting for ${PARENT} (${missing} years missing, ${queued} jobs queued). Check the run."
        exit 1
    fi
    echo "${PARENT} not finished (${missing} years missing, ${queued} jobs queued): checking again in ${POLL}"
    sbatch --begin=now+${POLL} --export=ALL,POLLS=$((POLLS + 1)) "$SELF" "$PARENT" "$CHILD" "${REST[@]}"
    exit 0
fi

# 2. Tune PARENT -> CHILD, create and submit the job of CHILD
cd "$ECTUNER_DIR"
python integrations/ecearth4/ecearth4_loop.py "$PARENT" "$CHILD" \
    -c "$CONFIG" --job_dir "$JOB_DIR" --quests_dir "$QUESTS_DIR" \
    --year1 "$YEAR1" --year2 "$YEAR2" --output_dir "$OUTPUT_DIR" \
    --manifest "${OUTPUT_DIR}/runs_manifest.yaml" $LOOP_OPTS --submit

# 3. Queue the next link
if [ ${#REST[@]} -gt 0 ]; then
    sbatch --begin=now+${POLL} --export=ALL,POLLS=0 "$SELF" "$CHILD" "${REST[@]}"
    echo "Next link queued: ${CHILD} -> ${REST[0]}"
else
    echo "Last link: ${CHILD} submitted, chain finished."
fi