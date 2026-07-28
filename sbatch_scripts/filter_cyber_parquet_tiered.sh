#!/bin/bash
#SBATCH --job-name=cyber-tier-pq
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/%A_%a.err
#SBATCH --array=0-99            # = (NUM_FILES * PARTS) - 1   ... see below
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# ---------------------------------------------------------------------------
# V4 = the SAME v3 tiered acceptance rule, applied to the FULL parquet
# FineWeb2 Arabic release (~112 GB, 25 x ~5 GB .parquet files).
#
# TASK MAPPING: with only ~25 files, one-task-per-file both under-uses the
# cluster and makes a failure cost a whole 5 GB file. Each file is therefore
# split into PARTS row-group slices:
#     file_index = TASK_ID / PARTS      part_index = TASK_ID % PARTS
# so 25 files x 4 parts = 100 tasks (same array width as the v3 run), and a
# failure costs ~1.3 GB of re-reading instead of 5 GB.
#
# !! BEFORE THE FIRST SUBMIT (SLURM opens the log files before this script
#    runs - a missing dir kills the task with NO log):
#      mkdir -p ~/fanar-data/cyber_filtered/{logs_v4,output_v4}
#
# !! SET --array CORRECTLY:
#      N=$(ls ~/fanar-data/fineweb-2-arabic/data/arb_Arab/train/*.parquet | wc -l)
#      echo $(( N * 4 - 1 ))        # <- put this in --array=0-<that>
#
# RESUMABLE: every finished (file,part) leaves a .done marker; just RESUBMIT
# this same file after any failure - finished slices exit in ~1 second.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
SCRIPT=$BASE/scripts/filter_cyber_parquet_tiered.py
TIERS_DIR=$BASE/keywords/tiers_clean
INPUT_DIR=$ROOT/fineweb-2-arabic/data/arb_Arab/train
OUT_DIR=$BASE/output_v4

PARTS=4                      # row-group slices per parquet file

mkdir -p "$BASE/logs_v4" "$OUT_DIR"

mapfile -d '' -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.parquet' -print0 | sort -z)
NFILES=${#FILES[@]}
TOTAL=$(( NFILES * PARTS ))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "task $SLURM_ARRAY_TASK_ID >= $TOTAL slices ($NFILES files x $PARTS parts) -> nothing to do."
    exit 0
fi

FILE_IDX=$(( SLURM_ARRAY_TASK_ID / PARTS ))
PART_IDX=$(( SLURM_ARRAY_TASK_ID % PARTS ))
FILE_PATH="${FILES[$FILE_IDX]}"
STEM=$(basename "$FILE_PATH" .parquet)
OUT_PREFIX=$OUT_DIR/cyber_v4_${STEM}_p${PART_IDX}

if [[ -f "$OUT_PREFIX.done" ]]; then
    echo "=== $STEM part $PART_IDX already done, skipping | $(date) ==="
    exit 0
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

# pyarrow must already be installed - by setup_and_inspect_parquet.sh, ONCE.
# Deliberately only a CHECK here: 100 array tasks running `conda install`
# concurrently would race on the same env and can corrupt it.
if ! python -c "import pyarrow" 2>/dev/null; then
    echo "ERROR: pyarrow is not installed in ocr-env."
    echo "       Run the one-time pre-flight job first:"
    echo "         sbatch sbatch_scripts/setup_and_inspect_parquet.sh"
    exit 1
fi

echo "=== node $(hostname) | task $SLURM_ARRAY_TASK_ID/$TOTAL | file $STEM | part $PART_IDX/$PARTS | $(date) ==="

python "$SCRIPT" \
    --input-file "$FILE_PATH" \
    --parts "$PARTS" --part "$PART_IDX" \
    --tiers-dir "$TIERS_DIR" \
    --out-prefix "$OUT_PREFIX"

echo "=== done $STEM part $PART_IDX | $(date) ==="
