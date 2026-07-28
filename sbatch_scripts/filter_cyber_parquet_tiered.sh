#!/bin/bash
#SBATCH --job-name=cyber-tier-pq
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/%A_%a.err
#SBATCH --array=0-199           # = (NUM_FILES * PARTS) - 1  = 25*8-1 ... see below
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00     # measured ~23 min/task at PARTS=8; 5x margin, and a
                            # short limit lets SLURM backfill you into gaps sooner

# ---------------------------------------------------------------------------
# V4 = the SAME v3 tiered acceptance rule, applied to the FULL parquet
# FineWeb2 Arabic release (~112 GB, 25 x ~5 GB .parquet files).
#
# TASK MAPPING: with only ~25 files, one-task-per-file both under-uses the
# cluster and makes a failure cost a whole 5 GB file. Each file is therefore
# split into PARTS contiguous row-group slices:
#     file_index = TASK_ID / PARTS      part_index = TASK_ID % PARTS
# At the default PARTS=8 that is 25 x 8 = 200 tasks, ~340k rows each, and a
# failure costs ~1/8 of a file instead of the whole 5 GB.
#
# !! BEFORE THE FIRST SUBMIT (SLURM opens the log files before this script
#    runs - a missing dir kills the task with NO log):
#      mkdir -p ~/fanar-data/cyber_filtered/{logs_v4,output_v4}
#
# !! THE ARRAY RANGE AND *PARTS* MUST AGREE: the array must cover exactly
#    0 .. (NFILES*PARTS - 1). If the array is SHORTER, the tail files are
#    silently never processed (e.g. PARTS=40 with --array=0-199 processes
#    only 5 of 25 files). This script warns when that happens, and
#    merge_v4.sh refuses to merge an incomplete set.
#      N=$(ls ~/fanar-data/fineweb-2-arabic/data/arb_Arab/train/*.parquet | wc -l)
#      echo $(( N * 8 - 1 ))        # <- put this in --array=0-<that>
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

# Row-group slices per parquet file. Override at submit time WITHOUT editing
# this file, e.g. 25 files x 40 parts = 1000 tasks:
#     sbatch --export=ALL,PARTS=40 --array=0-999 sbatch_scripts/filter_cyber_parquet_tiered.sh
# NOTE: PARTS is part of the output filename, so two different PARTS values
# never overwrite or silently mix with each other (the merge job refuses to
# merge a mixture). Changing PARTS = a fresh, independent set of outputs.
PARTS=${PARTS:-8}

mkdir -p "$BASE/logs_v4" "$OUT_DIR"

mapfile -d '' -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.parquet' -print0 | sort -z)
NFILES=${#FILES[@]}
TOTAL=$(( NFILES * PARTS ))

if (( NFILES == 0 )); then
    echo "ERROR: no *.parquet found in $INPUT_DIR - wrong path?"
    exit 1
fi

# Guard: the array must cover 0..TOTAL-1. A short array silently skips whole
# files (PARTS=40 with --array=0-199 would process only 5 of 25 files), and
# nothing downstream could tell the difference - every output that DOES exist
# looks complete. Warn loudly in every task; merge_v4.sh enforces it.
if [[ -n "${SLURM_ARRAY_TASK_MAX:-}" ]] && (( SLURM_ARRAY_TASK_MAX + 1 < TOTAL )); then
    echo "############################################################"
    echo "WARNING: array covers 0-$SLURM_ARRAY_TASK_MAX but there are $TOTAL slices"
    echo "         ($NFILES files x PARTS=$PARTS)."
    echo "         Slices $(( SLURM_ARRAY_TASK_MAX + 1 ))..$(( TOTAL - 1 )) will NOT run"
    echo "         => only $(( (SLURM_ARRAY_TASK_MAX + 1) / PARTS )) of $NFILES files processed."
    echo "         For a full run resubmit with:  --array=0-$(( TOTAL - 1 ))"
    echo "############################################################"
fi

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "task $SLURM_ARRAY_TASK_ID >= $TOTAL slices ($NFILES files x $PARTS parts) -> nothing to do."
    exit 0
fi

FILE_IDX=$(( SLURM_ARRAY_TASK_ID / PARTS ))
PART_IDX=$(( SLURM_ARRAY_TASK_ID % PARTS ))
FILE_PATH="${FILES[$FILE_IDX]}"
STEM=$(basename "$FILE_PATH" .parquet)
OUT_PREFIX=$OUT_DIR/cyber_v4_${STEM}_P${PARTS}_p${PART_IDX}

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
