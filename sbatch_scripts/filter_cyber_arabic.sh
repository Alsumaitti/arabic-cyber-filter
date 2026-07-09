#!/bin/bash
#SBATCH --job-name=cyber-filter-ar
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs/%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs/%A_%a.err
#SBATCH --array=0-99           # set to (num .gz files - 1); see instructions below
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# ARRAY JOB: one SLURM task per .jsonl.gz shard in fanar-data/FineWeb2.
# Each task scans exactly ONE shard for Arabic cybersecurity keywords and
# writes its own pair of output files:
#     output/cyber_arabic_<shard_stem>.json
#     output/cyber_arabic_<shard_stem>.csv
#
# The list of shards is rebuilt fresh every run (alphabetically sorted,
# null-byte safe for filenames with spaces) and indexed by
# SLURM_ARRAY_TASK_ID, same pattern as the OCR production job.
#
# BEFORE SUBMITTING, count the shards and fix --array above:
#   ls ~/fanar-data/FineWeb2/*.jsonl.gz | wc -l   -> set --array=0-<count-1>
#
# After all tasks finish, merge the per-shard outputs into one combined
# json/csv with the merge command shown at the bottom of this file's
# accompanying instructions.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
SCRIPT=$BASE/scripts/filter_cyber_arabic.py
KEYWORDS=$BASE/keywords/arabic_cyber_keywords.txt
INPUT_DIR=$ROOT/FineWeb2
OUT_DIR=$BASE/output

mkdir -p "$BASE/logs" "$OUT_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

# Build a stable, alphabetically-sorted array of shard paths and pick this
# task's file by SLURM_ARRAY_TASK_ID.
mapfile -d '' -t SHARDS < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.jsonl.gz' -print0 | sort -z)

TOTAL=${#SHARDS[@]}
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "task id $SLURM_ARRAY_TASK_ID >= $TOTAL shards found -> nothing to do, exiting."
    exit 0
fi

SHARD_PATH="${SHARDS[$SLURM_ARRAY_TASK_ID]}"
SHARD_STEM=$(basename "$SHARD_PATH")
SHARD_STEM=${SHARD_STEM%.jsonl.gz}
OUT_PREFIX=$OUT_DIR/cyber_arabic_${SHARD_STEM}

echo "=== node $(hostname) | task $SLURM_ARRAY_TASK_ID/$TOTAL | shard: $SHARD_STEM | $(date) ==="

python "$SCRIPT" \
    --input-file "$SHARD_PATH" \
    --keywords "$KEYWORDS" \
    --out-prefix "$OUT_PREFIX"

echo "=== done $SHARD_STEM | $(date) ==="
