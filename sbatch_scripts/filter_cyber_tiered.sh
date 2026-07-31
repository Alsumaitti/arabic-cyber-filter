#!/bin/bash
#SBATCH --job-name=cyber-tier-ar
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_tiered/%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_tiered/%A_%a.err
#SBATCH --array=0-99           # set to (num .gz files - 1); see instructions below
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# V3 TIERED FILTER - ARRAY JOB: one SLURM task per .jsonl.gz shard.
# Each task scans exactly ONE shard and applies the TIERED ACCEPTANCE RULE
# (learned from the 14,653-record LLM census) instead of "any keyword hit":
#     keep iff  >=1 tier-1 hit, or >=2 hits of tier<=2,
#               or >=3 hits of tier<=3, or >=4 tiered hits.
# Outputs per shard:
#     output_tiered/cyber_v3_<shard_stem>.json / .csv / .done
#
# RESUMABLE: each finished shard leaves a .done marker. If anything fails
# mid-run (node death, timeout, preemption), just RESUBMIT THIS SAME FILE -
# finished shards exit in ~1 second, only unfinished ones re-run.
#
# !! BEFORE THE FIRST SUBMIT, create the log/output dirs ONCE:
#      mkdir -p ~/fanar-data/cyber_filtered/{logs_tiered,output_tiered}
#    SLURM opens the --output/--error files BEFORE this script body runs, so
#    if logs_tiered/ does not exist the job dies instantly with NO log file.
#    (The mkdir below only helps for later runs - it is too late for logging.)
#
# BEFORE SUBMITTING, count the shards and fix --array above:
#   ls ~/fanar-data/FineWeb2/*.jsonl.gz | wc -l   -> set --array=0-<count-1>
#
# After all tasks finish, merge (as an sbatch job, NOT on the login node):
#   sbatch merge_cyber_tiered.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
SCRIPT=$BASE/scripts/filter_cyber_arabic_tiered.py
TIERS_DIR=$BASE/keywords/tiers_clean
INPUT_DIR=$ROOT/FineWeb2
OUT_DIR=$BASE/output_tiered

mkdir -p "$BASE/logs_tiered" "$OUT_DIR"

# Build a stable, alphabetically-sorted array of shard paths and pick this
# task's file by SLURM_ARRAY_TASK_ID (null-byte safe, same as the v1 job).
mapfile -d '' -t SHARDS < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.jsonl.gz' -print0 | sort -z)

TOTAL=${#SHARDS[@]}
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "task id $SLURM_ARRAY_TASK_ID >= $TOTAL shards found -> nothing to do, exiting."
    exit 0
fi

SHARD_PATH="${SHARDS[$SLURM_ARRAY_TASK_ID]}"
SHARD_STEM=$(basename "$SHARD_PATH")
SHARD_STEM=${SHARD_STEM%.jsonl.gz}
OUT_PREFIX=$OUT_DIR/cyber_v3_${SHARD_STEM}

# fast resume path: skip before even activating conda
if [[ -f "$OUT_PREFIX.done" ]]; then
    echo "=== shard $SHARD_STEM already done (marker found), skipping | $(date) ==="
    exit 0
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== node $(hostname) | task $SLURM_ARRAY_TASK_ID/$TOTAL | shard: $SHARD_STEM | $(date) ==="

python "$SCRIPT" \
    --input-file "$SHARD_PATH" \
    --tiers-dir "$TIERS_DIR" \
    --out-prefix "$OUT_PREFIX"

echo "=== done $SHARD_STEM | $(date) ==="
