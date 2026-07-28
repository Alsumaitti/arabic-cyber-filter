#!/bin/bash
#SBATCH --job-name=cyber-tier-merge
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_tiered/merge_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_tiered/merge_%j.err
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# MERGE JOB for the v3 tiered filter - run AFTER the whole filter array
# (filter_cyber_tiered.sh) has finished.
#
# Deliberately a proper sbatch job with 32 GB, NOT a login-node command
# (login-node limits killed the v1 merge - lesson learned).
#
# !! logs_tiered/ must already exist (the filter array creates it, or run
#    mkdir -p ~/fanar-data/cyber_filtered/logs_tiered) - SLURM opens the
#    --output/--error files before this script runs, so a missing directory
#    kills the job with no log at all.
#
# Sanity gate: refuses to merge if any shard output lacks its .done marker,
# so a half-finished array can never silently produce a truncated corpus.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
OUT_DIR=$BASE/output_tiered
RESULT_DIR=$BASE/result

mkdir -p "$RESULT_DIR"

# --- completeness gate: every .json must have a matching .done marker ------
missing=0
shopt -s nullglob
for j in "$OUT_DIR"/cyber_v3_*.json; do
    d="${j%.json}.done"
    if [[ ! -f "$d" ]]; then
        echo "MISSING marker for: $(basename "$j")"
        missing=$((missing+1))
    fi
done
NJSON=$("ls" "$OUT_DIR"/cyber_v3_*.json 2>/dev/null | wc -l)
NDONE=$("ls" "$OUT_DIR"/cyber_v3_*.done 2>/dev/null | wc -l)
echo "shard outputs: $NJSON json, $NDONE done markers"
if (( missing > 0 )); then
    echo "ABORT: $missing shard(s) unfinished - resubmit filter_cyber_tiered.sh first."
    exit 1
fi
if (( NJSON == 0 )); then
    echo "ABORT: no shard outputs found in $OUT_DIR"
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== merging $NJSON shard outputs | $(date) ==="
python "$BASE/scripts/merge_cyber_outputs_tiered.py" \
    --dir "$OUT_DIR" \
    --out-prefix "$RESULT_DIR/cyber_arabic_v3_ALL"
echo "=== merge done | $(date) ==="
