#!/bin/bash
#SBATCH --job-name=cyber-merge-ar
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs/merge_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs/merge_%j.err
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Merge all per-shard cyber_arabic_*.json / .csv files (produced by the
# array job) into one final combined json + csv.
# Run as a proper sbatch job (not directly on the login node) because the
# combined record count/text volume can be large enough to get OOM-killed
# or process-limit-killed if run interactively on panther-login.
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/cyber_filtered
SCRIPT=$BASE/scripts/merge_cyber_outputs.py
OUT_DIR=$BASE/output
OUT_PREFIX=$OUT_DIR/cyber_arabic_ALL

mkdir -p "$BASE/logs"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date) ==="
echo "merging shard outputs from: $OUT_DIR"
echo "final output prefix       : $OUT_PREFIX"

python "$SCRIPT" \
    --dir "$OUT_DIR" \
    --out-prefix "$OUT_PREFIX"

echo "=== done | $(date) ==="
