#!/bin/bash
#SBATCH --job-name=cyber-v4-merge
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/merge_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/merge_%j.err
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# MERGE for the v4 parquet run - run AFTER the whole filter array finishes.
#
# Uses the STREAMING merger: ~420k accepted records / ~5 GB of JSON is far too
# much to accumulate in memory, so shards are read one at a time and appended
# straight to the output. 32 GB is plenty for that.
#
# Completeness gate: refuses to merge unless every shard .json has its .done
# marker, so an unfinished array can never yield a silently truncated corpus.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
OUT_DIR=$BASE/output_v4
RESULT_DIR=$BASE/result

mkdir -p "$RESULT_DIR"

missing=0
shopt -s nullglob
for j in "$OUT_DIR"/cyber_v4_*.json; do
    [[ -f "${j%.json}.done" ]] || { echo "MISSING marker: $(basename "$j")"; missing=$((missing+1)); }
done
NJSON=$(ls "$OUT_DIR"/cyber_v4_*.json 2>/dev/null | wc -l)
NDONE=$(ls "$OUT_DIR"/cyber_v4_*.done 2>/dev/null | wc -l)
echo "shard outputs: $NJSON json, $NDONE done markers"
if (( missing > 0 )); then
    echo "ABORT: $missing shard(s) unfinished - resubmit filter_cyber_parquet_tiered.sh first."
    exit 1
fi
if (( NJSON == 0 )); then
    echo "ABORT: no shard outputs in $OUT_DIR"; exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== merging $NJSON shard outputs | $(date) ==="
python "$BASE/scripts/merge_v4_streaming.py" \
    --dir "$OUT_DIR" \
    --out-prefix "$RESULT_DIR/cyber_arabic_v4_ALL"
echo "=== merge done | $(date) ==="
