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
# Uses the STREAMING merger: ~420k accepted records / ~5 GB is far too much to
# accumulate in memory, so shards are read one at a time and appended straight
# to the output. 32 GB is plenty for that.
#
# Output is JSONL (one record per line) - at 5 GB a single JSON array would
# need ~16-21 GB of RAM just to read back, defeating the point. Add
# --format json if you specifically need an array, --csv for a CSV copy.
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

INPUT_DIR=$ROOT/fineweb-2-arabic/data/arb_Arab/train

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

# --- coverage gate --------------------------------------------------------
# "every output has its .done" is NOT enough: if the array was shorter than
# NFILES*PARTS, whole files were never attempted, so no output exists for them
# and the check above passes anyway. Verify the EXPECTED number of slices.
PARTS_SEEN=$(ls "$OUT_DIR"/cyber_v4_*_P*_p*.json 2>/dev/null \
             | sed -E 's/.*_P([0-9]+)_p[0-9]+\.json$/\1/' | sort -un)
NPARTS_VALUES=$(echo "$PARTS_SEEN" | grep -c . || true)
if (( NPARTS_VALUES != 1 )); then
    echo "ABORT: outputs mix PARTS values ($(echo $PARTS_SEEN | tr '\n' ' '))."
    echo "       Different PARTS slice differently - merging them would DUPLICATE"
    echo "       records. Keep only one set (e.g. mkdir old && mv *_P8_p*.* old/)."
    exit 1
fi
PARTS=$PARTS_SEEN
NFILES=$(ls "$INPUT_DIR"/*.parquet 2>/dev/null | wc -l)
EXPECTED=$(( NFILES * PARTS ))
echo "coverage: PARTS=$PARTS, input files=$NFILES -> expected $EXPECTED slices, have $NDONE"
if (( NDONE != EXPECTED )); then
    echo "ABORT: incomplete coverage - $NDONE/$EXPECTED slices done."
    echo "       Whole files may never have been attempted (a too-short --array"
    echo "       does exactly this). Resubmit the filter with:"
    echo "         sbatch --export=ALL,PARTS=$PARTS --array=0-$(( EXPECTED - 1 )) \\"
    echo "                filter_cyber_parquet_tiered.sh"
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== merging $NJSON shard outputs | $(date) ==="
python "$BASE/scripts/merge_v4_streaming.py" \
    --dir "$OUT_DIR" \
    --out-prefix "$RESULT_DIR/cyber_arabic_v4_ALL"
echo "=== merge done | $(date) ==="
