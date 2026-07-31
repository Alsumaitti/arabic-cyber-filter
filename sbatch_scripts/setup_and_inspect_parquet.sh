#!/bin/bash
#SBATCH --job-name=cyber-preflight
#SBATCH --output=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/preflight_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/cyber_filtered/logs_v4/preflight_%j.err
#SBATCH -p cpu-all
#SBATCH -q normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# ONE-TIME PRE-FLIGHT for the v4 parquet run. Run this FIRST, once.
#
# The login node kills interactive python, so everything that needs python
# has to happen inside a job. This single-task job does all of it:
#   1. installs pyarrow into ocr-env if it is missing (ONCE - never let the
#      100-task array do this, they would race on the same conda env)
#   2. prints each parquet file's schema, row count and row-group count
#   3. prints the exact --array range to use, given PARTS
#
#   mkdir -p ~/fanar-data/cyber_filtered/{logs_v4,output_v4}   # first!
#   sbatch setup_and_inspect_parquet.sh
#   tail -40 ~/fanar-data/cyber_filtered/logs_v4/preflight_*.out
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=/export/home/oalsumaitti/fanar-data
BASE=$ROOT/cyber_filtered
SCRIPT=$BASE/scripts/filter_cyber_parquet_tiered.py
INPUT_DIR=$ROOT/fineweb-2-arabic/data/arb_Arab/train
PARTS=8                       # keep in sync with filter_cyber_parquet_tiered.sh

mkdir -p "$BASE/logs_v4" "$BASE/output_v4"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== node $(hostname) | $(date) ==="
echo "python: $(which python)"

# --- 1. pyarrow ------------------------------------------------------------
if python -c "import pyarrow" 2>/dev/null; then
    echo "pyarrow already present: $(python -c 'import pyarrow;print(pyarrow.__version__)')"
else
    echo "pyarrow MISSING -> installing into ocr-env (one time) ..."
    conda install -y -c conda-forge pyarrow || pip install pyarrow
    python -c "import pyarrow;print('pyarrow installed:',pyarrow.__version__)"
fi

# --- 2. inspect every parquet file ----------------------------------------
echo
echo "=== PARQUET INVENTORY ==="
mapfile -d '' -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.parquet' -print0 | sort -z)
NFILES=${#FILES[@]}
echo "found $NFILES parquet file(s) in $INPUT_DIR"
echo

for f in "${FILES[@]}"; do
    python "$SCRIPT" --inspect --input-file "$f"
    echo "---"
done

# --- 3. the array size to use ---------------------------------------------
echo
echo "=== SUBMIT SETTINGS ==="
echo "files : $NFILES"
echo "parts : $PARTS   (row-group slices per file)"
echo "tasks : $(( NFILES * PARTS ))"
echo "=> set  #SBATCH --array=0-$(( NFILES * PARTS - 1 ))  in filter_cyber_parquet_tiered.sh"
echo
echo "If the per-file 'row groups' count above is much larger than $PARTS,"
echo "raising PARTS (both scripts) gives more parallelism and cheaper resume."
echo "=== preflight done | $(date) ==="
