#!/bin/bash
#SBATCH --job-name=gemma-judge
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/%A_%a.err
#SBATCH --array=0-1              # = PARTS-1; 2 H200s -> 2 tasks, 1 GPU each
#SBATCH -p gpu-all
#SBATCH -A H200_Intern
#SBATCH -q H200_Intern_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# STEP 1 of the "can an open 31B model replace a frontier model as judge?"
# study: judge the 1,992-record blind validation set with gemma-4-31B-it.
#
# Each array task loads the model on ITS OWN GPU and judges a contiguous slice.
# gemma-4-31B in bf16 is ~62GB and an H200 has 141GB, so the whole model fits
# on ONE GPU - two independent tasks beat tensor-parallel across two (no
# cross-GPU traffic, and a crash costs half the work instead of all of it).
# PARTS must equal the array width.
#
# !! FIRST TIME:
#      mkdir -p ~/fanar-data/judge_bench/{logs,out}
#      sbatch sbatch_scripts/setup_gemma_judge.sh     # installs vLLM, smoke test
#
# RESUMABLE: finished slices leave a .done marker; just resubmit this file.
#
# Scale PARTS to the GPUs you can get:
#   sbatch --export=ALL,PARTS=8 --array=0-7 sbatch_scripts/judge_gemma_validate.sh
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/judge_with_gemma.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=$BASE/validation_prompts.jsonl
OUT_DIR=$BASE/out

PARTS=${PARTS:-2}                # keep equal to the array width

mkdir -p "$BASE/logs" "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found - scp validation_prompts.jsonl up first."
    exit 1
fi

PART_IDX=$SLURM_ARRAY_TASK_ID
if (( PART_IDX >= PARTS )); then
    echo "task $PART_IDX >= PARTS=$PARTS -> nothing to do."
    exit 0
fi
if [[ -n "${SLURM_ARRAY_TASK_MAX:-}" ]] && (( SLURM_ARRAY_TASK_MAX + 1 != PARTS )); then
    echo "############################################################"
    echo "WARNING: array width $(( SLURM_ARRAY_TASK_MAX + 1 )) != PARTS=$PARTS."
    echo "         Slices would be skipped or duplicated. Use:"
    echo "           sbatch --export=ALL,PARTS=N --array=0-\$((N-1)) $0"
    echo "############################################################"
fi

OUT_PREFIX=$OUT_DIR/gemma_val_P${PARTS}_p${PART_IDX}
if [[ -f "$OUT_PREFIX.done" ]]; then
    echo "=== part $PART_IDX already done, skipping | $(date) ==="
    exit 0
fi

# Dedicated env for vLLM (override: --export=ALL,ENV=other-env).
# PYTHONNOUSERSITE stops ~/.local/lib/python3.10/site-packages leaking in -
# that dir is SHARED by every python3.10 env on the account, so a stray
# `pip install` there silently mixes environments (and broke this run once).
ENV=${ENV:-vllm-env}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "--- env diagnostics ---"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "python       : $(which python)"
python -c "import sys; print('sys.path[0:3] :', sys.path[:3])"
python -c "import torch; print('torch        :', torch.__version__, '| cuda', torch.cuda.is_available())" || echo "torch import FAILED"
echo "-----------------------"

if ! python -c "import vllm" 2>/tmp/vllm_err.$$; then
    echo "ERROR: vLLM not importable in conda env '$ENV'. The REAL error:"
    sed 's/^/    /' /tmp/vllm_err.$$ | tail -15
    rm -f /tmp/vllm_err.$$
    echo
    echo "Run the pre-flight ONCE first (it installs into the env, not ~/.local):"
    echo "  sbatch sbatch_scripts/setup_gemma_judge.sh"
    exit 1
fi
rm -f /tmp/vllm_err.$$

echo "=== node $(hostname) | part $PART_IDX/$PARTS | $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --parts "$PARTS" --part "$PART_IDX" \
    --out-prefix "$OUT_PREFIX"

echo "=== done part $PART_IDX | $(date) ==="
