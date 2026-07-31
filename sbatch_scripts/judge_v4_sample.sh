#!/bin/bash
#SBATCH --job-name=judge-v4
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/v4_%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/v4_%A_%a.err
#SBATCH --array=0-1              # = PARTS-1; 2 H200s -> 2 tasks, 1 GPU each
#SBATCH -p gpu-H200
#SBATCH -A h200_intern
#SBATCH -q h200_intern_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# STEP 2 of measuring v4: judge the rule-stratified sample with gemma-4-31B-it.
#
# This is the payoff of evaluation/JUDGE_BENCHMARK.md. We are NOT trusting
# gemma's per-document verdict - it over-calls `cyber` by ~1.9x. We are using it
# as a BINARY instrument with a measured operating point (sens 0.926 /
# spec 0.705) and inverting the bias with Rogan-Gladen afterwards. That yields
# an unbiased corpus-level precision estimate from a biased classifier.
#
# ZERO-SHOT ON PURPOSE: the few-shot arm changed 3-way kappa by +0.003 and the
# binary operating point by +0.003, so exemplars buy nothing here and cost ~2k
# prefill tokens per record. The calibration constants in
# estimate_v4_precision.py come from the few-shot arm; the two arms agreed to
# within 0.003, so the choice does not move the estimate.
#
# !! FIRST:
#      mkdir -p ~/fanar-data/judge_bench/{logs,out}
#      sbatch sbatch_scripts/sample_v4_for_judging.sh     # writes the sample
#
# WALL TIME is 8h, not 4h: this sample is larger than the 1,992-record
# validation set. Watch the first progress line and scale PARTS if it is tight.
#
# RESUMABLE: finished slices leave a .done marker; just resubmit.
#
# Scale to whatever GPUs you can get (PARTS must equal the array width):
#   sbatch --export=ALL,PARTS=4 --array=0-3 sbatch_scripts/judge_v4_sample.sh
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/judge_with_gemma.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=$BASE/v4_sample_prompts.jsonl
OUT_DIR=$BASE/out

PARTS=${PARTS:-2}                # keep equal to the array width

mkdir -p "$BASE/logs" "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found."
    echo "Run the sampler first:  sbatch sbatch_scripts/sample_v4_for_judging.sh"
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

OUT_PREFIX=$OUT_DIR/gemma_v4_P${PARTS}_p${PART_IDX}
if [[ -f "$OUT_PREFIX.done" ]]; then
    echo "=== part $PART_IDX already done, skipping | $(date) ==="
    exit 0
fi

# Dedicated env (override: --export=ALL,ENV=other-env).
# PYTHONNOUSERSITE stops ~/.local/lib/python3.10/site-packages leaking in -
# that dir is SHARED by every python3.10 env on the account, so a stray
# `pip install` there silently mixes environments (and broke this run once).
ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1

# The OOM traceback showed 11.7GB "reserved but unallocated" - allocator
# fragmentation, which expandable_segments largely removes. Costs nothing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "--- env diagnostics ---"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "python       : $(which python)"
python -c "import torch; print('torch        :', torch.__version__, '| cuda', torch.cuda.is_available())" || echo "torch import FAILED"
echo "-----------------------"

# This cluster's driver (12.8) cannot run the CUDA-13 vLLM wheels, so the
# transformers backend is the intended path - require THAT, not vLLM.
BACKEND=${BACKEND:-transformers}
if ! python -c "import torch, transformers" 2>/tmp/hf_err.$$; then
    echo "ERROR: torch/transformers not importable in conda env '$ENV':"
    sed 's/^/    /' /tmp/hf_err.$$ | tail -10
    rm -f /tmp/hf_err.$$
    exit 1
fi
rm -f /tmp/hf_err.$$

echo "=== node $(hostname) | part $PART_IDX/$PARTS | $(date) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

# NOTE: torch.cuda.is_available() returning True does NOT mean the GPU works -
# a previous run generated on CPU at 0.003 rec/s while reporting CUDA available.
# judge_with_gemma.py proves the device with a real matmul and aborts if any
# layer lands on cpu/disk. If you see "0.00 rec/s", kill it.
python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --parts "$PARTS" --part "$PART_IDX" \
    --backend "$BACKEND" \
    --out-prefix "$OUT_PREFIX"

echo "=== done part $PART_IDX | $(date) ==="
echo
echo "When ALL parts are done, pull them back and estimate:"
echo "  scp 'panther-login.qcri.org:$OUT_DIR/gemma_v4_P${PARTS}_p*.jsonl' evaluation/judge_benchmark/"
echo "  scp panther-login.qcri.org:$BASE/v4_sample_meta.jsonl evaluation/judge_benchmark/"
echo "  python evaluation/scripts/estimate_v4_precision.py \\"
echo "      --pred 'evaluation/judge_benchmark/gemma_v4_P${PARTS}_p*.jsonl' \\"
echo "      --meta evaluation/judge_benchmark/v4_sample_meta.jsonl --show-cal"
