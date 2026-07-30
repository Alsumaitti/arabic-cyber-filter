#!/bin/bash
#SBATCH --job-name=judge-v4-full
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/v4full_%A_%a.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/v4full_%A_%a.err
#SBATCH --array=0-7              # = PARTS-1. Only 2 run at once on 2 GPUs;
                                 # the rest queue and drain in waves.
#SBATCH -p gpu-H200
#SBATCH -A h200_intern
#SBATCH -q h200_intern_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# Judge the ENTIRE v4 corpus (~370k records), not a sample.
#
# WHY THIS IS WORTH THE GPU TIME
# The sample told us v4 is 0.715 precision but not WHICH records are bad, so it
# cannot improve the corpus. Per-record verdicts can. With the measured
# operating point (sens 0.9255 / spec 0.7048), keeping only gemma-positive
# records gives precision ~0.887 while retaining ~92.5% of genuine content -
# beating the best keyword-threshold rule (0.816 at 81.6%) on BOTH axes.
#
# !! FIRST:
#      sbatch sbatch_scripts/prepare_v4_full.sh      # writes v4_full_prompts.jsonl
#
# SIZING THE ARRAY - do this, do not guess:
#      grep 'rec/s' ~/fanar-data/judge_bench/logs/v4_*.out | tail -3
#   Each slice must fit in 8h (28,800s). With N records and R rec/s per GPU:
#      PARTS >= N / (28800 * R)
#   e.g. 370,000 records at 2 rec/s -> PARTS >= 6.4, so PARTS=8 with headroom.
#   Total wall clock is roughly (PARTS/2) * slice_time, since 2 GPUs run at once.
#
# RESUMABLE AND SAFE TO OVERSHOOT: every finished slice writes a .done marker and
# is skipped on resubmit. If a slice times out, raise PARTS and resubmit - the
# completed ones are not redone. Prefer too many parts over too few.
#
#   sbatch --export=ALL,PARTS=8  --array=0-7  sbatch_scripts/judge_v4_full.sh
#   sbatch --export=ALL,PARTS=16 --array=0-15 sbatch_scripts/judge_v4_full.sh
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/judge_with_gemma.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=$BASE/v4_full_prompts.jsonl
OUT_DIR=$BASE/out_full

PARTS=${PARTS:-8}                # keep equal to the array width

mkdir -p "$BASE/logs" "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found."
    echo "Run the prep job first:  sbatch sbatch_scripts/prepare_v4_full.sh"
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
    echo "         Records would be skipped or judged twice. Use:"
    echo "           sbatch --export=ALL,PARTS=N --array=0-\$((N-1)) $0"
    echo "############################################################"
fi

OUT_PREFIX=$OUT_DIR/gemma_v4full_P${PARTS}_p${PART_IDX}
if [[ -f "$OUT_PREFIX.done" ]]; then
    echo "=== part $PART_IDX already done, skipping | $(date) ==="
    exit 0
fi

# PYTHONNOUSERSITE stops ~/.local/lib/python3.10/site-packages leaking in - that
# dir is SHARED by every python3.10 env on the account.
ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

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
echo "records total: $(wc -l < "$INPUT")"

# NOTE: torch.cuda.is_available() returning True does NOT mean the GPU works.
# Watch the first progress line - "0.00 rec/s" means it fell back to CPU; kill it.
python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --parts "$PARTS" --part "$PART_IDX" \
    --backend "$BACKEND" \
    --out-prefix "$OUT_PREFIX"

echo "=== done part $PART_IDX | $(date) ==="
echo
echo "When every part has a .done marker:"
echo "  ls $OUT_DIR/*.done | wc -l      # must equal $PARTS"
echo "  python evaluation/scripts/apply_v4_verdicts.py \\"
echo "      --pred '$OUT_DIR/gemma_v4full_P${PARTS}_p*.jsonl' \\"
echo "      --meta $BASE/v4_full_meta.jsonl \\"
echo "      --corpus ~/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL.jsonl \\"
echo "      --out ~/fanar-data/cyber_filtered/result/cyber_arabic_v5_judged.jsonl"
