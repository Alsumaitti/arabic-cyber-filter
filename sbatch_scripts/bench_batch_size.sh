#!/bin/bash
#SBATCH --job-name=bench-batch
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/bench_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/bench_%j.err
#SBATCH -p gpu-H200
#SBATCH -A h200_intern
#SBATCH -q h200_intern_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# Find the fastest batch size BEFORE committing ~163 GPU-hours to the full run.
#
# Measured throughput was 0.63 rec/s at batch 8, which puts the full 370k corpus
# at ~3.4 days on 2 GPUs. Batch size is the biggest lever available - a 31B in
# bf16 occupies ~62GB of a 141GB card, so batch 8 left most of the KV cache
# unused. This sweep costs ~20 minutes and can remove a day or more.
#
# Loads the model once and times each batch size on the same records. OOM at the
# top of the sweep is expected and handled - it reports the largest size that
# worked and stops.
#
# !! Run prepare_v4_full.sh first (this reads v4_full_prompts.jsonl).
#
#   sbatch sbatch_scripts/bench_batch_size.sh
#   tail -30 ~/fanar-data/judge_bench/logs/bench_*.out
#
# It prints the exact sbatch line to use for the real run.
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/bench_judge_throughput.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=${INPUT:-$BASE/v4_full_prompts.jsonl}
SIZES=${SIZES:-8,16,32,48,64}

mkdir -p "$BASE/logs"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found."
    echo "Run:  sbatch sbatch_scripts/prepare_v4_full.sh"
    echo "Or point elsewhere:  sbatch --export=ALL,INPUT=/path/prompts.jsonl $0"
    exit 1
fi

ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --sizes "$SIZES" \
    --total-records "$(wc -l < "$INPUT")"

echo
echo "=== done | $(date) ==="
