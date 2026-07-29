#!/bin/bash
#SBATCH --job-name=gemma-judge-pre
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/preflight_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/preflight_%j.err
#SBATCH -p gpu-all
#SBATCH -q normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# ONE-TIME PRE-FLIGHT for the gemma-as-judge study. Run this FIRST.
#
# The login node kills interactive python, so every check happens in a job:
#   1. GPU visible?  which one?
#   2. vLLM importable? (installs it if missing - ONCE, never in the array)
#   3. does the model actually load, and does it emit parseable verdicts?
#      -> judges 5 real records and prints them
#
#   mkdir -p ~/fanar-data/judge_bench/{logs,out}
#   sbatch sbatch_scripts/setup_gemma_judge.sh
#   tail -60 ~/fanar-data/judge_bench/logs/preflight_*.out
#
# NOTE: adjust -p gpu-all / --gres if your GPU partition is named differently
#   sinfo -o "%P %G %N" | head        <- lists partitions and their GPUs
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/judge_with_gemma.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=$BASE/validation_prompts.jsonl

mkdir -p "$BASE/logs" "$BASE/out"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocr-env

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "!! no nvidia-smi"
echo "python: $(which python)"

# --- vLLM ------------------------------------------------------------------
if python -c "import vllm" 2>/dev/null; then
    echo "vLLM present: $(python -c 'import vllm;print(vllm.__version__)')"
else
    echo "vLLM MISSING -> installing (one time) ..."
    pip install vllm
    python -c "import vllm;print('vLLM installed:',vllm.__version__)"
fi

# --- model + 5-record smoke test ------------------------------------------
echo
echo "=== smoke test: 5 records through the real judging path ==="
python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --limit 5 --verbose \
    --out-prefix "$BASE/out/smoke" \
    --no-done-marker

echo
echo "=== preflight done | $(date) ==="
echo "If the 5 verdicts above look sane, launch the array:"
echo "  sbatch sbatch_scripts/judge_gemma_validate.sh"
