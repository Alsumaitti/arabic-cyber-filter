#!/bin/bash
#SBATCH --job-name=gemma-judge-pre
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/preflight_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/preflight_%j.err
#SBATCH -p gpu-H200
#SBATCH -A h200_intern
#SBATCH -q h200_intern_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# ONE-TIME PRE-FLIGHT for the gemma-as-judge study. Run this FIRST.
#
# The login node kills interactive python, so every check happens in a job:
#   1. GPU visible?  which one?
#   2. torch/transformers importable, and does the GPU really work?
#   3. does the model actually load, and does it emit parseable verdicts?
#      -> judges 5 real records and prints them
#
#   mkdir -p ~/fanar-data/judge_bench/{logs,out}
#   sbatch sbatch_scripts/setup_gemma_judge.sh
#   tail -60 ~/fanar-data/judge_bench/logs/preflight_*.out
#
# Allocation: partition gpu-H200, account h200_intern, qos h200_intern_qos
#   (2 H200s granted; 1 GPU per task - a 31B in bf16 fits on one 141GB card)
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/judge_with_gemma.py
MODEL=$HOME/models/gemma-4-31B-it
INPUT=$BASE/validation_prompts.jsonl

mkdir -p "$BASE/logs" "$BASE/out"

# Dedicated env (override: --export=ALL,ENV=other-env).
# PYTHONNOUSERSITE stops ~/.local/lib/python3.10/site-packages leaking in -
# that dir is SHARED by every python3.10 env on the account, so a stray
# `pip install` there silently mixes environments (and broke this run once).
ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "--- env diagnostics ---"
echo "CONDA_PREFIX : ${CONDA_PREFIX:-unset}"
echo "python       : $(which python)"
python -c "import sys; print('sys.path[0:3] :', sys.path[:3])"
python -c "import torch; print('torch        :', torch.__version__, '| cuda', torch.cuda.is_available())" || echo "torch import FAILED"
echo "-----------------------"

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "!! no nvidia-smi"
echo "python: $(which python)"

# --- backend deps -----------------------------------------------------------
# Deliberately does NOT install vLLM: its wheels need CUDA 13 while this
# driver is 12.8, and `pip install vllm` would drag torch to 2.11 and DESTROY
# a working cu124 env. transformers is the supported path on this cluster.
if python -c "import torch, transformers" 2>/tmp/hf_err.$$; then
    python - <<'PYCHK'
import torch, transformers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda        :", torch.cuda.is_available())
print("matmul      :", (torch.zeros(8,8,device="cuda")@torch.zeros(8,8,device="cuda")).sum().item())
print("gpu         :", torch.cuda.get_device_name(0))
PYCHK
else
    echo "ERROR: torch/transformers missing in $CONDA_PREFIX:"
    sed 's/^/    /' /tmp/hf_err.$$ | tail -10
    echo "  pip install --no-user transformers accelerate sentencepiece protobuf"
    echo "  pip install --no-user 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi
rm -f /tmp/hf_err.$$

# --- model + 5-record smoke test ------------------------------------------
echo
echo "=== smoke test: 5 records through the real judging path ==="
python "$SCRIPT" \
    --input "$INPUT" \
    --model "$MODEL" \
    --limit 5 --verbose \
    --backend "${BACKEND:-transformers}" \
    --out-prefix "$BASE/out/smoke" \
    --no-done-marker

echo
echo "=== preflight done | $(date) ==="
echo "If the 5 verdicts above look sane, launch the array:"
echo "  sbatch sbatch_scripts/judge_gemma_validate.sh"
