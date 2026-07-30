#!/bin/bash
#SBATCH --job-name=v4-prep-full
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/prep_full_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/prep_full_%j.err
#SBATCH -p cpu-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Build judge prompts for the WHOLE v4 corpus. CPU only - one streaming pass, no
# GPU needed, so it should not sit in the H200 queue behind the judging array.
#
# !! The log dir must exist before you submit, or the job dies with no log:
#      mkdir -p ~/fanar-data/judge_bench/{logs,out_full}
#
#   sbatch sbatch_scripts/prepare_v4_full.sh
#   tail -40 ~/fanar-data/judge_bench/logs/prep_full_*.out
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/prepare_v4_full_prompts.py
CORPUS=${CORPUS:-/export/home/oalsumaitti/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL.jsonl}

mkdir -p "$BASE/logs" "$BASE/out_full"

if [[ ! -f "$CORPUS" ]]; then
    echo "ERROR: merged v4 corpus not found at:"
    echo "  $CORPUS"
    echo "Check:  ls -la ~/fanar-data/cyber_filtered/result/"
    echo "Or:     sbatch --export=ALL,CORPUS=/path/to/corpus.jsonl $0"
    exit 1
fi

ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "=== node $(hostname) | $(date) ==="
echo "corpus: $CORPUS ($(du -h "$CORPUS" | cut -f1))"

python "$SCRIPT" --input "$CORPUS" --out-dir "$BASE"

echo
echo "=== done | $(date) ==="
