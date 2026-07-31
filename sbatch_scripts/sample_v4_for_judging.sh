#!/bin/bash
#SBATCH --job-name=v4-sample
#SBATCH --output=/export/home/oalsumaitti/fanar-data/judge_bench/logs/sample_%j.out
#SBATCH --error=/export/home/oalsumaitti/fanar-data/judge_bench/logs/sample_%j.err
#SBATCH -p cpu-all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# STEP 1 of measuring v4: draw a judgeable, rule-stratified sample from the
# merged 370k corpus. CPU only - this is one streaming pass over a JSONL file,
# no GPU needed, so it should not sit in the H200 queue.
#
# Runs as a job because the login node kills interactive python.
#
# !! THE LOG DIR MUST EXIST BEFORE YOU SUBMIT - slurm will not create it, and
#    the job dies silently with no log at all:
#      mkdir -p ~/fanar-data/judge_bench/{logs,out}
#
#   sbatch sample_v4_for_judging.sh
#   tail -40 ~/fanar-data/judge_bench/logs/sample_*.out
#
# Override the sample size (per acceptance rule, 4 rules -> x4 total):
#   sbatch --export=ALL,PER_RULE=1000 sample_v4_for_judging.sh
# ---------------------------------------------------------------------------
set -euo pipefail

BASE=/export/home/oalsumaitti/fanar-data/judge_bench
SCRIPT=$BASE/scripts/sample_v4_for_judging.py
CORPUS=${CORPUS:-/export/home/oalsumaitti/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL.jsonl}
PER_RULE=${PER_RULE:-750}

mkdir -p "$BASE/logs" "$BASE/out"

if [[ ! -f "$CORPUS" ]]; then
    echo "ERROR: merged v4 corpus not found at:"
    echo "  $CORPUS"
    echo
    echo "Either the merge has not run, or it wrote .json instead of .jsonl."
    echo "Check:  ls -la ~/fanar-data/cyber_filtered/result/"
    echo "Then:   sbatch merge_v4.sh      # defaults to jsonl"
    echo "Or point this job at the right file:"
    echo "  sbatch --export=ALL,CORPUS=/path/to/corpus.jsonl $0"
    exit 1
fi

ENV=${ENV:-gjudge}
export PYTHONNOUSERSITE=1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

echo "=== node $(hostname) | $(date) ==="
echo "corpus   : $CORPUS ($(du -h "$CORPUS" | cut -f1))"
echo "per rule : $PER_RULE"
echo "python   : $(which python)"

python "$SCRIPT" \
    --input "$CORPUS" \
    --out-dir "$BASE" \
    --per-rule "$PER_RULE"

echo
echo "=== done | $(date) ==="
echo "next:  sbatch judge_v4_sample.sh"
