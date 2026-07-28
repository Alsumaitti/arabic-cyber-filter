# Running the v3 tiered filter on the cluster — transfer & run guide

Step-by-step guide for re-running the FineWeb2 filtering on
**panther-login.qcri.org** using the **tiered acceptance rule** (the
second-generation filter learned from the 14,653-record LLM census), in
parallel on `cpu-all`, fully resumable.

**What changes vs the v1 run:** same scanning engine, but a record is kept
iff it hits **≥ 1 Tier-1 keyword, or ≥ 2 keywords of tier ≤ 2, or ≥ 3 of
tier ≤ 3, or ≥ 4 tiered keywords** — instead of "any keyword". Measured on
the census, this lifts lenient precision **0.39 → 0.64** while keeping 68%
of the genuine documents. On the same 100 shards you should get
**≈ 6,041 accepted records** (the census-measured number) — a big deviation
means wrong tier files or wrong input.

---

## 0 · What to move (6 small files + 4 keyword files)

Everything lives in this repo; nothing is large:

| Local file (repo) | Destination on panther |
|---|---|
| `scripts/filter_cyber_arabic_tiered.py` | `~/fanar-data/cyber_filtered/scripts/` |
| `scripts/merge_cyber_outputs_tiered.py` | `~/fanar-data/cyber_filtered/scripts/` |
| `sbatch_scripts/filter_cyber_tiered.sh` | `~/fanar-data/cyber_filtered/sbatch_scripts/` |
| `sbatch_scripts/merge_cyber_tiered.sh` | `~/fanar-data/cyber_filtered/sbatch_scripts/` |
| `keywords/tiers_clean/tier1_keywords.txt` … `tier4_keywords.txt` | `~/fanar-data/cyber_filtered/keywords/tiers_clean/` |

Use **`keywords/tiers_clean/`** (comment-free: pure keywords, one per line,
LF endings, UTF-8 no BOM — generated exactly for this) rather than
`keywords/tiers/`. The Python loader tolerates `#` comments anyway, so both
work — the clean files just remove every possible parsing surprise.

The FineWeb2 shards, conda env (`ocr-env`), and directory skeleton already
exist on the cluster from the v1 run.

## 1 · Upload (from the Windows machine, Git Bash)

```bash
cd /c/Users/osama/Downloads/arabic-cyber-filter
H=oalsumaitti@panther-login.qcri.org
B=/export/home/oalsumaitti/fanar-data/cyber_filtered

ssh $H "mkdir -p $B/scripts $B/sbatch_scripts $B/keywords/tiers_clean"

scp scripts/filter_cyber_arabic_tiered.py scripts/merge_cyber_outputs_tiered.py  $H:$B/scripts/
scp sbatch_scripts/filter_cyber_tiered.sh sbatch_scripts/merge_cyber_tiered.sh   $H:$B/sbatch_scripts/
scp keywords/tiers_clean/tier?_keywords.txt                                      $H:$B/keywords/tiers_clean/
```

## 2 · VERIFY the upload (do not skip)

The v1 run taught us the hard way (README, lesson 3): an upload can silently
not transfer, and SLURM will happily run a stale file. Compare checksums:

```bash
# local (Git Bash)
md5sum scripts/filter_cyber_arabic_tiered.py scripts/merge_cyber_outputs_tiered.py \
       sbatch_scripts/filter_cyber_tiered.sh sbatch_scripts/merge_cyber_tiered.sh \
       keywords/tiers_clean/tier?_keywords.txt

# remote — must print the SAME hashes
ssh $H "cd $B && md5sum scripts/filter_cyber_arabic_tiered.py scripts/merge_cyber_outputs_tiered.py \
        sbatch_scripts/filter_cyber_tiered.sh sbatch_scripts/merge_cyber_tiered.sh \
        keywords/tiers_clean/tier?_keywords.txt"
```

Then confirm the tier files carry the exact census-built counts —
**136 / 295 / 57 / 152**:

```bash
ssh $H "wc -l $B/keywords/tiers_clean/tier?_keywords.txt"
#   136 tier1  ·  295 tier2  ·  57 tier3  ·  152 tier4  ·  640 total
```

If the `.sh` files were ever touched by a Windows editor, strip CRLF once:
`ssh $H "sed -i 's/\r$//' $B/sbatch_scripts/filter_cyber_tiered.sh $B/sbatch_scripts/merge_cyber_tiered.sh"`

## 3 · Size the array & smoke-test one shard

```bash
ssh $H
cd ~/fanar-data/cyber_filtered

ls ~/fanar-data/FineWeb2/*.jsonl.gz | wc -l     # e.g. 100
# if not 100, edit  #SBATCH --array=0-99  in sbatch_scripts/filter_cyber_tiered.sh
# to 0-<count-1>

# smoke test: run ONLY shard 0 through the real sbatch path
sbatch --array=0-0 sbatch_scripts/filter_cyber_tiered.sh
squeue -u $USER                                  # wait for it to finish
tail -20 logs_tiered/*_0.out
```

The log must end with the acceptance summary, e.g.
`matched >=1 keyword : N  /  accepted (tiered) : M  (~40-45% of matched)`,
and `output_tiered/` must contain `cyber_v3_<shard>.json`, `.csv`, **`.done`**.

## 4 · Launch the full parallel run (`cpu-all`)

```bash
sbatch sbatch_scripts/filter_cyber_tiered.sh
```

100 tasks, one shard each, `-p cpu-all -q normal`, 4 CPUs / 16 GB / 6 h per
task (v1 tasks took ~1–2 min each; the limits are generous safety margins).
The shard-0 smoke test is automatically skipped in seconds via its `.done`
marker — no duplicate work.

Monitor:

```bash
squeue -u $USER                                   # queue state
ls output_tiered/*.done | wc -l                   # shards finished (target: 100)
sacct -j <jobid> --format=JobID,State,Elapsed -X  # per-task states
grep -l "done marker exists" logs_tiered/*.out | wc -l   # how many were resumes
```

## 5 · If ANYTHING fails mid-run → just resubmit

This is the resume story, and it is one command:

```bash
sbatch sbatch_scripts/filter_cyber_tiered.sh
```

- Every finished shard has a `.done` marker → its task exits in ~1 s.
- Unfinished/failed shards have **no** marker (outputs are written to `.tmp`
  and renamed only on success, so there are never half-written `.json`s) →
  they re-run from scratch.
- Safe to resubmit any number of times; also safe to resubmit only the
  failed indices if you prefer: `sbatch --array=17,42 sbatch_scripts/filter_cyber_tiered.sh`.

## 6 · Merge (as a job, never on the login node)

```bash
sbatch sbatch_scripts/merge_cyber_tiered.sh
tail -5 logs_tiered/merge_*.out
```

The merge job **refuses to run** if any shard `.json` lacks its `.done`
marker (so a half-finished array can never produce a silently truncated
corpus), then writes:

```
result/cyber_arabic_v3_ALL.json
result/cyber_arabic_v3_ALL.csv     (utf-8-sig → opens clean in Excel)
```

## 7 · Sanity-check the result

```bash
python - <<'EOF'
import json, collections
d = json.load(open("result/cyber_arabic_v3_ALL.json", encoding="utf-8"))
print("records:", len(d))                       # expect ~6,041 on the same 100 shards
print(collections.Counter(r["accept_rule"] for r in d))
EOF
```

Output schema per record: the v1 columns plus
`tiers_found` (tier of each matched keyword, aligned with `keywords_found`)
and `accept_rule` (`tier1_hit` / `two_tier2_hits` / `three_tier3_hits` /
`four_plus_hits`) — so every keep decision remains fully explainable, now
including *which clause of the tiered rule* fired.

---

**Optional flag:** add `--extra-keywords keywords/all_cyber_keywords.txt`
(upload that file too) to the `python` line in `filter_cyber_tiered.sh` if
you also want untiered lexicon keywords recorded in the evidence trail (tier
`"-"`); they never influence acceptance — the decision stays exactly the
measured rule.
