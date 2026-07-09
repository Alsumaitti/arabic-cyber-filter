# Evaluation — LLM-as-Judge Verification of the Filter

The keyword filter answers a mechanical question ("does this document contain a
cybersecurity term?"). That is not the same as the question we actually care
about ("is this document *about* cybersecurity?"). This evaluation layer adds a
second, semantic judgment on top of the keyword match and uses it to **measure
how good the keyword filter actually is** — its precision, its recall, and where
it fails.

An LLM (Claude) reads each sampled document and returns a structured verdict:

| Field | Meaning |
|---|---|
| `judge_label` | `cyber` (cybersecurity is a main topic), `borderline` (present but minor), or `not_cyber` (false positive) |
| `judge_confidence` | `high` / `medium` / `low` |
| `judge_justification` | one English sentence citing the concrete evidence in the text |
| `judge_summary` | one English sentence stating what the document is about |

The full labelling rubric is in [`JUDGE_PROTOCOL.md`](JUDGE_PROTOCOL.md).

## Folder layout

```
evaluation/
├── README.md               ← you are here (method overview)
├── JUDGE_PROTOCOL.md        ← the exact labelling rubric
├── RESULTS.md               ← written interpretation of the numbers
├── scripts/                 ← the pipeline
│   ├── make_samples.py          build the matched + unmatched samples
│   ├── make_unmatched_sample.py standalone pooled-negative (recall) sampler
│   └── compute_metrics.py       verdicts → precision / recall / F1 (Wilson CIs)
├── samples/                 ← judge-view inputs shown to the LLM
│   ├── sample_matched.json
│   └── sample_unmatched.json
├── judgments/               ← raw per-record verdicts
│   ├── matched_*.json           (300 matched records)
│   └── unmatched_*.json         (200 pooled records)
└── results/                 ← computed outputs
    ├── metrics.json             precision / recall / F1
    ├── population_counts.json   stratum & pool sizes
    ├── judged_sample_matched.json / .csv    original columns + 4 judge columns
    └── judged_sample_unmatched.json / .csv
```

## Why sampling instead of judging all 13,345

Judging every record with a frontier LLM is the job of
[`scripts/llm_judge_api.py`](../scripts/llm_judge_api.py) (batch API, ~USD 80–110
for the whole corpus). For a trustworthy *metric*, though, judging a
well-designed **sample** is both cheaper and statistically sufficient. We follow
the standard practice for estimating a classifier's precision/recall on
imbalanced data: **stratified sampling** of the positives plus **pooling** for
the negatives.

## Precision — how many kept records are truly cybersecurity

The 13,345 matched records are split into four strata by how many *distinct*
keywords each record hit, because hit-count is a strong proxy for how "on-topic"
a record is:

| Stratum (distinct hits) | Corpus population | Sampled & judged |
|---|---|---|
| 1 | 9,062 | 100 |
| 2 | 2,209 | 80 |
| 3–5 | 1,610 | 60 |
| 6+ | 464 | 60 |
| **Total** | **13,345** | **300** |

Each of the 300 records was judged blind (in randomized order, not by stratum).
Two precision figures are reported:

- **Strict precision** — only `cyber` counts as a true positive.
- **Lenient precision** — `cyber` + `borderline` count (a borderline record does
  genuinely contain cybersecurity content, just not as its main topic).

Per-stratum rates are combined using each stratum's true corpus weight
(9062 / 2209 / 1610 / 464) to get a single **corpus-weighted precision**, with
Wilson 95% confidence intervals. See [`results/metrics.json`](results/metrics.json)
for the numbers and [`RESULTS.md`](RESULTS.md) for the written interpretation.

## Recall — how many cybersecurity documents the filter missed

Precision alone can be gamed by a filter that keeps only the most obvious hits.
Recall asks the opposite: of all the genuinely-cyber documents in the million,
how many did the keyword filter *catch*? The rejected ~986k records are far too
many to judge, so we use **TREC-style pooling**:

1. A deliberately broad "safety-net" regex (looser Arabic + English cyber
   signals **not** in the strict lexicon) is run over every rejected record. Its
   hits form a **candidate false-negative pool** — the records most likely to be
   missed cybersecurity content.
2. A random sample of that pool is judged, giving the cyber rate *inside* the
   pool.
3. A pure-random sample of the records *outside* the pool is judged too, to
   catch anything the safety net itself missed.
4. False-negative mass is extrapolated from both rates over their populations,
   and recall is `TP / (TP + FN_est)`.

This is a lower-bound-friendly estimate: the safety net is intentionally wide, so
it over-collects candidates rather than under-collecting.

## Reproducing

All scripts derive their paths from their own location, so run them from anywhere.

```bash
# 1. Build the samples (scans the 100 shards; deterministic, seed=42)
#    → writes samples/sample_matched.json, samples/sample_unmatched.json
python evaluation/scripts/make_samples.py

# 2. Judge them — either with the batch API judge over the sample,
#    or (as done here) an in-session LLM writing evaluation/judgments/*.json

# 3. Compute the metrics → writes results/metrics.json
python evaluation/scripts/compute_metrics.py
```

`scripts/make_unmatched_sample.py` is a faster, standalone version of the
pooled-negative (recall) scan, kept separate so the slow full-corpus pass can run
on its own.

## Key files

| Path | What it is |
|---|---|
| [`JUDGE_PROTOCOL.md`](JUDGE_PROTOCOL.md) | The exact labelling rubric and metric conventions |
| [`RESULTS.md`](RESULTS.md) | Human-readable interpretation of the metrics |
| [`scripts/`](scripts) | `make_samples.py`, `make_unmatched_sample.py`, `compute_metrics.py` |
| [`samples/`](samples) | The judge views shown to the LLM (`sample_matched.json`, `sample_unmatched.json`) |
| [`judgments/`](judgments) | Raw per-record verdicts (`matched_*.json`, `unmatched_*.json`) |
| [`results/judged_sample_matched.json`](results/judged_sample_matched.json) / `.csv` | The 300 matched records with original columns **+** the four judge columns |
| [`results/judged_sample_unmatched.json`](results/judged_sample_unmatched.json) / `.csv` | The 200 pooled records with verdicts |
| [`results/metrics.json`](results/metrics.json) | Final computed precision / recall / F1 |
