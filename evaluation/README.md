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
│   ├── compute_metrics.py       verdicts → precision / recall / F1 (Wilson CIs)
│   ├── collect_pool_texts.py    gather all missed-record text for keyword mining
│   ├── mine_missing_keywords.py find cyber terms absent from the lexicon
│   └── verify_no_overlap.py     prove a keyword file is disjoint from the lexicon
├── samples/                 ← judge-view inputs shown to the LLM
│   ├── sample_matched.json
│   └── sample_unmatched.json
├── judgments/               ← raw per-record verdicts
│   ├── matched_*.json           (300 matched records)
│   └── unmatched_*.json         (200 pooled records)
└── results/                 ← computed outputs
    ├── metrics.json             precision / recall / F1
    ├── population_counts.json   stratum & pool sizes
    ├── candidate_terms.tsv      ranked keyword-gap candidates (df, surface, normalized)
    ├── judged_sample_matched.json / .csv    original columns + 4 judge columns
    └── judged_sample_unmatched.json / .csv
```

## Keyword-gap mining (turning the recall miss into lexicon fixes)

The recall analysis found the filter misses cyber documents written in vocabulary
the lexicon lacks. To turn that into concrete additions, three scripts mine the
**missed records** for cybersecurity terms that are *not* already in the lexicon:

```bash
# 1. Collect every rejected record that carries a broad cyber signal (~17k docs)
python evaluation/scripts/collect_pool_texts.py       # → scratch/pool_texts.jsonl

# 2. Extract cyber n-grams, dedupe against the lexicon (normalized), rank by df
python evaluation/scripts/mine_missing_keywords.py    # → results/candidate_terms.tsv

# 3. Hand-curate the ranked candidates into keywords/missing_cybersecurity_keywords.txt,
#    then prove it shares nothing with the existing lexicon:
python evaluation/scripts/verify_no_overlap.py keywords/missing_cybersecurity_keywords.txt
```

Dedup and overlap-checking normalize Arabic (strip tashkeel, unify alef-hamza and
alef-maqsura, strip the definite article `ال` per word) so a term is only counted
as "new" when **no** form of it already exists. The curated result —
[`keywords/missing_cybersecurity_keywords.txt`](../keywords/missing_cybersecurity_keywords.txt)
— is **24 verified-new keywords**, and the single biggest gap is a block of
**Latin/English terms** (`VPN`, `Antivirus`, `Firewall`, `Malware`, `Hacker`, …)
that the Arabic-only lexicon never contained.

### Why an "Arabic" lexicon was missing English words — the real reason

At first glance, Latin-script tokens in an *Arabic* cybersecurity lexicon look
wrong. They are not. Three facts about the domain explain it:

1. **The vocabulary of cybersecurity is coined in English.** Terms like *VPN*,
   *firewall*, *malware*, *ransomware*, *phishing*, *proxy* and *keylogger* enter
   Arabic writing **untranslated**. Sometimes no Arabic equivalent has settled;
   more often the English word simply *is* the term practitioners use, even when
   the surrounding sentence is fully Arabic. Real Arabic security writing is
   therefore **code-switched**: Arabic prose peppered with Latin-script technical
   tokens (`... يُنصح باستخدام VPN موثوق عند الاتصال بشبكة عامة ...`).
2. **Acronyms and product names are inherently Latin.** `VPN`, `DDoS`, `SSL`,
   `TLS`, `OTP`, `2FA` are initialisms of English phrases and are essentially
   never rewritten in Arabic letters; `OpenVPN` and similar tool names are proper
   nouns. They can only ever appear in Latin script.
3. **The original lexicon was built as an Arabic *terminology* list.** Its source —
   the [Arabic Cybersecurity Glossary](https://github.com/Alsumaitti/arabic-cyber-keywords)
   — was, by construction, a list of **Arabic** words and phrases. Latin-script
   tokens fell outside its definition of an entry, so they were never added. This
   is a **script/scope blind spot, not a conceptual gap**: the glossary "knows"
   the concept *VPN* (it has `الشبكة الافتراضية الخاصة`), it just never listed the
   three-letter form the documents actually use.

**Why the same words keep recurring:** because they are the *everyday* vocabulary
of the field in Arabic, not rare loanwords. `VPN` alone appears in **605** of the
missed records; `Antivirus` 70, `Proxy` 43, `Hacker` 32, `Firewall` 30. A single
Arabic article about VPNs may say "VPN" a dozen times and never once use
`الشبكة الافتراضية الخاصة`.

**What their absence cost:** pure **recall loss (false negatives).** A document
genuinely about cybersecurity, but that names the concept only through its English
token, matched *none* of the 1,209 Arabic entries and was silently rejected — even
though a human reads it instantly as on-topic. These are exactly the kind of
records behind the ~2,460 estimated false negatives in the [recall analysis](#recall--how-many-cybersecurity-documents-the-filter-missed);
`VPN`'s 605 missed documents make it the single highest-leverage addition.
`SSL`, `DDoS` and `TLS` are the exception — they already live *inside* lexicon
phrases (`بروتوكول SSL`, `هجمات DDoS`), so they were deliberately **not** re-added.

### The combined lexicon

For downstream use, [`keywords/all_cyber_keywords.txt`](../keywords/all_cyber_keywords.txt)
is a single **1,233-keyword** file: the original 1,209 entries (every surface
variant preserved) plus the 24 new terms, concatenated and de-duplicated on
**exact** lines. Dedup is exact — *not* normalized — on purpose: the filter matches
literally, so intentional variants like `الآلي`/`الالي` or a term with and without
`ال` must all survive. The two source files are left untouched; this is an additive
build artifact. The 7 ambiguous single roots in Group 3 of the missing-keywords
file (`اختراق`, `قرصنة`, …) stay commented out and are therefore **excluded** from
the combined file, matching their inactive status (single-root matches judged only
~21% precise — see [Precision](#precision--how-many-kept-records-are-truly-cybersecurity)).

## Tiered keywords — learning each keyword's evidential weight

The precision analysis showed the filter's real problem is treating all 1,209
keywords as equally trustworthy: one hit of `الأمن السيبراني` and one hit of
`حصان طروادة` both kept a record, yet the first is nearly always a cyber
document and the second is usually a political metaphor. The judge data lets us
**measure each keyword's individual reliability** and split the lexicon into
tiers by how much "company" a keyword needs before its match can be trusted:

| Tier | Meaning | Assignment rule (from the 300 judged records) | Size |
|---|---|---|---|
| **1** | self-sufficient — one hit is enough | appeared **alone** in a record judged cyber/borderline, and its solo records are majority-good (`solo_good > solo_bad`) | 38 |
| **2** | needs one companion | strongest good evidence is in 2-hit records (or demoted from Tier 1) | 64 |
| **3** | needs two companions | strongest good evidence is in 3-hit records | 34 |
| **4** | needs heavy company | only ever seen good in records with 4+ hits | 143 |

Built by [`scripts/build_keyword_tiers.py`](scripts/build_keyword_tiers.py) →
[`keywords/tiers/tier{1..4}_keywords.txt`](../keywords/tiers), with the full
per-keyword audit trail in
[`results/keyword_tiers_evidence.tsv`](results/keyword_tiers_evidence.tsv).

**The demotion rule is the crucial refinement.** A keyword seen solo in one
good record does *not* automatically reach Tier 1 — its solo record must not
usually be a false positive. Four keywords were demoted this way, exactly the
"traitors" the precision analysis exposed: `حصان طروادة` (solo: 1 good vs **5**
not_cyber — Trojan-horse metaphors in politics), `هاكر` (1 vs 3), `أمن البيانات`
and `اختراق البيانات` (1 vs 1 each). Without this rule the tier system would
re-import the very false positives the evaluation was built to catch.

Seven keywords earned **no tier at all** — every judged record they appeared in
was not_cyber: `الباب الخلفي` (literal back doors — 312 corpus records!),
`إرهاب إلكتروني`, `الترقيع الأمني`, `التصحيح الأمني`, `برمجيات التجسس`,
`تجزئة الشبكة`, `حوكمة الأمن السيبراني`. These are the lexicon's measured
liabilities.

**How to use the tiers:** instead of the flat `--min-keywords N` threshold,
keep a record iff it hits ≥ 1 Tier-1 keyword, **or** ≥ 2 keywords of Tier ≤ 2,
**or** ≥ 3 of Tier ≤ 3, **or** ≥ 4 of any tier. This keeps the high-precision
single-hit records (the 21%-precise stratum's *good* part) that a flat
`--min-keywords 2` would throw away, while still suppressing the metaphorical
single-hit noise.

**Caveat — coverage.** The tiers are evidence-based, and the evidence is the
300-record judged sample: only **286 of the 1,209** lexicon keywords were
observed at all, so 923 keywords remain unassigned (no evidence either way,
*not* judged bad). Judging the full corpus with
[`scripts/llm_judge_api.py`](../scripts/llm_judge_api.py) would tier every
keyword and firm up the small solo counts behind some Tier-1 assignments.

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
