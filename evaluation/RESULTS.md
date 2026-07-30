# Results — What the LLM Judge Found

Evaluation ran in two rounds. **Round 1** judged a stratified **sample** of 300
of the 13,345 keyword-matched records (plus a pooled sample of the rejected
records for recall). **Round 2** — after the lexicon was expanded to 1,233
keywords and the corpus re-filtered ("v2", 14,653 records) — judged the corpus
in **full**: a census of every single record. Every verdict, with its English
justification, is in [`judgments/`](judgments) (round 1) and
[`chat_judging/judged_corpus.json`](chat_judging/judged_corpus.json) (round 2).

**The census below is the authoritative measurement.** The round-1 sample
sections (precision by stratum, recall, false-positive patterns) are kept because
they still carry the recall estimate and the qualitative failure analysis, which
the census — which only covers *kept* records — does not replace.

How the census was run — chunking, model, the ~5-hour-limit-driven resumable
design, the multi-session timeline (~20 sessions / ~14 h active over 5 days), and
the token/cost accounting — is documented in
[`chat_judging/CAMPAIGN.md`](chat_judging/CAMPAIGN.md).

## Headline

The keyword filter has **high recall but modest precision**, and its precision is
almost entirely determined by **how many distinct keywords a record hit**. A
single incidental keyword is a weak signal; three or more is a strong one. The
census then turns that finding into a working **second-generation filter**: a
tiered acceptance rule that lifts lenient precision from 0.39 to **0.64** while
keeping **68%** of the genuine records — strictly beating a blind threshold on
both precision and recall.

## Full-corpus census (authoritative)

Every one of the **14,653** v2-corpus records was read and labelled. Because this
is the whole population, the numbers are **measured exactly** — no confidence
intervals.

| Label | Records | Share |
|---|---|---|
| `cyber` | 3,131 | 21.4% |
| `borderline` | 2,550 | 17.4% |
| `not_cyber` | 8,972 | 61.2% |
| **Total** | **14,653** | 100% |

### Precision by keyword-hit count (census)

| Distinct keyword hits | Records | Share | Strict precision (`cyber`) | Lenient (`cyber`+`borderline`) |
|---|---|---|---|---|
| **1** | 9,807 | 66.9% | **0.116** | **0.270** |
| **2** | 2,352 | 16.1% | **0.312** | **0.570** |
| **3–5** | 1,887 | 12.9% | **0.464** | **0.645** |
| **6+** | 607 | 4.1% | **0.631** | **0.778** |
| **Whole corpus** | **14,653** | 100% | **0.214** | **0.388** |

The monotonic rise with hit-count reproduces the round-1 sample at full resolution,
but at a **lower absolute level** (whole-corpus strict 0.21 vs the sample's ~0.35).
That is **by design**: v2 added the recall-driven Latin/English terms (`VPN`,
`Firewall`, `Proxy`, `Antivirus` …), which are the corpus's biggest false-positive
magnets — consumer-VPN/antivirus product pages, privacy-policy boilerplate, and
auto-generated adware "removal" pages. The census prices that precision-for-recall
trade exactly, per keyword — which is what the tiered filter below acts on.

## Second-generation filter — the tiered acceptance rule (measured)

The tiered lexicon ([`keywords/tiers/`](../keywords/tiers), rebuilt on the census:
**136 / 295 / 57 / 152** keywords in tiers 1–4) drives the proposal's acceptance
rule — *keep a record iff it matches ≥ 1 Tier-1 keyword, or ≥ 2 of tier ≤ 2, or
≥ 3 of tier ≤ 3, or ≥ 4 tiered keywords of any tier*. Because the census carries a
label for **every** record, [`scripts/apply_tier_filter.py`](scripts/apply_tier_filter.py)
measures the rule exactly and against the naive baselines it set out to beat:

| Rule | Kept | Strict P | Lenient P | Retention (of genuine) | F1 (lenient) |
|---|---|---|---|---|---|
| accept-all (v2 corpus as-is) | 14,653 | 0.214 | 0.388 | 1.000 | 0.559 |
| min distinct keywords ≥ 2 | 4,846 | 0.411 | 0.625 | 0.534 | 0.576 |
| min distinct keywords ≥ 3 | 2,494 | 0.504 | 0.678 | 0.297 | 0.413 |
| **Tiered acceptance rule** | **6,041** | **0.397** | **0.636** | **0.677** | **0.656** |

_Retention = share of all 5,681 genuine (`cyber`+`borderline`) records that the rule
keeps. F1 = harmonic mean of lenient precision and retention._

**The tiered rule strictly dominates the blind `min ≥ 2` threshold** — higher
lenient precision (0.636 vs 0.625) **and** higher retention (0.677 vs 0.534) at
once, and the best F1 of any rule (0.656). Against the raw v2 corpus it lifts
lenient precision from 0.388 to **0.636** (a 64% relative gain) while still keeping
**68%** of the genuine content. This is exactly the proposal's promise realised:
the rule keeps the good single-hit records a flat threshold throws away (a record
matched only by `الأمن السيبراني` is trusted; one matched only by `VPN` is not)
while suppressing the metaphorical/product noise.

The tiered rule kept 6,041 records = **cyber 2,398 + borderline 1,446 + not_cyber
2,197**. To emit that accepted corpus, run `apply_tier_filter.py --write-corpus`
(→ `result/cyber_arabic_v3_tiered.json`).

### Produced at scale — and independently validated

The rule above was not left on paper: it was implemented as a production filter
([`scripts/filter_cyber_arabic_tiered.py`](../scripts/filter_cyber_arabic_tiered.py))
and **re-run over the full 1,000,000-record FineWeb2 sample** on the cluster
(100-task `cpu-all` SLURM array, resumable — see
[`TIERED_RUN_GUIDE.md`](../TIERED_RUN_GUIDE.md)), producing
`result/cyber_arabic_v3_ALL.{json,csv}`.

That run is a genuine end-to-end check, because it reached its answer by a
completely different path than the table above (a fresh regex scan of 1M raw
records, vs. offline arithmetic over the census). The two agree **exactly**:

| | Offline, from the census | Cluster, scanning 1M raw records |
|---|---|---|
| Accepted | 6,041 | **6,041** |
| `tier1_hit` | 3,253 | **3,253** |
| `two_tier2_hits` | 2,780 | **2,780** |
| `three_tier3_hits` | 6 | **6** |
| `four_plus_hits` | 2 | **2** |

**Why they must agree** (and why this is a real test, not a tautology): the tier
lexicon (640 keywords) is a verified strict subset of the v2 lexicon (1,233), and
every clause of the rule needs ≥ 1 tier-keyword hit — so any accepted record was
necessarily already kept by v2, i.e. already inside the judged 14,653. A record
from the other ~985,000 *cannot* pass. Confirmed empirically by
[`join_v3_with_judgments.py`](scripts/join_v3_with_judgments.py): all 6,041
matched an existing verdict by text hash, **0 unmatched, 0 new**. Divergence would
have meant a truncated tier file, a stale upload, or the wrong input shards.

The v3 corpus is therefore a strict, higher-precision **subset** of v2 — a
precision refinement, never a recall extension. Relative to v2 it discards
**6,775 of the 8,972 junk records (75%)** while retaining **2,398 of the 3,131
strictly-cyber ones (77%)**; the discarded pile itself measures only 0.085 strict
/ 0.213 lenient precision, confirming the rule cuts along the intended seam.
Because every accepted record already carries a census verdict, the shipped
corpus is published **with labels attached**
(`result/cyber_arabic_v3_ALL_judged.json`).

> **Caveat (honest):** the tiers were learned from this same census, so the tiered
> row is **in-sample** (optimistic). The baseline rows use no labels, so they are
> unbiased — the fair reading is "tiered rule **vs those baselines**", where it wins
> on both axes.

## Measured deletion list — keywords that are a net liability

Crossing every keyword against the census also exposes the lexicon's dead weight:
**74** keywords whose *every* judged appearance (across all 14,653 records) landed
in a `not_cyber` record — zero good matches. They earn no tier (so the v3 rule
already ignores them) and are the measured **review-and-drop list**
([`keywords/deletion_candidates.txt`](../keywords/deletion_candidates.txt)). The
worst offenders are literal-sense or generic phrases: `تدقيق أمني` (20 not_cyber —
"audit" in a business sense), `حاجز الحماية` / `حاجز حماية` / `حواجز الحماية` (48
combined — a literal "protective barrier"), `الترقيع الأمني`, `فجوة هوائية` (a
literal "air gap/void"). A few are valid cyber concepts (`تقسيم الشبكات` network
segmentation, `اخفاء المعلومات` steganography) whose handful of corpus matches were
literal-sense or drowned by dominant `not_cyber` signals — so this is a
**human-review** list, not an automatic delete. Either way, their no-tier status
means they cannot trigger a false accept in the tiered filter.

## Precision by stratum (round-1 stratified sample, 300 records)

Judged blind, `cyber` = cybersecurity is a main topic, `borderline` = present but
minor, `not_cyber` = false positive. Confidence intervals are Wilson 95%. _(These
were the sample estimates; the census above supersedes them for precision.)_

| Distinct keyword hits | Corpus share | n judged | `cyber` | `borderline` | `not_cyber` | Strict precision | Lenient precision |
|---|---|---|---|---|---|---|---|
| **1** | 67.9% | 100 | 21 | 37 | 42 | **0.21** (0.14–0.30) | **0.58** (0.48–0.67) |
| **2** | 16.6% | 80 | 41 | 23 | 16 | **0.51** (0.41–0.62) | **0.80** (0.70–0.87) |
| **3–5** | 12.1% | 60 | 45 | 14 | 1 | **0.75** (0.63–0.84) | **0.98** (0.91–1.00) |
| **6+** | 3.5% | 60 | 58 | 2 | 0 | **0.97** (0.89–0.99) | **1.00** (0.94–1.00) |

Combining the strata with their true corpus weights (9062 / 2209 / 1610 / 464):

|  | Strict (`cyber` only) | Lenient (`cyber`+`borderline`) |
|---|---|---|
| **Corpus-weighted precision** | **≈ 0.35** | **≈ 0.68** |
| Estimated true positives in the 13,345 | ≈ 4,700 | ≈ 9,100 |

### Reading this

- **Single-hit records (68% of the corpus) are the weak point.** Only ~21% are
  substantively about cybersecurity; ~42% are outright false positives
  (metaphorical "الحرب الإلكترونية" in a military-hardware piece, "الباب الخلفي"
  meaning a literal back door, "حصان طروادة" as a political metaphor, boilerplate
  privacy-policy security clauses, forex/casino spam pages whose only cyber word
  is an incidental "firewall" or "VPN", a copyright footer citing the "computer
  crimes law").
- **Multi-hit records are excellent.** At 3+ distinct keywords, lenient precision
  is ≥ 98% and even strict precision is 75–97%. The evidence trail works exactly
  as intended: hit-count is a usable confidence score.
- **This is directly actionable.** Re-running the filter with `--min-keywords 2`
  (or filtering the existing corpus on `len(keywords_found) >= 2`) would lift
  strict precision from ~0.35 to roughly ~0.6 and lenient precision to ~0.85,
  discarding most of the single-hit noise while keeping essentially all of the
  genuine cybersecurity content. The single-hit stratum is where a human reviewer
  (or the full-corpus API judge) should focus.

## Recall

Recall was estimated by TREC-style pooling over the **986,655 rejected records**
(see [`README.md`](README.md) § Recall). A deliberately broad safety-net
regex (loose Arabic + English cyber signals *not* in the strict lexicon) flagged
**15,535** of them as candidate misses; a random **120** of that pool and a
random **80** from the ~971k records *outside* the pool were judged.

| Sample | n judged | cyber + borderline | rate | population | est. false negatives |
|---|---|---|---|---|---|
| Safety-net pool | 120 | 19 | 15.8% | 15,535 | ~2,460 |
| Outside the pool (random) | 80 | 0 | 0.0% | 971,120 | ~0 |
| **Total est. false negatives** | | | | | **~2,460** |

With ~9,070 true positives kept (lenient) and ~2,460 estimated missed:

|  | Value |
|---|---|
| **Estimated recall (lenient)** | **≈ 0.79** |
| **F1 (lenient precision × recall)** | **≈ 0.73** |

### What the misses look like

The genuine false negatives the filter missed were mostly documents that discuss
cybersecurity in **general Arabic terms the strict lexicon doesn't contain** — a
report on Pegasus spyware ("تجسّس إلكتروني"), Israel's Unit 8200 and the
information-security industry, a first-person account of cracking a school's WiFi
("اختراق الشبكة"), SEC cybersecurity-breach rules, and phone-surveillance apps.
These are exactly the kind of records a lexicon expansion (adding the broad terms)
or a recall-oriented second pass would recover.

### Caveat on the estimate

The recall point estimate rests on the random-outside-pool rate being ~0 (0 of 80
judged). That is plausible — the safety net is wide, so genuine Arabic
cybersecurity text almost always contains either a strict keyword or a broad-net
term — but 0/80 has a Wilson 95% upper bound near 4.6%. If the true outside-pool
rate were at that upper bound, recall would fall substantially. The robust,
low-variance finding is the **within-pool miss rate of ~16%**, which is what a
lexicon expansion should target; the single-number recall is a best estimate, not
a tight bound.

## Can an open 31B model replace the frontier judge?

Full write-up: [`JUDGE_BENCHMARK.md`](JUDGE_BENCHMARK.md). Summary of the result,
because it changes what the rest of this project can afford to do.

Every number above cost frontier judgments. That is the expensive part of careful
curation, and it is most expensive exactly where corpora most need curating — in
low-resource languages with small budgets. So we tested whether a locally-hosted
open model can take over, on a blind **1,992-record** stratified set (166 per cell
over 3 labels × 4 keyword-hit buckets), gold labels never leaving the local
machine, `gemma-4-31B-it` in bf16 on 2 × H200.

| metric | zero-shot | few-shot (11 exemplars) | delta |
|---|---|---|---|
| raw agreement (3-way) | 0.613 | 0.614 | +0.002 |
| corpus-weighted agreement | 0.749 | 0.752 | +0.004 |
| Cohen κ (3-way) | 0.419 | 0.422 | +0.002 |
| **binary agreement** | 0.851 | **0.852** | +0.001 |
| **Cohen κ (binary)** | 0.651 | **0.654** | +0.003 |
| `cyber` precision / recall | 0.526 / 0.992 | 0.526 / 0.992 | — |
| `borderline` precision / recall | 0.554 / 0.146 | 0.564 / **0.146** | +0.010 / **0** |
| `not_cyber` precision / recall | 0.826 / 0.700 | 0.825 / 0.705 | −0.001 / +0.005 |

**Both arms parsed 1,992/1,992 verdicts with 0 unparsed**, so nothing here is a
parsing artefact.

### The prompting fix was tried, and it failed

The few-shot arm was designed against the zero-shot failure mode: 5 of its 11
exemplars are `borderline`, each carrying the frontier model's own justification,
drawn from the census with all 1,992 evaluation ids excluded.

It moved 3-way κ by **+0.003** and `borderline` recall by **exactly nothing** —
while changing **483 of 1,992 (24.2%)** generated justifications and only
**5 of 1,992 (0.3%)** labels. The model saw different input, reasoned differently
in prose, and labelled the same. That rules out the boring explanation and makes
this a **capability limit, not a prompting limit** — a stronger claim than "few-shot
fixed it" would have been.

### Where it breaks, precisely

Confusion matrix, few-shot arm (rows = frontier gold, columns = gemma):

| | cyber | borderline | not_cyber |
|---|---|---|---|
| **cyber** | **659** | 5 | 0 |
| **borderline** | 468 | **97** | 99 |
| **not_cyber** | 126 | 70 | **468** |

Against a gold set that is 664/664/664 by construction, gemma predicts
**1,253 / 172 / 567**. It assigns `cyber` to 63% where the truth is 33%, and
effectively does not use the middle category.

The error is one-directional and diagnosable. `cyber` recall is 0.992 and the
`cyber → not_cyber` cell is **0** — across 664 documents, not one catastrophic
miss. But of 664 `borderline` documents it called **468 `cyber`**. It is not
confused about the domain; it is uncalibrated about *degree of topicality* —
presence of the subject reads to it as aboutness.

Agreement therefore **falls** as keyword density rises:

| distinct keyword hits | n | zero-shot | few-shot |
|---|---|---|---|
| 1 | 498 | 0.691 | 0.693 |
| 2 | 498 | 0.643 | 0.645 |
| 3 | 498 | 0.594 | 0.594 |
| 4+ | 498 | 0.524 | 0.526 |

Backwards from the naive expectation, and it confirms the mechanism: more lexicon
hits means more security vocabulary, which is the signal being over-read. The bias
is worst on exactly the population a keyword filter hands to a judge.

One caveat in the model's favour: the frontier census contains **zero
high-confidence `borderline` labels** — every one is medium or low. Part of the
3-way gap is irreducible disagreement about a genuinely fuzzy boundary, which caps
achievable 3-way κ for *any* judge. It does not rescue 0.146 recall.

### What it licenses — and the correction that makes it useful

Collapsed to the binary decision the tiered rule actually makes:

| | |
|---|---|
| agreement | **0.852** [95% CI 0.836–0.867] |
| Cohen κ | **0.654** (substantial) |
| cyber-ish P / R / F1 | 0.862 / 0.925 / 0.893 |
| sensitivity / specificity | **0.9255** / **0.7048** |

A biased classifier with a *measured* operating point still gives unbiased
population estimates. Rogan–Gladen inverts the misclassification:

`p_true = (p_obs − 0.2952) / 0.6303`

Validated against the evaluation set itself: observed 0.7154 → corrected
**0.667**, true gold **0.667**. Exact.

**So: yes for accept/reject, no for graded relevance.** The recipe is a hybrid —
spend frontier judgments once to build the reference standard and calibrate the
open model, then run the open model at corpus scale with its bias corrected
arithmetically. Do **not** rebuild the keyword tiers from open-model labels: tier
assignment depends on each keyword's good/bad ratio, and a judge that inflates
`cyber` by 1.9× would inflate every keyword's apparent quality, worst for the
high-frequency keywords that matter most.

### Next: this is how v4 gets measured

The v4 corpus (**370,019** records from ~70M parquet rows) has no quality numbers
yet — every figure above describes v2/v3 at 1M records. Judging 370k with a
frontier model is unaffordable; judging a calibrated sample locally is not:

```bash
sbatch sbatch_scripts/sample_v4_for_judging.sh   # rule-stratified sample, CPU
sbatch sbatch_scripts/judge_v4_sample.sh         # 2 H200s, resumable
python evaluation/scripts/estimate_v4_precision.py \
    --pred "evaluation/judge_benchmark/gemma_v4_P2_p*.jsonl" \
    --meta evaluation/judge_benchmark/v4_sample_meta.jsonl --show-cal
```

The sample is stratified by **acceptance rule** rather than drawn uniformly, so the
report gives both a corpus-level corrected precision (with sampling *and*
calibration uncertainty propagated) and a per-rule breakdown showing which of the
four doors into the corpus leaks. If one rule is clearly weaker with
non-overlapping CIs, dropping it raises corpus precision with **no further
judging**.

#### The tiered rule is two doors at scale, not four

Sampling revealed the v4 corpus composition, which is a result in its own right:

| acceptance rule | records | share of corpus |
|---|---|---|
| `tier1_hit` | 192,583 | 52.05% |
| `two_tier2_hits` | 177,351 | 47.93% |
| `three_tier3_hits` | 71 | 0.019% |
| `four_plus_hits` | 14 | 0.004% |

The third and fourth rules admitted **85 records out of 370,019** — small enough
to be judged exhaustively rather than sampled. This is not a bug, it follows from
the rule ordering: reaching `three_tier3_hits` requires a document with *no*
tier-1 keyword, *at most one* tier-≤2 hit, and *three or more* tier-≤3 hits.
Against 136 tier-1 and 295 tier-2 keywords, almost nothing threads that needle.

The practical consequence is that v4's precision is determined almost entirely by
two rules of near-equal volume, so equal allocation across strata is close to
optimal and the effective sample size stays high (**n_eff ≈ 1,498** from 1,585
judged records). The consequence for interpretation is that the two vestigial
rules cannot support any conclusion, and
[`estimate_v4_precision.py`](scripts/estimate_v4_precision.py) excludes strata
below 1% of the corpus from the per-rule verdict — otherwise a 14-record stratum
could be labelled "the weakest rule" and prompt a change that affects nothing.

## Most common false-positive patterns (single-hit stratum)

From the judged not_cyber records, the recurring causes were:

1. **Wrong sense / metaphor** — "الحرب الإلكترونية" (electronic warfare, military),
   "حصان طروادة" (Trojan horse as political metaphor or the Troy legend),
   "الباب الخلفي" (a literal back door), "فك التشفير" (media decoding).
2. **Boilerplate legal text** — privacy policies and terms-of-service whose only
   cyber content is a stock "we protect against unauthorized access" clause.
3. **Spam / SEO / aggregator pages** — forex and online-casino pages whose only
   keyword is an incidental "VPN", "firewall", or "SSL".
4. **Copyright footers** — site footers citing the "computer crimes law" against
   copying, attached to sports or entertainment articles.
5. **Passing mentions in general news** — one clause about "cyberattacks" inside a
   diplomacy or sanctions story.

Each of these is exactly what the `keywords_found` evidence column makes visible,
and what a min-keywords threshold or a second-pass LLM judge removes.
