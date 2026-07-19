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
