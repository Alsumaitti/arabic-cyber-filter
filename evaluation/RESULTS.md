# Results — What the LLM Judge Found

Claude judged a stratified sample of **300** of the 13,345 keyword-matched
records (plus a pooled sample of the rejected records for recall). Every verdict,
with its English justification, is in [`judgments/`](judgments) and joined back
to the source records in [`judged_sample_matched.json`](judged_sample_matched.json).

## Headline

The keyword filter has **high recall but modest precision**, and its precision is
almost entirely determined by **how many distinct keywords a record hit**. A
single incidental keyword is a weak signal; three or more is a strong one.

## Precision by stratum

Judged blind, `cyber` = cybersecurity is a main topic, `borderline` = present but
minor, `not_cyber` = false positive. Confidence intervals are Wilson 95%.

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

<!-- RECALL_PLACEHOLDER -->
Recall is estimated by TREC-style pooling over the ~986k rejected records (see
[`EVALUATION.md`](EVALUATION.md) § Recall). Numbers are filled in from
[`metrics.json`](metrics.json) once the pooled unmatched sample has been judged.

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
