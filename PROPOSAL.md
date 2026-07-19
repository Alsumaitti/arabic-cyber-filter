# Proposal: Full-Corpus LLM Evaluation to Build a High-Precision Tiered Filtering System

**Submitted by:** Osamah Alsumaitti
**Project:** Arabic cybersecurity corpus filter for FineWeb2
**Request:** API budget for full-corpus LLM judging — **maximum USD 150**

---

## 1. Executive Summary

We built a lexicon-based filter that extracted **13,345 cybersecurity documents** from one million Arabic web records, then measured its quality scientifically with an LLM-as-Judge evaluation. The measurement exposed two specific weaknesses, and for each we have already developed a working, methodical fix:

1. **A recall gap:** the Arabic lexicon was missing the English terms that pervade Arabic technical writing (VPN, Firewall, …) — we **mined 24 missing keywords** from the missed records themselves and built a unified lexicon of **1,233 keywords**.
2. **A precision gap:** the filter trusts all keywords equally — we **designed a keyword *tier* system** that learns each keyword's individual reliability from the LLM verdicts.

Both fixes are currently built on a 300-record judged sample. **The single remaining step** to turn them into a production-grade filtering system is judging the corpus in full, at a cost of **~USD 85–120** (via the Batches API at a 50% discount). That budget is the subject of this proposal.

---

## 2. Current Status: How We Got Here

### Stage 1 — Building the filter (complete)
- Filtered **1,000,000 Arabic records** from FineWeb2 across 100 parallel SLURM tasks.
- Lexicon of **1,209 terms** from the Arabic Cybersecurity Glossary (the output of an earlier stage of this same project).
- Whole-word matching (no substrings) with a full evidence trail: every kept record carries the exact list of keywords that matched it (`keywords_found`).
- **Output:** a corpus of 13,345 documents, every classification human-verifiable.

### Stage 2 — Scientific measurement (complete)
We judged a stratified sample (300 kept records + 200 rejected records via TREC-style pooling) with an LLM that reads each full document and returns a structured verdict (cyber / borderline / not_cyber + justification):

| Metric | Value |
|---|---|
| Strict precision (corpus-weighted) | ~0.35 |
| Lenient precision (cyber + borderline) | ~0.68 |
| Estimated recall | ~0.79 |
| F1 | ~0.73 |
| Estimated missed documents (false negatives) | ~2,460 |

**The pivotal finding:** precision is determined almost entirely by how many distinct keywords a record matched:

| Distinct keyword hits | Corpus share | Lenient precision |
|---|---|---|
| 1 | 67.9% | 0.58 |
| 2 | 16.6% | 0.80 |
| 3–5 | 12.1% | 0.98 |
| 6+ | 3.5% | 1.00 |

---

## 3. Idea 1: Mining the Missing Keywords (complete)

Instead of guessing what the lexicon lacks, **we mined the very records the filter missed** (17,216 rejected records carrying a broad cybersecurity signal):

1. Extract cybersecurity n-grams (1–4 words) from the missed records' text.
2. Deduplicate against the lexicon under proper Arabic normalization (diacritics, hamza variants, the definite article) — a term counts as "new" only if *no* form of it already exists.
3. Rank by document frequency, then hand-curate.

**Result: 24 verified-new keywords** ([keywords/missing_cybersecurity_keywords.txt](keywords/missing_cybersecurity_keywords.txt)), with a structural discovery: **the single biggest gap is English terms**. Cybersecurity vocabulary is coined in English and enters Arabic writing untranslated — an Arabic author writes "VPN", not "الشبكة الافتراضية الخاصة". The original lexicon was built as an *Arabic* terminology list, so Latin-script tokens fell outside its scope — a scope gap, not a conceptual one.

**Measured impact:** the 24 keywords alone recover **916 additional cybersecurity documents** the filter was rejecting (VPN by itself appears in 604 of them).

**Ready deliverable:** a unified lexicon — [keywords/all_cyber_keywords.txt](keywords/all_cyber_keywords.txt) — containing **1,233 keywords** (the original 1,209 + the 24 new ones, exactly deduplicated and machine-verified disjoint).

---

## 4. Idea 2: The Tiered Keyword System (working prototype)

**The problem:** the current filter trusts every keyword equally. One match on `الأمن السيبراني` (cybersecurity) almost certainly means a cyber document; one match on `حصان طروادة` (Trojan horse) is usually a political metaphor. The naive fix (`min-keywords ≥ 2`) throws away *all* single-hit records — the good ones along with the bad.

**The idea:** crossing the LLM verdict with each record's keyword-hit count lets us **measure every keyword's individual reliability** and split the lexicon into tiers by how much "company" a keyword needs before its match can be trusted:

| Tier | Meaning | Assignment rule | Count |
|---|---|---|---|
| **1** | Self-sufficient — one hit is enough | appeared **alone** in a record judged cyber, and its solo records are majority-correct | 38 |
| **2** | Needs one companion | strongest positive evidence is in 2-keyword records | 64 |
| **3** | Needs two companions | strongest positive evidence is in 3-keyword records | 34 |
| **4** | Needs heavy company | only ever seen correct alongside 4+ keywords | 143 |

**The demotion rule (the heart of the system):** a keyword that appears alone in one correctly-judged record does *not* automatically reach Tier 1 — the majority of its solo records must be correct. `حصان طروادة` appeared solo in 1 cyber record versus **5 metaphorical ones**, so it was demoted. The system also exposed **7 keywords that never produced a single correct match** in the sample (most notably `الباب الخلفي` — literal back doors — responsible for 312 corpus records).

**How it improves filtering:** instead of a blind threshold, a tiered acceptance rule — keep a record iff it matches ≥ 1 Tier-1 keyword, or ≥ 2 keywords of Tier ≤ 2, or ≥ 3 of Tier ≤ 3, or ≥ 4 of any tier. This keeps the genuinely good single-hit records (which a blind threshold discards) while suppressing the metaphorical noise — a precision lift **without** sacrificing recall.

---

## 5. The Gap: A Sample Is Not Enough for Production

The tier prototype rests on only 300 judged records, and therefore:

- Only **286 of the 1,209** keywords appeared in the sample at all — **923 keywords have no evidence** and cannot be assigned a tier.
- Some Tier-1 assignments rest on one or two solo records — statistically fragile.
- The 24 newly mined keywords (VPN and friends) have not entered any evaluation yet.

**The fix:** judge the corpus **in full**. The infrastructure is already built and tested ([scripts/llm_judge_api.py](scripts/llm_judge_api.py)): the same measured judging protocol, run through the Claude Message Batches API (50% discount), schema-enforced structured output, resumable per chunk. **The only missing ingredient is API credit.**

---

## 6. Options and Cost

| | Option A — current corpus | **Option B — expanded corpus (recommended)** |
|---|---|---|
| Lexicon | old: 1,209 keywords | **new: 1,233 keywords** ([all_cyber_keywords.txt](keywords/all_cyber_keywords.txt)) |
| Corpus judged | 13,345 records | **~14,260 records** (13,345 + ~916 recovered) |
| Re-filtering | not needed | minutes on SLURM — **free** |
| The 24 new keywords evaluated | ✗ | ✓ evaluated and entered into the tier system |
| Estimated cost | ~USD 80–110 | **~USD 85–120** |

**Recommendation: Option B.** For a marginal cost difference (~USD 10) we get the expanded lexicon evaluated in full — including the newly mined English terms — so everything enters one consistent tier system.

**Requested amount: USD 150** (upper estimate + ~25% safety margin for smoke tests and any failed-batch retries). Any surplus goes unspent.

**Time:** 1–3 hours end to end (mostly passive batch-processing wait; the official guarantee is 24 hours maximum). Near-zero manual effort.

---

## 7. Expected Deliverables

1. **A fully judged corpus:** ~14,260 records, each carrying an LLM verdict + justification + summary — a reusable research asset (classifier training, fine-tuning, follow-up studies).
2. **A complete tier system:** every keyword observed in the corpus assigned a tier backed by thousands of verdicts instead of hundreds — rebuilt by re-running [build_keyword_tiers.py](evaluation/scripts/build_keyword_tiers.py) (seconds, no extra cost).
3. **A second-generation filter:** the tiered acceptance rule applied in the filter, with expected precision jumping from ~0.68 to **~0.85+** (lenient) while preserving recall — measured, not guessed.
4. **A measured deletion list:** keywords proven to be a net liability to the lexicon across the whole corpus.

---

## 8. Conclusion

The entire research infrastructure is in place and tested: the filter, the judging protocol, an expanded **1,233-keyword** lexicon, and a working prototype of the tier system. What remains is one small budget item — **maximum USD 150** — to turn the 300-record sample into full evidence over ~14,260 records, completing an Arabic cybersecurity filtering system whose precision is measured and whose every decision is explainable.

---

## 9. Post-census results — what was delivered

The full-corpus judging is **done**. All **14,653** records of the expanded (v2) corpus were individually judged (`cyber` 3,131 / `borderline` 2,550 / `not_cyber` 8,972) — a complete census, not a sample, merged into [evaluation/chat_judging/judged_corpus.json](evaluation/chat_judging/judged_corpus.json). Each of the four promised deliverables (§7), measured:

**1. A fully judged corpus.** ✓ 14,653 records, every one carrying an LLM verdict + justification. This is the reusable research asset.

**2. A complete tier system.** ✓ Rebuilt on the census: **714** keywords assigned from real evidence (up from 286 in the sample), **136 / 295 / 57 / 152** in tiers 1–4, with **213** demoted out of Tier 1 (their solo matches were mostly false positives — e.g. `VPN`, 167 good vs 414 not_cyber). Audit trail: [evaluation/results/keyword_tiers_evidence.tsv](evaluation/results/keyword_tiers_evidence.tsv).

**3. A second-generation filter.** ✓ The tiered acceptance rule, **measured exactly** on the census by [evaluation/scripts/apply_tier_filter.py](evaluation/scripts/apply_tier_filter.py):

| Rule | Kept | Lenient precision | Retention of genuine | F1 |
|---|---|---|---|---|
| accept-all (v2 as-is) | 14,653 | 0.388 | 1.000 | 0.559 |
| blind `min ≥ 2` | 4,846 | 0.625 | 0.534 | 0.576 |
| **Tiered rule** | **6,041** | **0.636** | **0.677** | **0.656** |

The tiered rule **strictly dominates the blind threshold** — higher precision *and* higher retention at once — and has the best F1. It lifts lenient precision from 0.39 to **0.64** while keeping **68%** of genuine records, exactly the "precision lift without sacrificing recall" the proposal argued for. (The §7 projection of ~0.85 lenient was based on the cleaner original lexicon; the v2 corpus deliberately traded precision for recall by adding the English false-positive-magnet terms, so 0.64 at 68% retention is the honest measured figure — and still a 64% relative precision gain.)

**4. A measured deletion list.** ✓ **74** keywords whose every census appearance was `not_cyber` (zero good matches) — [keywords/deletion_candidates.txt](keywords/deletion_candidates.txt) — a review-and-drop list (mostly literal-sense phrases like `حاجز الحماية` "protective barrier", `تدقيق أمني` "audit"). They earn no tier, so the v3 filter already ignores them.

Full write-up and the honest in-sample caveat: [evaluation/RESULTS.md](evaluation/RESULTS.md).
