---
license: odc-by
language:
- ar
- en
multilinguality:
- multilingual
size_categories:
- 100K<n<1M
task_categories:
- text-classification
- text-generation
tags:
- cybersecurity
- arabic
- low-resource
- web-corpus
- fineweb
- llm-as-judge
- data-curation
pretty_name: Arabic Cybersecurity Web Corpus
configs:
- config_name: corpus
  data_files:
  - split: train
    path: corpus/train-*.parquet
- config_name: census
  data_files:
  - split: train
    path: census/train-*.parquet
- config_name: judge_benchmark
  data_files:
  - split: test
    path: judge_benchmark/test-*.parquet
---

# Arabic Cybersecurity Web Corpus

A domain-filtered Arabic web corpus for cybersecurity — plus, unusually, **the
labelled evidence and the measurement apparatus that show how good it is**.

Most domain corpora ship as a pile of documents with a keyword list and an
assurance of quality. This one ships with 14,653 documents individually labelled
by a frontier model against a written rubric, a measured precision with
confidence intervals, and a reproducible benchmark for the judging itself. If
you do not trust our numbers, the `census` config lets you recompute them.

> **Why this exists.** Cybersecurity NLP resources are overwhelmingly English.
> Arabic — ~400M speakers, and a region with acute and well-documented security
> needs — has almost none. The obstacle is not that Arabic security text does not
> exist on the web; it is that finding it inside a trillion-token crawl requires
> a filter, and building a *trustworthy* filter requires labelled data that also
> does not exist. This dataset is the attempt to break that loop.

---

## What's in it

| config | rows | what it is |
|---|---|---|
| `corpus` | **277,418** | Arabic web documents about cybersecurity — keyword-filtered from FineWeb2 **and individually LLM-judged** |
| `census` | 14,653 | **every** document of an earlier corpus version, labelled by a frontier LLM with a justification |
| `judge_benchmark` | 1,992 | documents carrying a frontier gold label *and* two open-model runs |

Plus `keywords/` — the 1,233-term lexicon and its four evidence-derived tiers, so
the filter is reproducible rather than merely described.

### `corpus` — the bulk data

```python
from datasets import load_dataset
ds = load_dataset("<YOUR-USERNAME>/arabic-cybersecurity-web", "corpus", split="train")
```

| field | description |
|---|---|
| `id` | `sha1(text)[:16]` — stable, and the join key across all three configs |
| `text` | document text |
| `url`, `date`, `dump` | provenance: source URL, crawl date, Common Crawl dump |
| `keywords_found` | **which lexicon terms matched** — the evidence for inclusion |
| `tiers_found` | the tier of each matched term (1 = strongest) |
| `accept_rule` | which acceptance rule admitted this record |
| `judge_label` | the judge's verdict: `cyber` or `borderline` (every record here passed) |

**Measured precision: 0.890** [0.863–0.914]. Every one of the 370,019
keyword-matched candidates was judged individually by a local 31B model with a
measured operating point, and the 25% it rejected were removed. See the story
below for what that means and does not mean.

> **Free precision, one line.** Document length is the strongest single quality
> signal in this corpus — and it is *inverted* from the naive expectation.
> Removed documents had a median length of 6,854 characters against 3,135 for
> kept ones, because long pages are forums, archives and aggregators that mention
> security in passing. Filtering `len(text) < 8000` raises precision from
> **0.890 to 0.943** while keeping 86% of the records:
>
> ```python
> ds = ds.filter(lambda r: len(r["text"]) < 8000)
> ```

`keywords_found` is the design decision we would defend hardest: every record
carries the reason it was kept, so you can audit or re-filter without rerunning
anything. Want only documents matched by a tier-1 term? Filter on `tiers_found`.

### `census` — the part we think is most reusable

**14,653 Arabic documents, each labelled `cyber` / `borderline` / `not_cyber` by
a frontier model against a written rubric, with a confidence and a
justification.** Not a sample — every record of the v2 corpus.

To our knowledge this is the first labelled Arabic cybersecurity
domain-relevance dataset. It is useful well beyond the corpus it came from:

- **train a classifier** to replace keyword filtering entirely
- **evaluate your own filter** against a fixed reference standard
- **study the boundary** — `borderline` marks documents that mention security
  without being about it, which is exactly where naive filters fail
- **study Arabic security discourse** with the justifications as a starting point

| field | description |
|---|---|
| `label` | `cyber` \| `borderline` \| `not_cyber` |
| `confidence` | `high` \| `medium` \| `low` |
| `justification` | short free-text reason for the label |
| `n_keywords` | count of distinct lexicon terms matched |

**Read the `borderline` class carefully.** Label × confidence across the whole
census:

| label | high | medium | low |
|---|---|---|---|
| `cyber` | 2,120 | 1,011 | 0 |
| `borderline` | **0** | 1,414 | **1,136** |
| `not_cyber` | 7,310 | 1,662 | 0 |

`borderline` has **no high-confidence labels at all**, and **every** low-confidence
label in the dataset is a `borderline` one. The uncertainty is not spread across
the classes — it is entirely concentrated in this boundary.

That is not sloppiness; it is the finding. Deciding whether a document is *about*
security or merely *mentions* it is genuinely uncertain, and any model or metric
built on this boundary inherits that ceiling. Treat 3-way agreement accordingly;
the binary collapse (`cyber`+`borderline` vs `not_cyber`) is the stable decision,
and `confidence` gives you a principled way to filter to the clear cases.

Class balance: `cyber` 3,131 · `borderline` 2,550 · `not_cyber` 8,972 — i.e. the
keyword filter that produced these documents had a lenient precision of 0.388,
which is what motivated everything that follows.

### `judge_benchmark` — reproduce the study, don't take our word

1,992 documents, stratified over 3 labels × 4 keyword-density buckets, each with
the frontier gold label **and** the labels a local open 31B model assigned in two
separate runs (zero-shot and few-shot). `sampling_weight` lets you recover
corpus-level estimates from the stratified sample.

This exists so the central methodological claim below is checkable by anyone with
the file and no GPU at all.

---

## The story

### 1. A keyword filter, and the discovery that it was not enough

We began with a 1,209-term Arabic cybersecurity glossary and scanned 1,000,000
FineWeb2 Arabic records, keeping any document containing a whole-word match.
That produced 13,345 documents across 96 Common Crawl dumps.

Then we measured it instead of trusting it — and **lenient precision was 0.388**.
Roughly three in five kept documents were not about cybersecurity. The failure
mode was consistent and, in hindsight, obvious: the word "الحرب الإلكترونية"
(electronic warfare) in a military history article, "حصان طروادة" (Trojan horse)
as political metaphor, "الباب الخلفي" (back door) meaning an actual door, privacy
boilerplate on shopping sites, and one-clause mentions of cyberattacks inside
diplomacy reporting.

A single keyword is a weak signal. That turned out to be measurable, and then
fixable.

### 2. The gap that only appeared when we looked for misses

Checking recall surfaced something structural: the biggest cause of missed
documents was **English**. `VPN` appeared in 605 missed records, then `Antivirus`,
`Proxy`, `Hacker`, `Firewall`, `Malware`.

Arabic security writing is heavily code-switched, because the vocabulary is coined
in English and enters Arabic **untranslated**. An Arabic-only glossary silently
drops exactly the documents most densely technical. Adding 24 verified English
terms produced the 1,233-term lexicon used here — a small change with an outsized
effect, and one that generalises to any language where technical vocabulary
arrives as loanwords.

### 3. Labelling everything, so keywords could be ranked by evidence

We then judged **all 14,653 documents** — the `census` config — rather than a
sample. That is what makes the next step possible: with a label on every
document, each keyword can be scored by the company it keeps.

A term appearing overwhelmingly in genuine security documents is trustworthy
alone. A term appearing mostly in false positives is only meaningful alongside
others. This produced **four evidence-derived tiers** (640 terms tiered) and a
tiered acceptance rule:

> accept if **≥1 tier-1 hit**, or **≥2 hits of tier ≤2**, or **≥3 of tier ≤3**,
> or **≥4 tiered hits**

It also produced a **deletion list** of 74 terms that never once appeared in a
genuine security document. `تدقيق أمني` ("security audit") appeared 20 times, and
was wrong every time.

The tiered rule lifted lenient precision from **0.388 to 0.636** while keeping
**77%** of the strictly-cyber documents — beating a blind "require N keywords"
threshold on precision *and* recall simultaneously, because it weights *which*
keywords matched rather than merely counting them.

### 4. Scaling 70×, and checking the rule survived

We then applied the same rule to the full FineWeb2 Arabic parquet release —
~70M rows, 112 GB — producing the **370,019-document** corpus published here.

Precision measured at this scale: **0.715 [95% CI 0.679–0.749]**, implying
~264,600 genuinely security-related documents. The rule did not merely survive
the 70× scale-up; it improved on it.

Sampling also revealed that the four-way acceptance rule is, at scale, really
**two** rules: `tier1_hit` (192,583 docs) and `two_tier2_hits` (177,351) carry
99.98% of the corpus, while the other two admit 85 documents between them.

### 5. Judging all 370,019 documents

The result above made the final step affordable: **every** keyword-matched
candidate was judged individually — 370,019 verdicts, 0 unparsed, 163 GPU-hours
across 32 slices on two H200s. 25% were rejected, giving the published corpus.

| | before judging | after judging |
|---|---|---|
| records | 370,019 | **277,418** |
| precision | 0.721 | **0.890** |
| genuine documents | 266,839 | 246,959 (92.5% retained) |

The census also validated the cheap path. A stratified sample of 1,585 documents
— 0.4% of the judging cost — had predicted 0.7152 [0.6793–0.7492]; the full
census came in at **0.7211**. Accurate to 0.6 percentage points. If you need a
corpus-precision number and cannot afford the full run, the sample is enough.

### 6. The methodological result: can an open model replace a frontier judge?

Frontier judging is the expensive part of careful curation, and the cost lands
hardest on the languages that most need curating. So we tested the substitution
directly — a locally-hosted **open 31B model** against the frontier census, on a
blind stratified set, gold labels never leaving the local machine.

**The answer is split, and the split is the useful part.**

| | zero-shot | few-shot (11 exemplars) |
|---|---|---|
| Cohen κ, 3-way | 0.419 | 0.422 |
| **Cohen κ, binary** | 0.651 | **0.654** |
| `cyber` recall | 0.992 | 0.992 |
| `borderline` recall | 0.146 | **0.146** |

For the **binary** accept/reject decision the open model is genuinely usable:
κ 0.654, 85.2% agreement, sensitivity 0.926 — and across 664 truly-cyber
documents it never once returned `not_cyber`.

For **graded relevance** it fails, and **prompting does not fix it**. It cannot
separate "about cybersecurity" from "mentions cybersecurity", calling 468 of 664
`borderline` documents `cyber`. Eleven worked exemplars aimed squarely at that
weakness moved κ by **+0.003** and `borderline` recall by **exactly zero** —
while changing 24.2% of the generated justifications, which proves the exemplars
were applied and simply did not help. This is a capability limit, not a prompting
limit.

Agreement also *falls* as keyword density rises (0.693 at one keyword → 0.526 at
four), because more security vocabulary is precisely the signal the model
over-reads. Its bias is worst on exactly the population a keyword filter produces.

**What makes it useful anyway:** a biased classifier with a *measured* operating
point still yields unbiased population estimates. Rogan–Gladen inverts it, and
reproduces the validation set's true prevalence exactly (0.667 vs 0.667). So the
recipe is a **hybrid** — spend frontier judgments once to build a reference
standard and calibrate the open model, then run the open model at corpus scale
with its bias corrected arithmetically.

That recipe, not the corpus, may be the most portable thing here. It is how a
lab without a frontier budget can still curate carefully in a low-resource
language.

---

## Limitations — please read before using

- **This is web text, not vetted reference material.** It is filtered for topic,
  not for accuracy, quality, or safety. It contains SEO spam, forum posts,
  outdated advice, and marketing copy alongside genuine technical writing.
- **Precision is ~0.890, not 1.0.** Roughly 11% of `corpus` documents are not
  genuinely about cybersecurity. Filter `len(text) < 8000` to reach ~0.943, or
  use `keywords_found` / `tiers_found` to filter on the keyword evidence.
- **Topic filtering is not quality filtering.** The largest single source domain
  is `2-remove-virus.com` (4,700 documents, ~100% kept), alongside
  `tips2-remove.com` and `fixmybrowsers.com`. These are SEO content farms
  producing templated malware-removal pages — genuinely *about* cybersecurity,
  so both the keyword filter and the judge accept them correctly, but low-value
  text. Consider domain-level deduplication or capping per-domain contributions.
  The corpus spans 43,443 distinct domains, so this is a head-of-distribution
  problem rather than a pervasive one.
- **Do not use `judge_label` confidence as a signal.** The open judge returned
  `high` confidence on 99.99% of the 370,019 verdicts. It is miscalibrated about
  its own certainty, and the field is not published for the corpus for that
  reason. (The `census` config's `confidence`, which comes from the frontier
  model, *is* meaningful.)
- **`borderline` documents are included.** The published corpus is
  `cyber` + `borderline` — documents the judge considered security-related at
  all. It is not a corpus of exclusively on-topic documents, because the open
  judge cannot reliably grade centrality (see the story above).
- **Recall is unmeasured at v4 scale.** We know the filter finds a lot; we do not
  know what fraction of all Arabic security web text it finds. The v1 recall
  study is in the source repository, but it does not transfer to this scale.
- **`borderline` is intrinsically fuzzy** — see the `census` note above.
- **Judge labels come from an LLM**, not human annotators. The rubric is
  published and the judging is reproducible, but no human gold standard exists.
  Frontier labels are treated as the reference standard here, which is a
  practical choice, not a claim that they are ground truth.
- **Dialect and register coverage are uncharacterised.** The lexicon is Modern
  Standard Arabic plus English loanwords; dialectal security discussion is
  probably under-represented.
- **Bias inherited from Common Crawl.** Source-site popularity, geography, and
  period all skew toward what the crawl captured. `dump` and `url` are included
  so you can characterise this yourself.

## Personal and sensitive information

This is web-crawled text and **may contain personal information** — names,
contact details, or other data published on the open web. It inherits FineWeb2's
processing and adds no further PII removal. Do not assume it is anonymised.

**Takedown:** if you own content here and want it removed, open a discussion on
this dataset's Hub page with the URL(s), and we will remove the matching records
and republish. `url` is preserved on every record specifically to make this
possible.

## Licensing and attribution

Released under **ODC-By 1.0**, inherited from FineWeb2. You may share and adapt
this dataset, including commercially, **provided you attribute** it. Use is also
subject to the [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use).

This dataset is derived from:

- **FineWeb2** ([HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)),
  ODC-By 1.0 — the source Arabic web text
- **Common Crawl** — the underlying web crawl

## Citation

```bibtex
@misc{alsumaitti_arabic_cybersecurity_web,
  title  = {Arabic Cybersecurity Web Corpus: a filtered corpus with a
            full-census LLM-judged reference standard},
  author = {Alsumaitti, Osamah},
  year   = {2026},
  url    = {https://huggingface.co/datasets/<YOUR-USERNAME>/arabic-cybersecurity-web},
  note   = {Derived from FineWeb2 (ODC-By 1.0)}
}
```

## Source code

Filters, judging protocol, evaluation scripts, and every number quoted above:
<https://github.com/Alsumaitti/arabic-cyber-filter>

Companion lexicon repository:
<https://github.com/Alsumaitti/arabic-cyber-keywords>
