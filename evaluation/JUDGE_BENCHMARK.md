# Can an open 31B model replace a frontier model for corpus curation in Arabic?

**Answer: for the binary decision, yes. For graded relevance, no — and prompting does not fix it.**

This document reports a controlled head-to-head between a frontier judge and a
locally-hosted open model on the exact task that gates this corpus: deciding
whether an Arabic web document is about cybersecurity.

It matters beyond this project. Frontier judging is what makes careful corpus
curation expensive, and that cost falls hardest on low-resource languages, where
the corpora most need curating and the budgets are smallest. If an open model
running on local hardware can do the job, the method travels. So we measured it
rather than assuming either way.

---

## 1. Setup

| | |
|---|---|
| **Reference standard** | frontier judge, full census of the v2 corpus — 14,653 records, [`JUDGE_PROTOCOL.md`](JUDGE_PROTOCOL.md) rubric |
| **Model under test** | `gemma-4-31B-it`, bf16, served locally |
| **Hardware** | 2 × NVIDIA H200 (141 GB), one model replica per GPU, 2 independent shards |
| **Evaluation set** | 1,992 records, stratified 166 per cell over 12 cells (3 labels × 4 keyword-hit buckets) |
| **Blinding** | gold labels never left the local machine; the cluster received text only |
| **Leakage guard** | few-shot exemplars drawn from the census *excluding* all 1,992 evaluation ids |
| **Decoding** | greedy, identical settings in both runs |

Because the evaluation set is stratified, raw agreement over-represents rare
strata. Every number below is reported both raw and **corpus-weighted** by
inverse sampling probability — the weighted figure is the one that describes the
corpus.

### The two conditions

1. **Zero-shot** — the rubric as a system prompt. The rubric *describes*
   `borderline`.
2. **Few-shot** — the same rubric plus **11 worked exemplars** (5 `borderline`,
   3 `cyber`, 3 `not_cyber`) injected as prior chat turns, each with its gold
   label and the frontier model's justification. The rubric now *shows*
   `borderline`.

Condition 2 was designed after seeing condition 1's failure mode. It targets the
weak class deliberately: nearly half the exemplars are `borderline`.

---

## 2. Result

```
                              zero-shot   few-shot     delta
raw agreement (3-way)             0.613      0.614    +0.002
corpus-weighted                   0.749      0.752    +0.004
Cohen kappa (3-way)               0.419      0.422    +0.002
binary agreement                  0.851      0.852    +0.001
Cohen kappa (binary)              0.651      0.654    +0.003
cyber      precision / recall  0.526/0.992  0.526/0.992  +0/+0
borderline precision / recall  0.554/0.146  0.564/0.146  +0.010/+0
not_cyber  precision / recall  0.826/0.700  0.825/0.705  -0.001/+0.005
```

Eleven targeted demonstrations moved 3-way κ by **+0.003** and `borderline`
recall by **exactly zero**.

### The exemplars were genuinely applied

A null this flat invites the boring explanation — that the few-shot prompt was
silently dropped and we ran the same thing twice. It was not:

| | |
|---|---|
| generated justifications that changed | **483 / 1,992 (24.2%)** |
| labels that changed | **5 / 1,992 (0.3%)** |

The model demonstrably saw different input and wrote different reasoning. It
then produced almost exactly the same labels. Reproduce with:

```bash
python evaluation/scripts/compare_judge_runs.py --md
```

All five flips: `cyber→not_cyber` ×2, `borderline→not_cyber` ×2,
`borderline→cyber` ×1.

**This is a capability limit, not a prompting limit.** That is a stronger and
more useful claim than "few-shot fixed it", because the obvious remedy was
tried, aimed squarely at the weak class, and failed.

---

## 3. Where it fails, precisely

Confusion matrix, few-shot run (rows = frontier gold, columns = gemma):

| | cyber | borderline | not_cyber |
|---|---|---|---|
| **cyber** | **659** | 5 | 0 |
| **borderline** | 468 | **97** | 99 |
| **not_cyber** | 126 | 70 | **468** |

Predicted class distribution against a gold set that is 664 / 664 / 664 by
construction:

| | cyber | borderline | not_cyber |
|---|---|---|---|
| gold | 664 | 664 | 664 |
| zero-shot | 1,254 | 175 | 563 |
| few-shot | 1,253 | 172 | 567 |

Gemma assigns `cyber` to **63%** of records where the truth is **33%**. It
essentially does not use the middle category: 172 predictions where 664 are
warranted.

The failure is one-directional and interpretable:

- **It detects security content almost perfectly.** `cyber` recall **0.992** —
  it missed 5 of 664, and never once called a `cyber` document `not_cyber`. The
  zero in that cell is the striking one: across 664 documents there is not a
  single catastrophic miss.
- **It cannot judge centrality.** Of 664 `borderline` documents — security
  vocabulary present, but the article is about something else — it called
  **468 of them `cyber`**.

So the model is not confused about the *domain*. It is uncalibrated about
*degree of topicality*: presence of the subject reads to it as aboutness.

### Agreement falls as keyword density rises

| keyword hits | n | zero-shot | few-shot |
|---|---|---|---|
| 1 | 498 | 0.691 | 0.693 |
| 2 | 498 | 0.643 | 0.645 |
| 3 | 498 | 0.594 | 0.594 |
| 4+ | 498 | 0.524 | 0.526 |

Backwards from the naive expectation, and it confirms the mechanism. More
lexicon hits means more security vocabulary on the page — which is exactly the
signal gemma over-reads. Its bias is *worst* where the lexicon fires hardest,
which is precisely the population a keyword filter hands to a judge. The
disagreement is structural, not noise.

### One honest caveat about the ceiling

`borderline` is the intrinsically uncertain class, and the frontier census
contains **zero high-confidence `borderline` labels** — every one is medium or
low. Some of the 3-way gap is irreducible disagreement about a genuinely fuzzy
boundary rather than gemma being wrong. This partly excuses the collapse and
caps the achievable 3-way κ for *any* judge. It does not rescue a `borderline`
recall of 0.146.

---

## 4. What this licenses

### Use it for the binary decision

Collapsing to *is this security content at all* (`cyber` + `borderline` vs
`not_cyber`) — which is the decision the tiered filter actually makes:

| | |
|---|---|
| agreement | **0.852** [95% CI 0.836–0.867] |
| Cohen κ | **0.654** — substantial |
| cyber-ish P / R / F1 | 0.862 / 0.925 / 0.893 |
| sensitivity | 0.9255 |
| specificity | 0.7048 |
| stability under prompt change | κ moved 0.003 |

κ = 0.65 with 0.93 sensitivity is a usable instrument, and its
insensitivity to prompting is a feature here: the operating point is stable, so
it can be characterised once and trusted across runs.

### Correct the bias rather than fight it

A biased-but-*measured* classifier still yields unbiased population estimates.
With sensitivity and specificity known, Rogan–Gladen inverts the bias:

$$p_{\text{true}} = \frac{p_{\text{obs}} - (1 - \text{spec})}{\text{sens} - (1 - \text{spec})} = \frac{p_{\text{obs}} - 0.2952}{0.6303}$$

Checked against the evaluation set itself: observed cyber-ish rate 0.7154 →
corrected **0.667**, versus true gold **0.667**. Exact.

This is what makes the open model genuinely useful. We cannot trust its
per-document `borderline` call, but we *can* trust what it says about the
composition of 370,019 documents — which is the question a corpus builder
actually has.

### Do not use it for

- **Per-document graded relevance.** 0.146 recall on the middle class.
- **Rebuilding the keyword tiers.** Tier assignment depends on the good/bad
  ratio per keyword, and a judge that inflates `cyber` by 1.9× would inflate
  every keyword's apparent quality — systematically, and worst for the
  high-frequency keywords that matter most (see the density table above).
  The frontier census stays the reference standard for anything tier-related.

---

## 5. Reproducing it

```bash
# 1. build the blind evaluation set (gold stays local)
python evaluation/scripts/make_judge_validation_set.py

# 2. build leakage-free exemplars for the few-shot arm
python evaluation/scripts/make_fewshot_examples.py

# 3. upload prompts + exemplars, never the gold
scp evaluation/judge_benchmark/validation_prompts.jsonl \
    evaluation/judge_benchmark/fewshot_examples.json \
    oalsumaitti@panther-login.qcri.org:~/fanar-data/judge_bench/

# 4. on the cluster - one-time check, then both arms
sbatch sbatch_scripts/setup_gemma_judge.sh
sbatch --export=ALL,ENV=gjudge            sbatch_scripts/judge_gemma_validate.sh
sbatch --export=ALL,ENV=gjudge,FEWSHOT=1  sbatch_scripts/judge_gemma_validate.sh

# 5. bring predictions back and score
python evaluation/scripts/score_judge_agreement.py \
    --pred "evaluation/judge_benchmark/gemma_val_P2_p*.jsonl"
python evaluation/scripts/score_judge_agreement.py \
    --pred "evaluation/judge_benchmark/gemma_fs_P2_p*.jsonl" \
    --out evaluation/judge_benchmark/agreement_report_fewshot.json
python evaluation/scripts/compare_judge_runs.py --md
```

Both arms parsed **1,992 / 1,992** verdicts with **0 unparsed** — the structured
output was completely reliable in both conditions, so no result here is
contaminated by parse failures.

### Engineering notes worth keeping

Two traps cost real time and are worth recording, because both fail *silently*:

- **`torch.cuda.is_available()` returning `True` does not mean the GPU works.**
  A run generated on CPU at 0.003 rec/s — 5 records in 28 minutes — while
  reporting CUDA available. The tell was a 31B model claiming to load in 14
  seconds. The judge script now proves the device with a real matmul and aborts
  if `hf_device_map` places any layer on `cpu` or `disk`.
- **`~/.local/lib/python3.10/site-packages` is shared by every python3.10 env
  on the account.** One `pip install` there silently rewires unrelated
  environments. Every job now exports `PYTHONNOUSERSITE=1` and installs with
  `--no-user`.

vLLM was abandoned deliberately: its current wheels require CUDA 13 against a
12.8 driver, and installing it drags in a torch build that breaks the working
environment. The transformers backend with batched greedy generation is the
supported path here.

---

## 6. Bottom line

An open 31B model **can** replace a frontier judge for the accept/reject
decision in Arabic corpus curation — κ 0.65, sensitivity 0.93, stable, and
free at the margin once the hardware exists.

It **cannot** replace one for graded topical relevance, and the deficit is a
property of the model rather than of the prompt: eleven worked examples aimed
directly at the failure moved it by 0.003.

The practical recipe that follows is a hybrid, and it is cheap where it needs to
be: spend frontier judgments once to build the reference standard and calibrate
the open model, then let the open model run at corpus scale with its bias
corrected arithmetically.

**Related:** [`RESULTS.md`](RESULTS.md) · [`JUDGE_PROTOCOL.md`](JUDGE_PROTOCOL.md) · [`chat_judging/CAMPAIGN.md`](chat_judging/CAMPAIGN.md)
