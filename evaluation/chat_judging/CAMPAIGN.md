# The in-chat judging campaign — how the whole corpus was judged

This is the operational record of **round 2** of the evaluation: judging **every**
record of the v2 corpus with an LLM, in-chat, at zero marginal API cost. It is the
"Option B without the API" path from [`../../PROPOSAL.md`](../../PROPOSAL.md).

## Why in-chat instead of the Batches API

The proposal budgeted **USD 85–150** to judge the full corpus through the Claude
**Message Batches API** (`scripts/llm_judge_api.py`). The same census was instead
produced **in a chat assistant session** — the assistant reads each record and
writes the verdict directly — for **zero marginal cost** (it is covered by the
Claude subscription rather than metered per token). The trade is **money for
wall-clock time**: the API would finish in ≤ 24 h hands-off; the in-chat route
spans several days of interactive sessions. For a one-off research asset with no
deadline, spending time instead of budget was the right call.

## How the chunks were made

[`../scripts/prepare_chat_judging.py`](../scripts/prepare_chat_judging.py) splits
the corpus into small, token-efficient units the assistant can judge inside one
turn, resumably:

| Parameter | Value |
|---|---|
| Input | `result/cyber_arabic_v2_ALL.json` — the v2 corpus, **14,653 records** |
| Chunk size | **25 records** per chunk |
| Number of chunks | **587** (chunk 587 holds the final 3 records) |
| Record text | truncated to **1,300 chars** (enough to judge topic; keeps tokens down) |
| Line format | one compact JSON record per line: `{"id","nk","kws","text"}` |
| Mode | **full** — no per-keyword caps, so *every* record is judged |

Chunks are **ordered by information value** so that if judging stopped early the
evidence would still be balanced, not clumped:

1. **Bucket 1 — single-keyword records (9,807)** come first. These are the *only*
   source of "solo" evidence — the entire tier system hinges on whether a keyword,
   *alone*, lands in a cyber record — so they are judged first.
2. Bucket 2 — two-keyword records (2,352), then bucket 3 — three (1,040), then
   bucket 4 — 4+ keywords (1,454), which are near-certain cyber and least informative.
3. Within each bucket, records are taken **round-robin over keywords** (every
   keyword's 1st record, then every keyword's 2nd, …) so per-keyword coverage grows
   evenly.

Progress is tracked by a **manifest** plus the presence of judgment files, so a
session can stop anywhere and the next one resumes at the correct chunk.

## The model

The judging was done by a **frontier Claude chat model** (the assistant reading
each document), not by a small/cheap classifier. The per-chunk commits credit
`Claude Fable 5` as co-author (the campaign convention); the final wrap-up and
merge session ran on **Claude Opus 4.8**.

## The per-chunk loop (what "judging one chunk" means)

For each of the 587 chunks the assistant ran a fixed, resumable cycle:

1. **Read** `chunks/chunk_NNNN.jsonl` (25 records).
2. **Judge** all 25 — for each: a **label** (`cyber` / `borderline` / `not_cyber`),
   a **confidence** (high / medium / low), and a one-sentence **English "why"**
   citing the evidence.
3. **Write** `judgments/chunk_NNNN.json`.
4. **Validate** (`merge_chat_judgments.py --validate` — checks every id present,
   labels legal, counts right).
5. **Commit + push** — so the chunk is durably saved before moving on.
6. **`--status`** reports the next chunk to do.

## Why this design — the ~5-hour usage limit

The Claude subscription enforces a **rolling ~5-hour usage window**. A single
long run judging 14,653 records would hit that window and lose whatever was
in-flight. The per-chunk **checkpoint** (validate → commit → push after *every*
one of the 587 chunks) is the answer: hitting the limit costs **at most one
in-progress chunk**, and the next session resumes at the exact next chunk with
**zero rework**, just by typing "continue". That resumability is what made a
full-corpus census feasible *under* the limit rather than in spite of it.

## Timeline (from git commit timestamps)

| | |
|---|---|
| First "Judge chunk" commit | **2026-07-14 10:18** |
| Last "Judge chunk" commit | **2026-07-19 11:04** |
| Calendar span | **5.0 days** (with **2026-07-16 taken off** — zero commits) |
| Chunks committed / commits | 587 chunks across **~500** judge-chunk commits (some commits carried >1 chunk) |
| Commits per active day | 53 (07-14) · 132 (07-15) · 125 (07-17) · 148 (07-18) · 42 (07-19) |

**Sessions and limit hits.** Consecutive-commit gaps imply **~20 sessions**
(a "session" = a run of chunks with no >30-min break). The gaps *between*
sessions cluster around **4–4.5 hours** (255–555 min) — consistent with the
rolling ~5-hour window resetting between sittings — plus one **~30-hour** gap that
is the deliberate day off (07-16). The number of *hard* limit hits isn't logged
directly, but those ~20 session boundaries with their recurring ~4-hour cadence
are the footprint of repeatedly working up to the window and resuming after it
reset — on the order of **~15–19 limit-driven resumes** (the rest being the
planned break).

**Active (hands-on) judging time.** Summing only the in-session gaps (≤ 30 min
between consecutive chunk commits) gives **≈ 14.3 hours** of actual judging work.
Median cadence was **~1.4 minutes per chunk** (25 records) — roughly **3.3 seconds
per record**, end to end including reasoning, writing, validating, and committing.

## Token / cost accounting

In-chat sessions do **not** expose a clean per-task token meter the way the
Batches API would, so exact billed tokens for this task are not recoverable. What
*is* measurable is the work content:

| Quantity | Value | Notes |
|---|---|---|
| Records judged | 14,653 | 100% of the v2 corpus |
| Unique input text shown to the judge | **18.1 M chars** (~34 MB) | record text, truncated to 1,300 chars each |
| ≈ input tokens (content) | **~8.2 M** | Arabic ≈ 2.2 chars/token |
| Output justifications written | **2.79 M chars** | the one-sentence "why" per record |
| ≈ output tokens (content) | **~0.7 M** | English ≈ 4 chars/token |

The **billed** total is larger than the ~8.2 M "content" input, because every turn
also re-sends the conversation scaffolding (system prompt, tool schemas, prior
turns) and the model spends reasoning tokens per chunk — an order-of-magnitude
**tens of millions of tokens** across the ~500+ turns, heavily discounted by
prompt caching. The economic headline is the comparison, not the absolute count:

- **Batches API path:** exact metering, ~**USD 85–120**, ≤ 24 h, hands-off.
- **In-chat path (used):** **USD 0 marginal**, ~14 h active work over ~20 sessions
  / 5 days, and it produced the identical deliverable — a fully judged 14,653-record
  corpus ([`judged_corpus.json`](judged_corpus.json)).

## Outputs

- [`chunks/`](chunks) — 587 input chunk files (gitignored).
- [`judgments/`](judgments) — 587 per-chunk verdict files (~3.5 MB, tracked).
- [`judged_corpus.json`](judged_corpus.json) — the merged census (14,653 records +
  4 judge columns; ~202 MB, gitignored) via `merge_chat_judgments.py --merge`.
- Downstream: rebuilt tiers ([`../../keywords/tiers/`](../../keywords/tiers)),
  the measured second-generation filter, and the deletion list — see
  [`../RESULTS.md`](../RESULTS.md).
