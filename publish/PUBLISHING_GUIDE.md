# Publishing to the Hugging Face Hub

Everything here is prepared so **you** run the upload. Publishing is public and
hard to walk back, and it needs your credentials — so the build is automated and
the push is deliberately manual.

---

## Before anything: two decisions

### 1. Publish the v4 corpus now, or wait for v5?

| | v4 (ready today) | v5 (needs ~25–30 GPU-hours) |
|---|---|---|
| documents | 370,019 | ~276,000 |
| precision | 0.715 | ~0.887 |
| genuine documents | ~264,600 | ~244,900 |

**Recommendation: publish v4 now, add v5 as a second config later.** The Hub is
versioned, `census` and `judge_benchmark` are finished and independently
valuable, and a dataset that exists gets used while a better one is still
queued. Adding a `corpus_v5` config later is a normal update, not a re-release —
and keeping both lets users choose their own precision/recall trade-off, which
is exactly the choice the numbers above frame.

If you would rather ship once, finish v5 first (task #15) and publish both
together.

### 2. Pick the repo name

The card uses `<YOUR-USERNAME>/arabic-cybersecurity-web` as a placeholder in
three places. Decide the name now and replace it everywhere — a dataset that
cites the wrong URL in its own citation block looks careless.

---

## Step 1 — build the parquet

```bash
python publish/build_hf_dataset.py --out-dir publish/hf_build
```

The corpus is ~5 GB of JSONL and takes a while; it compresses to roughly 1–2 GB
of zstd parquet in shards near 400 MB. To iterate on the small configs first:

```bash
python publish/build_hf_dataset.py --out-dir publish/hf_build --skip-corpus
```

To publish v5 instead once it exists:

```bash
python publish/build_hf_dataset.py --out-dir publish/hf_build --corpus result/cyber_arabic_v5_judged.jsonl --corpus-name v5
```

Expected layout:

```
publish/hf_build/
├── corpus/train-00000.parquet …
├── census/train-00000.parquet
├── judge_benchmark/test-00000.parquet
└── keywords/{all_cyber_keywords,tier1..4_keywords}.txt
```

## Step 2 — add the card

```bash
cp publish/DATASET_CARD.md publish/hf_build/README.md
```

Then edit `publish/hf_build/README.md` and replace every `<YOUR-USERNAME>`
(three occurrences: the load example, the citation URL, the citation `url`
field).

## Step 3 — install the client and log in

```bash
pip install huggingface_hub
```

```bash
hf auth login
```

You will need a **write** token from <https://huggingface.co/settings/tokens>.
Enter it in the prompt — do not paste it into a file, a command argument, or any
script in this repo.

## Step 4 — create the dataset repo

```bash
hf repo create arabic-cybersecurity-web --repo-type dataset
```

Consider creating it **private** first (`--private`), inspecting how the Hub
renders the card and the data preview, then flipping to public in the settings
once it looks right. The dataset viewer sometimes surfaces schema problems that
are invisible locally.

## Step 5 — upload

```bash
hf upload arabic-cybersecurity-web publish/hf_build . --repo-type dataset
```

Large uploads resume if interrupted. For a multi-GB corpus, expect this to take a
while and prefer a wired connection.

## Step 6 — verify it actually loads

```bash
python -c "from datasets import load_dataset; d=load_dataset('<YOUR-USERNAME>/arabic-cybersecurity-web','census',split='train'); print(d); print(d[0]['label'], d[0]['justification'])"
```

Check on the Hub page that: the viewer renders all three configs, Arabic text
displays right-to-left correctly, the license shows **odc-by**, and the config
dropdown lists `corpus`, `census`, `judge_benchmark`.

---

## Pre-publication checklist

Worth ten minutes, because these are the things that are awkward to fix after
people start downloading.

- [ ] `<YOUR-USERNAME>` replaced in all three places in `README.md`
- [ ] License is **odc-by**, and FineWeb2 + Common Crawl are both credited — this
      is a *requirement* of the source license, not a courtesy
- [ ] The takedown paragraph names a channel you will actually monitor
- [ ] `validation_gold.jsonl` labels are published deliberately. They are *in*
      `judge_benchmark` on purpose — the study is meant to be reproducible — but
      that means the blind evaluation set is now public. Any future open-model
      benchmark using this data must draw a fresh sample, or it is contaminated.
- [ ] Spot-check ~20 random `corpus` rows for anything you would not want to
      republish under your own name
- [ ] Numbers in the card match `evaluation/RESULTS.md`

## After publishing

- Link the dataset from both GitHub repos' READMEs
- If you write this up, the `judge_benchmark` config is the artifact reviewers
  will want — it is what makes the open-vs-frontier claim checkable
- Watch the Hub discussions tab for takedown requests

## A note on what you are publishing

The corpus is the bulk, but the `census` is the contribution. A filtered corpus
is one lab's judgment call; **14,653 individually labelled Arabic documents with
a published rubric** is infrastructure other people can build on — to train a
classifier, to evaluate a different filter, or to disagree with your labels in a
measurable way.

Frame it that way in the description, and it will be found by people who need it.
