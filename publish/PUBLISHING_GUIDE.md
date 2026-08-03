# Publishing to the Hugging Face Hub

Everything here is prepared so **you** run the upload. Publishing is public and
hard to walk back, and it needs your credentials — so the build is automated and
the push is deliberately manual.

---

## Status: built and ready

`publish/hf_build/` already contains the v5 release, validated:

| config | rows | size |
|---|---|---|
| `corpus` | 277,418 (v5, judged, precision 0.890) | 560 MB, 6 shards |
| `census` | 14,653 frontier-labelled | 52 MB |
| `judge_benchmark` | 1,992 gold + two open-model runs | 1.2 MB |
| `keywords/` | lexicon + 4 tiers | 72 KB |

The card is in place as `README.md` and names
`Alsumaitti/arabic-cybersecurity-web`. **Skip to Step 3** unless you are
rebuilding.

---

## Step 1 — build the parquet (already done)

```bash
python publish/build_hf_dataset.py --out-dir publish/hf_build --corpus result/cyber_arabic_v5_judged.jsonl --corpus-name v5
```

The v5 corpus is ~4 GB of JSONL; it compresses to ~560 MB of zstd parquet in
shards near 400 MB. To iterate on the small configs only:

```bash
python publish/build_hf_dataset.py --out-dir publish/hf_build --skip-corpus
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

The card already names `Alsumaitti/arabic-cybersecurity-web` in both places (the
load example and the citation `url`), so this needs no edit unless you are
changing the repo name.

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
python -c "from datasets import load_dataset; d=load_dataset('Alsumaitti/arabic-cybersecurity-web','census',split='train'); print(d); print(d[0]['label'], d[0]['justification'])"
```

Check on the Hub page that: the viewer renders all three configs, Arabic text
displays right-to-left correctly, the license shows **odc-by**, and the config
dropdown lists `corpus`, `census`, `judge_benchmark`.

---

## Pre-publication checklist

Worth ten minutes, because these are the things that are awkward to fix after
people start downloading.

- [x] Repo name set to `Alsumaitti/arabic-cybersecurity-web` in the card
- [ ] **The write token used for `hf auth login` is a fresh one** — never one that
      has been pasted into a chat, a file, or a command argument
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
