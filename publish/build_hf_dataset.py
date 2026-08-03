#!/usr/bin/env python3
"""
build_hf_dataset.py - package this project into a Hugging Face dataset.

FOUR CONFIGS
  cyber_all       (default) every document judged cybersecurity-related, from BOTH
                  judges, merged and deduplicated. This is what most people want:
                  no not_cyber, no need to know which corpus version made it.
  corpus          the v5 corpus alone - uniform provenance, one judge.
  census          14,653 documents, EVERY one labelled by a frontier model against
                  a written rubric, with a justification per record. This is a
                  labelled Arabic domain-relevance benchmark - as far as we know
                  the first of its kind for cybersecurity - and it is reusable far
                  beyond this project: train a classifier, evaluate your own
                  filter, or study Arabic security discourse.
  judge_benchmark 1,992 documents carrying BOTH the frontier gold label and two
                  open-model runs (zero-shot and few-shot), so the
                  "can an open 31B model replace a frontier judge?" result is
                  independently reproducible rather than merely reported.

Parquet, sharded near 400 MB, because the Hub serves and previews it efficiently
and it is what `datasets` expects.

Usage:
    python publish/build_hf_dataset.py --out-dir publish/hf_build
    python publish/build_hf_dataset.py --out-dir publish/hf_build --skip-corpus
    python publish/build_hf_dataset.py --out-dir publish/hf_build \
        --corpus result/cyber_arabic_v5_judged.jsonl --corpus-name v5
"""
import argparse, glob, hashlib, json, os, sys

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("error: pyarrow required.  pip install pyarrow")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SHARD_BYTES = 400 * 1024 * 1024
ROWS_PER_CHECK = 2000


def rid_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class ShardWriter:
    """Streams rows into parquet shards without holding the dataset in RAM.

    Shard size is tracked from a running estimate of the raw text bytes rather
    than by materialising the buffer to check. Building a pyarrow Table just to
    read .nbytes, every N rows, is quadratic in the shard size - on a 370k-row
    corpus that dominates the entire run.
    """

    def __init__(self, out_dir, split, schema):
        self.out_dir, self.split, self.schema = out_dir, split, schema
        os.makedirs(out_dir, exist_ok=True)
        self.buf, self.idx, self.rows, self.files = [], 0, 0, []
        self.est_bytes = 0

    def add(self, row):
        self.buf.append(row)
        self.rows += 1
        # Cheap size proxy: the text field dominates every schema here.
        t = row.get("text") or ""
        self.est_bytes += len(t.encode("utf-8", "ignore")) + 256
        if self.est_bytes >= TARGET_SHARD_BYTES:
            self._write(self._table())
            self.buf, self.est_bytes = [], 0

    def _table(self):
        cols = {f.name: [r.get(f.name) for r in self.buf] for f in self.schema}
        return pa.Table.from_pydict(cols, schema=self.schema)

    def _write(self, table):
        name = f"{self.split}-{self.idx:05d}.parquet"
        path = os.path.join(self.out_dir, name)
        pq.write_table(table, path, compression="zstd", compression_level=6)
        size = os.path.getsize(path)
        print(f"    {name}  {table.num_rows:,} rows  {size/1e6:.0f} MB", flush=True)
        self.files.append(name)
        self.idx += 1

    def close(self):
        if self.buf:
            self._write(self._table())
            self.buf = []
        return self.rows, self.files


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
CORPUS_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("url", pa.string()),
    ("date", pa.string()),
    ("dump", pa.string()),
    ("keywords_found", pa.list_(pa.string())),
    ("tiers_found", pa.list_(pa.int32())),
    ("accept_rule", pa.string()),
    ("judge_label", pa.string()),          # present only for judged corpora (v5)
])

CENSUS_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("url", pa.string()),
    ("date", pa.string()),
    ("dump", pa.string()),
    ("keywords_found", pa.list_(pa.string())),
    ("n_keywords", pa.int32()),
    ("tiers_found", pa.list_(pa.int32())),
    ("accept_rule", pa.string()),
    ("in_v3", pa.bool_()),
    ("label", pa.string()),
    ("confidence", pa.string()),
    ("justification", pa.string()),
])


def load_tiers(root):
    """keyword -> tier, from the published tier files."""
    t = {}
    for tier in (1, 2, 3, 4):
        p = os.path.join(root, "keywords", "tiers_clean", f"tier{tier}_keywords.txt")
        if not os.path.isfile(p):
            return {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                k = line.strip()
                if k and (k not in t or tier < t[k]):
                    t[k] = tier
    return t


def accept_rule(tiers):
    """The v3 tiered acceptance rule - identical to the filter's accept()."""
    if any(x == 1 for x in tiers):
        return "tier1_hit"
    if sum(1 for x in tiers if x <= 2) >= 2:
        return "two_tier2_hits"
    if sum(1 for x in tiers if x <= 3) >= 3:
        return "three_tier3_hits"
    if len(tiers) >= 4:
        return "four_plus_hits"
    return None

CYBER_ALL_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("url", pa.string()),
    ("date", pa.string()),
    ("dump", pa.string()),
    ("keywords_found", pa.list_(pa.string())),
    ("tiers_found", pa.list_(pa.int32())),
    ("accept_rule", pa.string()),
    ("label", pa.string()),              # cyber | borderline  (best available)
    ("label_source", pa.string()),       # frontier | open_31b
    ("confidence", pa.string()),         # frontier only - the open judge's is useless
    ("justification", pa.string()),
    ("frontier_label", pa.string()),     # both are kept where both exist, so the
    ("open_model_label", pa.string()),   # 3,619-record agreement set is usable
])

BENCH_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("n_keywords", pa.int32()),
    ("stratum", pa.string()),
    ("sampling_weight", pa.float64()),
    ("gold_label", pa.string()),
    ("gold_confidence", pa.string()),
    ("open_model_zeroshot", pa.string()),
    ("open_model_fewshot", pa.string()),
])


def build_corpus(path, out_dir, name):
    print(f"\n[corpus:{name}] {path}")
    if not os.path.isfile(path):
        print(f"  !! not found - skipping. Point --corpus at the merged .jsonl.")
        return None
    w = ShardWriter(os.path.join(out_dir, "corpus"), "train", CORPUS_SCHEMA)
    seen, dupes, n = set(), 0, 0
    for r in load_jsonl(path):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        rid = rid_of(text)
        if rid in seen:
            dupes += 1
            continue
        seen.add(rid)
        tiers = [t for t in (r.get("tiers_found") or []) if isinstance(t, int)]
        w.add({"id": rid, "text": text, "url": r.get("url"), "date": r.get("date"),
               "dump": r.get("dump"),
               "keywords_found": r.get("keywords_found") or [],
               "tiers_found": tiers, "accept_rule": r.get("accept_rule"),
               "judge_label": r.get("judge_label")})
        n += 1
        if n % 50000 == 0:
            print(f"  {n:,} rows...", flush=True)
    rows, files = w.close()
    print(f"  {rows:,} rows in {len(files)} shard(s); {dupes:,} duplicate texts dropped")
    return rows


def build_census(path, out_dir):
    print(f"\n[census] {path}")
    if not os.path.isfile(path):
        print(f"  !! not found - skipping.")
        return None
    recs = json.load(open(path, encoding="utf-8"))
    # Recompute the tier evidence so v3 - the tier-filtered corpus - is one
    # boolean away instead of requiring the reader to reimplement the rule.
    tier_of = load_tiers(ROOT)
    lower = {k.lower(): v for k, v in tier_of.items()}
    if not tier_of:
        print("  !! tier files not found - tiers_found/accept_rule/in_v3 will be null")

    w = ShardWriter(os.path.join(out_dir, "census"), "train", CENSUS_SCHEMA)
    n_v3 = 0
    for r in recs:
        text = (r.get("text") or "").strip()
        kws = r.get("keywords_found") or []
        tiers = []
        for k in kws:
            t = tier_of.get(k)
            if t is None:
                t = lower.get(k.lower())
            if t is not None:
                tiers.append(t)
        rule = accept_rule(tiers) if tier_of else None
        if rule:
            n_v3 += 1
        w.add({"id": rid_of(text), "text": text, "url": r.get("url"),
               "date": r.get("date"), "dump": r.get("dump"),
               "keywords_found": kws, "n_keywords": len(kws),
               "tiers_found": tiers, "accept_rule": rule,
               "in_v3": bool(rule) if tier_of else None,
               "label": r.get("judge_label"),
               "confidence": r.get("judge_confidence"),
               "justification": r.get("judge_justification")})
    rows, files = w.close()
    print(f"  {rows:,} rows in {len(files)} shard(s)")
    print(f"  in_v3 = True for {n_v3:,} rows  (v3 corpus was 6,041)")
    return rows


def build_bench(bench_dir, out_dir):
    print(f"\n[judge_benchmark] {bench_dir}")
    need = ["validation_prompts.jsonl", "validation_gold.jsonl"]
    for f in need:
        if not os.path.isfile(os.path.join(bench_dir, f)):
            print(f"  !! missing {f} - skipping.")
            return None
    prompts = {r["id"]: r for r in load_jsonl(os.path.join(bench_dir, "validation_prompts.jsonl"))}
    gold = {r["id"]: r for r in load_jsonl(os.path.join(bench_dir, "validation_gold.jsonl"))}

    def preds(pattern):
        out = {}
        for p in sorted(glob.glob(os.path.join(bench_dir, pattern))):
            for r in load_jsonl(p):
                out[r["id"]] = r.get("judge_label")
        return out

    zs, fs = preds("gemma_val_P2_p*.jsonl"), preds("gemma_fs_P2_p*.jsonl")
    print(f"  gold {len(gold):,} | zero-shot {len(zs):,} | few-shot {len(fs):,}")

    w = ShardWriter(os.path.join(out_dir, "judge_benchmark"), "test", BENCH_SCHEMA)
    for rid, g in gold.items():
        p = prompts.get(rid)
        if p is None:
            continue
        w.add({"id": rid, "text": p.get("text"), "n_keywords": g.get("nk"),
               "stratum": g.get("stratum"), "sampling_weight": g.get("weight"),
               "gold_label": g.get("gold_label"),
               "gold_confidence": g.get("gold_confidence"),
               "open_model_zeroshot": zs.get(rid),
               "open_model_fewshot": fs.get(rid)})
    rows, files = w.close()
    print(f"  {rows:,} rows in {len(files)} shard(s)")
    return rows


def build_cyber_all(out_dir):
    """Everything ever judged cybersecurity-related, from BOTH judges, in one place.

    This is the config most people actually want: `cyber` + `borderline` merged,
    no `not_cyber`, no need to know which corpus version produced a record. It
    unions the v5 corpus (open-model judged) with the cyber-ish part of the
    frontier census, which contributes 2,062 documents that v5 does not have -
    they predate the parquet crawl or were dropped by the tier rule.

    Where both judges saw the same document (3,619 records) the FRONTIER label
    wins, because it is the reference standard - but both labels are kept, so
    that overlap doubles as a ready-made agreement set.
    """
    print(f"\n[cyber_all] merging v5 corpus + frontier census")
    corp = sorted(glob.glob(os.path.join(out_dir, "corpus", "train-*.parquet")))
    cen = os.path.join(out_dir, "census", "train-00000.parquet")
    if not corp or not os.path.isfile(cen):
        print("  !! needs corpus and census built first - skipping")
        return None

    # Frontier labels first: they are the reference standard and win on conflict.
    c = pq.read_table(cen).to_pydict()
    best = {}
    for i in range(len(c["id"])):
        if c["label"][i] == "not_cyber":
            continue
        best[c["id"][i]] = {
            "id": c["id"][i], "text": c["text"][i], "url": c["url"][i],
            "date": c["date"][i], "dump": c["dump"][i],
            "keywords_found": c["keywords_found"][i],
            "tiers_found": c["tiers_found"][i], "accept_rule": c["accept_rule"][i],
            "label": c["label"][i], "label_source": "frontier",
            "confidence": c["confidence"][i], "justification": c["justification"][i],
            "frontier_label": c["label"][i], "open_model_label": None,
        }
    n_frontier = len(best)

    n_open = n_both = 0
    for f in corp:
        d = pq.read_table(f).to_pydict()
        for i in range(len(d["id"])):
            rid = d["id"][i]
            if rid in best:
                best[rid]["open_model_label"] = d["judge_label"][i]
                n_both += 1
                continue
            best[rid] = {
                "id": rid, "text": d["text"][i], "url": d["url"][i],
                "date": d["date"][i], "dump": d["dump"][i],
                "keywords_found": d["keywords_found"][i],
                "tiers_found": d["tiers_found"][i],
                "accept_rule": d["accept_rule"][i],
                "label": d["judge_label"][i], "label_source": "open_31b",
                "confidence": None, "justification": None,
                "frontier_label": None, "open_model_label": d["judge_label"][i],
            }
            n_open += 1

    w = ShardWriter(os.path.join(out_dir, "cyber_all"), "train", CYBER_ALL_SCHEMA)
    for r in best.values():
        w.add(r)
    rows, files = w.close()
    print(f"  frontier-labelled : {n_frontier:,}")
    print(f"  open-model only   : {n_open:,}")
    print(f"  judged by both    : {n_both:,}  (frontier label used; both kept)")
    print(f"  {rows:,} rows in {len(files)} shard(s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "publish", "hf_build"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "result", "cyber_arabic_v4_ALL.jsonl"))
    ap.add_argument("--corpus-name", default="v4")
    ap.add_argument("--census", default=os.path.join(ROOT, "evaluation", "chat_judging", "judged_corpus.json"))
    ap.add_argument("--bench-dir", default=os.path.join(ROOT, "evaluation", "judge_benchmark"))
    ap.add_argument("--skip-corpus", action="store_true",
                    help="the corpus is multi-GB; skip it while iterating on the others")
    args = ap.parse_args()

    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)
    print(f"building -> {out}")

    counts = {}
    if not args.skip_corpus:
        counts["corpus"] = build_corpus(args.corpus, out, args.corpus_name)
    counts["census"] = build_census(args.census, out)
    counts["judge_benchmark"] = build_bench(args.bench_dir, out)
    counts["cyber_all"] = build_cyber_all(out)

    # Copy the lexicon in - it is small, and a filter corpus without the filter
    # that produced it is not reproducible.
    kw_out = os.path.join(out, "keywords")
    os.makedirs(kw_out, exist_ok=True)
    copied = 0
    for src in [os.path.join(ROOT, "keywords", "all_cyber_keywords.txt")] + \
               [os.path.join(ROOT, "keywords", "tiers_clean", f"tier{t}_keywords.txt") for t in (1, 2, 3, 4)]:
        if os.path.isfile(src):
            dst = os.path.join(kw_out, os.path.basename(src))
            with open(src, encoding="utf-8") as a, open(dst, "w", encoding="utf-8", newline="\n") as b:
                b.write(a.read())
            copied += 1
    print(f"\n[keywords] copied {copied} lexicon file(s)")

    print("\n=== SUMMARY ===")
    for k, v in counts.items():
        print(f"  {k:<18}{'skipped' if v is None else f'{v:,} rows'}")
    print(f"\nNext: copy publish/DATASET_CARD.md to {out}/README.md, then follow")
    print(f"      publish/PUBLISHING_GUIDE.md")


if __name__ == "__main__":
    main()
