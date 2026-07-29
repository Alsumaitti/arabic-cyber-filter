#!/usr/bin/env python3
"""
make_judge_validation_set.py - build the blind validation set for the
"can an open 31B model replace a frontier model as judge?" study.

Runs LOCALLY (the judged census lives here, not on the cluster). Draws a
stratified sample from the 14,653-record Claude census and splits it into:

    validation_prompts.jsonl   {id, nk, text}          -> upload to the cluster
    validation_gold.jsonl      {id, nk, gold_label,     -> STAYS LOCAL
                                gold_confidence, stratum, weight}

The gold labels are deliberately kept OUT of the file the judge sees, so the
comparison is blind by construction, not by convention.

Stratification is by (label x keyword-hit bucket): 12 cells. Sampling evenly
across cells over-represents the rare/hard ones on purpose - that is where
judge disagreement lives - and each record carries the inverse-probability
`weight` needed to recover corpus-level numbers afterwards.

Usage:
    python evaluation/scripts/make_judge_validation_set.py --n 2000
"""
import argparse, collections, hashlib, json, os, random, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
CENSUS = os.path.join(EVAL, "chat_judging", "judged_corpus.json")
OUT_DIR = os.path.join(EVAL, "judge_benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=CENSUS)
    ap.add_argument("--n", type=int, default=2000, help="total records to sample")
    ap.add_argument("--text-chars", type=int, default=1300,
                    help="truncate text to this many chars (same view the "
                         "in-chat judge had, so the comparison is fair)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if not os.path.isfile(args.census):
        sys.exit(f"error: census not found: {args.census}")
    print(f"loading census: {args.census}")
    recs = json.load(open(args.census, encoding="utf-8"))
    print(f"  {len(recs):,} judged records")

    # ---- stratify by (label, keyword-hit bucket) --------------------------
    strata = collections.defaultdict(list)
    for r in recs:
        nk = min(len(set(r["keywords_found"])), 4)
        strata[(r["judge_label"], nk)].append(r)

    cells = sorted(strata)
    per_cell = max(1, args.n // len(cells))
    rng = random.Random(args.seed)

    prompts, gold = [], []
    for cell in cells:
        pool = strata[cell]
        take = min(per_cell, len(pool))
        chosen = rng.sample(pool, take)
        weight = len(pool) / take          # inverse sampling probability
        for r in chosen:
            rid = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:16]
            text = r["text"][: args.text_chars]
            prompts.append({"id": rid, "nk": cell[1], "text": text})
            gold.append({"id": rid, "nk": cell[1],
                         "gold_label": r["judge_label"],
                         "gold_confidence": r.get("judge_confidence", ""),
                         "stratum": f"{cell[0]}_nk{cell[1]}",
                         "weight": round(weight, 4)})
        print(f"  {str(cell):<24} pool {len(pool):>5}  take {take:>4}  weight {weight:>7.2f}")

    # ids must be unique or the join later is ambiguous
    if len({p["id"] for p in prompts}) != len(prompts):
        sys.exit("error: duplicate ids (identical texts sampled twice)")

    os.makedirs(args.out_dir, exist_ok=True)
    p_path = os.path.join(args.out_dir, "validation_prompts.jsonl")
    g_path = os.path.join(args.out_dir, "validation_gold.jsonl")
    with open(p_path, "w", encoding="utf-8", newline="\n") as f:
        for r in prompts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(g_path, "w", encoding="utf-8", newline="\n") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsampled {len(prompts):,} records across {len(cells)} strata")
    print(f"  UPLOAD to cluster : {p_path}  "
          f"({os.path.getsize(p_path)/2**20:.1f} MiB)")
    print(f"  KEEP local (gold) : {g_path}")
    print(f"\ngold distribution: "
          f"{dict(collections.Counter(g['gold_label'] for g in gold))}")


if __name__ == "__main__":
    main()
