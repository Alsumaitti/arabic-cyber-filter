#!/usr/bin/env python3
"""
make_fewshot_examples.py - pick few-shot exemplars for the gemma judge from the
frontier-model census.

Why: the zero-shot run scored kappa 0.419 with a clear failure mode - gemma
almost never used `borderline` (14.6% recall), collapsing it into `cyber`
(469 of 664 borderline docs). The rubric DESCRIBES borderline; it never SHOWS
it. This builds demonstrations, weighted toward the weak class.

Leakage guard: every record in the validation set is excluded by id, so the
few-shot examples and the evaluated records are disjoint. Without this the
comparison would be meaningless.

Selection: high-confidence gold labels, short enough not to bloat the prompt,
and (for borderline) preferring justifications that spell out WHY it is minor.

Usage:
    python evaluation/scripts/make_fewshot_examples.py
    python evaluation/scripts/make_fewshot_examples.py --per-class 3 --borderline 5
"""
import argparse, hashlib, json, os, random, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
CENSUS = os.path.join(EVAL, "chat_judging", "judged_corpus.json")
BENCH = os.path.join(EVAL, "judge_benchmark")
LABELS = ("cyber", "borderline", "not_cyber")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=CENSUS)
    ap.add_argument("--gold", default=os.path.join(BENCH, "validation_gold.jsonl"),
                    help="validation ids to EXCLUDE (no leakage)")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--borderline", type=int, default=5,
                    help="extra examples for the weak class")
    ap.add_argument("--text-chars", type=int, default=420,
                    help="truncate exemplar text (keeps the prompt small)")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", default=os.path.join(BENCH, "fewshot_examples.json"))
    args = ap.parse_args()

    excluded = set()
    if os.path.isfile(args.gold):
        excluded = {json.loads(l)["id"] for l in open(args.gold, encoding="utf-8") if l.strip()}
    print(f"excluding {len(excluded):,} validation ids from exemplar selection")

    recs = json.load(open(args.census, encoding="utf-8"))
    rng = random.Random(args.seed)

    pools = {l: [] for l in LABELS}
    for r in recs:
        rid = hashlib.sha1(r["text"].encode("utf-8")).hexdigest()[:16]
        if rid in excluded:
            continue
        # Prefer confident exemplars - but note the census contains NO
        # high-confidence `borderline` at all (all medium/low). That is not a
        # bug: borderline is intrinsically the uncertain class, which both
        # partly excuses gemma's collapse of it and caps the achievable 3-way
        # kappa. So accept medium here, and rank high first when choosing.
        if r.get("judge_confidence") not in ("high", "medium"):
            continue
        t = r["text"].strip()
        if not (250 <= len(t) <= 1500):
            continue                       # long enough to judge, short enough to show
        just = (r.get("judge_justification") or "").strip()
        if len(just) < 40:
            continue
        pools[r["judge_label"]].append((r.get("judge_confidence"), t, just))

    want = {l: args.per_class for l in LABELS}
    want["borderline"] = args.borderline

    out = []
    for lab in LABELS:
        pool = pools[lab]
        if len(pool) < want[lab]:
            sys.exit(f"error: only {len(pool)} usable exemplars for {lab}")
        # sample from the most confident tier available for this label
        high = [x for x in pool if x[0] == "high"]
        src = high if len(high) >= want[lab] else pool
        for conf, t, just in rng.sample(src, want[lab]):
            out.append({"label": lab,
                        "text": t[: args.text_chars],
                        "justification": just[:220],
                        "summary": ""})
        print(f"  {lab:<12} pool {len(pool):>5} (high: {len(high):>5}) -> chose "
              f"{want[lab]} from {'high' if src is high else 'high+medium'}")

    rng.shuffle(out)                        # avoid label-order artefacts
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    chars = sum(len(e["text"]) + len(e["justification"]) for e in out)
    print(f"\n{len(out)} exemplars -> {args.out}")
    print(f"added prompt size: ~{chars:,} chars (~{chars//3:,} tokens)")
    print("UPLOAD this file to the cluster next to validation_prompts.jsonl")


if __name__ == "__main__":
    main()
