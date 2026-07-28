#!/usr/bin/env python3
"""
join_v3_with_judgments.py - attach the EXISTING census verdicts to the v3
tiered corpus, with no re-judging.

Why this works: the v3 tier lexicon (640 keywords) is a strict subset of the
v2 lexicon (1,233), and every clause of the tiered acceptance rule needs at
least one tier-keyword hit. So every v3-accepted record was necessarily also
kept by v2 - and all 14,653 v2 records were judged in the full census
(evaluation/chat_judging/judged_corpus.json). Every v3 record therefore
already has a verdict; we just need to look it up.

Records are matched on a hash of their exact `text` (stable, and unlike URL
it is guaranteed unique per source record).

Usage:
    python evaluation/scripts/join_v3_with_judgments.py \
        --v3 result/cyber_arabic_v3_ALL.json \
        [--judged evaluation/chat_judging/judged_corpus.json] \
        [--out result/cyber_arabic_v3_ALL_judged.json]
"""
import argparse, collections, hashlib, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default=os.path.join(ROOT, "result", "cyber_arabic_v3_ALL.json"))
    ap.add_argument("--judged", default=os.path.join(EVAL, "chat_judging", "judged_corpus.json"))
    ap.add_argument("--out", default=None,
                    help="write the joined corpus here (default: <v3 stem>_judged.json)")
    args = ap.parse_args()

    for p in (args.v3, args.judged):
        if not os.path.isfile(p):
            sys.exit(f"error: not found: {p}")

    print(f"loading judged census: {args.judged}")
    judged = json.load(open(args.judged, encoding="utf-8"))
    verdict = {key(r["text"]): r for r in judged}
    print(f"  {len(judged):,} judged records indexed")

    print(f"loading v3 corpus: {args.v3}")
    v3 = json.load(open(args.v3, encoding="utf-8"))
    print(f"  {len(v3):,} v3 records")

    out_rows, unmatched = [], 0
    labels = collections.Counter()
    by_rule = collections.defaultdict(collections.Counter)
    for r in v3:
        j = verdict.get(key(r["text"]))
        row = dict(r)
        if j is None:
            unmatched += 1
            row["judge_label"] = row["judge_confidence"] = row["judge_justification"] = None
        else:
            row["judge_label"] = j["judge_label"]
            row["judge_confidence"] = j["judge_confidence"]
            row["judge_justification"] = j["judge_justification"]
            labels[j["judge_label"]] += 1
            by_rule[r.get("accept_rule", "?")][j["judge_label"]] += 1
        out_rows.append(row)

    n = len(v3)
    matched = n - unmatched
    print(f"\njoined {matched:,}/{n:,} records with an existing verdict "
          f"({unmatched:,} unmatched)")
    if unmatched:
        print("  ! unmatched records would mean the v3 run used a lexicon/corpus that is "
              "NOT a subset of the judged census - investigate before trusting the numbers.")

    good = labels["cyber"] + labels["borderline"]
    print("\n=== VERDICTS ON THE v3 TIERED CORPUS ===")
    for lab in ("cyber", "borderline", "not_cyber"):
        print(f"  {lab:<11} {labels[lab]:>6,}  ({100.0*labels[lab]/matched:5.1f}%)")
    print(f"  {'TOTAL':<11} {matched:>6,}")
    print(f"\n  strict precision  (cyber)              : {labels['cyber']/matched:.3f}")
    print(f"  lenient precision (cyber + borderline) : {good/matched:.3f}")

    print("\n=== BY ACCEPTANCE CLAUSE ===")
    print(f"  {'clause':<20}{'n':>7}{'cyber':>8}{'border':>8}{'not':>8}{'lenientP':>10}")
    for rule, c in sorted(by_rule.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        lp = (c["cyber"] + c["borderline"]) / tot if tot else 0
        print(f"  {rule:<20}{tot:>7,}{c['cyber']:>8,}{c['borderline']:>8,}"
              f"{c['not_cyber']:>8,}{lp:>10.3f}")

    out = args.out or (os.path.splitext(args.v3)[0] + "_judged.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(out_rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\njoined corpus -> {out}")


if __name__ == "__main__":
    main()
