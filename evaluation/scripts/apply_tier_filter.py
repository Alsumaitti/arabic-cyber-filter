#!/usr/bin/env python3
"""
apply_tier_filter.py - the second-generation ("v3") filter and its measurement.

Idea 2 of the proposal, now closed with full evidence. The tiered lexicon
(keywords/tiers/tier{1..4}_keywords.txt), learned from the full-corpus census
(evaluation/chat_judging/judged_corpus.json - every record LLM-judged), defines
how much "company" each keyword needs before its match can be trusted:

    Tier 1  self-sufficient        - one hit is enough
    Tier 2  needs one companion    - trust at 2 tiered hits (tier <= 2)
    Tier 3  needs two companions   - trust at 3 tiered hits (tier <= 3)
    Tier 4  needs heavy company    - only inside 4+ tiered hits
    (no tier) never carried a good record - contributes nothing

Tiered acceptance rule (keep a record iff ANY holds):
    >= 1 Tier-1 hit,  OR
    >= 2 hits of tier <= 2,  OR
    >= 3 hits of tier <= 3,  OR
    >= 4 tiered hits of any tier.

Because the census carries a judged label for EVERY record, this script MEASURES
the rule exactly (no sampling) and compares it against the naive baselines the
proposal set out to beat: accept-all (the v2 corpus as-is) and a flat
min-distinct-keywords >= 2 / >= 3 threshold.

    precision (strict)  = kept cyber                / kept
    precision (lenient) = kept (cyber+borderline)   / kept
    retention (lenient) = kept (cyber+borderline)   / all (cyber+borderline) in corpus
    F1 (lenient)        = harmonic mean of lenient precision and lenient retention

CAVEAT: the tiers were learned from this same census, so the tiered-rule numbers
are IN-SAMPLE (optimistic). The baseline rules use no labels, so their numbers
are unbiased; the honest comparison is "tiered rule vs those baselines".

Usage:
    python evaluation/scripts/apply_tier_filter.py                 # measure + print
    python evaluation/scripts/apply_tier_filter.py --write-corpus  # also emit the
        # accepted records to result/cyber_arabic_v3_tiered.json
"""
import argparse, collections, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)

JUDGED = os.path.join(EVAL, "chat_judging", "judged_corpus.json")
TIERS_DIR = os.path.join(ROOT, "keywords", "tiers")
GOOD = {"cyber", "borderline"}


def load_tiers():
    """keyword -> tier(1..4). Keywords absent from every tier file have no tier."""
    tier_of = {}
    for t in (1, 2, 3, 4):
        path = os.path.join(TIERS_DIR, f"tier{t}_keywords.txt")
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                tier_of[line] = t
    return tier_of


def accept_tiered(tiers):
    """tiers = list of tier ints (1..4) for a record's TIERED keyword hits."""
    t1 = sum(1 for t in tiers if t == 1)
    le2 = sum(1 for t in tiers if t <= 2)
    le3 = sum(1 for t in tiers if t <= 3)
    return t1 >= 1 or le2 >= 2 or le3 >= 3 or len(tiers) >= 4


def score(kept_labels, total_good):
    """kept_labels: Counter of judge labels among kept records."""
    kept = sum(kept_labels.values())
    if kept == 0:
        return dict(kept=0, strict_p=0.0, lenient_p=0.0, retention=0.0, f1=0.0,
                    cyber=0, borderline=0, not_cyber=0)
    cyber = kept_labels["cyber"]
    good = cyber + kept_labels["borderline"]
    strict_p = cyber / kept
    lenient_p = good / kept
    retention = good / total_good if total_good else 0.0
    f1 = (2 * lenient_p * retention / (lenient_p + retention)
          if (lenient_p + retention) else 0.0)
    return dict(kept=kept, strict_p=strict_p, lenient_p=lenient_p,
                retention=retention, f1=f1, cyber=cyber,
                borderline=kept_labels["borderline"], not_cyber=kept_labels["not_cyber"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=JUDGED)
    ap.add_argument("--write-corpus", action="store_true",
                    help="write accepted records to result/cyber_arabic_v3_tiered.json")
    args = ap.parse_args()

    recs = json.load(open(args.judged, encoding="utf-8"))
    tier_of = load_tiers()
    N = len(recs)
    total_good = sum(1 for r in recs if r["judge_label"] in GOOD)
    total_cyber = sum(1 for r in recs if r["judge_label"] == "cyber")

    rules = {
        "accept-all (v2 corpus)":   lambda r: True,
        "min distinct keywords >=2": lambda r: len(set(r["keywords_found"])) >= 2,
        "min distinct keywords >=3": lambda r: len(set(r["keywords_found"])) >= 3,
        "TIERED acceptance rule":   None,   # handled specially
    }

    results = {}
    accepted_tiered = []
    for name, pred in rules.items():
        kept = collections.Counter()
        for r in recs:
            if name == "TIERED acceptance rule":
                tiers = [tier_of[k] for k in set(r["keywords_found"]) if k in tier_of]
                ok = accept_tiered(tiers)
                if ok:
                    accepted_tiered.append(r)
            else:
                ok = pred(r)
            if ok:
                kept[r["judge_label"]] += 1
        results[name] = score(kept, total_good)

    # ---------------------------------------------------------------- report
    print(f"Census: {N:,} records  |  cyber {total_cyber:,}  "
          f"good(cyber+borderline) {total_good:,}\n")
    hdr = f"{'rule':<28}{'kept':>8}{'strictP':>9}{'lenientP':>10}{'retention':>11}{'F1':>7}"
    print(hdr); print("-" * len(hdr))
    for name in rules:
        s = results[name]
        print(f"{name:<28}{s['kept']:>8,}{s['strict_p']:>9.3f}"
              f"{s['lenient_p']:>10.3f}{s['retention']:>11.3f}{s['f1']:>7.3f}")
    print()
    for name in rules:
        s = results[name]
        print(f"{name}: kept {s['kept']:,} = "
              f"cyber {s['cyber']:,} + borderline {s['borderline']:,} + "
              f"not_cyber {s['not_cyber']:,}")

    if args.write_corpus:
        out = os.path.join(ROOT, "result", "cyber_arabic_v3_tiered.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(accepted_tiered, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nwrote {len(accepted_tiered):,} accepted records -> {out}")


if __name__ == "__main__":
    main()
