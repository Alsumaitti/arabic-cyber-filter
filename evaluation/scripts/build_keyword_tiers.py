#!/usr/bin/env python3
"""
build_keyword_tiers.py - derive a LAYERED keyword lexicon from the LLM-judge
evaluation: each keyword is assigned to the tier that reflects how much
"company" it needs before a match can be trusted.

The idea
--------
The judged sample (evaluation/results/judged_sample_matched.json, 300 records)
tells us, for every record, (a) exactly which lexicon keywords matched it and
(b) whether a frontier LLM that READ the document judged it cyber / borderline
/ not_cyber. Cross those two facts per keyword:

  Tier 1  keyword appeared ALONE (the record's only keyword hit) in a document
          judged cyber/borderline, AND its solo appearances are majority-good
          (solo_good > solo_bad). One hit of a Tier-1 keyword is, by measured
          evidence, enough to trust the record.
  Tier 2  strongest good evidence is in 2-keyword records (or a demoted Tier-1
          candidate whose solo record was usually a false positive). Needs one
          companion keyword.
  Tier 3  strongest good evidence is in 3-keyword records. Needs two companions.
  Tier 4  only ever seen in records with 4+ keyword hits. Never carries a
          document on its own in the observed data.

Demotion rule (the crucial refinement): a keyword seen solo in a good record
does NOT automatically reach Tier 1. Keywords like حصان طروادة appear solo in
1 good but 5 not_cyber records (political metaphors) — trusting them alone
would re-import exactly the false positives the evaluation exposed. Tier 1
therefore requires solo_good > solo_bad; failing that, the keyword drops to
its next-best evidence bucket (never higher than Tier 2).

Keywords observed ONLY in not_cyber records get NO tier (they earned zero
positive evidence); they are listed in the evidence TSV with tier "-".
Lexicon keywords never seen in the judged sample are simply not assigned
(no evidence either way) — judging the full corpus (scripts/llm_judge_api.py)
would tier all 1,209.

Outputs
-------
  keywords/tiers/tier1_keywords.txt ... tier4_keywords.txt
      (lexicon format: one keyword per line, '#' lines are comments)
  evaluation/results/keyword_tiers_evidence.tsv
      per-keyword evidence: tier, solo_good/solo_bad, good/bad record counts
      per bucket - the audit trail behind every tier assignment.
"""
import collections, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
# default: the 300-record judged sample; pass a path argument to build tiers
# from a bigger judged set (e.g. evaluation/chat_judging/judged_corpus.json)
JUDGED = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(EVAL, "results", "judged_sample_matched.json")
TIERS_DIR = os.path.join(ROOT, "keywords", "tiers")
EVIDENCE_OUT = os.path.join(EVAL, "results", "keyword_tiers_evidence.tsv")
os.makedirs(TIERS_DIR, exist_ok=True)

GOOD_LABELS = {"cyber", "borderline"}   # lenient: borderline counts as good


def bucket(n_keywords):
    """Map a record's distinct-keyword count to its tier bucket (4 = 4+)."""
    return min(n_keywords, 4)


def main():
    recs = json.load(open(JUDGED, encoding="utf-8"))

    # per keyword: good/bad record counts per bucket
    good = collections.defaultdict(collections.Counter)  # kw -> {bucket: n}
    bad = collections.defaultdict(collections.Counter)
    for r in recs:
        kws = r["keywords_found"]
        b = bucket(len(kws))
        target = good if r["judge_label"] in GOOD_LABELS else bad
        for kw in kws:
            target[kw][b] += 1

    all_kws = sorted(set(good) | set(bad))

    tiers = {}          # kw -> 1..4
    demoted = []        # (kw, solo_good, solo_bad) demoted out of tier 1
    for kw in all_kws:
        if not good[kw]:
            continue                          # no positive evidence -> no tier
        base = min(good[kw])                  # strongest good bucket
        if base == 1:
            solo_g, solo_b = good[kw][1], bad[kw][1]
            if solo_g <= solo_b:              # solo record is usually a false positive
                higher = [b for b in good[kw] if b >= 2]
                base = min(higher) if higher else 2
                demoted.append((kw, solo_g, solo_b))
        tiers[kw] = base

    # ------------------------------------------------------------- tier files
    HEADERS = {
        1: ("Tier 1 - SELF-SUFFICIENT keywords",
            "One hit of any keyword below, ALONE, was judged cyber/borderline by\n"
            "# the LLM in the evaluation sample (and its solo record was NOT usually\n"
            "# a false positive: solo_good > solo_bad). A single match is trustworthy."),
        2: ("Tier 2 - keywords that need ONE companion",
            "Strongest positive evidence for these is in records with exactly 2\n"
            "# distinct keyword hits (includes Tier-1 candidates demoted because their\n"
            "# solo records were mostly false positives, e.g. metaphorical uses)."),
        3: ("Tier 3 - keywords that need TWO companions",
            "Strongest positive evidence is in records with exactly 3 distinct\n"
            "# keyword hits."),
        4: ("Tier 4 - keywords that need HEAVY company (4+ hits)",
            "Only ever observed in records with 4 or more distinct keyword hits.\n"
            "# In the judged data these never carried a document on their own."),
    }
    per_tier = collections.defaultdict(list)
    for kw, t in tiers.items():
        per_tier[t].append(kw)

    def sort_key(kw):
        return (-good[kw][1], -sum(good[kw].values()), kw)

    for t in (1, 2, 3, 4):
        kws = sorted(per_tier[t], key=sort_key)
        path = os.path.join(TIERS_DIR, f"tier{t}_keywords.txt")
        title, body = HEADERS[t]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# {title}\n")
            f.write(f"# {body}\n")
            f.write("# Derived from evaluation/results/judged_sample_matched.json "
                    "(300 LLM-judged records)\n")
            f.write("# by evaluation/scripts/build_keyword_tiers.py. Evidence per keyword:\n")
            f.write("# evaluation/results/keyword_tiers_evidence.tsv\n")
            for kw in kws:
                f.write(kw + "\n")
        print(f"tier {t}: {len(kws):4d} keywords -> {path}")

    # --------------------------------------------------------- evidence table
    with open(EVIDENCE_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("keyword\ttier\tsolo_good\tsolo_bad\t"
                "good_b1\tgood_b2\tgood_b3\tgood_b4\t"
                "bad_b1\tbad_b2\tbad_b3\tbad_b4\tdemoted_from_tier1\n")
        def ev_key(kw):
            return (tiers.get(kw, 9), sort_key(kw))
        for kw in sorted(all_kws, key=ev_key):
            t = tiers.get(kw, "-")
            dem = "yes" if any(kw == d[0] for d in demoted) else ""
            f.write(f"{kw}\t{t}\t{good[kw][1]}\t{bad[kw][1]}\t"
                    f"{good[kw][1]}\t{good[kw][2]}\t{good[kw][3]}\t{good[kw][4]}\t"
                    f"{bad[kw][1]}\t{bad[kw][2]}\t{bad[kw][3]}\t{bad[kw][4]}\t{dem}\n")

    no_tier = [kw for kw in all_kws if kw not in tiers]
    print(f"\nkeywords observed in judged sample : {len(all_kws)} / 1209 lexicon")
    print(f"assigned a tier                    : {len(tiers)}")
    print(f"demoted out of tier 1              : {len(demoted)}")
    for kw, g, b in demoted:
        print(f"    {kw}   (solo: {g} good vs {b} not_cyber)")
    print(f"NO tier (only seen in not_cyber)   : {len(no_tier)}")
    for kw in no_tier:
        print(f"    {kw}")
    print(f"\nevidence table -> {EVIDENCE_OUT}")


if __name__ == "__main__":
    main()
