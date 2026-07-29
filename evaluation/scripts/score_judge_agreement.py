#!/usr/bin/env python3
"""
score_judge_agreement.py - answer the study question with numbers.

    "Can an open 31B model replace a frontier model for corpus curation
     in a low-resource language?"

Joins gemma's blind verdicts to the held-back frontier-model gold labels and
reports:
  * raw and corpus-weighted agreement (the sample is stratified, so the
    weighted figure is the one that describes the corpus)
  * Cohen's kappa - agreement corrected for chance, the headline number
  * per-class precision / recall / F1, treating the frontier labels as truth
  * the confusion matrix (this is where you SEE whether `borderline` is the
    weak spot, as expected)
  * the BINARY collapse (cyber+borderline vs not_cyber), which is the decision
    the tiered filter actually cares about - a model can be weak on the
    3-way split yet perfectly usable here
  * agreement by keyword-hit bucket, and unparsed-output rate

Usage (locally, after scp'ing the verdicts back):
    python evaluation/scripts/score_judge_agreement.py \
        --pred evaluation/judge_benchmark/gemma_verdicts.jsonl \
        --gold evaluation/judge_benchmark/validation_gold.jsonl
"""
import argparse, collections, glob, json, math, os, sys

LABELS = ("cyber", "borderline", "not_cyber")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
BENCH = os.path.join(EVAL, "judge_benchmark")


def load_jsonl(path):
    out = []
    for p in sorted(glob.glob(path)) or [path]:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def kappa(pairs, labels):
    """Cohen's kappa on (gold, pred) pairs."""
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for g, p in pairs if g == p) / n
    gc = collections.Counter(g for g, _ in pairs)
    pc = collections.Counter(p for _, p in pairs)
    pe = sum((gc[l] / n) * (pc[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def prf(pairs, label):
    tp = sum(1 for g, p in pairs if p == label and g == label)
    fp = sum(1 for g, p in pairs if p == label and g != label)
    fn = sum(1 for g, p in pairs if p != label and g == label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fn


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(BENCH, "gemma_verdicts.jsonl"),
                    help="gemma verdicts (accepts a glob for per-slice files)")
    ap.add_argument("--gold", default=os.path.join(BENCH, "validation_gold.jsonl"))
    ap.add_argument("--out", default=os.path.join(BENCH, "agreement_report.json"))
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(args.gold)}
    pred = {r["id"]: r for r in load_jsonl(args.pred)}
    if not gold:
        sys.exit(f"error: no gold records at {args.gold}")
    if not pred:
        sys.exit(f"error: no predictions at {args.pred}")

    ids = [i for i in gold if i in pred]
    missing = len(gold) - len(ids)
    unparsed = [i for i in ids if not pred[i].get("judge_label")]
    ok = [i for i in ids if pred[i].get("judge_label")]

    print(f"gold records      : {len(gold):,}")
    print(f"predictions joined: {len(ids):,}   (missing {missing})")
    print(f"unparsed verdicts : {len(unparsed):,} "
          f"({100.0*len(unparsed)/len(ids) if ids else 0:.2f}%)")
    if not ok:
        sys.exit("error: no parseable predictions to score")

    pairs = [(gold[i]["gold_label"], pred[i]["judge_label"]) for i in ok]
    w = [gold[i].get("weight", 1.0) for i in ok]

    agree = sum(1 for g, p in pairs if g == p)
    wagree = sum(wi for (g, p), wi in zip(pairs, w) if g == p)
    wtot = sum(w)
    lo, hi = wilson(agree, len(pairs))

    print("\n=== HEADLINE ===")
    print(f"  raw agreement      : {agree/len(pairs):.3f}  ({agree}/{len(pairs)}) "
          f"[95% CI {lo:.3f}-{hi:.3f}]")
    print(f"  corpus-weighted    : {wagree/wtot:.3f}   <- describes the corpus")
    print(f"  Cohen's kappa      : {kappa(pairs, LABELS):.3f}")

    print("\n=== PER-CLASS (frontier labels treated as truth) ===")
    print(f"  {'label':<12}{'prec':>7}{'recall':>8}{'F1':>7}{'support':>9}")
    for l in LABELS:
        p, r, f, s = prf(pairs, l)
        print(f"  {l:<12}{p:>7.3f}{r:>8.3f}{f:>7.3f}{s:>9}")

    print("\n=== CONFUSION (rows = frontier gold, cols = gemma) ===")
    cm = collections.Counter(pairs)
    print(f"  {'':<12}" + "".join(f"{l:>12}" for l in LABELS))
    for g in LABELS:
        row = "".join(f"{cm[(g, p)]:>12}" for p in LABELS)
        print(f"  {g:<12}{row}")

    print("\n=== BINARY COLLAPSE (cyber+borderline vs not_cyber) ===")
    b = lambda l: "not_cyber" if l == "not_cyber" else "cyber_ish"
    bp = [(b(g), b(p)) for g, p in pairs]
    bagree = sum(1 for g, p in bp if g == p)
    blo, bhi = wilson(bagree, len(bp))
    print(f"  agreement          : {bagree/len(bp):.3f} [95% CI {blo:.3f}-{bhi:.3f}]")
    print(f"  Cohen's kappa      : {kappa(bp, ('cyber_ish', 'not_cyber')):.3f}")
    p, r, f, s = prf(bp, "cyber_ish")
    print(f"  cyber_ish P/R/F1   : {p:.3f} / {r:.3f} / {f:.3f}   (support {s})")

    print("\n=== AGREEMENT BY KEYWORD-HIT BUCKET ===")
    byk = collections.defaultdict(list)
    for i in ok:
        byk[gold[i].get("nk", "?")].append(
            (gold[i]["gold_label"], pred[i]["judge_label"]))
    for k in sorted(byk, key=str):
        v = byk[k]
        a = sum(1 for g, p in v if g == p)
        print(f"  nk={k:<3} n={len(v):<5} agreement {a/len(v):.3f}")

    report = {
        "n_gold": len(gold), "n_joined": len(ids), "n_unparsed": len(unparsed),
        "raw_agreement": agree / len(pairs),
        "weighted_agreement": wagree / wtot,
        "kappa_3way": kappa(pairs, LABELS),
        "binary_agreement": bagree / len(bp),
        "kappa_binary": kappa(bp, ("cyber_ish", "not_cyber")),
        "per_class": {l: dict(zip(("precision", "recall", "f1", "support"),
                                  prf(pairs, l))) for l in LABELS},
        "confusion": {f"{g}->{p}": cm[(g, p)] for g in LABELS for p in LABELS},
    }
    json.dump(report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nreport -> {args.out}")

    k3 = report["kappa_3way"]
    verdict = ("STRONG - usable as a drop-in judge" if k3 >= 0.7 else
               "MODERATE - usable for the binary decision, check the confusion matrix"
               if k3 >= 0.4 else
               "WEAK - do not use these labels to rebuild tiers")
    print(f"\nverdict: kappa={k3:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
