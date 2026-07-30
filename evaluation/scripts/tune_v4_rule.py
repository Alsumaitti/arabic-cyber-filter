#!/usr/bin/env python3
"""
tune_v4_rule.py - find the best tiered acceptance rule using ONLY the judging
already paid for.

WHY
estimate_v4_precision.py reports that `two_tier2_hits` is much weaker than
`tier1_hit` (corrected 0.583 vs 0.837, non-overlapping CIs) and notes that
dropping it would raise corpus precision. That framing is precision-myopic:
`two_tier2_hits` admits 177,351 records of which ~103k are genuinely cyber, so
dropping it buys +12pp precision at the cost of ~39% of the good content. For a
pretraining corpus that is usually the wrong trade.

The useful question is not "drop or keep" but "what threshold". This sweeps a
family of stricter rules over the sample and reports the precision/recall
frontier, so the choice is made against numbers instead of instinct.

WHAT IT CAN AND CANNOT EVALUATE
The sample was drawn from the v4 corpus, i.e. from records the CURRENT rule
already accepted. So candidate rules must be SUBSETS of the current one -
tightening is measurable, loosening is not (those records were never sampled and
their judgments do not exist). Rules that would admit new records are therefore
out of scope here, and the script refuses to pretend otherwise.

ESTIMATION
Every sampled record carries its stratum weight (population / n_sampled). For a
candidate rule R:
    population kept   = sum of weights over sampled records passing R
    observed positive = weighted mean of gemma's cyber-ish verdict among them
    precision         = Rogan-Gladen(observed, sens, spec)
    genuine kept      = precision x population kept
"Recall" is relative to the genuine content of the CURRENT corpus, not of
FineWeb2 - the filter's absolute recall is a separate measurement.

Usage:
    python evaluation/scripts/tune_v4_rule.py \
        --pred "evaluation/judge_benchmark/gemma_v4_P2_p*.jsonl" \
        --meta evaluation/judge_benchmark/v4_sample_meta.jsonl
"""
import argparse, glob, json, math, os, sys

CYBER_ISH = {"cyber", "borderline"}
DEF_SENS, DEF_SPEC = 0.9255, 0.7048


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def rogan_gladen(p_obs, sens, spec):
    denom = sens - (1 - spec)
    return min(1.0, max(0.0, (p_obs - (1 - spec)) / denom)) if denom > 0 else float("nan")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def counts(tiers):
    """(n_tier1, n_tier<=2, n_tier<=3, n_tiered) for one record."""
    ts = [t for t in (tiers or []) if isinstance(t, int)]
    return (sum(1 for t in ts if t == 1),
            sum(1 for t in ts if t <= 2),
            sum(1 for t in ts if t <= 3),
            len(ts))


def make_rules():
    """Candidate rules, all subsets of the current one. (name, predicate)."""
    rules = [("current (t1>=1 | t<=2>=2 | t<=3>=3 | any>=4)",
              lambda c: c[0] >= 1 or c[1] >= 2 or c[2] >= 3 or c[3] >= 4)]
    for b in (3, 4, 5):
        rules.append((f"t1>=1 | t<=2 hits >= {b}",
                      lambda c, b=b: c[0] >= 1 or c[1] >= b))
    rules.append(("t1>=1 | (t<=2>=2 & any>=4)",
                  lambda c: c[0] >= 1 or (c[1] >= 2 and c[3] >= 4)))
    rules.append(("t1>=1 | (t<=2>=3 & any>=4)",
                  lambda c: c[0] >= 1 or (c[1] >= 3 and c[3] >= 4)))
    rules.append(("t1>=1 only (drop rule 2)", lambda c: c[0] >= 1))
    rules.append(("t1>=2", lambda c: c[0] >= 2))
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--sens", type=float, default=DEF_SENS)
    ap.add_argument("--spec", type=float, default=DEF_SPEC)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = {r["id"]: r for r in load_jsonl(os.path.expanduser(args.meta))}
    pred = {}
    for p in sorted(glob.glob(os.path.expanduser(args.pred))):
        for r in load_jsonl(p):
            pred[r["id"]] = r
    if not meta or not pred:
        sys.exit("error: need both --meta and --pred")

    recs = []
    for rid, m in meta.items():
        pr = pred.get(rid)
        if pr is None or pr.get("judge_label") not in ("cyber", "borderline", "not_cyber"):
            continue
        recs.append((counts(m.get("tiers_found")), m["weight"],
                     1 if pr["judge_label"] in CYBER_ISH else 0))
    if not recs:
        sys.exit("error: no judged records to evaluate")

    # Baseline = the current corpus, for the recall denominator.
    base_pop = sum(w for _, w, _ in recs)
    base_p = rogan_gladen(sum(w * y for _, w, y in recs) / base_pop, args.sens, args.spec)
    base_genuine = base_p * base_pop
    print(f"judged sample      : {len(recs):,} records")
    print(f"current corpus     : {base_pop:,.0f} records, corrected precision "
          f"{base_p:.3f}, genuine ~{base_genuine:,.0f}")
    print(f"\n(recall below is relative to the CURRENT corpus's genuine content,\n"
          f" not to FineWeb2 - only rules STRICTER than the current one are\n"
          f" measurable from this sample.)\n")

    print(f"{'rule':<34}{'kept':>10}{'prec':>7}{'95% CI':>15}{'genuine':>10}"
          f"{'recall':>8}{'F1':>7}")
    print("-" * 91)
    out = []
    for name, fn in make_rules():
        sel = [(w, y) for c, w, y in recs if fn(c)]
        if not sel:
            print(f"{name:<34}{'0':>10}   - accepts nothing")
            continue
        pop = sum(w for w, _ in sel)
        n = len(sel)
        p_obs = sum(w * y for w, y in sel) / pop
        prec = rogan_gladen(p_obs, args.sens, args.spec)
        # CI on p_obs via effective n for this subset, then propagated.
        sw = sum(w for w, _ in sel)
        sw2 = sum(w * w for w, _ in sel)
        n_eff = sw * sw / sw2 if sw2 else n
        _, lo, hi = wilson(p_obs * n_eff, n_eff)
        clo, chi = rogan_gladen(lo, args.sens, args.spec), rogan_gladen(hi, args.sens, args.spec)
        genuine = prec * pop
        rec = genuine / base_genuine if base_genuine else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"rule": name, "kept": round(pop), "n_sample": n,
                    "precision": prec, "ci": [clo, chi],
                    "genuine": round(genuine), "recall_vs_current": rec, "f1": f1})
        print(f"{name:<34}{pop:>10,.0f}{prec:>7.3f}{f'{clo:.3f}-{chi:.3f}':>15}"
              f"{genuine:>10,.0f}{rec:>8.3f}{f1:>7.3f}")

    print("\n=== READING THIS ===")
    print("  'recall' is share of the current corpus's GENUINE records retained.")
    print("  A rule is only worth adopting if it raises precision materially")
    print("  WITHOUT giving up genuine content you wanted - check the 'genuine'")
    print("  column, not just 'prec'. F1 balances the two, but the right choice")
    print("  depends on use:")
    print("    - pretraining data  -> favour recall; noise is tolerable")
    print("    - eval/benchmark set -> favour precision; every bad record costs")
    # The baseline has recall 1.000 BY CONSTRUCTION (it defines the denominator),
    # so it wins on F1 unless a tightening is spectacular. Ranking it against the
    # alternatives would be an artefact of the metric, not a finding - compare
    # only the genuine alternatives, and say what the baseline costs instead.
    alts = [r for r in out if not r["rule"].startswith("current")]
    if alts:
        best = max(alts, key=lambda r: r["f1"])
        print(f"\n  NOTE: the current rule scores recall 1.000 by construction - it")
        print(f"  defines the denominator - so its F1 is not comparable. Among the")
        print(f"  tightenings, best F1 is:")
        print(f"    {best['rule']}")
        print(f"    precision {best['precision']:.3f} (from {base_p:.3f}), "
              f"keeps {best['kept']:,} records")
        print(f"    retains {best['recall_vs_current']:.1%} of genuine content, "
              f"giving up ~{base_genuine - best['genuine']:,.0f} good records")
    if any(r["precision"] >= 1.0 for r in out):
        print(f"\n  CAUTION: a precision of 1.000 is the Rogan-Gladen estimator")
        print(f"  clamped at its ceiling, not a measurement of perfection. It means")
        print(f"  the observed rate exceeded what the operating point can explain -")
        print(f"  treat those rows as 'very high', with the CI as the honest range.")

    dest = args.out or os.path.join(os.path.dirname(os.path.abspath(args.meta)),
                                    "v4_rule_tuning.json")
    json.dump({"baseline": {"kept": round(base_pop), "precision": base_p,
                            "genuine": round(base_genuine)}, "rules": out},
              open(dest, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nreport -> {dest}")


if __name__ == "__main__":
    main()
