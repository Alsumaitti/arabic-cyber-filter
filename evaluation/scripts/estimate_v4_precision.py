#!/usr/bin/env python3
"""
estimate_v4_precision.py - turn gemma's biased binary verdicts on the v4 sample
into a corrected, CI-bounded precision estimate for the whole 370k corpus.

THE ARGUMENT
gemma over-calls `cyber` by roughly 1.9x (see evaluation/JUDGE_BENCHMARK.md), so
its raw accept rate is NOT the corpus precision. But its bias was measured on a
blind 1,992-record set against frontier labels:

    sensitivity = 0.9255      (P[says cyber-ish | truly cyber-ish])
    specificity = 0.7048      (P[says not_cyber | truly not_cyber])

A classifier with known sensitivity and specificity yields an unbiased
prevalence estimate by inverting the misclassification (Rogan-Gladen 1978):

    p_true = (p_obs - (1 - spec)) / (sens - (1 - spec))

Sanity check that this is not hand-waving: applied to the validation set itself,
observed 0.7154 -> corrected 0.667, against a true gold rate of 0.667. Exact.

TWO SOURCES OF UNCERTAINTY, both reported:
  1. sampling error on p_obs        -> Wilson interval, propagated through the
                                       correction
  2. error in sens/spec themselves  -> they came from n=1,992, so they carry
                                       their own Wilson intervals; --show-cal
                                       reports how much that widens things

Weighting: the sample is stratified by acceptance rule with unequal probability,
so p_obs is the inverse-probability-weighted mean, not the raw mean. The raw
number is printed too, clearly labelled, because it is the one people quote by
accident.

Usage:
    python evaluation/scripts/estimate_v4_precision.py \
        --pred "evaluation/judge_benchmark/gemma_v4_P2_p*.jsonl" \
        --meta evaluation/judge_benchmark/v4_sample_meta.jsonl
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict

CYBER_ISH = {"cyber", "borderline"}

# Operating point from the few-shot arm of the judge benchmark (n=1,992).
# Both arms agreed to within 0.003, so this is not sensitive to the choice.
DEF_SENS, DEF_SENS_N = 0.9255, 1328
DEF_SPEC, DEF_SPEC_N = 0.7048, 664


def wilson(k, n, z=1.96):
    """Wilson score interval - behaves sanely near 0 and 1, unlike normal-approx."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def wilson_p(p, n, z=1.96):
    """Wilson bounds from a proportion + n (for effective/weighted samples)."""
    return wilson(p * n, n, z)


def rogan_gladen(p_obs, sens, spec):
    """Bias-corrected prevalence, clamped to [0,1] (the estimator can overshoot)."""
    denom = sens - (1 - spec)
    if denom <= 0:
        return float("nan")
    return min(1.0, max(0.0, (p_obs - (1 - spec)) / denom))


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="gemma verdicts (glob ok, e.g. 'out/gemma_v4_P2_p*.jsonl')")
    ap.add_argument("--meta", required=True,
                    help="v4_sample_meta.jsonl from sample_v4_for_judging.py")
    ap.add_argument("--sens", type=float, default=DEF_SENS)
    ap.add_argument("--spec", type=float, default=DEF_SPEC)
    ap.add_argument("--show-cal", action="store_true",
                    help="also propagate uncertainty in sens/spec")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = {r["id"]: r for r in load_jsonl(os.path.expanduser(args.meta))}
    if not meta:
        sys.exit(f"error: no records in {args.meta}")

    files = sorted(glob.glob(os.path.expanduser(args.pred)))
    if not files:
        sys.exit(f"error: no prediction files match {args.pred}")
    pred = {}
    for p in files:
        for r in load_jsonl(p):
            pred[r["id"]] = r
    print(f"prediction files : {len(files)}")
    print(f"sample records   : {len(meta):,}")
    print(f"judged           : {len(pred):,}")

    missing = [i for i in meta if i not in pred]
    if missing:
        print(f"!! {len(missing):,} sampled records have no verdict - the estimate "
              f"below covers only the judged ones, which biases it if the gap is "
              f"not random. Finish the array before quoting these numbers.")
    unparsed = sum(1 for r in pred.values()
                   if r.get("judge_label") not in ("cyber", "borderline", "not_cyber"))
    if unparsed:
        print(f"!! {unparsed:,} unparsed verdicts ({100*unparsed/len(pred):.2f}%) - excluded")

    # --- aggregate per stratum ---------------------------------------------
    strata = defaultdict(lambda: {"n": 0, "pos": 0, "weight": 0.0, "pop": 0})
    for rid, m in meta.items():
        r = pred.get(rid)
        if r is None:
            continue
        lab = r.get("judge_label")
        if lab not in ("cyber", "borderline", "not_cyber"):
            continue
        s = strata[m["accept_rule"]]
        s["n"] += 1
        s["pos"] += 1 if lab in CYBER_ISH else 0
        s["weight"] = m["weight"]
        s["pop"] = m["population"]

    if not strata:
        sys.exit("error: no usable (sampled, judged, parsed) records")

    total_pop = sum(s["pop"] for s in strata.values())
    print(f"corpus represented: {total_pop:,}")

    # --- per-rule ----------------------------------------------------------
    print(f"\n{'accept rule':<20}{'pop':>10}{'n':>7}{'raw':>8}{'corrected':>11}"
          f"{'95% CI':>18}")
    print("-" * 74)
    rows = {}
    for rule in sorted(strata, key=lambda r: -strata[r]["pop"]):
        s = strata[rule]
        p, lo, hi = wilson(s["pos"], s["n"])
        c, clo, chi = (rogan_gladen(x, args.sens, args.spec) for x in (p, lo, hi))
        rows[rule] = {"population": s["pop"], "n": s["n"], "raw": p,
                      "corrected": c, "ci": [clo, chi],
                      "est_true_cyber": round(c * s["pop"])}
        print(f"{rule:<20}{s['pop']:>10,}{s['n']:>7,}{p:>8.3f}{c:>11.3f}"
              f"{f'{clo:.3f}-{chi:.3f}':>18}")

    # --- corpus level, inverse-probability weighted -------------------------
    w_pos = sum(s["pos"] * s["weight"] for s in strata.values())
    w_n = sum(s["n"] * s["weight"] for s in strata.values())
    p_w = w_pos / w_n
    raw_pos = sum(s["pos"] for s in strata.values())
    raw_n = sum(s["n"] for s in strata.values())

    # Effective sample size for a stratified weighted mean (Kish). Using raw_n
    # here would overstate precision, so the CI is built on n_eff instead.
    sw = sum(s["n"] * s["weight"] for s in strata.values())
    sw2 = sum(s["n"] * s["weight"] ** 2 for s in strata.values())
    n_eff = sw * sw / sw2 if sw2 else raw_n

    _, lo_w, hi_w = wilson_p(p_w, n_eff)
    corr = rogan_gladen(p_w, args.sens, args.spec)
    c_lo = rogan_gladen(lo_w, args.sens, args.spec)
    c_hi = rogan_gladen(hi_w, args.sens, args.spec)

    print(f"\n=== V4 CORPUS PRECISION ===")
    print(f"  gemma raw accept rate (unweighted) : {raw_pos/raw_n:.4f} "
          f"({raw_pos:,}/{raw_n:,})  <- do not quote this")
    print(f"  gemma accept rate (weighted)       : {p_w:.4f}")
    print(f"  effective sample size              : {n_eff:,.0f}")
    print(f"  operating point                    : sens {args.sens:.4f} / "
          f"spec {args.spec:.4f}")
    print(f"  CORRECTED precision                : {corr:.4f} "
          f"[95% CI {c_lo:.4f}-{c_hi:.4f}]   <- the number")
    print(f"  implied genuine cyber records      : {corr*total_pop:,.0f} "
          f"of {total_pop:,}")
    print(f"                                       [{c_lo*total_pop:,.0f} - "
          f"{c_hi*total_pop:,.0f}]")

    cal = None
    if args.show_cal:
        # Worst/best case over the calibration intervals - deliberately
        # conservative: it moves sens and spec in the same adverse direction.
        _, s_lo, s_hi = wilson(round(args.sens * DEF_SENS_N), DEF_SENS_N)
        _, k_lo, k_hi = wilson(round(args.spec * DEF_SPEC_N), DEF_SPEC_N)
        worst = rogan_gladen(lo_w, s_hi, k_hi)
        best = rogan_gladen(hi_w, s_lo, k_lo)
        cal = {"sens_ci": [s_lo, s_hi], "spec_ci": [k_lo, k_hi],
               "range": sorted([worst, best])}
        print(f"\n  including calibration uncertainty (sens/spec from n=1,992):")
        print(f"    sens 95% CI {s_lo:.4f}-{s_hi:.4f} | spec 95% CI {k_lo:.4f}-{k_hi:.4f}")
        print(f"    precision range {min(worst,best):.4f}-{max(worst,best):.4f}")

    # --- what it implies for the filter ------------------------------------
    weakest = min(rows, key=lambda r: rows[r]["corrected"])
    strongest = max(rows, key=lambda r: rows[r]["corrected"])
    print(f"\n=== WHAT TO DO WITH THIS ===")
    print(f"  strongest rule: {strongest} ({rows[strongest]['corrected']:.3f})")
    print(f"  weakest rule  : {weakest} ({rows[weakest]['corrected']:.3f}, "
          f"{rows[weakest]['population']:,} records)")
    spread = rows[strongest]["corrected"] - rows[weakest]["corrected"]
    if spread > 0.15 and rows[weakest]["ci"][1] < rows[strongest]["ci"][0]:
        kept = total_pop - rows[weakest]["population"]
        num = corr * total_pop - rows[weakest]["corrected"] * rows[weakest]["population"]
        print(f"  -> the gap is {spread:.3f} and the CIs do not overlap, so dropping "
              f"'{weakest}'\n     leaves {kept:,} records at precision "
              f"~{num/kept:.3f} (from {corr:.3f}).")
        print(f"     That is a real improvement available with no further judging.")
    else:
        print(f"  -> spread {spread:.3f}; no single rule is clearly the weak link, "
              f"so the\n     tiered rule is behaving consistently across doors. "
              f"Keep all four.")

    report = {"total_population": total_pop, "n_judged": raw_n,
              "n_missing": len(missing), "n_unparsed": unparsed,
              "sens": args.sens, "spec": args.spec,
              "raw_accept_unweighted": raw_pos / raw_n,
              "accept_weighted": p_w, "n_eff": n_eff,
              "precision_corrected": corr, "precision_ci": [c_lo, c_hi],
              "implied_cyber_records": round(corr * total_pop),
              "per_rule": rows, "calibration": cal}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.meta)),
                                   "v4_precision_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
