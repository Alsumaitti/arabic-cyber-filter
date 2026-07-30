#!/usr/bin/env python3
"""
apply_v4_verdicts.py - join full-corpus gemma verdicts back onto v4 and emit v5.

This is where per-record judging pays off. The sample could only ever report a
number; per-record verdicts let us actually remove the bad records.

WHAT COMES OUT
  v5 corpus  - v4 records gemma called cyber-ish, with the verdict columns
               appended (judge_label / judge_confidence / judge_justification),
               so every kept record carries its own evidence
  a report   - what was removed, and the corrected precision of the result

PRECISION OF THE RESULT
With the whole corpus judged there is NO sampling error left - p_obs is the
census accept rate, not an estimate. Only the calibration of sens/spec carries
uncertainty. Writing pi for true prevalence:

    p_obs      = pi*sens + (1-pi)*(1-spec)        -> invert for pi (Rogan-Gladen)
    kept       = N * p_obs
    genuine    = N * pi * sens
    precision  = pi * sens / p_obs

so the reported precision of v5 follows from the operating point plus the
observed accept rate, and the CI comes from propagating the sens/spec intervals
(both measured on n=1,992 in evaluation/JUDGE_BENCHMARK.md).

HONEST LIMIT: gemma cannot grade topical centrality (borderline recall 0.146),
so v5 is a cleaner corpus, NOT a corpus of only-strictly-on-topic documents. The
`borderline` label it does assign is unreliable and should not be used to
sub-filter further - keep the binary decision, which is what was validated.

Usage:
    python evaluation/scripts/apply_v4_verdicts.py \
        --pred   "out_full/gemma_v4full_P8_p*.jsonl" \
        --meta   v4_full_meta.jsonl \
        --corpus result/cyber_arabic_v4_ALL.jsonl \
        --out    result/cyber_arabic_v5_judged.jsonl
"""
import argparse, glob, hashlib, json, math, os, sys
from collections import Counter

CYBER_ISH = {"cyber", "borderline"}
DEF_SENS, DEF_SENS_N = 0.9255, 1328
DEF_SPEC, DEF_SPEC_N = 0.7048, 664


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


def precision_of_kept(p_obs, sens, spec):
    """Precision among records the judge accepted."""
    pi = rogan_gladen(p_obs, sens, spec)
    return (pi * sens / p_obs) if p_obs > 0 else float("nan"), pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="verdict shards (glob)")
    ap.add_argument("--corpus", required=True, help="merged v4 corpus .jsonl")
    ap.add_argument("--out", required=True, help="v5 output .jsonl")
    ap.add_argument("--meta", default=None, help="optional, for per-rule reporting")
    ap.add_argument("--sens", type=float, default=DEF_SENS)
    ap.add_argument("--spec", type=float, default=DEF_SPEC)
    ap.add_argument("--keep-all", action="store_true",
                    help="write every record with its verdict instead of only positives")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.expanduser(args.pred)))
    if not files:
        sys.exit(f"error: no verdict files match {args.pred}")
    pred = {}
    for p in files:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    pred[r["id"]] = r
    print(f"verdict shards : {len(files)}")
    print(f"verdicts loaded: {len(pred):,}")

    corpus = os.path.expanduser(args.corpus)
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    labels = Counter()
    rules_in, rules_kept = Counter(), Counter()
    total = kept = missing = unparsed = 0

    with open(corpus, encoding="utf-8") as fh, \
         open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            rid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            rule = r.get("accept_rule") or "unknown"
            rules_in[rule] += 1

            v = pred.get(rid)
            if v is None:
                missing += 1
                continue
            lab = v.get("judge_label")
            if lab not in ("cyber", "borderline", "not_cyber"):
                unparsed += 1
                continue
            labels[lab] += 1
            positive = lab in CYBER_ISH
            if not positive and not args.keep_all:
                continue
            r["judge_label"] = lab
            r["judge_confidence"] = v.get("judge_confidence")
            r["judge_justification"] = v.get("judge_justification")
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
            if positive:
                rules_kept[rule] += 1

    print(f"corpus records : {total:,}")
    if missing:
        print(f"!! {missing:,} records ({100*missing/total:.2f}%) have no verdict - "
              f"the judging array is incomplete.\n"
              f"   Finish it before quoting these numbers; every unjudged record "
              f"was DROPPED from the output.")
    if unparsed:
        print(f"!! {unparsed:,} unparsed verdicts - dropped")

    judged = sum(labels.values())
    if not judged:
        sys.exit("error: nothing judged")
    pos = sum(labels[l] for l in CYBER_ISH)
    p_obs = pos / judged

    print(f"\n{'label':<14}{'records':>10}{'share':>9}")
    for lab in ("cyber", "borderline", "not_cyber"):
        print(f"{lab:<14}{labels[lab]:>10,}{labels[lab]/judged:>9.3f}")

    prec, pi = precision_of_kept(p_obs, args.sens, args.spec)
    # Calibration-only CI: no sampling error remains once the census is complete.
    _, s_lo, s_hi = wilson(round(args.sens * DEF_SENS_N), DEF_SENS_N)
    _, k_lo, k_hi = wilson(round(args.spec * DEF_SPEC_N), DEF_SPEC_N)
    lo = min(precision_of_kept(p_obs, a, b)[0] for a in (s_lo, s_hi) for b in (k_lo, k_hi))
    hi = max(precision_of_kept(p_obs, a, b)[0] for a in (s_lo, s_hi) for b in (k_lo, k_hi))

    print(f"\n=== V5 ===")
    print(f"  judged                 : {judged:,}")
    print(f"  observed accept rate   : {p_obs:.4f}  (census, no sampling error)")
    print(f"  implied true prevalence: {pi:.4f}  (v4 corpus precision)")
    print(f"  records written        : {kept:,}"
          f"{'  (all, --keep-all)' if args.keep_all else ''}")
    print(f"  removed                : {judged - pos:,} "
          f"({100*(judged-pos)/judged:.1f}% of judged)")
    print(f"  V5 PRECISION           : {prec:.4f} [{lo:.4f}-{hi:.4f}]")
    print(f"    (interval is calibration uncertainty only - sens/spec from n=1,992)")
    print(f"  genuine records kept   : {pi*judged*args.sens:,.0f} of "
          f"{pi*judged:,.0f} in v4  ({args.sens:.1%} of them)")

    if rules_in:
        print(f"\n{'accept rule':<20}{'v4':>10}{'v5':>10}{'kept':>8}")
        for rule in sorted(rules_in, key=lambda r: -rules_in[r]):
            k = rules_kept[rule]
            print(f"{rule:<20}{rules_in[rule]:>10,}{k:>10,}"
                  f"{k/rules_in[rule]:>8.3f}")

    print(f"\nv5 -> {out_path}")
    print("\nNOTE: gemma cannot grade topical centrality (borderline recall 0.146),")
    print("so v5 is a cleaner corpus, not an only-strictly-on-topic one. Do NOT")
    print("sub-filter on its `borderline` label - only the binary call was validated.")

    rep = {"corpus_records": total, "judged": judged, "missing": missing,
           "unparsed": unparsed, "labels": dict(labels), "observed_accept": p_obs,
           "implied_v4_precision": pi, "v5_records": kept,
           "v5_precision": prec, "v5_precision_ci": [lo, hi],
           "sens": args.sens, "spec": args.spec,
           "per_rule": {r: {"v4": rules_in[r], "v5": rules_kept[r]} for r in rules_in}}
    rep_path = os.path.splitext(out_path)[0] + "_report.json"
    json.dump(rep, open(rep_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"report -> {rep_path}")


if __name__ == "__main__":
    main()
