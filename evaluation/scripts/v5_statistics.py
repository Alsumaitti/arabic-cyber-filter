#!/usr/bin/env python3
"""
v5_statistics.py - full profile of the judged corpus, in one streaming pass.

With a per-record verdict on all 370,019 v4 documents we can finally answer
things a sample never could: which keywords actually survive contact with a
judge, how the crawl distributes over time, and whether the tiered rule behaves
the same way at every keyword density.

Everything here streams - the corpus is multi-GB and never lands in RAM.

A NOTE ON WHAT THESE NUMBERS ARE
The verdicts come from the open 31B judge, which over-calls `cyber` by ~1.9x
(evaluation/JUDGE_BENCHMARK.md). Corpus-level rates are corrected for that bias
via Rogan-Gladen. The PER-KEYWORD table is deliberately NOT corrected and is
labelled a diagnostic: the bias inflates every keyword's apparent quality by
roughly the same factor, so the RANKING is informative while the absolute values
are not. Do not rebuild the tiers from this table - the tiers stay derived from
the frontier census.

Usage:
    python evaluation/scripts/v5_statistics.py \
        --corpus result/cyber_arabic_v4_ALL.jsonl \
        --pred "evaluation/out_full/gemma_v4full_P32_p*.jsonl"
"""
import argparse, collections, glob, hashlib, json, math, os, sys
from urllib.parse import urlparse

CYBER_ISH = {"cyber", "borderline"}
SENS, SPEC = 0.9255, 0.7048


def rogan_gladen(p, sens=SENS, spec=SPEC):
    d = sens - (1 - spec)
    return min(1.0, max(0.0, (p - (1 - spec)) / d)) if d > 0 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def bar(frac, width=28):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="result/cyber_arabic_v4_ALL.jsonl")
    ap.add_argument("--pred", default="evaluation/out_full/gemma_v4full_P32_p*.jsonl")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-kw", type=int, default=200,
                    help="min appearances before a keyword enters the diagnostic")
    ap.add_argument("--out", default="evaluation/judge_benchmark/v5_statistics.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pred))
    if not files:
        sys.exit(f"error: no verdicts match {args.pred}")
    verdict = {}
    for p in files:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    verdict[r["id"]] = r.get("judge_label")
    print(f"verdicts: {len(verdict):,} from {len(files)} shards", flush=True)

    corpus = os.path.expanduser(args.corpus)
    if not os.path.isfile(corpus):
        sys.exit(f"error: {corpus} not found")

    labels = collections.Counter()
    by_rule = collections.defaultdict(lambda: collections.Counter())
    by_nk = collections.defaultdict(lambda: collections.Counter())
    by_dump = collections.defaultdict(lambda: collections.Counter())
    by_domain = collections.defaultdict(lambda: collections.Counter())
    kw = collections.defaultdict(lambda: [0, 0])          # keyword -> [kept, total]
    tier_hits = collections.Counter()
    len_kept, len_drop = [], []
    n = miss = 0

    print("streaming corpus...", flush=True)
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,}", flush=True)
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            lab = verdict.get(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
            if lab is None:
                miss += 1
                continue
            keep = lab in CYBER_ISH
            labels[lab] += 1

            rule = r.get("accept_rule") or "unknown"
            by_rule[rule][keep] += 1
            kws = r.get("keywords_found") or []
            nk = min(len(kws), 6)
            by_nk[nk][keep] += 1
            if r.get("dump"):
                by_dump[r["dump"]][keep] += 1
            url = r.get("url") or ""
            if url:
                try:
                    host = urlparse(url).netloc.lower().removeprefix("www.")
                    if host:
                        by_domain[host][keep] += 1
                except ValueError:
                    pass
            for k in kws:
                e = kw[k]
                e[1] += 1
                if keep:
                    e[0] += 1
            for t in (r.get("tiers_found") or []):
                if isinstance(t, int):
                    tier_hits[t] += 1
            (len_kept if keep else len_drop).append(len(text))

    judged = sum(labels.values())
    pos = sum(labels[l] for l in CYBER_ISH)
    p_obs = pos / judged
    pi = rogan_gladen(p_obs)
    print(f"\nscanned {n:,} | judged {judged:,} | no verdict {miss:,}")

    print(f"\n{'='*64}\nCORPUS-LEVEL\n{'='*64}")
    print(f"  v4 records                : {judged:,}")
    print(f"  observed cyber-ish rate   : {p_obs:.4f}   (census - no sampling error)")
    print(f"  corrected v4 precision    : {pi:.4f}")
    print(f"  v5 records (kept)         : {pos:,}")
    print(f"  removed                   : {judged-pos:,} ({(judged-pos)/judged:.1%})")
    v5_prec = pi * SENS / p_obs
    print(f"  v5 precision              : {v5_prec:.4f}")
    print(f"  genuine kept              : {pi*judged*SENS:,.0f} of {pi*judged:,.0f} "
          f"({SENS:.1%})")

    print(f"\n{'label':<14}{'n':>10}{'share':>9}")
    for l in ("cyber", "borderline", "not_cyber"):
        print(f"{l:<14}{labels[l]:>10,}{labels[l]/judged:>9.4f}")

    print(f"\n{'='*64}\nBY ACCEPTANCE RULE\n{'='*64}")
    print(f"{'rule':<20}{'v4':>10}{'v5':>10}{'keep':>8}  {'':28}")
    rows_rule = {}
    for rule in sorted(by_rule, key=lambda r: -sum(by_rule[r].values())):
        c = by_rule[rule]
        tot = c[True] + c[False]
        f = c[True] / tot if tot else 0
        rows_rule[rule] = {"v4": tot, "v5": c[True], "keep_rate": f}
        print(f"{rule:<20}{tot:>10,}{c[True]:>10,}{f:>8.3f}  {bar(f)}")

    print(f"\n{'='*64}\nBY DISTINCT KEYWORD HITS\n{'='*64}")
    print("More keywords should mean better documents. If the keep rate is flat,")
    print("the lexicon's density signal is weaker than the tier rule assumes.")
    print(f"\n{'hits':<8}{'v4':>10}{'v5':>10}{'keep':>8}  {'':28}")
    rows_nk = {}
    for k in sorted(by_nk):
        c = by_nk[k]
        tot = c[True] + c[False]
        f = c[True] / tot if tot else 0
        lbl = f"{k}+" if k == 6 else str(k)
        rows_nk[lbl] = {"v4": tot, "v5": c[True], "keep_rate": f}
        print(f"{lbl:<8}{tot:>10,}{c[True]:>10,}{f:>8.3f}  {bar(f)}")

    print(f"\n{'='*64}\nKEYWORD DIAGNOSTIC (uncorrected - ranking only)\n{'='*64}")
    elig = {k: v for k, v in kw.items() if v[1] >= args.min_kw}
    print(f"{len(elig):,} keywords with >= {args.min_kw} appearances "
          f"(of {len(kw):,} that fired at all)")
    ranked = sorted(elig.items(), key=lambda x: (x[1][0] / x[1][1], -x[1][1]))
    print(f"\nWORST {args.top} - most likely to drag in junk:")
    print(f"{'keyword':<34}{'appears':>9}{'kept':>8}{'rate':>7}")
    worst = []
    for k, (kept, tot) in ranked[: args.top]:
        worst.append({"keyword": k, "appears": tot, "kept": kept, "rate": kept / tot})
        print(f"{k:<34}{tot:>9,}{kept:>8,}{kept/tot:>7.3f}")
    print(f"\nBEST {args.top} - strongest standalone evidence:")
    best = []
    for k, (kept, tot) in list(reversed(ranked))[: args.top]:
        best.append({"keyword": k, "appears": tot, "kept": kept, "rate": kept / tot})
        print(f"{k:<34}{tot:>9,}{kept:>8,}{kept/tot:>7.3f}")

    print(f"\n{'='*64}\nTOP SOURCE DOMAINS (by v5 records)\n{'='*64}")
    dom = sorted(by_domain.items(), key=lambda x: -x[1][True])[: args.top]
    print(f"{'domain':<40}{'v5':>9}{'v4':>9}{'keep':>7}")
    doms = []
    for d, c in dom:
        tot = c[True] + c[False]
        doms.append({"domain": d, "v5": c[True], "v4": tot, "keep_rate": c[True] / tot})
        print(f"{d:<40}{c[True]:>9,}{tot:>9,}{c[True]/tot:>7.3f}")

    print(f"\n{'='*64}\nCRAWL COVERAGE\n{'='*64}")
    print(f"  distinct Common Crawl dumps : {len(by_dump):,}")
    print(f"  distinct source domains     : {len(by_domain):,}")
    top_dumps = sorted(by_dump.items(), key=lambda x: -x[1][True])[:8]
    print(f"\n{'dump':<26}{'v5':>9}{'keep':>7}")
    for d, c in top_dumps:
        tot = c[True] + c[False]
        print(f"{d:<26}{c[True]:>9,}{c[True]/tot:>7.3f}")

    print(f"\n{'='*64}\nDOCUMENT LENGTH\n{'='*64}")

    def q(v, p):
        if not v:
            return 0
        s = sorted(v)
        return s[min(len(s) - 1, int(p * len(s)))]
    for name, v in (("kept (v5)", len_kept), ("removed", len_drop)):
        if v:
            print(f"  {name:<12} n={len(v):>9,}  median {q(v,0.5):>7,} chars  "
                  f"p10 {q(v,0.1):>6,}  p90 {q(v,0.9):>7,}  mean {sum(v)//len(v):>7,}")
    if len_kept and len_drop:
        mk, md = q(len_kept, .5), q(len_drop, .5)
        print(f"\n  Kept documents are {mk/md:.2f}x the median length of removed ones"
              f" - {'longer, as expected for substantive articles' if mk > md else 'SHORTER, which is the opposite of the naive expectation'}.")

    # --- LENGTH AS A FILTER -------------------------------------------------
    # This turned out to be the single strongest signal in the whole corpus, and
    # the filter does not use it at all. Long documents are aggregators, forum
    # threads, news archives and terms-of-service pages that mention security in
    # passing among a lot of other content; a focused article is ~3k chars.
    BUCKETS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000),
               (5000, 8000), (8000, 15000), (15000, 30000), (30000, 1 << 62)]
    bl = collections.defaultdict(lambda: [0, 0])
    for L in len_kept:
        for b in BUCKETS:
            if b[0] <= L < b[1]:
                bl[b][0] += 1
                bl[b][1] += 1
                break
    for L in len_drop:
        for b in BUCKETS:
            if b[0] <= L < b[1]:
                bl[b][1] += 1
                break
    print(f"\n{'='*64}\nKEEP RATE BY LENGTH - the strongest signal found\n{'='*64}")
    print(f"{'length (chars)':<18}{'v4':>10}{'v5':>10}{'keep':>8}  {'':28}")
    len_rows = {}
    for b in BUCKETS:
        k, t = bl[b]
        if not t:
            continue
        lbl = f"{b[0]:,}-{b[1]:,}" if b[1] < (1 << 62) else f"{b[0]:,}+"
        len_rows[lbl] = {"v4": t, "v5": k, "keep_rate": k / t}
        print(f"{lbl:<18}{t:>10,}{k:>10,}{k/t:>8.3f}  {bar(k/t)}")

    print(f"\nWhat a plain length cap would buy, with NO judging at all:")
    print(f"{'cap':<12}{'records':>10}{'% corpus':>10}{'precision':>12}"
          f"{'  + judging':>13}")
    caps = {}
    for cap in (5000, 8000, 15000, 30000):
        k = sum(bl[b][0] for b in BUCKETS if b[1] <= cap)
        t = sum(bl[b][1] for b in BUCKETS if b[1] <= cap)
        if not t:
            continue
        po = k / t
        p_cap = rogan_gladen(po)
        p_both = p_cap * SENS / po
        caps[cap] = {"records": t, "precision_capped": p_cap,
                     "records_judged": k, "precision_capped_judged": p_both}
        print(f"<{cap:,}{'':<6}{t:>10,}{100*t/judged:>9.1f}%{p_cap:>12.4f}"
              f"{p_both:>13.4f}")
    print(f"\n  for comparison: v4 as-is {pi:.4f} at {judged:,} records; "
          f"v5 (judged) {v5_prec:.4f} at {pos:,}")
    print("  The '+ judging' column is a length cap AND the judge together -")
    print("  they are largely independent signals, so they compose.")

    print(f"\n  tier hits across the corpus: "
          + ", ".join(f"tier{t}={tier_hits[t]:,}" for t in sorted(tier_hits)))

    rep = {"v4_records": judged, "no_verdict": miss, "labels": dict(labels),
           "observed_accept": p_obs, "v4_precision_corrected": pi,
           "v5_records": pos, "v5_precision": v5_prec,
           "sens": SENS, "spec": SPEC,
           "by_rule": rows_rule, "by_nk": rows_nk,
           "keyword_worst": worst, "keyword_best": best,
           "top_domains": doms, "n_dumps": len(by_dump),
           "n_domains": len(by_domain),
           "median_len_kept": q(len_kept, .5), "median_len_removed": q(len_drop, .5),
           "by_length": len_rows, "length_caps": caps}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
