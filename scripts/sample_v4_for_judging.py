#!/usr/bin/env python3
"""
sample_v4_for_judging.py - draw a judgeable sample from the merged v4 corpus.

WHY THIS EXISTS
The v4 corpus (370k records out of ~70M parquet rows) has never been measured.
Every quality number this project reports comes from the v2/v3 census. v4 uses
the same tiered rule on a 70x larger crawl, and we have no evidence that the
rule's precision survives that change of distribution.

Judging all 370k with a frontier model is not affordable. But the judge
benchmark (see evaluation/JUDGE_BENCHMARK.md) established that the local 31B
model is a usable BINARY instrument - kappa 0.654, sensitivity 0.926,
specificity 0.705 - and that a known operating point can be inverted with
Rogan-Gladen. So: sample, judge locally, correct arithmetically.

STRATIFIED BY ACCEPTANCE RULE, on purpose
The tiered rule has four independent doors into the corpus:

    tier1_hit | two_tier2_hits | three_tier3_hits | four_plus_hits

A single overall precision number would hide which door leaks. Sampling equally
from each and weighting back by inverse probability gives BOTH the corpus-level
estimate and a per-rule breakdown - which is directly actionable: if one rule is
much worse than the others, drop or tighten that rule and the corpus improves
without any new judging.

STREAMING: reservoir sampling per stratum, so a multi-GB JSONL never lands in
RAM. One pass, memory bounded by the sample itself.

RUN THIS ON THE CLUSTER (the merged corpus lives there, and the login node kills
interactive python) - see sbatch_scripts/sample_v4_for_judging.sh.

Usage:
    python scripts/sample_v4_for_judging.py \
        --input    ~/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL.jsonl \
        --out-dir  ~/fanar-data/judge_bench \
        --per-rule 750
"""
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict

# Must match accept() in filter_cyber_parquet_tiered.py exactly.
RULES = ("tier1_hit", "two_tier2_hits", "three_tier3_hits", "four_plus_hits")


def rid_of(text: str) -> str:
    """Same id scheme as the census join - sha1 of the text, first 16 hex."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="merged v4 corpus (.jsonl - one record per line)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-rule", type=int, default=750,
                    help="sample size per acceptance rule (default 750)")
    ap.add_argument("--text-chars", type=int, default=6000,
                    help="truncate text sent to the judge (prompt budget)")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip stubs too short to judge")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    inp = os.path.expanduser(args.input)
    out_dir = os.path.expanduser(args.out_dir)
    if not os.path.isfile(inp):
        sys.exit(f"error: {inp} not found. Run the v4 merge first "
                 f"(sbatch_scripts/merge_v4.sh) and check --format jsonl.")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    # reservoir[stratum] holds at most per-rule records; seen[stratum] is the
    # true population count, which is what turns the sample into a weight.
    reservoir = defaultdict(list)
    seen = Counter()
    skipped_short = 0
    dupes = 0
    ids_seen = set()
    total = 0

    print(f"streaming {inp}", flush=True)
    with open(inp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if total % args.progress_every == 0:
                print(f"  {total:,} records | strata {dict(seen)}", flush=True)
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("text") or "").strip()
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            rid = rid_of(text)
            if rid in ids_seen:
                dupes += 1          # exact-duplicate text across shards
                continue
            ids_seen.add(rid)

            rule = r.get("accept_rule") or "unknown"
            seen[rule] += 1
            res = reservoir[rule]
            if len(res) < args.per_rule:
                res.append((rid, text, r))
            else:
                # standard reservoir: keep each seen item with equal probability
                j = rng.randrange(seen[rule])
                if j < args.per_rule:
                    res[j] = (rid, text, r)

    if not total:
        sys.exit(f"error: {inp} had no records")

    print(f"\nscanned {total:,} lines | judgeable {sum(seen.values()):,} "
          f"| too short {skipped_short:,} | duplicate text {dupes:,}")

    # --- write prompts (text only) + a local meta file (weights, rules) ------
    prompts_path = os.path.join(out_dir, "v4_sample_prompts.jsonl")
    meta_path = os.path.join(out_dir, "v4_sample_meta.jsonl")
    n_written = 0
    with open(prompts_path, "w", encoding="utf-8", newline="\n") as pf, \
         open(meta_path, "w", encoding="utf-8", newline="\n") as mf:
        print(f"\n{'rule':<20}{'population':>12}{'sampled':>9}{'weight':>10}")
        print("-" * 51)
        for rule in list(RULES) + [r for r in sorted(seen) if r not in RULES]:
            pop = seen.get(rule, 0)
            res = reservoir.get(rule, [])
            if not pop:
                print(f"{rule:<20}{0:>12}{0:>9}{'-':>10}")
                continue
            # inverse sampling probability: each sampled record stands for this
            # many corpus records. Sums to the population by construction.
            weight = pop / len(res)
            print(f"{rule:<20}{pop:>12,}{len(res):>9,}{weight:>10.1f}")
            for rid, text, r in res:
                pf.write(json.dumps(
                    {"id": rid, "text": text[: args.text_chars]},
                    ensure_ascii=False) + "\n")
                mf.write(json.dumps(
                    {"id": rid, "accept_rule": rule, "weight": weight,
                     "population": pop, "n_sampled": len(res),
                     "nk": len(r.get("keywords_found") or []),
                     "tiers_found": r.get("tiers_found"),
                     "url": r.get("url"), "dump": r.get("dump")},
                    ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\n{n_written:,} records -> {prompts_path}")
    print(f"{n_written:,} records -> {meta_path}   (weights/rules - keep for scoring)")
    print(f"corpus represented: {sum(seen.values()):,}")
    print("\nnext:  sbatch sbatch_scripts/judge_v4_sample.sh")


if __name__ == "__main__":
    main()
