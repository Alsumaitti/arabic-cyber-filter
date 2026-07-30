#!/usr/bin/env python3
"""
prepare_v4_full_prompts.py - turn the whole merged v4 corpus into judge prompts.

WHY JUDGE ALL OF IT
The sample established that v4 is 0.715 precision. Sampling can only ever give
that number - it cannot tell you WHICH records are the bad ones, so it cannot
improve the corpus. Judging every record can.

The arithmetic favours it decisively. With the measured operating point
(sens 0.9255 / spec 0.7048), keeping only the records gemma calls cyber-ish
yields precision ~0.887 while retaining ~92.5% of the genuine content. Compare
the alternatives available from keyword counts alone:

    current rule           precision 0.715, genuine 264,645 (100%)
    tighten to t<=2 >= 3   precision 0.816, genuine 215,924 (81.6%)
    per-record judging     precision 0.887, genuine 244,922 (92.5%)

Per-record judging beats the best threshold rule on BOTH axes, because a
threshold can only count keywords while the model reads the document.

WHAT THIS SCRIPT DOES
Streams the merged corpus, assigns each record the same sha1-based id used
everywhere else in this project, drops exact-duplicate texts and unjudgeable
stubs, and writes:

    v4_full_prompts.jsonl   id + truncated text   -> fed to the GPUs
    v4_full_meta.jsonl      id + rule/nk/url/dump -> for joining verdicts back

Memory stays flat: one pass, nothing accumulated except the id set for dedup.

RUN THIS ON THE CLUSTER (the corpus lives there; the login node kills
interactive python) - see sbatch_scripts/judge_v4_full.sh.

Usage:
    python scripts/prepare_v4_full_prompts.py \
        --input   ~/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL.jsonl \
        --out-dir ~/fanar-data/judge_bench
"""
import argparse, hashlib, json, os, sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--text-chars", type=int, default=6000,
                    help="truncate text sent to the judge (prompt budget)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    inp = os.path.expanduser(args.input)
    out_dir = os.path.expanduser(args.out_dir)
    if not os.path.isfile(inp):
        sys.exit(f"error: {inp} not found - run the v4 merge first (jsonl format).")
    os.makedirs(out_dir, exist_ok=True)

    p_path = os.path.join(out_dir, "v4_full_prompts.jsonl")
    m_path = os.path.join(out_dir, "v4_full_meta.jsonl")

    seen = set()
    rules = Counter()
    total = kept = short = dupes = 0

    print(f"streaming {inp}", flush=True)
    with open(inp, encoding="utf-8") as fh, \
         open(p_path, "w", encoding="utf-8", newline="\n") as pf, \
         open(m_path, "w", encoding="utf-8", newline="\n") as mf:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if total % args.progress_every == 0:
                print(f"  {total:,} read | {kept:,} written", flush=True)
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("text") or "").strip()
            if len(text) < args.min_chars:
                short += 1
                continue
            rid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            if rid in seen:
                dupes += 1
                continue
            seen.add(rid)

            rule = r.get("accept_rule") or "unknown"
            rules[rule] += 1
            pf.write(json.dumps({"id": rid, "text": text[: args.text_chars]},
                                ensure_ascii=False) + "\n")
            mf.write(json.dumps(
                {"id": rid, "accept_rule": rule,
                 "nk": len(r.get("keywords_found") or []),
                 "tiers_found": r.get("tiers_found"),
                 "url": r.get("url"), "dump": r.get("dump"),
                 "date": r.get("date")}, ensure_ascii=False) + "\n")
            kept += 1

    if not kept:
        sys.exit("error: nothing written - check the input format")

    print(f"\nread {total:,} | judgeable {kept:,} | too short {short:,} "
          f"| duplicate text {dupes:,}")
    print(f"\n{'accept rule':<20}{'records':>12}")
    for rule, n in rules.most_common():
        print(f"{rule:<20}{n:>12,}")

    print(f"\n{kept:,} prompts -> {p_path}")
    print(f"{kept:,} meta    -> {m_path}   (keep - needed to join verdicts back)")
    print(f"\nSize the array so each slice fits the wall clock. At R rec/s per GPU,")
    print(f"one slice of {kept:,}/PARTS records takes {kept:,}/(PARTS*R) seconds.")
    print(f"Measure R from an earlier run:  grep 'rec/s' logs/v4_*.out | tail -3")
    print(f"\nnext:  sbatch --export=ALL,PARTS=8 --array=0-7 sbatch_scripts/judge_v4_full.sh")


if __name__ == "__main__":
    main()
