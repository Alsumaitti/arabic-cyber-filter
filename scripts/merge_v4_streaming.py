#!/usr/bin/env python3
"""
merge_v4_streaming.py - memory-safe merge for the v4 (parquet) run.

merge_cyber_outputs_tiered.py accumulates every record in a Python list before
writing, which is fine for the ~78 MB v3 corpus but not for v4: ~70M input
records are expected to yield ~420k accepted ones (~5 GB of JSON), and holding
that in memory as Python objects would need tens of GB.

This version streams: it reads ONE shard file at a time (each ~30 MB) and
appends its records straight to the output, so peak memory stays at roughly
one shard regardless of how many shards there are.

It also reports the summary you actually want at the end: totals, the
acceptance-clause breakdown, and per-shard record counts.

Usage:
  python merge_v4_streaming.py \
      --dir        ~/fanar-data/cyber_filtered/output_v4 \
      --out-prefix ~/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

COLUMNS = ["text", "dump", "url", "date", "file_path",
           "keywords_found", "tiers_found", "accept_rule"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--no-csv", action="store_true",
                    help="skip the CSV copy (halves the output size / time)")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.dir, "*.json"))
                   if not p.endswith(".json.tmp"))
    if not paths:
        sys.exit(f"error: no .json shard outputs in {args.dir}")
    print(f"merging {len(paths)} shard file(s) (streaming) ...", flush=True)

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path = args.out_prefix + ".json"
    csv_path = args.out_prefix + ".csv"

    total = 0
    rules = collections.Counter()
    empty_shards = []

    jf = open(json_path + ".tmp", "w", encoding="utf-8")
    jf.write("[\n")
    first = True

    cf = None
    writer = None
    if not args.no_csv:
        cf = open(csv_path + ".tmp", "w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(cf, fieldnames=COLUMNS)
        writer.writeheader()

    for i, p in enumerate(paths, 1):
        try:
            rows = json.load(open(p, encoding="utf-8"))
        except json.JSONDecodeError as e:
            jf.close()
            if cf:
                cf.close()
            sys.exit(f"\nerror: {os.path.basename(p)} is not valid JSON ({e}).\n"
                     f"       That shard was interrupted - re-run the array, then merge.")
        if not rows:
            empty_shards.append(os.path.basename(p))
        for r in rows:
            if not first:
                jf.write(",\n")
            jf.write(json.dumps(r, ensure_ascii=False))
            first = False
            rules[r.get("accept_rule", "?")] += 1
            if writer:
                cr = {c: r.get(c, "") for c in COLUMNS}
                for k in ("keywords_found", "tiers_found"):
                    if isinstance(cr[k], (list, dict)):
                        cr[k] = json.dumps(cr[k], ensure_ascii=False)
                writer.writerow(cr)
        total += len(rows)
        if i % 25 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {total:,} records so far", flush=True)

    jf.write("\n]\n")
    jf.close()
    os.replace(json_path + ".tmp", json_path)
    if cf:
        cf.close()
        os.replace(csv_path + ".tmp", csv_path)

    print(f"\ntotal accepted records : {total:,}")
    print(f"acceptance clauses     : {dict(rules)}")
    if empty_shards:
        print(f"shards with 0 records  : {len(empty_shards)}")
    print(f"json -> {json_path} ({os.path.getsize(json_path)/2**30:.2f} GiB)")
    if not args.no_csv:
        print(f"csv  -> {csv_path} ({os.path.getsize(csv_path)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
