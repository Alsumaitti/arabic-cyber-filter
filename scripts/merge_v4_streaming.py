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

OUTPUT FORMAT - why JSONL is the default here
---------------------------------------------
v3's corpus (78 MB) was a single JSON array, which is convenient and fine at
that size. At v4's ~5 GB it is not: json.load() on a 5 GB array needs ~16-21 GB
of RAM, which would undo the whole point of merging without loading. So the
default output is JSONL - one JSON record per line:

    * read it line-by-line with ~one record of memory
    * wc -l / head / split / grep work on it directly
    * it is FineWeb2's own native format (.jsonl.gz)

Use --format json if you specifically need a single array (e.g. to reuse the
v3-era tooling, which expects one); at v4 scale prefer jsonl.

It also reports the summary you actually want at the end: totals and the
acceptance-clause breakdown.

Usage:
  python merge_v4_streaming.py \
      --dir        ~/fanar-data/cyber_filtered/output_v4 \
      --out-prefix ~/fanar-data/cyber_filtered/result/cyber_arabic_v4_ALL

Reading the result:
  import json
  with open("cyber_arabic_v4_ALL.jsonl", encoding="utf-8") as f:
      for line in f:                     # constant memory
          rec = json.loads(line)
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
    ap.add_argument("--format", choices=("jsonl", "json"), default="jsonl",
                    help="jsonl (default, streamable - recommended at v4 scale) "
                         "or json (single array, needs ~4x its size in RAM to read)")
    ap.add_argument("--csv", action="store_true",
                    help="also write a CSV copy (off by default: at ~5 GB it "
                         "doubles disk use and is too big for Excel anyway)")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.dir, "*.json"))
                   if not p.endswith(".json.tmp"))
    if not paths:
        sys.exit(f"error: no .json shard outputs in {args.dir}")

    # Guard: outputs are named ..._P<PARTS>_p<idx>.json. Two different PARTS
    # values slice the same row groups differently, so merging a mixture would
    # duplicate records. Refuse, and say exactly which sets are present.
    import re as _re
    seen = collections.Counter()
    for p in paths:
        m = _re.search(r"_P(\d+)_p\d+\.json$", os.path.basename(p))
        seen[m.group(1) if m else "unlabelled"] += 1
    if len(seen) > 1:
        sys.exit("error: output dir mixes runs with different PARTS values: "
                 + ", ".join(f"P{k} ({v} files)" for k, v in sorted(seen.items()))
                 + "\n       Merging them would DUPLICATE records. Keep only one set,\n"
                   "       e.g.:  mkdir old && mv *_P8_p*.* old/")
    print(f"merging {len(paths)} shard file(s), PARTS={list(seen)[0]} (streaming) ...",
          flush=True)

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    jsonl = args.format == "jsonl"
    json_path = args.out_prefix + (".jsonl" if jsonl else ".json")
    csv_path = args.out_prefix + ".csv"

    total = 0
    rules = collections.Counter()
    empty_shards = []

    jf = open(json_path + ".tmp", "w", encoding="utf-8")
    if not jsonl:
        jf.write("[\n")
    first = True

    cf = None
    writer = None
    if args.csv:
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
            if jsonl:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
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

    if not jsonl:
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
    print(f"{args.format:<5}-> {json_path} ({os.path.getsize(json_path)/2**30:.2f} GiB)")
    if args.csv:
        print(f"csv  -> {csv_path} ({os.path.getsize(csv_path)/2**30:.2f} GiB)")
    if jsonl:
        print("\nread it without loading it all:")
        print("  with open(path, encoding='utf-8') as f:")
        print("      for line in f: rec = json.loads(line)")


if __name__ == "__main__":
    main()
