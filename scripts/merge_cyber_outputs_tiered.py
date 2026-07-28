#!/usr/bin/env python3
"""
merge_cyber_outputs_tiered.py - merge the per-shard outputs of
filter_cyber_arabic_tiered.py into one combined corpus (JSON + CSV).

Same job as merge_cyber_outputs.py but schema-aware for the v3 columns
(keywords_found, tiers_found, accept_rule) - the column list is inferred
from the first record seen, so it also works on v1-schema outputs.

Usage:
  python merge_cyber_outputs_tiered.py \
      --dir        ~/fanar-data/cyber_filtered/output_tiered \
      --out-prefix ~/fanar-data/cyber_filtered/result/cyber_arabic_v3_ALL
"""
import argparse
import csv
import glob
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="folder with per-shard *.json outputs")
ap.add_argument("--out-prefix", required=True)
args = ap.parse_args()

# ignore leftover .tmp files from interrupted runs; only merge final .json
paths = sorted(p for p in glob.glob(os.path.join(args.dir, "*.json"))
               if not p.endswith(".json.tmp"))
if not paths:
    sys.exit(f"error: no .json files found in {args.dir}")
print(f"merging {len(paths)} per-shard file(s) ...")

all_rows = []
for i, p in enumerate(paths, 1):
    with open(p, encoding="utf-8") as f:
        try:
            rows = json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"error: {p} is not valid JSON ({e}) - "
                     f"was its shard interrupted? re-run the array first.")
    all_rows.extend(rows)
    print(f"  [{i}/{len(paths)}] {os.path.basename(p)}: {len(rows)} record(s)")

if not all_rows:
    sys.exit("error: zero records across all shards - nothing to merge")

columns = list(all_rows[0].keys())
print(f"total: {len(all_rows)} records | columns: {columns}")

out_dir = os.path.dirname(args.out_prefix)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
json_path = args.out_prefix + ".json"
csv_path = args.out_prefix + ".csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_rows, f, ensure_ascii=False, indent=1)

with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    for row in all_rows:
        r = dict(row)
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                r[k] = json.dumps(v, ensure_ascii=False)
        w.writerow(r)

print(f"json -> {json_path}")
print(f"csv  -> {csv_path}")
