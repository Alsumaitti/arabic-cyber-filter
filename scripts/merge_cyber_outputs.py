#!/usr/bin/env python3
"""
merge_cyber_outputs.py - combine all per-shard cyber_arabic_*.json / .csv
files (produced by the array job) into one final json + csv.

Usage:
  python merge_cyber_outputs.py --dir ~/fanar-data/cyber_filtered/output \
      --out-prefix ~/fanar-data/cyber_filtered/output/cyber_arabic_ALL
"""
import argparse, csv, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()

json_files = sorted(f for f in glob.glob(os.path.join(a.dir, "cyber_arabic_*.json"))
                     if not f.endswith("_ALL.json"))
csv_files = sorted(f for f in glob.glob(os.path.join(a.dir, "cyber_arabic_*.csv"))
                    if not f.endswith("_ALL.csv"))

all_records = []
for jf in json_files:
    all_records.extend(json.load(open(jf, encoding="utf-8")))

json.dump(all_records, open(a.out_prefix + ".json", "w", encoding="utf-8"),
           ensure_ascii=False, indent=2)

columns = ["text", "dump", "url", "date", "file_path", "keywords_found"]
with open(a.out_prefix + ".csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    for r in all_records:
        row = dict(r)
        row["keywords_found"] = json.dumps(row["keywords_found"], ensure_ascii=False)
        w.writerow(row)

print(f"merged {len(json_files)} shard json files -> {len(all_records)} total records")
print(f"-> {a.out_prefix}.json")
print(f"-> {a.out_prefix}.csv")
