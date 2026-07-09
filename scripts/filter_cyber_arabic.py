#!/usr/bin/env python3
"""
filter_cyber_arabic.py
-----------------------
Scan every arb_Arab_*.jsonl.gz file inside a FineWeb2 folder, flag every
record whose "text" field contains one or more Arabic cybersecurity
keywords/phrases (loaded from a plain-text file, one keyword per line),
and write the matching records to BOTH a .json and a .csv file.

Kept columns (from the source jsonl) + 1 extra column:
    text, dump, url, date, file_path, keywords_found

"keywords_found" is a JSON array of the exact keyword phrases that were
found in that record's text - this is the evidence trail used to justify
why a record was classified as cybersecurity-related.

Matching notes:
  - Keywords are matched as whole tokens (not sub-string false positives),
    e.g. the keyword "أمن" will NOT match inside "الأمنية" - Arabic word
    boundaries are handled by treating any Arabic letter run as one word
    (regex \b relies on \w which, in Python 3 unicode mode, already
    includes Arabic letters, so \b works correctly here).
  - Matching is case-irrelevant (Arabic has no case) and is done on the
    raw text without diacritics stripping (add --strip-tashkeel if your
    keyword list/text mixes diacritized and non-diacritized forms).
  - Multi-word keyword phrases (e.g. "اختراق الشبكات") are matched as a
    literal phrase in normal reading order.
  - All keywords are compiled into ONE regex (alternation) for speed,
    since FineWeb2 shards can contain millions of records.

Usage (batch mode - scan every .gz file in a folder):
  python filter_cyber_arabic.py \
      --input-dir ~/fanar-data/FineWeb2 \
      --keywords ~/arabic_cyber_keywords.txt \
      --out-prefix ~/fanar-data/cyber_filtered/cyber_arabic

Usage (single-shard mode - used by the SLURM array job):
  python filter_cyber_arabic.py \
      --input-file ~/fanar-data/FineWeb2/some_shard.jsonl.gz \
      --keywords ~/arabic_cyber_keywords.txt \
      --out-prefix ~/fanar-data/cyber_filtered/output/cyber_arabic_some_shard

  # produces:
  #   <out-prefix>.json
  #   <out-prefix>.csv

Optional:
  --strip-tashkeel   strip Arabic diacritics from BOTH text and keywords
                      before matching (helps if sources are inconsistently
                      diacritized)
  --min-keywords N   only keep records with at least N distinct keyword
                      hits (default 1)
  --limit-files N    only process the first N .gz files (batch mode only,
                      useful for a quick test run before scanning the
                      whole folder)
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys
import unicodedata

TASHKEEL_RE = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D4-\u08E1\u08E3-\u08FF]"
)


def strip_tashkeel(s: str) -> str:
    return TASHKEEL_RE.sub("", s)


def load_keywords(path: str, do_strip: bool) -> list[str]:
    kws = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if not kw or kw.startswith("#"):
                continue
            if do_strip:
                kw = strip_tashkeel(kw)
            kws.append(kw)
    # de-dup while preserving order, longest-first so multi-word phrases
    # are checked before their shorter sub-phrases match inside them
    seen = set()
    uniq = []
    for kw in sorted(kws, key=len, reverse=True):
        if kw not in seen:
            seen.add(kw)
            uniq.append(kw)
    return uniq


def build_pattern(keywords: list[str]) -> re.Pattern:
    escaped = [re.escape(kw) for kw in keywords]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.UNICODE)


def find_keywords(text: str, keywords: list[str], pattern: re.Pattern) -> list[str]:
    hits = pattern.findall(text)
    # findall on a non-grouped alternation returns full matches directly
    # since the whole alternation IS the single group here (no nested
    # groups inside each escaped keyword), so hits are the matched strings.
    return sorted(set(hits), key=lambda h: keywords.index(h) if h in keywords else 0)


def iter_records(gz_path: str):
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ! skip malformed line {lineno} in {os.path.basename(gz_path)}: {e}",
                      file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir")
    ap.add_argument("--input-file")
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--strip-tashkeel", action="store_true")
    ap.add_argument("--min-keywords", type=int, default=1)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()

    keywords = load_keywords(args.keywords, args.strip_tashkeel)
    if not keywords:
        sys.exit(f"error: no keywords loaded from {args.keywords}")
    pattern = build_pattern(keywords)
    print(f"loaded {len(keywords)} keywords, e.g.: {keywords[:5]}")

    if args.input_file:
        if not os.path.isfile(args.input_file):
            sys.exit(f"error: --input-file not found: {args.input_file}")
        gz_files = [args.input_file]
    elif args.input_dir:
        gz_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl.gz")))
        if args.limit_files:
            gz_files = gz_files[: args.limit_files]
        if not gz_files:
            sys.exit(f"error: no *.jsonl.gz files found in {args.input_dir}")
    else:
        sys.exit("error: pass either --input-dir (batch mode) or --input-file (single-shard mode)")
    print(f"found {len(gz_files)} .gz shard(s) to scan")

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path = args.out_prefix + ".json"
    csv_path = args.out_prefix + ".csv"

    columns = ["text", "dump", "url", "date", "file_path", "keywords_found"]

    total_records = 0
    total_matched = 0

    json_out = open(json_path, "w", encoding="utf-8")
    json_out.write("[\n")
    first_json_row = True

    csv_out = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_out, fieldnames=columns)
    writer.writeheader()

    for gi, gz_path in enumerate(gz_files, 1):
        print(f"[{gi}/{len(gz_files)}] scanning {os.path.basename(gz_path)} ...")
        file_matched = 0
        for rec in iter_records(gz_path):
            total_records += 1
            text = rec.get("text", "")
            if args.strip_tashkeel:
                text = strip_tashkeel(text)

            hits = find_keywords(text, keywords, pattern)
            if len(hits) < args.min_keywords:
                continue

            total_matched += 1
            file_matched += 1

            row = {
                "text": rec.get("text", ""),
                "dump": rec.get("dump", ""),
                "url": rec.get("url", ""),
                "date": rec.get("date", ""),
                "file_path": rec.get("file_path", ""),
                "keywords_found": hits,
            }

            if not first_json_row:
                json_out.write(",\n")
            json_out.write(json.dumps(row, ensure_ascii=False))
            first_json_row = False

            csv_row = dict(row)
            csv_row["keywords_found"] = json.dumps(hits, ensure_ascii=False)
            writer.writerow(csv_row)

        print(f"    -> {file_matched} matching record(s) in this shard")

    json_out.write("\n]\n")
    json_out.close()
    csv_out.close()

    print(f"\ndone. scanned {total_records} records total across "
          f"{len(gz_files)} shard(s).")
    print(f"matched {total_matched} cybersecurity-related record(s) "
          f"(min {args.min_keywords} keyword hit(s) each).")
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
