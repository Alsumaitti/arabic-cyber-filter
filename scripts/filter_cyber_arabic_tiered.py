#!/usr/bin/env python3
"""
filter_cyber_arabic_tiered.py  -  the SECOND-GENERATION (v3) FineWeb2 filter.

Same scanning engine as filter_cyber_arabic.py (whole-word Arabic matching,
one compiled alternation regex, evidence trail per record) but the keep/drop
decision is the TIERED ACCEPTANCE RULE learned from the full-corpus LLM census
(evaluation/chat_judging/judged_corpus.json, 14,653 judged records):

    tier 1  self-sufficient        - one hit alone is trustworthy
    tier 2  needs one companion    - trust at >= 2 hits of tier <= 2
    tier 3  needs two companions   - trust at >= 3 hits of tier <= 3
    tier 4  needs heavy company    - only counts toward the >= 4 clause

    ACCEPT a record iff ANY of:
        >= 1 tier-1 hit
        >= 2 distinct hits of tier <= 2
        >= 3 distinct hits of tier <= 3
        >= 4 distinct tiered hits (any tier)

Measured on the census, this rule lifts lenient precision 0.39 -> 0.64 while
keeping 68% of the genuinely-cyber documents - strictly dominating a blind
min-2-keywords threshold on both precision and recall.

Keywords are loaded from FOUR files inside --tiers-dir:
    tier1_keywords.txt ... tier4_keywords.txt
(one keyword per line; blank lines and '#' comment lines are ignored, so both
keywords/tiers/ and the comment-free keywords/tiers_clean/ work).

Optionally, --extra-keywords adds annotate-only keywords (e.g. the rest of
all_cyber_keywords.txt): they are matched and recorded in the evidence trail
with tier "-", but NEVER count toward acceptance - the decision stays exactly
the measured rule.

Output columns (JSON + CSV, superset of the v1 schema):
    text, dump, url, date, file_path,
    keywords_found   - evidence trail (exact keywords matched, deduped)
    tiers_found      - parallel array: tier of each keyword (1..4 or "-")
    accept_rule      - which clause accepted the record:
                       tier1_hit | two_tier2_hits | three_tier3_hits | four_plus_hits

RESUMABILITY (for SLURM re-submission):
    Outputs are written to *.tmp and atomically renamed on success, then a
    <out-prefix>.done marker is written. If the marker already exists the
    script exits 0 immediately - so re-submitting the whole array after ANY
    failure re-does only the unfinished shards.

Usage (single-shard mode - what the SLURM array uses):
  python filter_cyber_arabic_tiered.py \
      --input-file  ~/fanar-data/FineWeb2/<shard>.jsonl.gz \
      --tiers-dir   ~/fanar-data/cyber_filtered/keywords/tiers_clean \
      --out-prefix  ~/fanar-data/cyber_filtered/output_tiered/cyber_v3_<shard>

Usage (batch mode - scan every .gz in a folder on one machine):
  python filter_cyber_arabic_tiered.py \
      --input-dir  ~/fanar-data/FineWeb2 \
      --tiers-dir  ~/fanar-data/cyber_filtered/keywords/tiers_clean \
      --out-prefix ~/fanar-data/cyber_filtered/output_tiered/cyber_v3
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys

TASHKEEL_RE = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D4-\u08E1\u08E3-\u08FF]"
)

TIER_FILES = {t: f"tier{t}_keywords.txt" for t in (1, 2, 3, 4)}


def strip_tashkeel(s: str) -> str:
    return TASHKEEL_RE.sub("", s)


def read_keyword_file(path: str, do_strip: bool) -> list[str]:
    kws = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if not kw or kw.startswith("#"):
                continue
            if do_strip:
                kw = strip_tashkeel(kw)
            kws.append(kw)
    return kws


def load_tiers(tiers_dir: str, extra_path: str | None, do_strip: bool) -> dict[str, object]:
    """Return {keyword: tier} with tier in {1,2,3,4} or '-' for annotate-only.
    If a keyword somehow appears in several tier files, the LOWEST (strongest)
    tier wins."""
    tier_of: dict[str, object] = {}
    for t in (1, 2, 3, 4):
        path = os.path.join(tiers_dir, TIER_FILES[t])
        if not os.path.isfile(path):
            sys.exit(f"error: missing tier file: {path}")
        for kw in read_keyword_file(path, do_strip):
            if kw not in tier_of or (isinstance(tier_of[kw], int) and t < tier_of[kw]):
                tier_of[kw] = t
    n_tiered = len(tier_of)
    if extra_path:
        for kw in read_keyword_file(extra_path, do_strip):
            tier_of.setdefault(kw, "-")
    print(f"loaded {n_tiered} tiered keywords "
          f"(+{len(tier_of) - n_tiered} annotate-only) from {tiers_dir}")
    return tier_of


def build_pattern(keywords: list[str]) -> re.Pattern:
    # longest-first so multi-word phrases win over sub-phrases inside them
    ordered = sorted(set(keywords), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in ordered) + r")\b"
    # IGNORECASE so Latin-script keywords (VPN, Firewall, ...) match all
    # casings; Arabic has no case, so Arabic matching is unaffected.
    return re.compile(pattern, re.UNICODE | re.IGNORECASE)


def accept(tiers_hit: list[int]) -> str | None:
    """tiers_hit = tiers (ints only) of this record's distinct tiered hits.
    Returns the accepting clause name, or None to reject."""
    if any(t == 1 for t in tiers_hit):
        return "tier1_hit"
    if sum(1 for t in tiers_hit if t <= 2) >= 2:
        return "two_tier2_hits"
    if sum(1 for t in tiers_hit if t <= 3) >= 3:
        return "three_tier3_hits"
    if len(tiers_hit) >= 4:
        return "four_plus_hits"
    return None


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
    ap.add_argument("--tiers-dir", required=True,
                    help="directory containing tier1_keywords.txt .. tier4_keywords.txt")
    ap.add_argument("--extra-keywords", default=None,
                    help="optional keyword file matched for evidence only (tier '-', "
                         "never counts toward acceptance)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--strip-tashkeel", action="store_true")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--no-done-marker", action="store_true",
                    help="disable the .done skip/marker mechanism")
    args = ap.parse_args()

    done_path = args.out_prefix + ".done"
    if not args.no_done_marker and os.path.isfile(done_path):
        print(f"done marker exists ({done_path}) -> shard already filtered, skipping.")
        return

    tier_of = load_tiers(args.tiers_dir, args.extra_keywords, args.strip_tashkeel)
    if not tier_of:
        sys.exit("error: no keywords loaded")
    keywords = list(tier_of)
    pattern = build_pattern(keywords)
    canon = {kw.lower(): kw for kw in keywords}

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
    json_tmp = json_path + ".tmp"
    csv_tmp = csv_path + ".tmp"

    columns = ["text", "dump", "url", "date", "file_path",
               "keywords_found", "tiers_found", "accept_rule"]

    total_records = 0
    total_matched_any = 0     # had >= 1 keyword hit (v1 would have kept these)
    total_accepted = 0        # passed the tiered rule (v3 keeps these)
    rule_counts: dict[str, int] = {}

    json_out = open(json_tmp, "w", encoding="utf-8")
    json_out.write("[\n")
    first_json_row = True

    csv_out = open(csv_tmp, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_out, fieldnames=columns)
    writer.writeheader()

    for gi, gz_path in enumerate(gz_files, 1):
        print(f"[{gi}/{len(gz_files)}] scanning {os.path.basename(gz_path)} ...")
        file_accepted = 0
        for rec in iter_records(gz_path):
            total_records += 1
            text = rec.get("text", "")
            if args.strip_tashkeel:
                text = strip_tashkeel(text)

            raw_hits = pattern.findall(text)
            if not raw_hits:
                continue
            # canonicalize case variants (vpn/Vpn/VPN) to lexicon spelling and dedup
            hits = sorted({canon.get(h.lower(), h) for h in raw_hits})
            total_matched_any += 1

            tiers = [tier_of[h] for h in hits]
            tier_ints = [t for t in tiers if isinstance(t, int)]
            rule = accept(tier_ints)
            if rule is None:
                continue

            total_accepted += 1
            file_accepted += 1
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

            row = {
                "text": rec.get("text", ""),
                "dump": rec.get("dump", ""),
                "url": rec.get("url", ""),
                "date": rec.get("date", ""),
                "file_path": rec.get("file_path", ""),
                "keywords_found": hits,
                "tiers_found": tiers,
                "accept_rule": rule,
            }

            if not first_json_row:
                json_out.write(",\n")
            json_out.write(json.dumps(row, ensure_ascii=False))
            first_json_row = False

            csv_row = dict(row)
            csv_row["keywords_found"] = json.dumps(hits, ensure_ascii=False)
            csv_row["tiers_found"] = json.dumps(tiers, ensure_ascii=False)
            writer.writerow(csv_row)

        print(f"    -> {file_accepted} accepted record(s) in this shard")

    json_out.write("\n]\n")
    json_out.close()
    csv_out.close()
    os.replace(json_tmp, json_path)
    os.replace(csv_tmp, csv_path)
    if not args.no_done_marker:
        with open(done_path, "w") as f:
            f.write(f"records={total_records} matched_any={total_matched_any} "
                    f"accepted={total_accepted}\n")

    print(f"\ndone. scanned {total_records} records across {len(gz_files)} shard(s).")
    print(f"matched >=1 keyword : {total_matched_any}  (what the v1 filter would keep)")
    print(f"accepted (tiered)   : {total_accepted}  "
          f"({(100.0 * total_accepted / total_matched_any) if total_matched_any else 0:.1f}% of matched)")
    print(f"acceptance clauses  : {rule_counts}")
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
