#!/usr/bin/env python3
"""
filter_cyber_parquet_tiered.py - the v3 TIERED filter for PARQUET FineWeb2
dumps (the full arb_Arab release: 25 x ~5 GB .parquet files, ~112 GB).

Identical decision logic to filter_cyber_arabic_tiered.py - same tier files,
same acceptance rule, same output schema, same .done resumability. The ONLY
difference is the reader: parquet is columnar, so instead of streaming
gzipped JSON lines we stream Arrow record batches.

Three things this buys at 112 GB scale:

 1. COLUMN PRUNING - only the 5 columns we actually need are read off disk
    (text, dump, url, date, file_path). FineWeb2 parquet also carries
    language_score, minhash signatures, top_langs, etc.; never touching them
    saves a large fraction of the I/O.
 2. BOUNDED MEMORY - records stream in batches (--batch-size rows), so a
    5 GB file is never materialised in RAM.
 3. SUB-FILE PARALLELISM - with only ~25 files, one-task-per-file would
    under-use the cluster and make a failure cost a whole 5 GB file.
    --parts N / --part I splits each file by ROW GROUP (task I takes row
    groups where idx % N == I), so 25 files x 4 parts = 100 tasks, matching
    the old array width, with finer-grained resume.

ACCEPT a record iff ANY of:
    >= 1 tier-1 hit | >= 2 hits of tier<=2 | >= 3 of tier<=3 | >= 4 tiered hits

Usage (one array task):
  python filter_cyber_parquet_tiered.py \
      --input-file  .../arb_Arab/train/000_00000.parquet \
      --parts 4 --part 0 \
      --tiers-dir   ~/fanar-data/cyber_filtered/keywords/tiers_clean \
      --out-prefix  ~/fanar-data/cyber_filtered/output_v4/cyber_v4_000_00000_p0

Inspect a file's schema / row count without filtering:
  python filter_cyber_parquet_tiered.py --inspect --input-file <file.parquet>
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("error: pyarrow is required for parquet input.\n"
             "       conda install -c conda-forge pyarrow   (or: pip install pyarrow)")

TASHKEEL_RE = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭࣔ-ࣣ࣡-ࣿ]"
)
TIER_FILES = {t: f"tier{t}_keywords.txt" for t in (1, 2, 3, 4)}
WANT_COLS = ["text", "dump", "url", "date", "file_path"]


def strip_tashkeel(s: str) -> str:
    return TASHKEEL_RE.sub("", s)


def read_keyword_file(path: str, do_strip: bool) -> list[str]:
    kws = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if not kw or kw.startswith("#"):
                continue
            kws.append(strip_tashkeel(kw) if do_strip else kw)
    return kws


def load_tiers(tiers_dir: str, extra_path: str | None, do_strip: bool) -> dict:
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
          f"(+{len(tier_of) - n_tiered} annotate-only) from {tiers_dir}", flush=True)
    return tier_of


def build_pattern(keywords) -> re.Pattern:
    ordered = sorted(set(keywords), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in ordered) + r")\b",
                      re.UNICODE | re.IGNORECASE)


def canonicalize(hit: str, canon: dict, keywords: list) -> str | None:
    """Map a matched substring back to the exact lexicon keyword.

    Usually a simple .lower() lookup. But re.IGNORECASE folds some Unicode
    pairs that .lower() does NOT invert - notably Turkish dotless i: the
    pattern 'Hacking' matches the text 'hackıng' (because 'ı'.upper() == 'I'),
    yet 'hackıng'.lower() is still 'hackıng', so the lookup misses. Returning
    the raw text there would break the invariant that every hit is a lexicon
    key, which is what crashed a task with KeyError: 'hackıng'.

    On a miss, recover the keyword by case-insensitive comparison. Misses are
    very rare, so the linear scan costs nothing in practice.
    """
    kw = canon.get(hit.lower())
    if kw is not None:
        return kw
    for k in keywords:
        if len(k) == len(hit) and re.fullmatch(re.escape(k), hit, re.IGNORECASE | re.UNICODE):
            return k
    return None          # unmappable - ignore this hit rather than crash


def accept(tiers_hit) -> str | None:
    if any(t == 1 for t in tiers_hit):
        return "tier1_hit"
    if sum(1 for t in tiers_hit if t <= 2) >= 2:
        return "two_tier2_hits"
    if sum(1 for t in tiers_hit if t <= 3) >= 3:
        return "three_tier3_hits"
    if len(tiers_hit) >= 4:
        return "four_plus_hits"
    return None


def pick_columns(pf: pq.ParquetFile) -> list[str]:
    """Read only the columns we need AND that exist in this file."""
    have = set(pf.schema_arrow.names)
    cols = [c for c in WANT_COLS if c in have]
    if "text" not in cols:
        sys.exit(f"error: no 'text' column in file; available: {sorted(have)}")
    missing = [c for c in WANT_COLS if c not in have]
    if missing:
        print(f"  note: columns absent from this file (emitted empty): {missing}", flush=True)
    return cols


def inspect(path: str):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"file        : {path}")
    print(f"rows        : {md.num_rows:,}")
    print(f"row groups  : {md.num_row_groups}")
    print(f"size on disk: {os.path.getsize(path)/2**30:.2f} GiB")
    print(f"columns     : {pf.schema_arrow.names}")
    rg = md.row_group(0)
    print(f"row group 0 : {rg.num_rows:,} rows, "
          f"{rg.total_byte_size/2**20:.0f} MiB uncompressed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file")
    ap.add_argument("--input-dir")
    ap.add_argument("--tiers-dir")
    ap.add_argument("--extra-keywords", default=None)
    ap.add_argument("--out-prefix")
    ap.add_argument("--parts", type=int, default=1,
                    help="split this file into N row-group slices")
    ap.add_argument("--part", type=int, default=0,
                    help="which slice (0..parts-1) this task handles")
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="rows per Arrow batch (memory knob)")
    ap.add_argument("--strip-tashkeel", action="store_true")
    ap.add_argument("--no-done-marker", action="store_true")
    ap.add_argument("--no-csv", action="store_true",
                    help="skip the per-shard CSV. The merge reads only *.json, so "
                         "at v4 scale these CSVs are ~5 GB written and never read; "
                         "the merge can regenerate one from the final corpus (--csv).")
    ap.add_argument("--inspect", action="store_true",
                    help="print schema/row counts and exit (no filtering)")
    ap.add_argument("--progress-every", type=int, default=200_000,
                    help="log a heartbeat every N records")
    args = ap.parse_args()

    if args.inspect:
        if not args.input_file:
            sys.exit("error: --inspect needs --input-file")
        inspect(args.input_file)
        return

    for req in ("tiers_dir", "out_prefix"):
        if not getattr(args, req):
            sys.exit(f"error: --{req.replace('_','-')} is required")

    done_path = args.out_prefix + ".done"
    if not args.no_done_marker and os.path.isfile(done_path):
        print(f"done marker exists ({done_path}) -> already filtered, skipping.")
        return

    tier_of = load_tiers(args.tiers_dir, args.extra_keywords, args.strip_tashkeel)
    keywords = list(tier_of)
    pattern = build_pattern(keywords)
    canon = {kw.lower(): kw for kw in keywords}

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
        if not files:
            sys.exit(f"error: no *.parquet in {args.input_dir}")
    else:
        sys.exit("error: pass --input-file or --input-dir")

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path, csv_path = args.out_prefix + ".json", args.out_prefix + ".csv"
    columns = ["text", "dump", "url", "date", "file_path",
               "keywords_found", "tiers_found", "accept_rule"]

    total = matched_any = accepted = 0
    rule_counts: dict[str, int] = {}
    t0 = time.time()

    json_out = open(json_path + ".tmp", "w", encoding="utf-8")
    json_out.write("[\n")
    first = True
    csv_out = writer = None
    if not args.no_csv:
        csv_out = open(csv_path + ".tmp", "w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(csv_out, fieldnames=columns)
        writer.writeheader()

    for path in files:
        pf = pq.ParquetFile(path)
        cols = pick_columns(pf)
        n_rg = pf.metadata.num_row_groups
        # CONTIGUOUS slices, not interleaved: each task reads one sequential
        # region of the file. Row groups here are uniform (1,000 rows), so
        # contiguous balances just as well as round-robin while giving
        # sequential I/O - which matters when PARTS tasks read the same
        # multi-GB file concurrently off a shared filesystem.
        per, rem = divmod(n_rg, args.parts)
        start = args.part * per + min(args.part, rem)
        count = per + (1 if args.part < rem else 0)
        mine = list(range(start, start + count))
        print(f"[{os.path.basename(path)}] {pf.metadata.num_rows:,} rows, "
              f"{n_rg} row groups -> this task takes {len(mine)} "
              f"(row groups {start}..{start + count - 1}, "
              f"part {args.part}/{args.parts})", flush=True)
        if not mine:
            continue

        for batch in pf.iter_batches(batch_size=args.batch_size,
                                     row_groups=mine, columns=cols):
            d = batch.to_pydict()
            texts = d["text"]
            for i, text in enumerate(texts):
                total += 1
                if not text:
                    continue
                hay = strip_tashkeel(text) if args.strip_tashkeel else text
                raw = pattern.findall(hay)
                if not raw:
                    continue
                mapped = {canonicalize(h, canon, keywords) for h in raw}
                mapped.discard(None)
                if not mapped:
                    continue
                hits = sorted(mapped)
                matched_any += 1
                tiers = [tier_of[h] for h in hits]
                rule = accept([t for t in tiers if isinstance(t, int)])
                if rule is None:
                    continue
                accepted += 1
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
                row = {c: (d[c][i] if c in d else "") for c in WANT_COLS}
                row["keywords_found"] = hits
                row["tiers_found"] = tiers
                row["accept_rule"] = rule
                if not first:
                    json_out.write(",\n")
                json_out.write(json.dumps(row, ensure_ascii=False))
                first = False
                if writer:
                    cr = dict(row)
                    cr["keywords_found"] = json.dumps(hits, ensure_ascii=False)
                    cr["tiers_found"] = json.dumps(tiers, ensure_ascii=False)
                    writer.writerow(cr)

                if total % args.progress_every == 0:
                    el = time.time() - t0
                    print(f"  ... {total:,} scanned | {accepted:,} accepted | "
                          f"{total/el:,.0f} rec/s | {el/60:.1f} min", flush=True)

    json_out.write("\n]\n")
    json_out.close()
    os.replace(json_path + ".tmp", json_path)
    if csv_out:
        csv_out.close()
        os.replace(csv_path + ".tmp", csv_path)
    if not args.no_done_marker:
        with open(done_path, "w") as f:
            f.write(f"records={total} matched_any={matched_any} accepted={accepted}\n")

    el = time.time() - t0
    print(f"\ndone in {el/60:.1f} min. scanned {total:,} records.")
    print(f"matched >=1 keyword : {matched_any:,}")
    print(f"accepted (tiered)   : {accepted:,} "
          f"({100.0*accepted/matched_any if matched_any else 0:.1f}% of matched)")
    print(f"acceptance clauses  : {rule_counts}")
    print(f"json -> {json_path}")
    if csv_out is not None:
        print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
