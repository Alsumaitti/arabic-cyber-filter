#!/usr/bin/env python3
"""
prepare_chat_judging.py - split the (v2) corpus into small, token-efficient
chunks so an LLM assistant can judge the WHOLE corpus inside a chat session,
resumably, with zero API cost ("Option B without the API").

Design goals
------------
1. RESUMABLE: judging happens one chunk file at a time; a manifest + the
   presence of judgment files under judgments/ record progress. A session can
   stop at any chunk (usage-limit hit) and resume later with "continue".
2. INFORMATION-VALUE ORDER: chunks are ordered so the most informative records
   for keyword tiering come first:
     bucket 1 (single-keyword records)  -> the ONLY source of solo evidence
     bucket 2 (two keywords)            -> pair evidence
     bucket 3 (three keywords)
     bucket 4 (4+ keywords)             -> near-certain cyber, least informative
   Within each bucket, records are ROUND-ROBIN over keywords (every keyword's
   1st record, then every keyword's 2nd, ...) so per-keyword coverage grows
   evenly - if judging stops early, evidence is still balanced, not clumped.
3. TOKEN-EFFICIENT: text truncated to --text-chars (default 1300), compact
   JSON keys, --chunk-size records per chunk (default 25).
4. OPTIONAL CAPS (recommended): per-keyword caps per bucket
   (--cap-solo/--cap-pair/--cap-triple/--cap-heavy). Statistically, ~12 solo
   records per keyword are enough to tier it by majority rule; judging the
   9th..900th solo VPN record adds almost nothing. Caps cut the workload
   roughly in half with near-identical tier quality. Use --mode full to
   disable caps and judge every record.

Outputs
-------
  evaluation/chat_judging/chunks/chunk_0001.jsonl ...   (gitignored)
      one record per line: {"id","nk","kws","text"}
  evaluation/chat_judging/manifest.json
      params, per-bucket counts, chunk count, corpus fingerprint
  judgments/ is created empty; the assistant writes chunk_0001.json ... there.
"""
import argparse, collections, hashlib, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
CJ = os.path.join(EVAL, "chat_judging")


def rec_id(rec):
    h = hashlib.sha1((rec.get("url", "") + "|" + rec.get("date", "")).encode("utf-8"))
    return h.hexdigest()[:16]


def bucket(n):
    return min(n, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "result", "cyber_arabic_v2_ALL.json"),
                    help="merged corpus JSON (list of records)")
    ap.add_argument("--mode", choices=["capped", "full"], default="capped")
    ap.add_argument("--cap-solo", type=int, default=12, help="max 1-keyword records per keyword")
    ap.add_argument("--cap-pair", type=int, default=8, help="max 2-keyword records per keyword")
    ap.add_argument("--cap-triple", type=int, default=5, help="max 3-keyword records per keyword")
    ap.add_argument("--cap-heavy", type=int, default=3, help="max 4+-keyword records per keyword")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--text-chars", type=int, default=1300)
    args = ap.parse_args()

    print(f"loading {args.input} ...", file=sys.stderr)
    with open(args.input, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"corpus: {len(corpus)} records", file=sys.stderr)

    # ---- group by bucket, then round-robin by (rarest-first) keyword --------
    by_bucket = collections.defaultdict(list)
    for rec in corpus:
        kws = rec["keywords_found"]
        if isinstance(kws, str):
            kws = json.loads(kws)
            rec["keywords_found"] = kws
        by_bucket[bucket(len(kws))].append(rec)

    caps = {1: args.cap_solo, 2: args.cap_pair, 3: args.cap_triple, 4: args.cap_heavy}
    ordered = []
    stats = {}
    for b in (1, 2, 3, 4):
        recs = by_bucket.get(b, [])
        # index records under their rarest keyword (fewest records in bucket)
        kw_count = collections.Counter(kw for r in recs for kw in r["keywords_found"])
        per_kw = collections.defaultdict(list)
        for r in recs:
            anchor = min(r["keywords_found"], key=lambda k: kw_count[k])
            per_kw[anchor].append(r)
        cap = caps[b] if args.mode == "capped" else None
        # round-robin: every keyword's 1st record, then every keyword's 2nd...
        taken, rnd = 0, 0
        picked = []
        while True:
            layer = [lst[rnd] for lst in per_kw.values() if len(lst) > rnd]
            if not layer or (cap is not None and rnd >= cap):
                break
            picked.extend(layer)
            taken += len(layer)
            rnd += 1
        ordered.extend((b, r) for r in picked)
        stats[b] = {"total": len(recs), "selected": len(picked), "keywords": len(per_kw)}
        print(f"bucket {b}: {len(recs):6d} records, {len(per_kw):4d} anchor keywords "
              f"-> selected {len(picked):6d}", file=sys.stderr)

    # ---- write chunks --------------------------------------------------------
    chunks_dir = os.path.join(CJ, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(os.path.join(CJ, "judgments"), exist_ok=True)
    # clear stale chunks from a previous prepare
    for fn in os.listdir(chunks_dir):
        os.remove(os.path.join(chunks_dir, fn))

    n_chunks = 0
    for i in range(0, len(ordered), args.chunk_size):
        n_chunks += 1
        path = os.path.join(chunks_dir, f"chunk_{n_chunks:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for b, rec in ordered[i:i + args.chunk_size]:
                row = {
                    "id": rec_id(rec),
                    "nk": len(rec["keywords_found"]),
                    "kws": rec["keywords_found"],
                    "text": rec.get("text", "")[:args.text_chars],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input": os.path.relpath(args.input, ROOT),
        "corpus_records": len(corpus),
        "mode": args.mode,
        "caps": caps if args.mode == "capped" else None,
        "chunk_size": args.chunk_size,
        "text_chars": args.text_chars,
        "selected_records": len(ordered),
        "n_chunks": n_chunks,
        "buckets": stats,
    }
    with open(os.path.join(CJ, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nselected {len(ordered)} records -> {n_chunks} chunks of {args.chunk_size} "
          f"(text capped at {args.text_chars} chars)")
    print(f"chunks:    {chunks_dir}")
    print(f"manifest:  {os.path.join(CJ, 'manifest.json')}")


if __name__ == "__main__":
    main()
