#!/usr/bin/env python3
"""
merge_chat_judgments.py - progress tracker + merger for in-chat corpus judging.

Usage:
  python merge_chat_judgments.py --status
      Print progress: which chunk is next, records judged, label breakdown.
      This is the entry point of every "continue" session.

  python merge_chat_judgments.py --validate
      Check every judgment file: ids must match the chunk exactly, labels and
      confidences must be valid. Reports problems per chunk.

  python merge_chat_judgments.py --merge
      Validate, then join judgments back to the full corpus records and write
        evaluation/chat_judging/judged_corpus.json
      in the same shape build_keyword_tiers.py / compute_metrics expect
      (record + judge_label + judge_confidence + judge_justification).

Judgment file format (written by the assistant, one per chunk):
  evaluation/chat_judging/judgments/chunk_0001.json
  {"chunk": 1, "records": [
      {"id": "<16-hex>", "label": "cyber|borderline|not_cyber",
       "conf": "high|medium|low", "why": "<short reason>"}, ...]}
"""
import argparse, collections, hashlib, json, os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
CJ = os.path.join(EVAL, "chat_judging")
CHUNKS = os.path.join(CJ, "chunks")
JUDG = os.path.join(CJ, "judgments")

LABELS = {"cyber", "borderline", "not_cyber"}
CONFS = {"high", "medium", "low"}


def rec_id(rec):
    h = hashlib.sha1((rec.get("url", "") + "|" + rec.get("date", "")).encode("utf-8"))
    return h.hexdigest()[:16]


def chunk_ids(path):
    ids = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            ids.append(json.loads(line)["id"])
    return ids


def load_judgment(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_chunk(n, problems):
    cpath = os.path.join(CHUNKS, f"chunk_{n:04d}.jsonl")
    jpath = os.path.join(JUDG, f"chunk_{n:04d}.json")
    want = chunk_ids(cpath)
    try:
        j = load_judgment(jpath)
    except (json.JSONDecodeError, OSError) as e:
        problems.append(f"chunk {n:04d}: unreadable judgment file ({e})")
        return None
    got = {}
    for r in j.get("records", []):
        if r.get("label") not in LABELS:
            problems.append(f"chunk {n:04d}: bad label {r.get('label')!r} for {r.get('id')}")
        if r.get("conf") not in CONFS:
            problems.append(f"chunk {n:04d}: bad conf {r.get('conf')!r} for {r.get('id')}")
        got[r.get("id")] = r
    missing = [i for i in want if i not in got]
    extra = [i for i in got if i not in set(want)]
    if missing:
        problems.append(f"chunk {n:04d}: {len(missing)} record(s) not judged: {missing[:5]}")
    if extra:
        problems.append(f"chunk {n:04d}: {len(extra)} unknown id(s): {extra[:5]}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "result", "cyber_arabic_v2_ALL.json"))
    args = ap.parse_args()
    if not (args.status or args.validate or args.merge):
        args.status = True

    manifest = json.load(open(os.path.join(CJ, "manifest.json"), encoding="utf-8"))
    n_chunks = manifest["n_chunks"]
    done = sorted(int(m.group(1)) for fn in os.listdir(JUDG)
                  if (m := re.match(r"chunk_(\d{4})\.json$", fn)))
    pending = [n for n in range(1, n_chunks + 1) if n not in set(done)]

    if args.status:
        judged = 0
        labels = collections.Counter()
        for n in done:
            try:
                j = load_judgment(os.path.join(JUDG, f"chunk_{n:04d}.json"))
                for r in j.get("records", []):
                    judged += 1
                    labels[r.get("label")] += 1
            except (json.JSONDecodeError, OSError):
                pass
        pct = 100.0 * len(done) / n_chunks if n_chunks else 0
        print(f"mode={manifest['mode']}  chunks: {len(done)}/{n_chunks} done ({pct:.1f}%)")
        print(f"records judged: {judged}/{manifest['selected_records']}")
        if labels:
            print("labels:", dict(labels))
        print(f"NEXT CHUNK: {'chunk_%04d.jsonl' % pending[0] if pending else 'ALL DONE'}")
        return

    problems = []
    all_judg = {}
    for n in done:
        got = validate_chunk(n, problems)
        if got:
            all_judg.update(got)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        if args.merge:
            sys.exit("fix problems before merging")
    else:
        print(f"validated {len(done)} chunk(s), {len(all_judg)} judgments - all clean")

    if args.merge:
        with open(args.corpus, encoding="utf-8") as f:
            corpus = json.load(f)
        out, matched = [], 0
        for rec in corpus:
            j = all_judg.get(rec_id(rec))
            if j is None:
                continue
            row = dict(rec)
            kws = row["keywords_found"]
            if isinstance(kws, str):
                row["keywords_found"] = json.loads(kws)
            row["judge_label"] = j["label"]
            row["judge_confidence"] = j["conf"]
            row["judge_justification"] = j.get("why", "")
            out.append(row)
            matched += 1
        dst = os.path.join(CJ, "judged_corpus.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"merged {matched} judged records -> {dst}")


if __name__ == "__main__":
    main()
