#!/usr/bin/env python3
"""
collect_pool_texts.py - collect every rejected record that hits the broad
safety-net regex (the ~15,535 "candidate false-negative" pool), so the missed
cybersecurity vocabulary can be mined from them.

One pass over the FineWeb2 shards; skips records that the strict keyword filter
already matched (identified by url+date, read from the per-shard outputs). Saves
each pool record's capped text to a gitignored scratch file — decoupling the slow
scan from the (fast, tunable) keyword-mining step.

Output: evaluation/scratch/pool_texts.jsonl   (one {rec_id,url,text} per line)
"""
import glob, gzip, json, os, re, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
OUT_DIR = os.path.join(ROOT, "output")
FW_DIR = os.path.join(ROOT, "FineWeb2")
SCRATCH = os.path.join(EVAL, "scratch")
os.makedirs(SCRATCH, exist_ok=True)

SCAN_CHARS = 6000   # cap per record: enough to determine topic, bounds cost/size

# Same broad safety net used for the recall pool (make_unmatched_sample.py).
BROAD = re.compile(
    r"فيروسات الحاسوب|فيروس إلكتروني|الفيروسات الإلكترونية|اختراق|قرصنة|تجسس إلكتروني|"
    r"هجمات إلكترونية|هجوم إلكتروني|أمن رقمي|الأمن الرقمي|حماية البيانات|سرقة البيانات|"
    r"تسريب بيانات|تسريب البيانات|كلمات المرور|كلمة المرور|جرائم المعلوماتية|"
    r"الجرائم المعلوماتية|أمن الشبكات|حماية إلكترونية|\bhack(?:er|ing|ed)?\b|\bmalware\b|"
    r"\bransomware\b|\bphishing\b|\bspyware\b|\bfirewall\b|\bcyber\s?security\b|"
    r"\bcyber\s?attack\b|\bantivirus\b|\bvpn\b|\bexploit\b|\bbotnet\b|\bddos\b",
    re.IGNORECASE)

matched_keys = set()
for sf in sorted(glob.glob(os.path.join(OUT_DIR, "cyber_arabic_*.json"))):
    if sf.endswith("_ALL.json"):
        continue
    for r in json.load(open(sf, encoding="utf-8")):
        matched_keys.add((r.get("url", ""), r.get("date", "")))
print(f"loaded {len(matched_keys)} matched keys", file=sys.stderr)

out = open(os.path.join(SCRATCH, "pool_texts.jsonl"), "w", encoding="utf-8")
pool = 0
gz_files = sorted(glob.glob(os.path.join(FW_DIR, "*.jsonl.gz")))
for gi, gz in enumerate(gz_files):
    shard = os.path.basename(gz)[:-len(".jsonl.gz")]
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("url", ""), rec.get("date", "")) in matched_keys:
                continue
            text = rec.get("text", "")
            if BROAD.search(text[:SCAN_CHARS]):
                pool += 1
                out.write(json.dumps({
                    "rec_id": f"{shard}#{i}",
                    "url": rec.get("url", ""),
                    "text": text[:SCAN_CHARS],
                }, ensure_ascii=False) + "\n")
    print(f"  [{gi+1}/{len(gz_files)}] {shard}: pool={pool}", file=sys.stderr)

out.close()
print(f"done. wrote {pool} pool records to scratch/pool_texts.jsonl")
