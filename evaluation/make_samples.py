#!/usr/bin/env python3
"""
make_samples.py - build the two evaluation samples for the LLM-as-judge pass.

1. sample_matched.json   : 300 records stratified-sampled from the 13,345 matched
                           records (strata by number of distinct keyword hits:
                           1 / 2 / 3-5 / 6+), with a truncated "judge view".
2. sample_unmatched.json : records sampled from the ~986k rejected records:
                           - "pool"   : 120 records that hit a broad safety-net regex
                                        (candidate false negatives, TREC-style pooling)
                           - "random" : 80 records outside the safety net
                           Also reports the pool size for extrapolation.

Deterministic: random.seed(42).
"""
import glob, gzip, json, os, random, re, sys

random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
FW_DIR = os.path.join(ROOT, "FineWeb2")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

VIEW_HEAD = 700          # chars from the start of the doc
VIEW_WIN = 400           # chars around the first keyword hit if it is beyond the head

def make_view(text, first_hit_pos=None):
    head = text[:VIEW_HEAD]
    if first_hit_pos is not None and first_hit_pos > VIEW_HEAD:
        a = max(0, first_hit_pos - VIEW_WIN // 2)
        win = text[a:a + VIEW_WIN]
        return head + "\n[...]\n" + win
    return head

# --------------------------------------------------------------------------
# 1. stratified sample of matched records
# --------------------------------------------------------------------------
def stratum_of(n):
    if n == 1: return "1"
    if n == 2: return "2"
    if n <= 5: return "3-5"
    return "6+"

STRATUM_SIZES = {"1": 100, "2": 80, "3-5": 60, "6+": 60}

by_stratum = {s: [] for s in STRATUM_SIZES}
shard_files = sorted(f for f in glob.glob(os.path.join(OUT_DIR, "cyber_arabic_*.json"))
                     if not f.endswith("_ALL.json"))
for sf in shard_files:
    shard = os.path.basename(sf)[len("cyber_arabic_"):-len(".json")]
    for i, rec in enumerate(json.load(open(sf, encoding="utf-8"))):
        rec["rec_id"] = f"{shard}#{i}"
        by_stratum[stratum_of(len(rec["keywords_found"]))].append(rec)

matched_sample = []
for s, size in STRATUM_SIZES.items():
    pop = by_stratum[s]
    picks = random.sample(pop, min(size, len(pop)))
    for rec in picks:
        text = rec["text"]
        # locate first keyword occurrence for the context window
        first_pos = min((text.find(kw) for kw in rec["keywords_found"]
                         if text.find(kw) != -1), default=None)
        matched_sample.append({
            "rec_id": rec["rec_id"],
            "stratum": s,
            "url": rec["url"],
            "keywords_found": rec["keywords_found"],
            "text_len": len(text),
            "judge_view": make_view(text, first_pos),
        })

random.shuffle(matched_sample)  # judge in random order, not stratum order
json.dump(matched_sample,
          open(os.path.join(EVAL_DIR, "sample_matched.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"matched sample: {len(matched_sample)} records "
      f"({ {s: len(v) for s, v in by_stratum.items()} } population)")

# --------------------------------------------------------------------------
# 2. unmatched records: safety-net pool + pure random
# --------------------------------------------------------------------------
# Identity of matched records (url+date is unique enough within this corpus)
matched_keys = set()
for s, recs in by_stratum.items():
    for r in recs:
        matched_keys.add((r["url"], r["date"]))

# Broad safety net: loose Arabic + English cyber signals NOT requiring the strict lexicon
BROAD = re.compile(
    r"فيروسات الحاسوب|فيروس إلكتروني|الفيروسات الإلكترونية|اختراق|قرصنة|تجسس إلكتروني|"
    r"هجمات إلكترونية|هجوم إلكتروني|أمن رقمي|الأمن الرقمي|حماية البيانات|سرقة البيانات|"
    r"تسريب بيانات|تسريب البيانات|كلمات المرور|كلمة المرور|جرائم المعلوماتية|"
    r"الجرائم المعلوماتية|أمن الشبكات|حماية إلكترونية|\bhack(?:er|ing|ed)?\b|\bmalware\b|"
    r"\bransomware\b|\bphishing\b|\bspyware\b|\bfirewall\b|\bcyber\s?security\b|"
    r"\bcyber\s?attack\b|\bantivirus\b|\bvpn\b|\bexploit\b|\bbotnet\b|\bddos\b",
    re.IGNORECASE)

pool = []        # reservoir for broad-net hits
randoms = []     # reservoir for everything else
POOL_K, RAND_K = 120, 80
pool_seen = rand_seen = 0

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
                continue  # this record was matched by the strict filter
            text = rec.get("text", "")
            m = BROAD.search(text)
            item = {
                "rec_id": f"{shard}#{i}",
                "url": rec.get("url", ""),
                "text_len": len(text),
            }
            if m:
                pool_seen += 1
                item["group"] = "pool"
                item["broad_hit"] = m.group(0)
                item["judge_view"] = make_view(text, m.start())
                if len(pool) < POOL_K:
                    pool.append(item)
                else:
                    j = random.randrange(pool_seen)
                    if j < POOL_K:
                        pool[j] = item
            else:
                rand_seen += 1
                item["group"] = "random"
                item["judge_view"] = text[:500]
                if len(randoms) < RAND_K:
                    randoms.append(item)
                else:
                    j = random.randrange(rand_seen)
                    if j < RAND_K:
                        randoms[j] = item
    print(f"  [{gi+1}/{len(gz_files)}] {shard}: pool_seen={pool_seen} rand_seen={rand_seen}",
          file=sys.stderr)

unmatched_sample = pool + randoms
random.shuffle(unmatched_sample)
json.dump(unmatched_sample,
          open(os.path.join(EVAL_DIR, "sample_unmatched.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

counts = {
    "matched_total": 13345,
    "strata_population": {s: len(v) for s, v in by_stratum.items()},
    "strata_sampled": {s: min(sz, len(by_stratum[s])) for s, sz in STRATUM_SIZES.items()},
    "unmatched_total": pool_seen + rand_seen,
    "unmatched_pool_size": pool_seen,
    "unmatched_random_size": rand_seen,
    "pool_sampled": len(pool),
    "random_sampled": len(randoms),
}
json.dump(counts, open(os.path.join(EVAL_DIR, "population_counts.json"), "w",
                       encoding="utf-8"), indent=2)
print(json.dumps(counts, indent=2))
