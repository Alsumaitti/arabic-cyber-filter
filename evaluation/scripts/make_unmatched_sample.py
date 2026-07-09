#!/usr/bin/env python3
"""
make_unmatched_sample.py - build ONLY the pooled unmatched (recall) sample.

Split out from make_samples.py so the slow shard scan can run on its own. The
broad "safety-net" regex is applied to the first SCAN_CHARS characters of each
record (the topic is determinable there, it matches the window the judge sees,
and it bounds the regex cost so a full-corpus scan finishes in a few minutes).

Deterministic: random.seed(42).
Writes evaluation/samples/sample_unmatched.json and merges pool sizes into
evaluation/results/population_counts.json.
"""
import glob, gzip, json, os, random, re, sys

random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)               # evaluation/
ROOT = os.path.dirname(EVAL)                       # repo root
OUT_DIR = os.path.join(ROOT, "output")
FW_DIR = os.path.join(ROOT, "FineWeb2")
SAMPLES_DIR = os.path.join(EVAL, "samples")
RESULTS_DIR = os.path.join(EVAL, "results")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SCAN_CHARS = 4000
VIEW_HEAD = 700
VIEW_WIN = 400
POOL_K, RAND_K = 120, 80

BROAD = re.compile(
    r"فيروسات الحاسوب|فيروس إلكتروني|الفيروسات الإلكترونية|اختراق|قرصنة|تجسس إلكتروني|"
    r"هجمات إلكترونية|هجوم إلكتروني|أمن رقمي|الأمن الرقمي|حماية البيانات|سرقة البيانات|"
    r"تسريب بيانات|تسريب البيانات|كلمات المرور|كلمة المرور|جرائم المعلوماتية|"
    r"الجرائم المعلوماتية|أمن الشبكات|حماية إلكترونية|\bhack(?:er|ing|ed)?\b|\bmalware\b|"
    r"\bransomware\b|\bphishing\b|\bspyware\b|\bfirewall\b|\bcyber\s?security\b|"
    r"\bcyber\s?attack\b|\bantivirus\b|\bvpn\b|\bexploit\b|\bbotnet\b|\bddos\b",
    re.IGNORECASE)


def make_view(text, first_hit_pos=None):
    head = text[:VIEW_HEAD]
    if first_hit_pos is not None and first_hit_pos > VIEW_HEAD:
        a = max(0, first_hit_pos - VIEW_WIN // 2)
        return head + "\n[...]\n" + text[a:a + VIEW_WIN]
    return head


# identity of matched records, read straight from the per-shard outputs (fast)
matched_keys = set()
for sf in sorted(glob.glob(os.path.join(OUT_DIR, "cyber_arabic_*.json"))):
    if sf.endswith("_ALL.json"):
        continue
    for r in json.load(open(sf, encoding="utf-8")):
        matched_keys.add((r.get("url", ""), r.get("date", "")))
print(f"loaded {len(matched_keys)} matched keys", file=sys.stderr)

pool, randoms = [], []
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
                continue
            text = rec.get("text", "")
            m = BROAD.search(text[:SCAN_CHARS])
            item = {"rec_id": f"{shard}#{i}", "url": rec.get("url", ""),
                    "text_len": len(text)}
            if m:
                pool_seen += 1
                item.update(group="pool", broad_hit=m.group(0),
                            judge_view=make_view(text, m.start()))
                if len(pool) < POOL_K:
                    pool.append(item)
                else:
                    j = random.randrange(pool_seen)
                    if j < POOL_K:
                        pool[j] = item
            else:
                rand_seen += 1
                item.update(group="random", judge_view=text[:500])
                if len(randoms) < RAND_K:
                    randoms.append(item)
                else:
                    j = random.randrange(rand_seen)
                    if j < RAND_K:
                        randoms[j] = item
    print(f"  [{gi+1}/{len(gz_files)}] {shard}: pool={pool_seen} rand={rand_seen}",
          file=sys.stderr)

unmatched = pool + randoms
random.shuffle(unmatched)
json.dump(unmatched, open(os.path.join(SAMPLES_DIR, "sample_unmatched.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

pc_path = os.path.join(RESULTS_DIR, "population_counts.json")
counts = {}
if os.path.exists(pc_path):
    counts = json.load(open(pc_path, encoding="utf-8"))
counts.update({
    "matched_total": 13345,
    "strata_population": {"1": 9062, "2": 2209, "3-5": 1610, "6+": 464},
    "strata_sampled": {"1": 100, "2": 80, "3-5": 60, "6+": 60},
    "unmatched_total": pool_seen + rand_seen,
    "unmatched_pool_size": pool_seen,
    "unmatched_random_size": rand_seen,
    "pool_sampled": len(pool),
    "random_sampled": len(randoms),
    "scan_chars": SCAN_CHARS,
})
json.dump(counts, open(pc_path, "w", encoding="utf-8"), indent=2)
print(json.dumps(counts, indent=2))
