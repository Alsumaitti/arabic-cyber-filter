#!/usr/bin/env python3
"""
mine_missing_keywords.py - find cybersecurity terms present in the missed-record
pool (scratch/pool_texts.jsonl) that are NOT already in the keyword lexicon.

Pipeline:
  1. Load the 1,209-term lexicon; build a normalized dedup set (tashkeel stripped,
     alef/ya unified, definite article "ال" stripped per word) so a candidate is
     only "new" if NO form of it already exists.
  2. For each pool document, tokenize, generate 1-4-word n-grams that contain a
     cybersecurity lexeme (a curated set of Arabic/English cyber morphemes), and
     trim leading/trailing stopwords.
  3. Count DOCUMENT frequency per normalized candidate (once per doc), track the
     most common raw surface form.
  4. Drop candidates already covered by the lexicon; keep df >= MIN_DF; rank.

Output: evaluation/results/candidate_terms.tsv   (df, surface_form, normalized)
        -> reviewed by hand into the final missing-keywords list.
"""
import collections, json, os, re, sys, unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(EVAL)
LEX_PATH = os.path.join(ROOT, "keywords", "arabic_cyber_keywords.txt")
POOL_PATH = os.path.join(EVAL, "scratch", "pool_texts.jsonl")
RESULTS_DIR = os.path.join(EVAL, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_DF = 15          # keep candidates seen in at least this many pool docs
MAX_WORDS = 4        # longest candidate phrase
TOP_N = 700          # cap the review file

# ---------------------------------------------------------------- normalization

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))   # drop tashkeel
    s = s.replace("ـ", "").replace("ٱ", "ا").replace("ى", "ي")
    return s.strip().lower()

def strip_al(tok):
    return tok[2:] if (len(tok) > 4 and tok.startswith("ال")) else tok

def dedup_keys(phrase):
    """All normalized forms of a phrase used for lexicon matching."""
    n = norm(phrase)
    toks = n.split()
    core = " ".join(strip_al(t) for t in toks)
    return {n, core}

# ------------------------------------------------------------------- cyber gate
# A token/phrase is a cyber candidate if it contains one of these normalized
# morphemes. Broad on purpose (recall of candidates); precision comes from the
# hand-review of the ranked output.
CYBER_LEXEMES = [
    "سيبران", "تشفير", "مشفر", "اختراق", "مخترق", "خترق", "هاكر", "هكرز", "هكر",
    "قرصن", "قرصان", "قراصن", "فيروس", "فايروس", "برمجيات خبيث", "برمجية خبيث",
    "برنامج خبيث", "برامج خبيث", "كود خبيث", "شيفرة خبيث", "شفرة خبيث", "ملف خبيث",
    "ملفات خبيث", "رابط خبيث", "موقع خبيث", "مواقع خبيث", "تطبيق خبيث", "خبيث",
    "ضار", "تجسس", "جاسوس", "تصيد", "فيشين", "ثغر", "فدي", "رانسوم", "طروادة",
    "دودة", "ديدان", "بوت نت", "بوتنت", "بوت‌نت", "جدار الحماي", "جدار ناري",
    "جدار حماي", "فايروول", "مصادق", "كلمة المرور", "كلمة السر", "كلمات المرور",
    "كلمات السر", "رمز المرور", "رمز التحقق", "الرقم السري", "رقم سري", "تسريب",
    "تسرب", "هجوم الكترون", "هجمات الكترون", "هجوم سيبران", "هجمات سيبران",
    "احتيال الكترون", "نصب الكترون", "ابتزاز الكترون", "ابتزاز الالكترون",
    "بريد مزعج", "بريد عشوائ", "رسائل مزعج", "سبام", "مكافحة الفيروس",
    "مضاد الفيروس", "مضاد فيروس", "انتي فيروس", "الباب الخلف", "باب خلف",
    "برامج التجسس", "برنامج تجسس", "برمجيات تجسس", "برامج تجسس", "سرقة البيانات",
    "سرقة بيانات", "سرقة المعلومات", "سرقة الهوي", "انتحال الهوي", "انتحال الشخص",
    "الوصول غير المصرح", "دخول غير مصرح", "وصول غير مصرح", "شبكة افتراض",
    "امن المعلومات", "امن الشبك", "امن سيبران", "امن رقمي", "امن البيانات",
    "امن الحاسوب", "امن الكترون", "امن المعلوماتي", "جريمة الكترون",
    "جرائم الكترون", "جريمة معلوماتي", "جرائم معلوماتي", "جريمة سيبران",
    "جرائم سيبران", "جريمة الحاسوب", "جرائم الحاسوب", "مصادقة ثنائي",
    "تحقق بخطوتين", "التحقق بخطوتين", "مصادقة متعدد", "تشفير من طرف",
    "حصان طروادة", "برمجية الفدي", "برامج الفدي", "فيروس الفدي", "بيانات الاعتماد",
    "حقن الشيفر", "حقن اس كيو", "استغلال الثغر", "اختبار الاختراق", "الهندسة الاجتماع",
    "متصفح تور", "شبكة تور", "دارك ويب", "الويب المظلم", "الانترنت المظلم",
    "الويب العميق", "روت كيت", "كي لوغر", "كيلوغر", "راصد لوحة المفاتيح",
    "بصمة رقمي", "توقيع رقمي", "شهادة رقمي", "شهادة امان", "طبقة المقابس",
    # latin / abbreviations (already lowercased by norm)
    "vpn", "ssl", "tls", "ddos", "botnet", "phishing", "malware", "ransomware",
    "spyware", "firewall", "exploit", "hacker", "hacking", "backdoor",
    "keylogger", "rootkit", "trojan", "antivirus", "cyber", "encryption",
]
CYBER_RE = re.compile("|".join(re.escape(x) for x in CYBER_LEXEMES))

STOP = set()   # populated (normalized) in main()
STOP_RAW = """في من على الى إلى عن مع هذا هذه ذلك تلك التي الذي الذين اللذين ما لا
و أو او ثم قد كما بين حول عبر عند لدى كان كانت هو هي هم أن إن انه أنه إلي لكن بل
كل بعض غير عبر او إذا اذا حتى منذ نحو ضد قبل بعد فوق تحت أي اي عن كي لكي إذ اذ يتم
تم كذلك ايضا أيضا هناك حيث عليه عليها له لها به بها منها منه فيها فيه هذا التى"""

TOKEN_RE = re.compile(r"[A-Za-z؀-ۿ]+")

def tokens_of(text):
    raw = TOKEN_RE.findall(text)
    return raw, [norm(t) for t in raw]

def main():
    if not os.path.exists(POOL_PATH):
        sys.exit(f"missing {POOL_PATH} - run collect_pool_texts.py first")

    STOP.update(norm(w) for w in STOP_RAW.split())

    lex_forms = set()
    n_lex = 0
    for line in open(LEX_PATH, encoding="utf-8"):
        kw = line.strip()
        if not kw or kw.startswith("#"):
            continue
        n_lex += 1
        lex_forms |= dedup_keys(kw)
    print(f"lexicon: {n_lex} entries -> {len(lex_forms)} normalized forms", file=sys.stderr)

    df = collections.Counter()          # normalized candidate -> document frequency
    surface = collections.defaultdict(collections.Counter)  # norm -> raw form counts
    n_docs = 0
    for line in open(POOL_PATH, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_docs += 1
        raw, ntok = tokens_of(rec.get("text", ""))
        is_cyber = [bool(CYBER_RE.search(t)) for t in ntok]
        seen = {}                        # norm phrase -> raw phrase (first in doc)
        L = len(ntok)
        for n in range(1, MAX_WORDS + 1):
            for i in range(L - n + 1):
                if not any(is_cyber[i:i + n]):
                    continue
                # trim leading/trailing stopwords
                a, b = i, i + n
                while a < b and ntok[a] in STOP:
                    a += 1
                while b > a and ntok[b - 1] in STOP:
                    b -= 1
                if b - a == 0 or not any(is_cyber[a:b]):
                    continue
                nphrase = " ".join(ntok[a:b])
                if len(nphrase) < 4 or len(nphrase) > 45:
                    continue
                if nphrase not in seen:
                    seen[nphrase] = " ".join(raw[a:b])
        for nphrase, rawphrase in seen.items():
            df[nphrase] += 1
            surface[nphrase][rawphrase] += 1
        if n_docs % 2000 == 0:
            print(f"  ...{n_docs} docs, {len(df)} raw candidates", file=sys.stderr)

    # keep candidates not already covered by the lexicon
    kept = {}   # nphrase -> freq
    for nphrase, freq in df.items():
        if freq < MIN_DF:
            continue
        core = " ".join(strip_al(t) for t in nphrase.split())
        if nphrase in lex_forms or core in lex_forms:
            continue
        kept[nphrase] = freq

    # collapse Arabic proclitic variants: drop a phrase whose first word carries a
    # leading و/ف/ب/ك/ل (or لل) IF stripping it yields a form that is already a
    # kept candidate or a lexicon entry (safe: only removes true prefixed dups,
    # never mangles words like "بريد" whose base "ريد" exists nowhere).
    def declitic_variants(nphrase):
        toks = nphrase.split()
        first = toks[0]
        outs = []
        for pre in ("لل", "و", "ف", "ب", "ك", "ل"):
            if first.startswith(pre) and len(first) > len(pre) + 2:
                cand = " ".join([first[len(pre):]] + toks[1:])
                outs.append(cand)
                outs.append(" ".join(strip_al(t) for t in cand.split()))
        return outs

    rows = []
    for nphrase, freq in kept.items():
        variants = declitic_variants(nphrase)
        # also collapse a definite-article form onto its bare candidate
        core = " ".join(strip_al(t) for t in nphrase.split())
        if core != nphrase:
            variants.append(core)
        if any(v in kept or v in lex_forms for v in variants):
            continue
        best_raw = surface[nphrase].most_common(1)[0][0]
        rows.append((freq, best_raw, nphrase))
    rows.sort(key=lambda r: (-r[0], r[2]))
    rows = rows[:TOP_N]

    out = os.path.join(RESULTS_DIR, "candidate_terms.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("df\tsurface_form\tnormalized\n")
        for freq, raw, nph in rows:
            f.write(f"{freq}\t{raw}\t{nph}\n")
    print(f"scanned {n_docs} pool docs; wrote {len(rows)} candidates (df>={MIN_DF}) -> {out}")


if __name__ == "__main__":
    main()
