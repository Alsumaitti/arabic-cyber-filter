#!/usr/bin/env python3
"""
verify_no_overlap.py - confirm a proposed keyword file shares NO entry with the
existing lexicon, under both exact and normalized (tashkeel/alef/ya/definite-
article-insensitive) comparison.

Usage:
  python verify_no_overlap.py <new_keywords.txt>
  (defaults to keywords/missing_cybersecurity_keywords.txt)

Exit code 0 and "OK" if disjoint; 1 and a list of collisions otherwise.
"""
import os, re, sys, unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LEX_PATH = os.path.join(ROOT, "keywords", "arabic_cyber_keywords.txt")


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))   # drop tashkeel
    s = s.replace("ـ", "").replace("ٱ", "ا").replace("ى", "ي")
    return s.strip().lower()

def strip_al(tok):
    return tok[2:] if (len(tok) > 4 and tok.startswith("ال")) else tok

def forms(kw):
    n = norm(kw)
    core = " ".join(strip_al(t) for t in n.split())
    return {n, core}

def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        w = line.strip()
        if w and not w.startswith("#"):
            out.append(w)
    return out

def main():
    new_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "keywords", "missing_cybersecurity_keywords.txt")

    lex = load(LEX_PATH)
    lex_forms = set()
    for kw in lex:
        lex_forms |= forms(kw)

    new = load(new_path)
    seen_new, dup_within, collisions = set(), [], []
    for kw in new:
        fs = forms(kw)
        if fs & lex_forms:
            collisions.append(kw)
        key = norm(kw)
        if key in seen_new:
            dup_within.append(kw)
        seen_new.add(key)

    print(f"lexicon entries      : {len(lex)}")
    print(f"proposed new keywords : {len(new)}  (unique normalized: {len(seen_new)})")
    print(f"collisions with lexicon: {len(collisions)}")
    print(f"duplicates within new : {len(dup_within)}")
    if collisions:
        print("\nCOLLISIONS (already in lexicon):")
        for c in collisions:
            print("  ", c)
    if dup_within:
        print("\nDUPLICATES within the new file:")
        for d in dup_within:
            print("  ", d)
    if not collisions and not dup_within:
        print("\nOK - the two lists are disjoint and the new file has no internal duplicates.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
