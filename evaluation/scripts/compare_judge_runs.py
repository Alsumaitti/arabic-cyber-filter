#!/usr/bin/env python3
"""
compare_judge_runs.py - put the zero-shot and few-shot gemma judge runs side by
side, and check the few-shot prompt actually reached the model.

That last check matters more than it sounds. The two runs scored almost
identically, so before reporting "prompting does not fix this" we have to rule
out the boring explanation: that the exemplars were silently dropped and we ran
the same prompt twice. The generated justifications settle it - if the model saw
different input, the free text moves even where the label does not.

Usage:
    python evaluation/scripts/compare_judge_runs.py
    python evaluation/scripts/compare_judge_runs.py --md   # markdown table
"""
import argparse, collections, glob, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(SCRIPT_DIR)
BENCH = os.path.join(EVAL, "judge_benchmark")
LABELS = ("cyber", "borderline", "not_cyber")


def load_preds(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot", default=os.path.join(BENCH, "gemma_val_P2_p*.jsonl"))
    ap.add_argument("--few-shot", default=os.path.join(BENCH, "gemma_fs_P2_p*.jsonl"))
    ap.add_argument("--report-zs", default=os.path.join(BENCH, "agreement_report.json"))
    ap.add_argument("--report-fs", default=os.path.join(BENCH, "agreement_report_fewshot.json"))
    ap.add_argument("--gold", default=os.path.join(BENCH, "validation_gold.jsonl"))
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    args = ap.parse_args()

    zs, fs = load_preds(args.zero_shot), load_preds(args.few_shot)
    if not zs or not fs:
        sys.exit("error: need both runs; missing "
                 + ("zero-shot " if not zs else "") + ("few-shot" if not fs else ""))
    ids = [i for i in zs if i in fs]
    print(f"records in both runs : {len(ids):,}  "
          f"(zero-shot {len(zs):,} / few-shot {len(fs):,})")

    # --- did the intervention land? -----------------------------------------
    lab_diff = [i for i in ids if zs[i]["judge_label"] != fs[i]["judge_label"]]
    txt_diff = [i for i in ids
                if (zs[i].get("judge_justification") or "")
                != (fs[i].get("judge_justification") or "")]
    print(f"justification changed: {len(txt_diff):,} "
          f"({100*len(txt_diff)/len(ids):.1f}%)  <- proves the prompt differed")
    print(f"label changed        : {len(lab_diff):,} "
          f"({100*len(lab_diff)/len(ids):.1f}%)  <- the effect of 11 exemplars")
    if txt_diff and len(lab_diff) / len(ids) < 0.02:
        print("  => intervention applied, effect negligible: a CAPABILITY limit,\n"
              "     not a prompting limit.")
    elif not txt_diff:
        print("  => WARNING: identical output text. The exemplars may not have been\n"
              "     applied at all - check the run log for the few-shot line.")

    if lab_diff:
        print("\nlabel flips (zero-shot -> few-shot):")
        flips = collections.Counter(
            (zs[i]["judge_label"], fs[i]["judge_label"]) for i in lab_diff)
        for (a, b), n in flips.most_common():
            print(f"  {a:<11} -> {b:<11} {n}")

    print("\npredicted label distribution:")
    gold_dist = None
    if os.path.isfile(args.gold):
        g = {json.loads(l)["id"]: json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()}
        gold_dist = collections.Counter(g[i]["gold_label"] for i in ids if i in g)
    for name, d in (("gold (frontier)", gold_dist),
                    ("zero-shot", collections.Counter(zs[i]["judge_label"] for i in ids)),
                    ("few-shot", collections.Counter(fs[i]["judge_label"] for i in ids))):
        if d is None:
            continue
        cells = "  ".join(f"{l}={d.get(l, 0):>5}" for l in LABELS)
        print(f"  {name:<16} {cells}")

    # --- metric table -------------------------------------------------------
    if not (os.path.isfile(args.report_zs) and os.path.isfile(args.report_fs)):
        print("\n(skipping metric table - run score_judge_agreement.py for both runs)")
        return
    a = json.load(open(args.report_zs, encoding="utf-8"))
    b = json.load(open(args.report_fs, encoding="utf-8"))

    rows = [
        ("raw agreement (3-way)", a["raw_agreement"], b["raw_agreement"]),
        ("corpus-weighted", a["weighted_agreement"], b["weighted_agreement"]),
        ("Cohen kappa (3-way)", a["kappa_3way"], b["kappa_3way"]),
        ("binary agreement", a["binary_agreement"], b["binary_agreement"]),
        ("Cohen kappa (binary)", a["kappa_binary"], b["kappa_binary"]),
    ]
    for lab in LABELS:
        for m in ("precision", "recall"):
            rows.append((f"{lab} {m}", a["per_class"][lab][m], b["per_class"][lab][m]))

    print()
    if args.md:
        print("| metric | zero-shot | few-shot | delta |")
        print("|---|---|---|---|")
        for name, x, y in rows:
            print(f"| {name} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")
    else:
        print(f"{'metric':<26}{'zero-shot':>10}{'few-shot':>10}{'delta':>9}")
        print("-" * 55)
        for name, x, y in rows:
            print(f"{name:<26}{x:>10.3f}{y:>10.3f}{y-x:>+9.3f}")

    # --- what the binary numbers buy you -----------------------------------
    sens = b["per_class"]["cyber"]  # placeholder to keep key access explicit
    tp = b["confusion"]["cyber->cyber"] + b["confusion"]["cyber->borderline"] \
        + b["confusion"]["borderline->cyber"] + b["confusion"]["borderline->borderline"]
    n_pos = b["per_class"]["cyber"]["support"] + b["per_class"]["borderline"]["support"]
    tn = b["confusion"]["not_cyber->not_cyber"]
    n_neg = b["per_class"]["not_cyber"]["support"]
    sensitivity, specificity = tp / n_pos, tn / n_neg
    print(f"\nbinary operating point (few-shot run):")
    print(f"  sensitivity (cyber-ish recall) : {sensitivity:.4f}")
    print(f"  specificity (not_cyber recall) : {specificity:.4f}")
    print(f"  Rogan-Gladen: p_true = (p_obs - {1-specificity:.4f}) / "
          f"{sensitivity - (1 - specificity):.4f}")
    print("  -> feed these to estimate_v4_precision.py")


if __name__ == "__main__":
    main()
