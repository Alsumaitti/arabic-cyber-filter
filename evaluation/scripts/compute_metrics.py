#!/usr/bin/env python3
"""
compute_metrics.py - turn the LLM-as-judge verdicts into precision/recall/F1.

Reads:
  evaluation/judgments/matched_*.json   (per-record judge_label for the 300
                                          stratified matched sample)
  evaluation/judgments/unmatched_*.json  (verdicts for the recall pool + random)
  evaluation/results/population_counts.json      (stratum populations + pool sizes)

Computes:
  - Per-stratum precision (strict = cyber only; lenient = cyber+borderline),
    with Wilson 95% CIs.
  - Corpus-weighted precision by combining strata with their real weights
    (9062/2209/1610/464 of 13,345).
  - Recall via TREC-style pooling: false-negative mass estimated from the
    broad-net "pool" hit-rate and the pure-random hit-rate over the ~986k
    rejected records, then Recall = TP / (TP + FN_est).
  - F1 from weighted precision and estimated recall.

Writes evaluation/results/metrics.json and prints a summary.
"""
import glob, json, math, os

EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evaluation/
SAMPLES_DIR = os.path.join(EVAL, "samples")
RESULTS_DIR = os.path.join(EVAL, "results")
MATCHED_TOTAL = 13345
STRATUM_WEIGHTS = {"1": 9062, "2": 2209, "3-5": 1610, "6+": 464}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(EVAL, "judgments", pattern))):
        out.extend(json.load(open(f, encoding="utf-8")))
    return out


def main():
    counts = json.load(open(os.path.join(RESULTS_DIR, "population_counts.json"), encoding="utf-8"))
    matched = load("matched_*.json")
    unmatched = load("unmatched_*.json")

    # ----- precision on the matched sample, per stratum -----
    by_stratum = {}
    id_to_stratum = {r["rec_id"]: r["stratum"]
                     for r in json.load(open(os.path.join(SAMPLES_DIR, "sample_matched.json"),
                                             encoding="utf-8"))}
    for rec in matched:
        s = id_to_stratum.get(rec["rec_id"], "?")
        by_stratum.setdefault(s, []).append(rec["judge_label"])

    strata_metrics = {}
    for s, labels in by_stratum.items():
        n = len(labels)
        cyber = sum(1 for x in labels if x == "cyber")
        border = sum(1 for x in labels if x == "borderline")
        p_strict, lo_s, hi_s = wilson(cyber, n)
        p_len, lo_l, hi_l = wilson(cyber + border, n)
        strata_metrics[s] = {
            "n_judged": n, "cyber": cyber, "borderline": border,
            "not_cyber": n - cyber - border,
            "precision_strict": round(p_strict, 4),
            "precision_strict_ci95": [round(lo_s, 4), round(hi_s, 4)],
            "precision_lenient": round(p_len, 4),
            "precision_lenient_ci95": [round(lo_l, 4), round(hi_l, 4)],
        }

    # corpus-weighted precision (weight each stratum's rate by its true share)
    tot_w = sum(STRATUM_WEIGHTS.values())
    w_strict = sum(strata_metrics[s]["precision_strict"] * STRATUM_WEIGHTS[s]
                   for s in STRATUM_WEIGHTS) / tot_w
    w_len = sum(strata_metrics[s]["precision_lenient"] * STRATUM_WEIGHTS[s]
                for s in STRATUM_WEIGHTS) / tot_w
    # pooled sample-level precision (all judged matched records, unweighted)
    all_labels = [r["judge_label"] for r in matched]
    n_all = len(all_labels)
    pooled_strict = sum(1 for x in all_labels if x == "cyber") / n_all
    pooled_len = sum(1 for x in all_labels if x in ("cyber", "borderline")) / n_all

    # weighted precision as true positives kept in the corpus
    est_true_positives_strict = w_strict * MATCHED_TOTAL
    est_true_positives_lenient = w_len * MATCHED_TOTAL

    # ----- recall via pooling over the ~986k rejected records -----
    pool = [r for r in unmatched if r.get("group") == "pool"]
    rand = [r for r in unmatched if r.get("group") == "random"]

    def cyber_rate(recs):
        n = len(recs)
        k = sum(1 for r in recs if r.get("judge_label") in ("cyber", "borderline"))
        return k, n

    pool_k, pool_n = cyber_rate(pool)
    rand_k, rand_n = cyber_rate(rand)

    pool_size = counts["unmatched_pool_size"]
    rand_size = counts["unmatched_random_size"]

    recall_block = {"note": "unmatched sample not yet judged"}
    if pool_n and rand_n:
        pool_rate = pool_k / pool_n
        rand_rate = rand_k / rand_n
        fn_pool = pool_rate * pool_size
        fn_rand = rand_rate * rand_size
        fn_est = fn_pool + fn_rand
        # true positives kept = lenient weighted precision * corpus
        tp = est_true_positives_lenient
        recall = tp / (tp + fn_est) if (tp + fn_est) else 0.0
        recall_block = {
            "pool_sampled": pool_n, "pool_cyber": pool_k,
            "pool_rate": round(pool_rate, 4), "pool_population": pool_size,
            "random_sampled": rand_n, "random_cyber": rand_k,
            "random_rate": round(rand_rate, 5), "random_population": rand_size,
            "estimated_false_negatives": round(fn_est),
            "estimated_true_positives_kept": round(tp),
            "estimated_recall_lenient": round(recall, 4),
        }
        prec = w_len
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        recall_block["f1_lenient"] = round(f1, 4)

    metrics = {
        "matched_sample_size": n_all,
        "corpus_size": MATCHED_TOTAL,
        "stratum_weights": STRATUM_WEIGHTS,
        "per_stratum": strata_metrics,
        "pooled_precision_strict": round(pooled_strict, 4),
        "pooled_precision_lenient": round(pooled_len, 4),
        "weighted_precision_strict": round(w_strict, 4),
        "weighted_precision_lenient": round(w_len, 4),
        "estimated_true_positives_strict": round(est_true_positives_strict),
        "estimated_true_positives_lenient": round(est_true_positives_lenient),
        "recall": recall_block,
    }
    json.dump(metrics, open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
