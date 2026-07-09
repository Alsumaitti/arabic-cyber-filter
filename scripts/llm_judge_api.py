#!/usr/bin/env python3
"""
llm_judge_api.py - LLM-as-judge verification layer for the FULL filtered corpus.

Sends every record of the keyword-filtered corpus (cyber_arabic_ALL.json) to
Claude via the Message Batches API (50% of standard price) and asks a single
question: "is this document substantively about cybersecurity?"

Each record gains four new columns:
    judge_label          cyber | borderline | not_cyber
    judge_confidence     high | medium | low
    judge_justification  one English sentence citing evidence from the text
    judge_summary        one English sentence stating what the document is about

Output:
    <out-prefix>.json    original columns + the four judge columns
    <out-prefix>.csv     same, UTF-8 with BOM (Excel-safe)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...      # or `ant auth login`
    python scripts/llm_judge_api.py \
        --input result/cyber_arabic_ALL.json \
        --out-prefix result/cyber_arabic_ALL_judged

    # Smoke test on the first 20 records first:
    python scripts/llm_judge_api.py --input ... --out-prefix ... --limit 20

Notes:
    - Requires: pip install anthropic
    - Uses claude-opus-4-8 with structured outputs (JSON schema enforced).
    - Texts are truncated to --max-chars (default 6000) before judging; that is
      more than enough to classify topic and keeps cost bounded.
    - Batches usually finish within an hour; the script polls until done and
      is resumable per chunk (already-judged chunks are cached on disk).
    - Rough cost at defaults: ~13k docs x ~2-3k input tokens at Opus batch
      pricing ($2.50/MTok input, $12.50/MTok output) ~= USD 80-110 total.
"""
import argparse
import csv
import json
import os
import sys
import time

import anthropic

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are an expert annotator verifying an Arabic cybersecurity text corpus.
Each user message contains one Arabic web document (possibly truncated).
Decide whether the document is SUBSTANTIVELY about cybersecurity.

Cybersecurity covers: cyber attacks, malware (viruses, trojans, ransomware,
spyware...), hacking/hackers, data breaches, vulnerabilities and exploits,
phishing/scams/social engineering, network and system security, encryption and
privacy protection, digital forensics, cybercrime and cyber law, cyber warfare,
security products (antivirus, firewalls, VPNs), and protect-yourself guides.

Labels:
- "cyber": cybersecurity is a main topic or a substantial section of the document.
- "borderline": cybersecurity content is genuinely present but tangential/minor
  (one paragraph in a general news roundup, a passing mention).
- "not_cyber": the document is not about cybersecurity; the keyword match was a
  false positive (different sense, metaphor, physical/military security, spam page).

Confidence: "high" | "medium" | "low" ("low" = truncation left real doubt).
Write judge_justification and judge_summary in English, one sentence each.
The justification must cite the concrete evidence in the text."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "judge_label": {"type": "string", "enum": ["cyber", "borderline", "not_cyber"]},
        "judge_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "judge_justification": {"type": "string"},
        "judge_summary": {"type": "string"},
    },
    "required": ["judge_label", "judge_confidence", "judge_justification", "judge_summary"],
    "additionalProperties": False,
}


def build_request(idx: int, text: str, max_chars: int) -> dict:
    doc = text[:max_chars]
    if len(text) > max_chars:
        doc += "\n[... document truncated ...]"
    return {
        "custom_id": f"rec-{idx}",
        "params": {
            "model": MODEL,
            "max_tokens": 500,
            "system": [{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            "messages": [{"role": "user", "content": doc}],
        },
    }


def run_chunk(client, requests, cache_path):
    """Submit one batch chunk and poll until it ends; returns {custom_id: verdict}."""
    if os.path.exists(cache_path):
        print(f"  chunk cached -> {cache_path}")
        return json.load(open(cache_path, encoding="utf-8"))

    batch = client.messages.batches.create(requests=requests)
    print(f"  submitted batch {batch.id} ({len(requests)} requests)")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"    ...processing (done {c.succeeded + c.errored}/{len(requests)})")
        time.sleep(60)

    verdicts = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                verdicts[result.custom_id] = json.loads(text)
            except json.JSONDecodeError:
                verdicts[result.custom_id] = {"error": "unparseable", "raw": text[:500]}
        else:
            verdicts[result.custom_id] = {"error": result.result.type}

    json.dump(verdicts, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="cyber_arabic_ALL.json")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N records (smoke test)")
    args = ap.parse_args()

    client = anthropic.Anthropic()

    records = json.load(open(args.input, encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} records to judge with {MODEL}")

    cache_dir = args.out_prefix + "_batch_cache"
    os.makedirs(cache_dir, exist_ok=True)

    all_verdicts = {}
    for start in range(0, len(records), args.chunk_size):
        chunk = records[start:start + args.chunk_size]
        print(f"chunk {start}-{start + len(chunk) - 1}:")
        requests = [build_request(start + i, r["text"], args.max_chars)
                    for i, r in enumerate(chunk)]
        cache_path = os.path.join(cache_dir, f"chunk_{start}.json")
        all_verdicts.update(run_chunk(client, requests, cache_path))

    judged = []
    errors = 0
    for i, rec in enumerate(records):
        v = all_verdicts.get(f"rec-{i}", {"error": "missing"})
        row = dict(rec)
        if "error" in v:
            errors += 1
            row.update(judge_label="ERROR", judge_confidence="",
                       judge_justification=v.get("error", ""), judge_summary="")
        else:
            row.update(v)
        judged.append(row)

    json_path = args.out_prefix + ".json"
    json.dump(judged, open(json_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    csv_path = args.out_prefix + ".csv"
    columns = ["text", "dump", "url", "date", "file_path", "keywords_found",
               "judge_label", "judge_confidence", "judge_justification", "judge_summary"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in judged:
            out = {k: row.get(k, "") for k in columns}
            out["keywords_found"] = json.dumps(row.get("keywords_found", []),
                                               ensure_ascii=False)
            w.writerow(out)

    labels = {}
    for row in judged:
        labels[row["judge_label"]] = labels.get(row["judge_label"], 0) + 1
    print(f"\ndone. {len(judged)} records judged, {errors} errors.")
    print(f"label distribution: {labels}")
    print(f"json -> {json_path}\ncsv  -> {csv_path}")


if __name__ == "__main__":
    main()
