#!/usr/bin/env python3
"""
judge_with_gemma.py - run a LOCAL open model (gemma-4-31B-it) as the corpus
judge, using exactly the rubric the frontier model used
(evaluation/JUDGE_PROTOCOL.md).

Purpose: answer "can an open 31B model replace a frontier model for corpus
curation in a low-resource language?" - by judging records whose frontier-model
labels we already hold, then measuring agreement.

Engine: vLLM (batched, GPU). Output is constrained to a JSON schema via guided
decoding when available, with a tolerant parser as a second line of defence -
a single malformed generation must never kill a slice.

Input : JSONL of {"id", "nk", "text"}   (NO gold labels - the run is blind)
Output: JSONL of {"id", "judge_label", "judge_confidence",
                  "judge_justification", "judge_summary", "raw"?}

Parallelism / resume: identical pattern to the filter jobs - --parts/--part
take a contiguous slice, outputs are written .tmp then renamed, and a .done
marker makes re-submission skip finished slices.

Usage (one array task):
  python judge_with_gemma.py \
      --input   ~/fanar-data/judge_bench/validation_prompts.jsonl \
      --model   ~/models/gemma-4-31B-it \
      --parts 4 --part 0 \
      --out-prefix ~/fanar-data/judge_bench/out/gemma_val_P4_p0

Smoke test (5 records, prints them):
  python judge_with_gemma.py --input ... --model ... --limit 5 --verbose \
      --out-prefix /tmp/smoke
"""
import argparse, json, os, re, sys, time

LABELS = ("cyber", "borderline", "not_cyber")
CONFS = ("high", "medium", "low")

SYSTEM = """You are judging whether Arabic web documents are substantively about CYBERSECURITY.

"Cybersecurity" covers: cyber attacks, malware (viruses, trojans, ransomware, spyware), \
hacking/hackers, data breaches, vulnerabilities & exploits, phishing/scams/social \
engineering, network/system security, encryption & privacy protection, digital forensics, \
cybercrime & cyber law, cyber warfare, security products (antivirus, firewalls, VPNs), \
security awareness/training, account/device protection guides.

Assign exactly one label:
- "cyber": cybersecurity is a MAIN topic of the document, or a substantial section of it \
(a security news story, a malware removal guide, an infosec explainer, cyber-attack \
coverage, a how-to-protect-yourself article, a security product review).
- "borderline": cybersecurity is genuinely present but tangential or minor - one paragraph \
in a general news roundup, a passing mention of hacking in a tech/politics article, general \
IT content with light security elements.
- "not_cyber": the document is NOT about cybersecurity. The keyword match was a false \
positive: metaphorical use ("Trojan horse" in politics, a literal back door), a different \
sense, boilerplate privacy-policy/terms text, or a spam/aggregator/product page whose \
content is not security material.

Judge the document by its SUBSTANCE, not by the presence of individual words. A consumer \
VPN or antivirus PRODUCT/download page is "not_cyber" unless it actually discusses security \
subject matter. The text may be truncated; judge what you can see.

Reply with ONLY a JSON object, no other text:
{"label": "cyber|borderline|not_cyber", "confidence": "high|medium|low", \
"justification": "one English sentence citing concrete evidence in the text", \
"summary": "one English sentence stating what the document is about"}"""

USER_TMPL = "Document:\n\"\"\"\n{text}\n\"\"\"\n\nJSON:"

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "string", "enum": list(CONFS)},
        "justification": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["label", "confidence", "justification", "summary"],
}


def slice_of(n, parts, part):
    per, rem = divmod(n, parts)
    start = part * per + min(part, rem)
    count = per + (1 if part < rem else 0)
    return start, start + count


def parse_verdict(text: str) -> dict | None:
    """Tolerant parser: guided decoding should make this unnecessary, but a
    single bad generation must not lose the record."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)          # first JSON-looking span
    if m:
        try:
            o = json.loads(m.group(0))
            lab = str(o.get("label", "")).strip().lower()
            if lab in LABELS:
                conf = str(o.get("confidence", "")).strip().lower()
                return {"judge_label": lab,
                        "judge_confidence": conf if conf in CONFS else "medium",
                        "judge_justification": str(o.get("justification", ""))[:600],
                        "judge_summary": str(o.get("summary", ""))[:600]}
        except json.JSONDecodeError:
            pass
    # last resort: a bare label somewhere in the text
    low = text.lower()
    for lab in ("not_cyber", "borderline", "cyber"):   # longest first
        if lab in low:
            return {"judge_label": lab, "judge_confidence": "low",
                    "judge_justification": "", "judge_summary": ""}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL of {id, nk, text}")
    ap.add_argument("--model", required=True, help="path to the local model")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N records")
    ap.add_argument("--few-shot", default=None,
                    help="JSON list of {label,text,justification} exemplars, "
                         "injected as prior chat turns. Fixes the zero-shot "
                         "collapse of `borderline` into `cyber`.")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="8192 by default: few-shot exemplars add ~2k tokens")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="GPUs per task (1 = one model per GPU, best throughput "
                         "for a 31B on H200)")
    ap.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto",
                    help="auto = try vLLM, fall back to transformers. Use "
                         "transformers when the cluster driver is older than the "
                         "CUDA the vLLM wheels were built for.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="transformers backend only: prompts per forward pass")
    ap.add_argument("--no-done-marker", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    done_path = args.out_prefix + ".done"
    if not args.no_done_marker and os.path.isfile(done_path):
        print(f"done marker exists ({done_path}) -> slice already judged, skipping.")
        return

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    lo, hi = slice_of(len(rows), args.parts, args.part)
    rows = rows[lo:hi]
    print(f"input {args.input}: taking records {lo}..{hi - 1} "
          f"(part {args.part}/{args.parts}) = {len(rows)} records", flush=True)
    if not rows:
        print("nothing to do for this slice.")
        return

    # few-shot exemplars become prior user/assistant turns, so the model sees
    # the task DEMONSTRATED (in its own output format) rather than only described
    shots = []
    if args.few_shot:
        ex = json.load(open(args.few_shot, encoding="utf-8"))
        for e in ex:
            shots.append({"role": "user",
                          "content": USER_TMPL.format(text=e["text"])})
            shots.append({"role": "assistant",
                          "content": json.dumps(
                              {"label": e["label"], "confidence": "high",
                               "justification": e.get("justification", ""),
                               "summary": e.get("summary", "")},
                              ensure_ascii=False)})
        import collections as _c
        print(f"few-shot: {len(ex)} exemplars "
              f"({dict(_c.Counter(e['label'] for e in ex))})", flush=True)

    def build_prompts(tok):
        out = []
        for r in rows:
            msgs = [{"role": "system", "content": SYSTEM}] + shots + \
                   [{"role": "user", "content": USER_TMPL.format(text=r["text"])}]
            try:
                p = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            except Exception:                  # model without a chat template
                # keep the exemplars: dropping them here would silently make a
                # "few-shot" run zero-shot and invalidate the comparison
                flat = "".join(f"{m['content']}\n\n" for m in shots)
                p = SYSTEM + "\n\n" + flat + USER_TMPL.format(text=r["text"])
            out.append(p)
        return out

    # ---------------------------------------------------------------- vLLM
    def run_vllm():
        from vllm import LLM, SamplingParams
        print(f"[vllm] loading model: {args.model}", flush=True)
        t0 = time.time()
        llm = LLM(model=args.model, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util,
                  tensor_parallel_size=args.tensor_parallel,
                  trust_remote_code=True)
        print(f"[vllm] model loaded in {time.time() - t0:.0f}s", flush=True)
        sp_kwargs = dict(temperature=0.0, max_tokens=args.max_tokens)
        try:
            from vllm.sampling_params import GuidedDecodingParams
            sp_kwargs["guided_decoding"] = GuidedDecodingParams(json=JSON_SCHEMA)
            print("[vllm] guided JSON decoding: ENABLED", flush=True)
        except Exception as e:
            print(f"[vllm] guided JSON unavailable ({e}); tolerant parser", flush=True)
        prompts = build_prompts(llm.get_tokenizer())
        print(f"[vllm] generating {len(prompts)} verdicts ...", flush=True)
        t1 = time.time()
        outs = llm.generate(prompts, SamplingParams(**sp_kwargs))
        el = time.time() - t1
        print(f"[vllm] done in {el/60:.1f} min ({len(prompts)/el:.1f} rec/s)", flush=True)
        return [o.outputs[0].text.strip() for o in outs]

    # -------------------------------------------------------- transformers
    def run_transformers():
        """Fallback that works with the SAME cu12x torch the OCR pipeline used.
        Slower than vLLM but has no CUDA-13 wheel requirement."""
        import torch
        from transformers import AutoTokenizer
        print(f"[hf] torch {torch.__version__} | cuda avail "
              f"{torch.cuda.is_available()}", flush=True)
        if not torch.cuda.is_available():
            sys.exit("error: CUDA not available to torch - check the driver/wheel match")

        # is_available() can return True even when CUDA cannot actually
        # initialise (driver older than the wheel's CUDA). Prove the GPU works
        # with a real op - otherwise device_map='auto' silently puts a 31B model
        # on the CPU and 'generation' runs for hours with no output.
        try:
            _ = (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
            print(f"[hf] GPU op OK on {torch.cuda.get_device_name(0)}", flush=True)
        except Exception as e:
            sys.exit(f"error: torch cannot USE the GPU ({type(e).__name__}: {e})\n"
                     f"       torch {torch.__version__} vs this node's driver.\n"
                     f"       Install a torch built for the driver's CUDA, e.g.:\n"
                     f"         pip install --no-user torch==2.6.0 "
                     f"--index-url https://download.pytorch.org/whl/cu124")
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"              # required for batched decoder gen

        model = None
        errs = []
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                import transformers as tf
                cls = getattr(tf, cls_name)
                print(f"[hf] loading with {cls_name} ...", flush=True)
                t0 = time.time()
                model = cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                            device_map="auto", trust_remote_code=True)
                print(f"[hf] loaded in {time.time() - t0:.0f}s via {cls_name}", flush=True)
                break
            except Exception as e:
                errs.append(f"{cls_name}: {type(e).__name__}: {e}")
        if model is None:
            sys.exit("error: could not load the model.\n  " + "\n  ".join(errs))
        model.eval()

        # Refuse to generate if any weights landed on cpu/disk: a 31B model
        # doing that is ~1000x slower and looks exactly like a hang.
        dmap = getattr(model, "hf_device_map", None)
        if dmap:
            offloaded = {k: v for k, v in dmap.items() if str(v) in ("cpu", "disk")}
            print(f"[hf] device map: {len(dmap)} blocks, "
                  f"{len(offloaded)} on cpu/disk", flush=True)
            if offloaded:
                sys.exit(f"error: {len(offloaded)} module(s) offloaded to cpu/disk "
                         f"(e.g. {list(offloaded)[:3]}).\n"
                         f"       Generation would take hours. Give the task more GPU "
                         f"memory (--gres=gpu:2 with --tensor-parallel 2), or use a "
                         f"smaller model.")
        else:
            dev = next(model.parameters()).device
            print(f"[hf] weights on: {dev}", flush=True)
            if dev.type != "cuda":
                sys.exit(f"error: model is on {dev}, not cuda - generation would "
                         f"take hours. Fix the torch/driver mismatch first.")

        prompts = build_prompts(tok)
        raws, bs = [], max(1, args.batch_size)
        t1 = time.time()
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_model_len).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                new_tokens = gen[j][enc["input_ids"].shape[1]:]
                raws.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
            done = min(i + bs, len(prompts))
            el = time.time() - t1
            print(f"[hf]   {done}/{len(prompts)} | {done/el:.2f} rec/s | "
                  f"{el/60:.1f} min", flush=True)
        return raws

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "transformers"
        print(f"backend auto-selected: {backend}", flush=True)

    if backend == "vllm":
        try:
            raws = run_vllm()
        except Exception as e:
            print(f"\n!! vLLM failed ({type(e).__name__}: {e})", flush=True)
            print("!! falling back to the transformers backend\n", flush=True)
            raws = run_transformers()
    else:
        raws = run_transformers()

    out_path = args.out_prefix + ".jsonl"
    n_ok = n_bad = 0
    with open(out_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for r, raw in zip(rows, raws):
            v = parse_verdict(raw)
            if v is None:
                n_bad += 1
                v = {"judge_label": None, "judge_confidence": None,
                     "judge_justification": "", "judge_summary": ""}
                v["raw"] = raw[:400]
            else:
                n_ok += 1
            rec = {"id": r["id"], "nk": r.get("nk"), **v}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.verbose:
                print(f"  {r['id']} -> {v['judge_label']} ({v['judge_confidence']}): "
                      f"{v['judge_justification'][:90]}")
    os.replace(out_path + ".tmp", out_path)
    if not args.no_done_marker:
        with open(done_path, "w") as f:
            f.write(f"records={len(rows)} parsed={n_ok} unparsed={n_bad}\n")

    print(f"\nparsed OK : {n_ok}\nunparsed  : {n_bad}")
    print(f"jsonl -> {out_path}")


if __name__ == "__main__":
    main()
