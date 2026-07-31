#!/usr/bin/env python3
"""
bench_judge_throughput.py - find the fastest batch size before committing days
of GPU time to the full corpus.

WHY
The full v4 run is ~370k records. At the measured 0.63 rec/s (batch 8) that is
163 GPU-hours, i.e. ~3.4 days on 2 GPUs. Batch size is the single biggest lever:
a 31B in bf16 is ~62GB of a 141GB H200, so batch 8 left most of the KV cache
idle. Twenty minutes spent measuring here can remove a day or two from the run.

The model is loaded ONCE and each batch size is timed against the same records,
so the comparison is clean and the whole sweep costs one model load rather than
one per setting.

OOM is expected at the top of the sweep and is not a failure - the script traps
it, reports the last size that worked, and moves on.

Usage (inside a job - the login node kills python):
    python scripts/bench_judge_throughput.py \
        --input ~/fanar-data/judge_bench/v4_full_prompts.jsonl \
        --model ~/models/gemma-4-31B-it \
        --sizes 8,16,32,48,64 --per-size 96
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="16384,24576,32768,40960,49152",
                    help="TOKEN BUDGETS to sweep (batch_size x longest prompt), "
                         "which is what the real run actually controls - not a "
                         "fixed batch count.")
    ap.add_argument("--max-batch", type=int, default=64,
                    help="upper bound on prompts per batch regardless of budget")
    ap.add_argument("--per-size", type=int, default=96,
                    help="records timed per setting")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--total-records", type=int, default=370019,
                    help="corpus size, for projecting the full run")
    ap.add_argument("--gpus", type=int, default=2)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    path = os.path.expanduser(args.input)
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
            if len(recs) >= max(sizes) * 8:
                break
    if not recs:
        sys.exit(f"error: no records in {path}")
    print(f"loaded {len(recs):,} sample records from {os.path.basename(path)}")

    model_path = os.path.expanduser(args.model)
    print(f"loading {model_path} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s on {torch.cuda.get_device_name(0)}",
          flush=True)

    # THE REAL PROMPT, not a bare text message. Without the SYSTEM rubric the
    # model has no JSON to emit, so it generates the full max_new_tokens every
    # time instead of stopping at EOS after a short verdict - which understates
    # throughput badly and makes the whole projection wrong.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from judge_with_gemma import SYSTEM, USER_TMPL

    def build(rs):
        out = []
        for r in rs:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(text=r["text"][:6000])}]
            out.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        return out

    all_prompts = build(recs[: args.per_size])
    lens = [len(tok(p, truncation=True,
                    max_length=args.max_model_len)["input_ids"]) for p in all_prompts]
    order = sorted(range(len(all_prompts)), key=lambda i: lens[i])
    print(f"prompt tokens: min {min(lens)} / median {sorted(lens)[len(lens)//2]} "
          f"/ max {max(lens)}", flush=True)

    results = []
    for budget in sizes:
        # Form batches exactly as the real run does.
        batches, cur = [], []
        for k in order:
            trial = cur + [k]
            if cur and (len(trial) * lens[k] > budget or len(trial) > args.max_batch):
                batches.append(cur)
                cur = [k]
            else:
                cur = trial
        if cur:
            batches.append(cur)
        avg_bs = len(all_prompts) / len(batches)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t1 = time.time()
            for idxs in batches:
                batch = [all_prompts[k] for k in idxs]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_model_len).to(model.device)
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=args.max_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
                del enc
            el = time.time() - t1
            rate = len(all_prompts) / el
            peak = torch.cuda.max_memory_allocated() / 1e9
            gpu_h = args.total_records / rate / 3600
            results.append((budget, rate, peak, gpu_h))
            print(f"  budget {budget:>6,}: {rate:5.2f} rec/s | avg batch "
                  f"{avg_bs:4.1f} | peak {peak:5.1f} GB | full run {gpu_h:5.0f} "
                  f"GPU-h = {gpu_h/args.gpus/24:.1f} days", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  budget {budget:>6,}: OOM - the real run would split and "
                  f"recover here, but slowly. Use a lower budget.", flush=True)
            torch.cuda.empty_cache()
            break

    if not results:
        sys.exit("error: every batch size failed")

    best = max(results, key=lambda r: r[1])
    # Back off one step from whatever peaked highest: the fastest setting is
    # usually the one closest to OOM, and paying for OOM splits mid-run costs
    # more than the margin saves. If the best IS the lowest budget, keep it.
    safe = best
    for b, rate, peak, gh in results:
        if b < best[0] and rate >= best[1] * 0.95:
            safe = (b, rate, peak, gh)
            break

    print(f"\n=== FASTEST: budget {best[0]:,} at {best[1]:.2f} rec/s "
          f"(peak {best[2]:.1f} GB) ===")
    if safe[0] != best[0]:
        print(f"=== RECOMMENDED: budget {safe[0]:,} at {safe[1]:.2f} rec/s "
              f"(peak {safe[2]:.1f} GB) - within 5% and further from OOM ===")
    print(f"  full {args.total_records:,}-record run: {safe[3]:.0f} GPU-hours "
          f"= {safe[3]/args.gpus/24:.1f} days on {args.gpus} GPUs")
    slice_max = 8 * 3600 * safe[1]
    parts = int(args.total_records / slice_max) + 1
    parts = max(parts + 2, 4)
    print(f"  max records per 8h slice: {slice_max:,.0f}")
    print(f"  -> PARTS >= {args.total_records/slice_max:.1f}, use PARTS={parts} for margin")
    print(f"\nsbatch --export=ALL,PARTS={parts},TOKEN_BUDGET={safe[0]} "
          f"--array=0-{parts-1} sbatch_scripts/judge_v4_full.sh")


if __name__ == "__main__":
    main()
