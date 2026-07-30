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
    ap.add_argument("--sizes", default="8,16,32,48,64")
    ap.add_argument("--per-size", type=int, default=96,
                    help="records timed per batch size (rounded up to whole batches)")
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

    # Same shape of prompt the real run uses, so the timing transfers.
    def build(rs):
        out = []
        for r in rs:
            msgs = [{"role": "user", "content": r["text"][:6000]}]
            out.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
        return out

    results = []
    for bs in sizes:
        n = max(bs, (args.per_size // bs) * bs)
        prompts = build(recs[:n])
        # Length-sort, exactly as the real run does, or the measurement flatters
        # the unsorted case and the projection is wrong.
        prompts.sort(key=len)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t1 = time.time()
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i + bs]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_model_len).to(model.device)
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=args.max_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            el = time.time() - t1
            rate = len(prompts) / el
            peak = torch.cuda.max_memory_allocated() / 1e9
            gpu_h = args.total_records / rate / 3600
            results.append((bs, rate, peak, gpu_h))
            print(f"  batch {bs:>3}: {rate:5.2f} rec/s | peak {peak:5.1f} GB | "
                  f"full run {gpu_h:5.0f} GPU-h = {gpu_h/args.gpus/24:.1f} days "
                  f"on {args.gpus} GPUs", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>3}: OOM - stop here, use the largest size below",
                  flush=True)
            torch.cuda.empty_cache()
            break

    if not results:
        sys.exit("error: every batch size failed")

    best = max(results, key=lambda r: r[1])
    base = results[0]
    print(f"\n=== BEST: batch {best[0]} at {best[1]:.2f} rec/s ===")
    print(f"  speedup vs batch {base[0]}: {best[1]/base[1]:.2f}x")
    print(f"  full {args.total_records:,}-record run: {best[3]:.0f} GPU-hours "
          f"= {best[3]/args.gpus/24:.1f} days on {args.gpus} GPUs")
    slice_max = 8 * 3600 * best[1]
    parts = int(args.total_records / slice_max) + 1
    print(f"  max records per 8h slice: {slice_max:,.0f}")
    print(f"  -> PARTS >= {args.total_records/slice_max:.1f}, use PARTS={max(parts+2, 4)} "
          f"for margin")
    print(f"\nsbatch --export=ALL,PARTS={max(parts+2,4)},BATCH={best[0]} "
          f"--array=0-{max(parts+2,4)-1} sbatch_scripts/judge_v4_full.sh")


if __name__ == "__main__":
    main()
