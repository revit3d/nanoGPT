"""Measure real optimizer throughput; never modifies a training checkpoint."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import GPT
from .training import (
    TokenData, optimizer_step, read_config, setup_runtime, synchronize,
)


def benchmark(args):
    model_config, config = read_config(args.config)
    device = setup_runtime(args.device, config.seed, config.cpu_threads)
    data = TokenData(args.data, model_config.block_size, model_config.vocab_size)
    results = []
    if args.steps < 1 or args.warmup < 1:
        raise ValueError("steps and warmup must be positive")
    for mode in args.modes:
        if device.type != "cuda" and mode != "fp32":
            results.append({"mode": mode, "skipped": "BF16/compile GPU experiment requires CUDA"})
            continue
        run_config = replace(
            config, precision="fp32" if mode == "fp32" else "bf16",
            compile=mode == "bf16-compile",
        )
        setup_runtime(device, config.seed, config.cpu_threads)
        model = GPT(model_config).to(device)
        optimizer = model.configure_optimizer(
            config.learning_rate, config.weight_decay, (config.beta1, config.beta2)
        )
        runner = torch.compile(model) if run_config.compile else model
        rng = np.random.default_rng(config.seed)
        synchronize(device)
        warmup_start = time.perf_counter()
        for _ in range(args.warmup):
            optimizer_step(runner, optimizer, data, rng, run_config, device)
        synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses = []
        for _ in range(args.steps):
            loss, _ = optimizer_step(runner, optimizer, data, rng, run_config, device)
            losses.append(loss)
        synchronize(device)
        elapsed = time.perf_counter() - start
        tokens_per_step = (
            config.micro_batch_size * config.gradient_accumulation_steps * model_config.block_size
        )
        rate = tokens_per_step * args.steps / elapsed
        budget = int(min(1_500_000_000, 5 * len(data.arrays["train"]), 0.85 * rate * 48 * 3600))
        budget = budget // tokens_per_step * tokens_per_step
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        kernels = []
        if args.profile and device.type == "cuda":
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
            ]) as prof:
                optimizer_step(runner, optimizer, data, rng, run_config, device)
            kernels = sorted({
                event.key for event in prof.key_averages()
                if any(part in event.key.lower() for part in ("attention", "flash", "fmha"))
            })
        results.append({
            "mode": mode, "steps": args.steps, "warmup_steps": args.warmup,
            "warmup_seconds": warmup_seconds,
            "seconds": elapsed, "tokens_per_second": rate, "tokens_per_step": tokens_per_step,
            "first_measured_loss": losses[0], "last_measured_loss": losses[-1],
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "suggested_48h_max_tokens": budget if device.type == "cuda" else None,
            "attention_profiler_events": kernels,
        })
        del runner, optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "data_fingerprint": data.fingerprint, "results": results,
        "notes": "Warmup/compilation excluded. Includes real loading, forward, backward, optimizer. "
                 "Evaluation/checkpoint overhead is not measured; budget reserves 15%.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--modes", nargs="+", choices=["fp32", "bf16", "bf16-compile"],
                        default=["fp32", "bf16", "bf16-compile"])
    parser.add_argument("--profile", action="store_true", help="Record actual attention kernel names on CUDA")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
