"""Single-GPU pretraining. Run `python -m icelandic_gpt.train --help`."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .model import GPT
from .training import (
    TokenData, config_dict, learning_rate_at, optimizer_step, read_config,
    setup_runtime, synchronize, validation_loss,
)


def run(args):
    model_config, config = read_config(args.config)
    if args.compile is not None:
        config.compile = args.compile
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
        config.__post_init__()
    device = setup_runtime(args.device, config.seed, config.cpu_threads)
    data = TokenData(args.data, model_config.block_size, model_config.vocab_size)
    if not data.meta.get("tokenizer_sha256"):
        raise ValueError("Token metadata must include tokenizer_sha256")
    output = Path(args.out)
    if not args.resume and any((output / name).exists() for name in ("last.pt", "initial.pt")):
        raise FileExistsError("Run already exists. Use --resume or choose a fresh --out.")
    output.mkdir(parents=True, exist_ok=True)
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizer(
        config.learning_rate, config.weight_decay, (config.beta1, config.beta2)
    )
    sampler_rng = np.random.default_rng(config.seed)
    step = 0
    best_val_loss = float("inf")
    tokens_per_step = (
        config.micro_batch_size * config.gradient_accumulation_steps * model_config.block_size
    )
    total_steps = math.ceil(config.max_tokens / tokens_per_step)
    final_step = total_steps
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        final_step = min(final_step, args.max_steps)
    if args.stop_at_tokens is not None:
        if args.stop_at_tokens < 1:
            raise ValueError("--stop-at-tokens must be positive")
        final_step = min(final_step, math.ceil(args.stop_at_tokens / tokens_per_step))
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours must be positive")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["model_config"] != asdict(model_config):
            raise ValueError("Resume architecture differs from saved architecture")
        if checkpoint["data_fingerprint"] != data.fingerprint:
            raise ValueError("Resume token data manifest differs from checkpoint")
        # Changing scheduling/precision/batch order is a new experiment, not an exact resume.
        flexible = {"compile", "cpu_threads", "log_interval", "eval_interval",
                    "checkpoint_interval"}
        for key, value in asdict(config).items():
            if key not in flexible and checkpoint["train_config"][key] != value:
                raise ValueError(f"Resume config changed {key}; use the original config")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        sampler_rng.bit_generator.state = checkpoint["sampler_rng"]
        restore_rng(checkpoint["rng"])
    if final_step < step:
        raise ValueError("Requested stopping step is before the resumed checkpoint")
    trained_model = torch.compile(model) if config.compile else model
    # Save the requested and resolved configuration alongside every run.
    resolved = config_dict(model_config, config)
    resolved.update({
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tokens_per_step": tokens_per_step, "schedule_steps": total_steps,
        "stop_step": final_step, "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": data.meta["tokenizer_sha256"],
        "training_tokens_in_corpus": len(data.arrays["train"]),
        "precision_effective": config.precision if device.type == "cuda" else "fp32",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    })
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stopping after the current optimizer step and saving a checkpoint.", flush=True)

    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)

    def payload():
        return {
            "format_version": 1,
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step, "tokens_seen": step * tokens_per_step,
            "best_val_loss": best_val_loss,
            "rng": capture_rng(), "sampler_rng": sampler_rng.bit_generator.state,
            "tokenizer_sha256": data.meta["tokenizer_sha256"],
            "data_fingerprint": data.fingerprint,
        }

    writer = SummaryWriter(
        str(output / "tensorboard"),
        purge_step=step * tokens_per_step if args.resume else None,
    )
    log_handle = (output / "metrics.jsonl").open("a", buffering=1)

    def log(values):
        record = {"step": step, "tokens_seen": step * tokens_per_step, **values}
        print(json.dumps(record), flush=True)
        log_handle.write(json.dumps(record) + "\n")
        for key, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                writer.add_scalar(key, value, step * tokens_per_step)

    def evaluate():
        nonlocal best_val_loss
        loss = validation_loss(trained_model, data, config, device)
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        log({"val/loss": loss, "val/perplexity": math.exp(min(loss, 80))})
        if loss < best_val_loss:
            best_val_loss = loss
            save_checkpoint(output / "best.pt", payload())

    save_steps = {
        math.ceil(tokens / tokens_per_step)
        for tokens in args.save_at_tokens if tokens > 0
    }
    try:
        if not args.resume:
            save_checkpoint(output / "initial.pt", payload())
            evaluate()
            save_checkpoint(output / "last.pt", payload())
        while step < final_step and not stop_requested:
            if args.max_hours is not None and time.perf_counter() - start_time >= args.max_hours * 3600:
                break
            lr = learning_rate_at(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            synchronize(device)
            tick = time.perf_counter()
            loss, grad_norm = optimizer_step(
                trained_model, optimizer, data, sampler_rng, config, device
            )
            synchronize(device)
            duration = time.perf_counter() - tick
            step += 1
            if step % config.log_interval == 0 or step == 1:
                record = {
                    "train/loss": loss, "train/lr": lr, "train/grad_norm": grad_norm,
                    "perf/step_seconds": duration,
                    "perf/tokens_per_second": tokens_per_step / duration,
                    "train/corpus_exposures": step * tokens_per_step / len(data.arrays["train"]),
                }
                if device.type == "cuda":
                    record["memory/peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    record["memory/peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                log(record)
            if step % config.eval_interval == 0:
                evaluate()
            if step % config.checkpoint_interval == 0:
                save_checkpoint(output / "last.pt", payload())
            if step in save_steps:
                save_checkpoint(output / f"tokens_{step * tokens_per_step}.pt", payload())
        if step % config.eval_interval or step == 0:
            evaluate()
        save_checkpoint(output / "last.pt", payload())
        writer.flush()
        log({"run/seconds": time.perf_counter() - start_time, "run/stopped_cleanly": True})
    finally:
        log_handle.close()
        writer.close()
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    return output / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Directory containing token .bin files and meta.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Absolute stopping step; does not change LR schedule")
    parser.add_argument("--stop-at-tokens", type=int, help="Stop at this budget, preserving full LR horizon")
    parser.add_argument("--max-tokens", type=int, help="Override the full LR horizon before starting a run")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--save-at-tokens", type=int, nargs="*", default=[])
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
