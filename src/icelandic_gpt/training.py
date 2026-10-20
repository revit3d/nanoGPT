"""Shared configuration, token loader and numerical training primitives."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .checkpoint import file_sha256
from .model import GPTConfig


@dataclass
class TrainConfig:
    seed: int = 42
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    max_tokens: int = 1_000_013_824
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"
    compile: bool = True
    eval_interval: int = 150
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 150
    cpu_threads: int = 4

    def __post_init__(self):
        for name in ("micro_batch_size", "gradient_accumulation_steps", "max_tokens",
                     "eval_interval", "eval_batches", "log_interval", "checkpoint_interval",
                     "cpu_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0,1)")
        if not 0 < self.min_lr <= self.learning_rate:
            raise ValueError("Require 0 < min_lr <= learning_rate")


def read_config(path):
    config = json.loads(Path(path).read_text())
    unexpected = set(config) - {"model", "train", "description"}
    if unexpected:
        raise ValueError(f"Unknown config sections: {sorted(unexpected)}")
    return GPTConfig(**config["model"]), TrainConfig(**config.get("train", {}))


def config_dict(model_config, train_config):
    return {"model": asdict(model_config), "train": asdict(train_config)}


def setup_runtime(device, seed, cpu_threads=4):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu only for a smoke test.")
    torch.set_num_threads(cpu_threads)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def autocast_context(device, precision):
    device = torch.device(device)
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but this GPU/runtime does not support it")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # CPU smoke tests intentionally use FP32, for portable numerical checks.
    return nullcontext()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def learning_rate_at(step, total_steps, config):
    warmup = min(total_steps - 1, max(1, round(total_steps * config.warmup_ratio)))
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    progress = min(1.0, max(0.0, progress))
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


class TokenData:
    def __init__(self, directory, block_size, vocab_size, verify=True):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text())
        if int(self.meta["vocab_size"]) != vocab_size:
            raise ValueError("Config vocab_size differs from tokenized data")
        self.block_size = block_size
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        if self.dtype not in (np.dtype("uint16"), np.dtype("uint32")):
            raise ValueError("Only uint16 and uint32 token files are supported")
        self.fingerprint = file_sha256(self.directory / "meta.json")
        self.arrays = {}
        for split in ("train", "val"):
            path = self.directory / f"{split}.bin"
            if path.stat().st_size % self.dtype.itemsize:
                raise ValueError(f"Truncated token file: {path}")
            if path.stat().st_size <= self.block_size * self.dtype.itemsize:
                raise ValueError(f"{split} needs at least block_size+1 tokens")
            info = self.meta.get("splits", {}).get(split, {})
            if verify and isinstance(info, dict) and info.get("sha256"):
                if file_sha256(path) != info["sha256"]:
                    raise ValueError(f"Checksum mismatch for {split}.bin")
            self.arrays[split] = np.memmap(path, dtype=self.dtype, mode="r")

    def batch(self, split, batch_size, rng, device):
        values = self.arrays[split]
        starts = rng.integers(0, len(values) - self.block_size, size=batch_size)
        x = np.stack([values[i:i + self.block_size] for i in starts]).astype(np.int64)
        y = np.stack([values[i + 1:i + self.block_size + 1] for i in starts]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if torch.device(device).type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def optimizer_step(model, optimizer, data, rng, config, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    for _ in range(config.gradient_accumulation_steps):
        x, y = data.batch("train", config.micro_batch_size, rng, device)
        with autocast_context(device, config.precision):
            _, loss = model(x, y)
        # Accumulate on-device: .item() in each microbatch would synchronize CUDA.
        total_loss += loss.detach() / config.gradient_accumulation_steps
        (loss / config.gradient_accumulation_steps).backward()
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite loss; last checkpoint is preserved")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    return float(total_loss), float(norm)


@torch.no_grad()
def validation_loss(model, data, config, device):
    was_training = model.training
    model.eval()
    # Reset the independent RNG each time: always evaluate exactly the same windows.
    rng = np.random.default_rng(123456)
    losses = []
    try:
        for _ in range(config.eval_batches):
            x, y = data.batch("val", config.micro_batch_size, rng, device)
            with autocast_context(device, config.precision):
                _, loss = model(x, y)
            losses.append(float(loss))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)
