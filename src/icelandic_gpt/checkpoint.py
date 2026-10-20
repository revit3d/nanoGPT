"""Atomic checkpoints containing only tensors and ordinary Python values."""
import hashlib
import os
from pathlib import Path
import random

import torch

from .model import GPT, GPTConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    return data


def load_model(checkpoint, device="cpu"):
    data = load_checkpoint(checkpoint, device)
    model = GPT(GPTConfig(**data["model_config"])).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()
    return model, data


def capture_rng():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
