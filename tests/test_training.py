from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from icelandic_gpt.benchmark import benchmark
from icelandic_gpt.checkpoint import file_sha256, load_checkpoint, load_model
from icelandic_gpt.train import run
from icelandic_gpt.training import TokenData, TrainConfig, learning_rate_at


def fixture_run(tmp_path):
    data = tmp_path / "tokens"
    data.mkdir()
    meta = {"vocab_size": 32, "dtype": "uint16", "eos_id": 0,
            "tokenizer_sha256": "fixture-tokenizer", "splits": {}}
    for split, size in (("train", 1000), ("val", 300)):
        path = data / f"{split}.bin"
        (np.arange(size) % 32).astype(np.uint16).tofile(path)
        meta["splits"][split] = {"tokens": size, "sha256": file_sha256(path)}
    (data / "meta.json").write_text(json.dumps(meta))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8,
                  "vocab_size": 32, "dropout": 0.2},
        "train": {"max_tokens": 128, "micro_batch_size": 2,
                  "gradient_accumulation_steps": 2, "precision": "fp32",
                  "compile": False, "eval_interval": 2, "eval_batches": 1,
                  "log_interval": 1, "checkpoint_interval": 1, "cpu_threads": 1},
    }))
    return data, config


def args_for(data, config, out, **kwargs):
    values = dict(data=str(data), config=str(config), out=str(out), device="cpu",
                  resume=None, max_steps=None, stop_at_tokens=None, max_tokens=None,
                  max_hours=None, save_at_tokens=[64], compile=False)
    values.update(kwargs)
    return Namespace(**values)


def test_resume_matches_uninterrupted_training_with_dropout(tmp_path):
    data, config = fixture_run(tmp_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    run(args_for(data, config, uninterrupted))
    run(args_for(data, config, resumed, max_steps=2))
    path = run(args_for(data, config, resumed, resume=str(resumed / "last.pt")))
    a = load_checkpoint(uninterrupted / "last.pt")
    b = load_checkpoint(path)
    assert a["step"] == b["step"] == 4
    assert a["tokens_seen"] == b["tokens_seen"] == 128
    for name, tensor in a["model_state_dict"].items():
        torch.testing.assert_close(tensor, b["model_state_dict"][name], rtol=0, atol=0)
    assert a["sampler_rng"] == b["sampler_rng"]
    assert (resumed / "initial.pt").exists()
    assert (resumed / "tokens_64.pt").exists()
    assert (resumed / "best.pt").exists()
    loaded, _ = load_model(path)
    assert loaded.config.block_size == 8


def test_existing_run_and_changed_schedule_rejected(tmp_path):
    data, config = fixture_run(tmp_path)
    output = tmp_path / "run"
    run(args_for(data, config, output, max_steps=1))
    with pytest.raises(FileExistsError):
        run(args_for(data, config, output))
    with pytest.raises(ValueError, match="max_tokens"):
        run(args_for(data, config, output, resume=str(output / "last.pt"), max_tokens=256))


def test_token_loader_shift_checksum_and_reproducibility(tmp_path):
    data, config = fixture_run(tmp_path)
    loader = TokenData(data, 8, 32)
    x, y = loader.batch("train", 3, np.random.default_rng(42), "cpu")
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    xx, yy = loader.batch("train", 3, np.random.default_rng(42), "cpu")
    assert torch.equal(x, xx) and torch.equal(y, yy)
    with pytest.raises(ValueError, match="vocab_size"):
        TokenData(data, 8, 33)
    path = data / "train.bin"
    binary = bytearray(path.read_bytes())
    binary[0] ^= 1
    path.write_bytes(binary)
    with pytest.raises(ValueError, match="Checksum"):
        TokenData(data, 8, 32)


def test_cpu_benchmark_does_not_claim_gpu_budget(tmp_path):
    data, config = fixture_run(tmp_path)
    report = benchmark(Namespace(
        config=str(config), data=str(data), device="cpu", output=str(tmp_path / "bench.json"),
        steps=2, warmup=1, modes=["fp32", "bf16"], profile=False,
    ))
    result = report["results"][0]
    assert result["tokens_per_second"] > 0
    assert result["suggested_48h_max_tokens"] is None
    assert result["peak_allocated_bytes"] is None
    assert "skipped" in report["results"][1]


def test_lr_has_warmup_and_reaches_floor():
    config = TrainConfig(warmup_ratio=0.1)
    assert learning_rate_at(0, 100, config) < learning_rate_at(9, 100, config)
    assert learning_rate_at(9, 100, config) == config.learning_rate
    assert learning_rate_at(99, 100, config) == config.min_lr
