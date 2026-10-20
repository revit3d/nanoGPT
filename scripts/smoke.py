#!/usr/bin/env python3
"""Exercise every pipeline stage offline on synthetic data, in a new directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("smoke") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a fresh path to avoid overwriting a run")
    output.mkdir(parents=True)

    def command(*parts):
        parts = [str(part) for part in parts]
        print("+ " + " ".join([sys.executable, *parts]), flush=True)
        subprocess.run([sys.executable, *parts], cwd=root, check=True)

    def module(name, *parts):
        command("-m", "icelandic_gpt." + name, *parts)

    raw, clean, tokens = (output / name for name in ("raw", "clean", "tokens"))
    tokenizers = output / "tokenizers"
    tokenizer = tokenizers / "512" / "tokenizer.json"
    run = output / "run"
    checkpoint = run / "last.pt"
    command(root / "scripts" / "make_fixture.py", "--output", raw)
    module("data", "prepare", "--input", raw, "--output", clean,
           "--skip-lid", "--min-chars", 40, "--split-ratios", .9, .05, .05)
    module("tokenizer", "train", "--data", clean, "--output", tokenizers,
           "--vocab-sizes", 512, "--sample-mb", 1)
    module("tokenizer", "encode", "--data", clean, "--tokenizer", tokenizer, "--output", tokens)
    module("train", "--config", root / "configs" / "smoke.json", "--data", tokens,
           "--out", run, "--device", "cpu", "--no-compile", "--max-steps", 5)
    module("train", "--config", root / "configs" / "smoke.json", "--data", tokens,
           "--out", run, "--device", "cpu", "--no-compile", "--resume", checkpoint)
    shared = ("--checkpoint", checkpoint, "--tokenizer", tokenizer, "--device", "cpu")
    module("generate", *shared, "--prompt", "Ísland er", "--max-new-tokens", 16)
    module("evaluate", "lm", *shared, "--data", clean / "test.jsonl",
           "--output", output / "lm.json", "--stride", 16, "--max-tokens", 1000)
    module("evaluate", "benchmark-build", "--data", clean / "test.jsonl", "--tokenizer", tokenizer,
           "--groups", 1, "--output", output / "benchmark.jsonl")
    module("evaluate", "benchmark", *shared, "--data", output / "benchmark.jsonl",
           "--output", output / "benchmark-results.json", "--stride", 16)
    module("evaluate", "gallery", *shared, "--data", clean / "test.jsonl",
           "--prompts", output / "prompts.jsonl", "--output", output / "gallery.json",
           "--count", 2, "--seeds", 42, 43, "--max-new-tokens", 16)
    module("benchmark", "--config", root / "configs" / "smoke.json", "--data", tokens,
           "--output", output / "throughput.json", "--device", "cpu",
           "--modes", "fp32", "--steps", 2, "--warmup", 1)
    module("inspect", "--config", root / "configs" / "smoke.json",
           "--device", "cpu", "--output", output / "memory.json")
    result = {"complete": True, "synthetic_data_only": True, "device": "cpu",
              "evidence_of_icelandic_language_quality": False}
    (output / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"All smoke stages passed. Synthetic artifacts: {output}")


if __name__ == "__main__":
    main()
