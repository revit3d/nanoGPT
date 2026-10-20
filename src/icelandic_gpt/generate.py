"""Generate continuations using a locally trained checkpoint and its tokenizer."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_model

EOS_TOKEN = "<|endoftext|>"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(checkpoint: str | Path, tokenizer_path: str | Path, device: str):
    model, metadata = load_model(checkpoint, device=device)
    expected = metadata.get("tokenizer_sha256")
    actual = file_sha256(tokenizer_path)
    if not expected or actual != expected:
        raise ValueError(
            "Tokenizer SHA-256 does not match the checkpoint. Use the exact "
            "tokenizer.json used for training; a matching vocabulary size is insufficient."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration.")
    if tokenizer.token_to_id(EOS_TOKEN) is None:
        raise ValueError(f"Tokenizer is missing {EOS_TOKEN!r}.")
    model.eval()
    return model, tokenizer, metadata


def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int = 42,
) -> dict:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be non-negative and top_p must be in (0, 1].")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive.")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing {EOS_TOKEN!r}.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = prompt_ids or [eos_id]
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, eos_id=eos_id, generator=generator,
        )
    generated_ids = output[0, len(input_ids):].tolist()
    # Decode the entire non-special token stream together: a byte-level tokenizer
    # may emit a Unicode character across several generated tokens.
    full_text = tokenizer.decode(prompt_ids + generated_ids, skip_special_tokens=True)
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "continuation": continuation,
        "text": full_text,
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "stopped_on_eos": bool(generated_ids and generated_ids[-1] == eos_id),
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--prompt", default="Ísland er")
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", default=4, type=int,
                        help="PyTorch CPU worker threads; ignored for CUDA (default: 4).")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, tokenizer, _ = load_bundle(args.checkpoint, args.tokenizer, args.device)
    result = generate_text(
        model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
    )
    print(result["text"])


if __name__ == "__main__":
    main()
