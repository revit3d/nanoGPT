"""Reproducible held-out language modeling, continuation ranking, and generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .generate import EOS_TOKEN, file_sha256, generate_text, load_bundle


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_jsonl(path: str | Path, values) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def truncate_text(text: str, tokenizer: Tokenizer, max_tokens: int) -> tuple[str, list[int]]:
    """Truncate at a Unicode character boundary and then re-encode that exact text."""
    if max_tokens <= 0:
        return "", []
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= max_tokens:
        return text, encoded.ids
    # Several byte-level tokens can have the same character offset. Re-encoding
    # is essential: slicing IDs alone can leave only part of a UTF-8 character.
    end = encoded.offsets[max_tokens - 1][1]
    candidate = text[:end]
    ids = tokenizer.encode(candidate, add_special_tokens=False).ids
    while candidate and len(ids) > max_tokens:
        candidate = candidate[:-1]
        ids = tokenizer.encode(candidate, add_special_tokens=False).ids
    return candidate, ids


@torch.inference_mode()
def score_token_ids(
    model, token_ids: list[int], *, eos_id: int, score_start: int = 0,
    stride: int = 256,
) -> tuple[float, int]:
    """Sum NLL for token_ids[score_start:] exactly once.

    EOS is an input-only beginning-of-document context. Each input at position t
    predicts the explicitly shifted target at t+1. Overlapping windows provide
    context but their already-scored targets do not contribute a second time.
    """
    block_size = model.config.block_size
    if not 1 <= stride <= block_size:
        raise ValueError("stride must be between 1 and the model context length.")
    if not 0 <= score_start <= len(token_ids):
        raise ValueError("score_start is outside the token sequence.")
    if score_start == len(token_ids):
        return 0.0, 0
    device = next(model.parameters()).device
    stream = [eos_id] + token_ids
    previous_end = score_start
    end = min(len(token_ids), max(block_size, score_start + stride))
    total_nll, count = 0.0, 0
    model.eval()
    while previous_end < len(token_ids):
        begin = max(0, end - block_size)
        inputs = torch.tensor([stream[begin:end]], dtype=torch.long, device=device)
        logits, _ = model(inputs)
        offset = previous_end - begin
        targets = torch.tensor(
            stream[previous_end + 1:end + 1], dtype=torch.long, device=device,
        )
        loss = F.cross_entropy(logits[0, offset:].float(), targets, reduction="sum")
        total_nll += float(loss.item())
        count += targets.numel()
        previous_end = end
        end = min(len(token_ids), end + stride)
    return total_nll, count


def summarize_loss(nll: float, tokens: int, utf8_bytes: int) -> dict:
    if tokens < 1 or utf8_bytes < 1:
        raise ValueError("Evaluation contained no non-empty text.")
    loss = nll / tokens
    return {
        "nll_nats": nll,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "loss_nats_per_token": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
        "bits_per_byte": nll / (math.log(2) * utf8_bytes),
    }


def evaluate_lm(model, tokenizer: Tokenizer, data: str | Path,
                *, stride: int = 256, max_tokens: int | None = 2_000_000) -> dict:
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing {EOS_TOKEN!r}.")
    sources = defaultdict(lambda: {"nll": 0.0, "tokens": 0, "bytes": 0, "documents": 0})
    total_tokens = 0
    partial_documents = 0
    # Split files can be grouped by source. A checkpoint-independent hash order
    # prevents a token budget from evaluating only the first source in the file.
    records = sorted(
        read_jsonl(data),
        key=lambda item: hashlib.sha256(str(item["id"]).encode("utf-8")).digest(),
    )
    for record in records:
        text = record["text"]
        if not text:
            continue
        if max_tokens is None:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
        else:
            if total_tokens >= max_tokens:
                break
            original_text = text
            text, ids = truncate_text(text, tokenizer, max_tokens - total_tokens)
            partial_documents += int(text != original_text)
        if not ids:
            break
        if eos_id in ids:
            raise ValueError("Evaluation text contains the reserved EOS token.")
        nll, count = score_token_ids(model, ids, eos_id=eos_id, stride=stride)
        source = sources[str(record.get("source", "unknown"))]
        source["nll"] += nll
        source["tokens"] += count
        source["bytes"] += len(text.encode("utf-8"))
        source["documents"] += 1
        total_tokens += count
        # A partially scored final document is the fixed token budget boundary;
        # do not skip its final character and spend leftover budget elsewhere.
        if max_tokens is not None and text != original_text:
            break
    result = summarize_loss(
        sum(s["nll"] for s in sources.values()),
        sum(s["tokens"] for s in sources.values()),
        sum(s["bytes"] for s in sources.values()),
    )
    result.update({
        "documents": sum(s["documents"] for s in sources.values()),
        "partial_documents": partial_documents,
        "stride": stride,
        "context_length": model.config.block_size,
        "max_tokens": max_tokens,
        "document_order": "ascending SHA-256 of UTF-8 document ID",
        "method": "Document-reset sliding windows; input-only EOS; no EOS targets; "
                  "all scored UTF-8 text bytes included exactly once.",
        "sources": {
            name: {**summarize_loss(s["nll"], s["tokens"], s["bytes"]),
                   "documents": s["documents"]}
            for name, s in sorted(sources.items())
        },
    })
    return result


def split_continuation(text: str, tokenizer: Tokenizer,
                       prefix_tokens: int = 128, suffix_tokens: int = 64):
    prefix, prefix_ids = truncate_text(text, tokenizer, prefix_tokens)
    if len(prefix_ids) < prefix_tokens // 2:
        return None
    # Prefer a word boundary. Keeping the separator in the suffix means the
    # concatenated strings still equal the original document prefix.
    whitespace = max((i for i, char in enumerate(prefix) if char.isspace()), default=-1)
    if whitespace > len(prefix) // 2:
        prefix = prefix[:whitespace]
    suffix, suffix_ids = truncate_text(text[len(prefix):], tokenizer, suffix_tokens)
    if len(suffix_ids) < suffix_tokens // 2 or not suffix.strip():
        return None
    return prefix, suffix


def build_benchmark(data: str | Path, tokenizer: Tokenizer, *,
                    groups: int = 50, seed: int = 42) -> list[dict]:
    if groups <= 0:
        raise ValueError("groups must be positive.")
    rng = random.Random(seed)
    # A bounded per-source reservoir keeps build memory independent of corpus size.
    reservoirs = defaultdict(list)
    seen = defaultdict(int)
    for record in read_jsonl(data):
        if len(record["text"]) < 400:
            continue
        source = str(record.get("source", "unknown"))
        seen[source] += 1
        bucket = reservoirs[source]
        if len(bucket) < 2000:
            bucket.append(record)
        else:
            index = rng.randrange(seen[source])
            if index < len(bucket):
                bucket[index] = record
    candidate_groups = []
    for source, records in sorted(reservoirs.items()):
        eligible = []
        for record in records:
            pair = split_continuation(record["text"], tokenizer)
            if pair is not None:
                eligible.append({**record, "prefix": pair[0], "suffix": pair[1]})
        # Similar document lengths and common source reduce easy stylistic cues.
        eligible.sort(key=lambda item: (len(item["text"]), str(item["id"])))
        for start in range(0, len(eligible) - 3, 4):
            group = eligible[start:start + 4]
            if len({item["suffix"] for item in group}) == 4:
                candidate_groups.append((source, group))
    rng.shuffle(candidate_groups)
    if len(candidate_groups) < groups:
        raise ValueError(
            f"Need {groups} groups of four sufficiently long documents; "
            f"found {len(candidate_groups)}. Supply more held-out text or lower --groups."
        )
    tasks = []
    for group_number, (source, group) in enumerate(candidate_groups[:groups]):
        rng.shuffle(group)
        candidates = [item["suffix"] for item in group]
        for answer, item in enumerate(group):
            tasks.append({
                "id": f"g{group_number:04d}-q{answer}",
                "group_id": f"g{group_number:04d}",
                "source": source,
                "document_id": str(item["id"]),
                "candidate_document_ids": [str(candidate["id"]) for candidate in group],
                "prefix": item["prefix"],
                "candidates": candidates,
                "answer": answer,
                "build_seed": seed,
            })
    return tasks


def group_bootstrap(values: list[float], *, seed: int = 42,
                    samples: int = 2000) -> list[float]:
    if not values:
        raise ValueError("No groups to bootstrap.")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples)
    )
    return [estimates[int(0.025 * (samples - 1))],
            estimates[int(0.975 * (samples - 1))]]


def evaluate_benchmark(model, tokenizer: Tokenizer, data: str | Path,
                       *, stride: int = 256, seed: int = 42) -> dict:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing {EOS_TOKEN!r}.")
    tasks = list(read_jsonl(data))
    groups = defaultdict(list)
    for task in tasks:
        groups[task["group_id"]].append(task)
    if not groups:
        raise ValueError("Benchmark contains no tasks.")
    for group_tasks in groups.values():
        if len(group_tasks) != 4 or sorted(t["answer"] for t in group_tasks) != [0, 1, 2, 3]:
            raise ValueError("Each benchmark group must have four counterbalanced answers.")
        if any(t["candidates"] != group_tasks[0]["candidates"] for t in group_tasks):
            raise ValueError("Each group must reuse the same ordered candidates.")
        if len(group_tasks[0]["candidates"]) != 4:
            raise ValueError("Each task needs exactly four candidates.")
    predictions = []
    group_accuracies, group_baselines, group_differences = [], [], []
    for group_id, group_tasks in sorted(groups.items()):
        conditional_correct = baseline_correct = 0
        baseline_scores = None
        for task in group_tasks:
            prefix_ids = tokenizer.encode(task["prefix"], add_special_tokens=False).ids
            scores, independent_scores = [], []
            for index, suffix in enumerate(task["candidates"]):
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
                if not suffix_ids or not suffix.encode("utf-8"):
                    raise ValueError("Benchmark candidates must be non-empty.")
                if eos_id in prefix_ids or eos_id in suffix_ids:
                    raise ValueError("Benchmark contains the reserved EOS token.")
                nll, _ = score_token_ids(
                    model, prefix_ids + suffix_ids, eos_id=eos_id,
                    score_start=len(prefix_ids), stride=stride,
                )
                scores.append(-nll / len(suffix.encode("utf-8")))
                if baseline_scores is None:
                    unconditioned, _ = score_token_ids(
                        model, suffix_ids, eos_id=eos_id, stride=stride,
                    )
                    independent_scores.append(-unconditioned / len(suffix.encode("utf-8")))
            if baseline_scores is None:
                baseline_scores = independent_scores
            prediction = max(range(4), key=lambda i: scores[i])
            baseline_prediction = max(range(4), key=lambda i: baseline_scores[i])
            correct = int(prediction == task["answer"])
            baseline_is_correct = int(baseline_prediction == task["answer"])
            conditional_correct += correct
            baseline_correct += baseline_is_correct
            predictions.append({
                "id": task["id"], "group_id": group_id,
                "answer": task["answer"], "prediction": prediction,
                "without_context_prediction": baseline_prediction,
                "log_probability_per_byte": scores,
                "without_context_log_probability_per_byte": baseline_scores,
                "correct": bool(correct),
            })
        group_accuracies.append(conditional_correct / 4)
        group_baselines.append(baseline_correct / 4)
        group_differences.append((conditional_correct - baseline_correct) / 4)
    return {
        "tasks": len(tasks),
        "groups": len(groups),
        "accuracy": sum(group_accuracies) / len(groups),
        "without_context_accuracy": sum(group_baselines) / len(groups),
        "context_gain": sum(group_differences) / len(groups),
        "chance_accuracy": 0.25,
        "accuracy_95pct_group_bootstrap": group_bootstrap(group_accuracies, seed=seed),
        "context_gain_95pct_group_bootstrap": group_bootstrap(group_differences, seed=seed),
        "seed": seed,
        "stride": stride,
        "tokenization": "Prefix and suffix encoded separately, then concatenated; "
                        "identical suffix token IDs scored with and without context. "
                        "Scores include suffix tokens only and normalize by suffix UTF-8 bytes.",
        "predictions": predictions,
    }


def choose_gallery_prompts(data: str | Path, tokenizer: Tokenizer, *,
                           count: int = 20, seed: int = 42) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    reservoir = []
    eligible_count = 0
    for record in read_jsonl(data):
        prefix, ids = truncate_text(record["text"], tokenizer, 48)
        if len(ids) < 24:
            continue
        eligible_count += 1
        prompt = {
            "id": str(record["id"]), "source": record.get("source", "unknown"),
            "prompt": prefix, "selection_seed": seed,
        }
        if len(reservoir) < count:
            reservoir.append(prompt)
        else:
            index = rng.randrange(eligible_count)
            if index < count:
                reservoir[index] = prompt
    if len(reservoir) != count:
        raise ValueError(f"Need {count} eligible prompts; found {len(reservoir)}.")
    return reservoir


def repetition_fraction(text: str, n: int = 3) -> float:
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lm", "benchmark-build", "benchmark", "gallery"):
        sub = commands.add_parser(name)
        sub.add_argument("--tokenizer", required=True, type=Path)
        sub.add_argument("--data", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        if name != "benchmark-build":
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            sub.add_argument("--cpu-threads", default=4, type=int,
                             help="PyTorch CPU worker threads; ignored for CUDA (default: 4).")
        if name in ("lm", "benchmark"):
            sub.add_argument("--stride", default=256, type=int)
        if name != "lm":
            sub.add_argument("--seed", default=42, type=int)
        if name == "lm":
            sub.add_argument("--max-tokens", default=2_000_000, type=int)
        elif name == "benchmark-build":
            sub.add_argument("--groups", default=50, type=int)
        elif name == "gallery":
            sub.add_argument("--prompts", required=True, type=Path)
            sub.add_argument("--count", default=20, type=int)
            sub.add_argument("--seeds", nargs="+", default=[42, 43], type=int)
            sub.add_argument("--max-new-tokens", default=128, type=int)
            sub.add_argument("--temperature", default=0.8, type=float)
            sub.add_argument("--top-p", default=0.95, type=float)
    args = parser.parse_args()
    if args.command == "benchmark-build":
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        result = build_benchmark(args.data, tokenizer, groups=args.groups, seed=args.seed)
        tokenizer_hash = file_sha256(args.tokenizer)
        for task in result:
            task["tokenizer_sha256"] = tokenizer_hash
        save_jsonl(args.output, result)
        print(f"Saved {len(result)} tasks in {args.groups} groups to {args.output}")
        return
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, tokenizer, metadata = load_bundle(args.checkpoint, args.tokenizer, args.device)
    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
    }
    if args.command == "lm":
        result = evaluate_lm(
            model, tokenizer, args.data, stride=args.stride, max_tokens=args.max_tokens,
        )
    elif args.command == "benchmark":
        for task in read_jsonl(args.data):
            expected = task.get("tokenizer_sha256")
            if expected is not None and expected != metadata["tokenizer_sha256"]:
                raise ValueError("Benchmark was built with a different tokenizer.")
        result = evaluate_benchmark(
            model, tokenizer, args.data, stride=args.stride, seed=args.seed,
        )
    else:
        if args.prompts.exists():
            prompts = list(read_jsonl(args.prompts))
            if len(prompts) != args.count:
                raise ValueError("Existing prompt count differs from --count; use a different path.")
        else:
            prompts = choose_gallery_prompts(
                args.data, tokenizer, count=args.count, seed=args.seed,
            )
            for prompt in prompts:
                prompt["tokenizer_sha256"] = metadata["tokenizer_sha256"]
                prompt["data_sha256"] = provenance["data_sha256"]
            # Freeze the entire prompt set before any model output is inspected.
            save_jsonl(args.prompts, prompts)
        for prompt in prompts:
            if prompt.get("tokenizer_sha256") != metadata["tokenizer_sha256"]:
                raise ValueError("Frozen gallery prompts were selected with a different tokenizer.")
            if prompt.get("data_sha256") != provenance["data_sha256"]:
                raise ValueError("Frozen gallery prompts refer to a different evaluation corpus.")
        result = {
            "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
            "count": len(prompts), "seeds": args.seeds,
            "expected_generations": len(prompts) * len(args.seeds),
            "complete": False, "generations": [], **provenance,
        }
        for prompt in prompts:
            for seed in args.seeds:
                generation = generate_text(
                    model, tokenizer, prompt["prompt"], max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, seed=seed,
                )
                result["generations"].append({
                    **generation, "document_id": prompt["id"], "source": prompt["source"],
                    "repeated_word_trigram_fraction": repetition_fraction(generation["continuation"]),
                })
                save_json(args.output, result)
        result["complete"] = True
    save_json(args.output, {**result, **provenance})
    print(f"Saved {args.command} results to {args.output}")


if __name__ == "__main__":
    main()
