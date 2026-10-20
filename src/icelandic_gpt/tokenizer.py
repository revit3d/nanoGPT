"""Train byte-level BPE only on train data, audit it, and write uint16 token streams."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import WORD_RE, sha256_file, write_json

EOS_TOKEN = "<|endoftext|>"
SPLITS = ("train", "val", "test")


def iter_documents(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Expected a document with text at {path}:{line_number}")
            if not record["text"]:
                raise ValueError(f"Empty document at {path}:{line_number}")
            yield record


def _build_sample(train_path: Path, sample_path: Path, index_path: Path,
                  sample_bytes: int, seed: int) -> dict:
    """Disk-sort hash priorities, giving an input-order independent document sample."""
    db = sqlite3.connect(index_path)
    db.execute("CREATE TABLE offsets(priority TEXT,offset INTEGER,n_bytes INTEGER)")
    db.execute("PRAGMA cache_size=-16384")
    documents = 0
    with train_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["text"]
            identity = record.get("id", hashlib.sha256(text.encode("utf-8")).hexdigest())
            priority = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
            db.execute("INSERT INTO offsets VALUES (?,?,?)", (priority, offset, len(text.encode("utf-8"))))
            documents += 1
    db.commit()
    db.execute("CREATE INDEX sorted_offsets ON offsets(priority)")
    sampled_bytes, selected, source_bytes = 0, 0, Counter()
    identity_hash = hashlib.sha256()
    with train_path.open("rb") as handle, sample_path.open("w", encoding="utf-8") as output:
        for offset, n_bytes in db.execute("SELECT offset,n_bytes FROM offsets ORDER BY priority"):
            if sampled_bytes >= sample_bytes:
                break
            handle.seek(offset)
            record = json.loads(handle.readline())
            # The final sample document may be shortened at a valid UTF-8 boundary.
            remaining = sample_bytes - sampled_bytes
            text = record["text"].encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
            if not text:
                continue
            original_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            record = {**record, "text": text, "sample_truncated": len(text.encode("utf-8")) < n_bytes}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            identity_hash.update((original_hash + "\n").encode())
            used = len(text.encode("utf-8"))
            sampled_bytes += used
            selected += 1
            source_bytes[record.get("source", "unknown")] += used
    db.close()
    if selected == 0:
        raise ValueError("No train text is available for tokenizer training")
    return {
        "seed": seed, "train_documents": documents, "sample_documents": selected,
        "sample_utf8_bytes": sampled_bytes, "sample_document_hashes_sha256": identity_hash.hexdigest(),
        "sample_sha256": sha256_file(sample_path), "train_sha256": sha256_file(train_path),
        "sample_source_bytes": dict(source_bytes), "requested_sample_bytes": sample_bytes,
        "selection": "documents ordered by SHA-256(seed:content-id); final document may be truncated",
        "split_used_for_training": "train",
    }


def tokenizer_metrics(tokenizer: Tokenizer, path: Path, *, max_bytes: int | None = None) -> dict:
    frequencies = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    n_documents = n_bytes = n_words = n_tokens = failed = 0
    long_words: dict[str, dict] = {}
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    for record in iter_documents(path):
        if max_bytes is not None and n_bytes >= max_bytes:
            break
        text = record["text"]
        if max_bytes is not None:
            text = text.encode("utf-8")[:max_bytes - n_bytes].decode("utf-8", errors="ignore")
        if not text:
            continue
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        failed += int(decoded != text)
        ids = np.asarray(encoding.ids, dtype=np.int64)
        np.add.at(frequencies, ids, 1)
        words = WORD_RE.findall(text)
        n_documents += 1
        n_bytes += len(text.encode("utf-8"))
        n_words += len(words)
        n_tokens += len(encoding.ids)
        for word in words:
            if len(word) < 16 or word in long_words:
                continue
            if len(long_words) < 20:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                long_words[word] = {"word": word, "tokens": tokens.tokens, "length": len(tokens.ids)}
    lexical_frequencies = np.delete(frequencies, eos_id) if eos_id is not None else frequencies
    sorted_ids = sorted(range(len(frequencies)), key=lambda i: (-int(frequencies[i]), i))
    histogram = {
        "zero": int((lexical_frequencies == 0).sum()),
        "1_to_10": int(((lexical_frequencies >= 1) & (lexical_frequencies <= 10)).sum()),
        "11_to_100": int(((lexical_frequencies > 10) & (lexical_frequencies <= 100)).sum()),
        "over_100": int((lexical_frequencies > 100).sum()),
    }
    return {
        "documents": n_documents, "utf8_bytes": n_bytes, "words": n_words, "tokens": n_tokens,
        "tokens_per_word": n_tokens / n_words if n_words else None,
        "bytes_per_token": n_bytes / n_tokens if n_tokens else None,
        "roundtrip_failures": failed, "frequency_histogram_excluding_eos": histogram,
        "inactive_token_fraction": histogram["zero"] / max(1, len(lexical_frequencies)),
        "rare_observed_token_fraction": histogram["1_to_10"] / max(1, len(lexical_frequencies)),
        "top_20_tokens": [{"id": i, "token": tokenizer.id_to_token(i), "count": int(frequencies[i])}
                          for i in sorted_ids[:20]],
        "long_word_examples": list(long_words.values()),
    }


def train_tokenizers(
    data: str | Path, output: str | Path, *, vocab_sizes: list[int] | tuple[int, ...] = (8192, 16384, 32768),
    sample_mb: float = 100, seed: int = 42, min_frequency: int = 2, eval_mb: float = 10,
) -> dict:
    if not vocab_sizes or any(size < 257 or size > 65536 for size in vocab_sizes):
        raise ValueError("Vocabulary must contain 256 byte symbols + EOS, and fit in uint16 (257..65536)")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("Repeated vocabulary sizes")
    if sample_mb <= 0 or eval_mb <= 0 or min_frequency < 1:
        raise ValueError("Sample/evaluation sizes and minimum frequency must be positive")
    data, output = Path(data), Path(output)
    for split in ("train", "val"):
        if not (data / f"{split}.jsonl").exists():
            raise FileNotFoundError(data / f"{split}.jsonl")
    if (output / "report.json").exists() or any((output / str(size) / "tokenizer.json").exists()
                                               for size in vocab_sizes):
        raise FileExistsError("Tokenizer outputs exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "algorithm": "byte-level BPE", "seed": seed,
              "special_tokens": [EOS_TOKEN], "normalizer": None, "metrics": []}
    with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=output) as work:
        sample = Path(work) / "sample.jsonl"
        manifest = _build_sample(data / "train.jsonl", sample, Path(work) / "offsets.sqlite3",
                                 int(sample_mb * 1_000_000), seed)
        write_json(output / "sample_manifest.json", manifest)
        for size in vocab_sizes:
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=size, min_frequency=min_frequency, special_tokens=[EOS_TOKEN],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
            )
            tokenizer.train_from_iterator(
                (record["text"] for record in iter_documents(sample)),
                trainer=trainer, length=manifest["sample_documents"],
            )
            directory = output / str(size)
            directory.mkdir(exist_ok=True)
            path = directory / "tokenizer.json"
            tokenizer.save(str(path))
            train_metrics = tokenizer_metrics(tokenizer, sample)
            validation_metrics = tokenizer_metrics(tokenizer, data / "val.jsonl",
                                                    max_bytes=int(eval_mb * 1_000_000))
            if train_metrics["roundtrip_failures"] or validation_metrics["roundtrip_failures"]:
                raise ValueError(f"Tokenizer {size} failed exact encode/decode roundtrip")
            result = {
                "requested_vocab_size": size, "actual_vocab_size": tokenizer.get_vocab_size(),
                "tokenizer_sha256": sha256_file(path), "eos_id": tokenizer.token_to_id(EOS_TOKEN),
                "train_sample": train_metrics, "validation_sample": validation_metrics,
            }
            write_json(directory / "metrics.json", result)
            report["metrics"].append(result)
            write_json(output / "report.json", report)
    return report


def encode_corpus(data: str | Path, tokenizer_path: str | Path, output: str | Path) -> dict:
    data, tokenizer_path, output = Path(data), Path(tokenizer_path), Path(output)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError("Vocabulary does not fit uint16")
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer has no required EOS token {EOS_TOKEN!r}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta.json").exists() or any((output / f"{split}.bin").exists() for split in SPLITS):
        raise FileExistsError("Token outputs already exist; choose a new output directory")
    for split in SPLITS:
        if not (data / f"{split}.jsonl").exists():
            raise FileNotFoundError(data / f"{split}.jsonl")
    meta = {
        "schema_version": 1, "dtype": "uint16", "byte_order": "little",
        "vocab_size": vocab_size, "eos_id": eos_id,
        "tokenizer_sha256": sha256_file(tokenizer_path), "splits": {},
        "document_boundaries": "one EOS before first document and one after every document",
    }
    stats_path = data / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        meta["clean_data_stats_sha256"] = sha256_file(stats_path)
        meta["dev_only_skip_lid"] = stats.get("dev_only_skip_lid", False)
    # Persist document offsets for analysis/benchmark preparation without seeking raw text.
    # The trainer consumes plain streams and may attend across EOS boundaries.
    for split in SPLITS:
        bin_path = output / f"{split}.bin"
        index_path = output / f"{split}.index.jsonl"
        tokens, documents, utf8_bytes = 0, 0, 0
        source_counts, source_tokens = Counter(), Counter()
        with bin_path.open("wb") as binary, index_path.open("w", encoding="utf-8") as index:
            for record in iter_documents(data / f"{split}.jsonl"):
                if EOS_TOKEN in record["text"]:
                    raise ValueError("Reserved EOS spelling appears in corpus text")
                if documents == 0:
                    binary.write(np.asarray([eos_id], dtype="<u2").tobytes())
                    tokens += 1
                ids = tokenizer.encode(record["text"], add_special_tokens=False).ids
                if not ids or min(ids) < 0 or max(ids) >= vocab_size:
                    raise ValueError("Invalid or empty token sequence")
                binary.write(np.asarray(ids + [eos_id], dtype="<u2").tobytes())
                source = record.get("source", "unknown")
                identity = record.get("id", hashlib.sha256(record["text"].encode("utf-8")).hexdigest())
                index.write(json.dumps({
                    "id": identity, "source": source, "start": tokens,
                    "tokens": len(ids), "eos_offset": tokens + len(ids),
                }, ensure_ascii=False) + "\n")
                tokens += len(ids) + 1
                documents += 1
                utf8_bytes += len(record["text"].encode("utf-8"))
                source_counts[source] += 1
                source_tokens[source] += len(ids) + 1
        meta["splits"][split] = {
            "tokens": tokens, "documents": documents, "utf8_bytes": utf8_bytes,
            "sha256": sha256_file(bin_path), "jsonl_sha256": sha256_file(data / f"{split}.jsonl"),
            "index_sha256": sha256_file(index_path), "source_counts": dict(source_counts),
            "source_tokens": dict(source_tokens),
        }
    if meta["splits"]["train"]["documents"] == 0:
        raise ValueError("Empty train split")
    write_json(output / "meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--vocab-sizes", nargs="+", type=int, default=[8192, 16384, 32768])
    train.add_argument("--sample-mb", type=float, default=100)
    train.add_argument("--eval-mb", type=float, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--min-frequency", type=int, default=2)
    encode = commands.add_parser("encode")
    encode.add_argument("--data", type=Path, required=True)
    encode.add_argument("--tokenizer", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        result = train_tokenizers(
            args.data, args.output, vocab_sizes=args.vocab_sizes, sample_mb=args.sample_mb,
            eval_mb=args.eval_mb, seed=args.seed, min_frequency=args.min_frequency,
        )
        summary = {
            "tokenizers": [
                {"requested_vocab_size": item["requested_vocab_size"],
                 "actual_vocab_size": item["actual_vocab_size"],
                 "validation_bytes_per_token": item["validation_sample"]["bytes_per_token"],
                 "roundtrip_failures": item["train_sample"]["roundtrip_failures"]
                     + item["validation_sample"]["roundtrip_failures"]}
                for item in result["metrics"]
            ], "report": str(args.output / "report.json"),
        }
    else:
        result = encode_corpus(args.data, args.tokenizer, args.output)
        summary = {"vocab_size": result["vocab_size"],
                   "split_tokens": {name: split["tokens"] for name, split in result["splits"].items()},
                   "metadata": str(args.output / "meta.json")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
