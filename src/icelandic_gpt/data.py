"""Download and clean document-level Icelandic data without loading a corpus in RAM.

Raw archives are read in place. SQLite holds texts, duplicate clusters and LSH
buckets. Near-duplicate discovery is approximate; candidate matches are always
confirmed with actual word-shingle Jaccard similarity before clustering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import struct
import sys
import tempfile
import unicodedata
import zipfile

REPO_ID = "arnastofnun/IGC-2024"
REVISION = "c98e8fba2ac8ec7d75643ec9a601d19eaeefc047"
SOURCES = {
    "visir": ("igc_news1/igc_news1_visir.zip", 2178105336,
              "2885e013bdb5efe616e161d52a36efd796cffc01e8f3cd6365f8c8fc6e0d43ce"),
    "ruv": ("igc_news1/igc_news1_ruv.zip", 867657881,
            "0be44f99a03413abf4311083459e749c147b640384b3a77c0faf346eebe5c4e5"),
    "frettabladid": ("igc_news1/igc_news1_frettabladid_is.zip", 385039303,
                    "36790620584ff9e0ab7e62efe474e1c7db9e7f7e480d96ce211c6e6b65462409"),
    "wiki": ("igc_wiki/igc_wiki.zip", 108375634,
             "4acecd6357147c0e5090224dce159a96d6861fe476a6403c7b9d3119edfe4835"),
}
DEFAULT_WEIGHTS = {"visir": .60, "ruv": .25, "frettabladid": .10, "wiki": .05}
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_corpus(output: str | Path, sources: list[str] | None = None) -> dict:
    """Use the HF cache's resume support, then verify committed LFS SHA-256s."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.json"
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "license": "CC-BY-4.0", "files": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("repo_id") != REPO_ID or manifest.get("revision") != REVISION:
            raise ValueError("Output contains a download manifest for a different dataset revision")
    for source in sources or list(SOURCES):
        name, expected_size, expected_sha = SOURCES[source]
        path = Path(hf_hub_download(
            repo_id=REPO_ID, filename=name, repo_type="dataset",
            revision=REVISION, local_dir=output,
        ))
        actual_sha = sha256_file(path)
        if path.stat().st_size != expected_size or actual_sha != expected_sha:
            raise ValueError(f"Archive integrity check failed: {path}")
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != name]
        manifest["files"].append({
            "source": source, "path": name, "bytes": expected_size, "sha256": actual_sha,
            "url": f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{name}",
        })
        manifest["files"].sort(key=lambda entry: entry["path"])
        write_json(manifest_path, manifest)
    return manifest


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        c if c in "\n\t" or not unicodedata.category(c).startswith("C") else ""
        for c in text
    )
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _canonical_source(path: Path, metadata: dict, record: dict) -> str:
    for source, (filename, _, _) in SOURCES.items():
        if path.name == Path(filename).name:
            return source
    source = record.get("source") or metadata.get("source") or path.stem
    if not isinstance(source, str):
        source = json.dumps(source, ensure_ascii=False, sort_keys=True)
    lowered = source.lower()
    for key, needles in {
        "visir": ("visir", "vísir"),
        "ruv": ("ruv", "rúv"),
        "frettabladid": ("frettablad", "fréttabla"),
        "wiki": ("wiki",),
    }.items():
        if any(needle in lowered for needle in needles):
            return key
    return source


def _records_from_lines(lines, path: Path, member: str):
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON at {path}:{member}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{member}:{line_number}")
        text = record.get("document", record.get("text"))
        if not isinstance(text, str):
            raise ValueError(f"Missing string document/text at {path}:{member}:{line_number}")
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"original_metadata": metadata}
        yield {
            "text": text, "source": _canonical_source(path, metadata, record),
            "origin": f"{path.name}:{member}:{line_number}",
            "original_id": record.get("uuid", record.get("id", metadata.get("xml_id"))),
            "publish_timestamp": metadata.get("publish_timestamp"),
        }


def iter_raw_documents(input_dir: str | Path):
    """Support the verified IGC JSONL-in-ZIP schema and local JSONL fixtures."""
    input_dir = Path(input_dir)
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in {".zip", ".jsonl"})
    if not paths:
        raise ValueError(f"No ZIP or JSONL inputs found under {input_dir}")
    for path in paths:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = sorted(name for name in archive.namelist()
                                 if name.lower().endswith((".jsonl", ".ndjson")))
                if not members:
                    raise ValueError(f"No JSONL members in {path}")
                for member in members:
                    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as handle:
                        yield from _records_from_lines(handle, path, member)
        else:
            with path.open(encoding="utf-8") as handle:
                yield from _records_from_lines(handle, path, "")


def inspect_inputs(input_dir: Path) -> list[dict]:
    """Record what is actually consumed, including custom files absent from a manifest."""
    records = []
    known = {Path(name).name: (size, digest) for name, size, digest in SOURCES.values()}
    for path in sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in {".zip", ".jsonl"}):
        size, digest = path.stat().st_size, sha256_file(path)
        if path.name in known and (size, digest) != known[path.name]:
            raise ValueError(f"Pinned IGC archive integrity check failed: {path}")
        records.append({"path": str(path.relative_to(input_dir)), "bytes": size, "sha256": digest,
                        "pinned_igc_archive": path.name in known})
    return records


class LanguageIdentifier:
    def __init__(self, path: str | Path):
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
        self.model = fasttext.load_model(str(path))

    def __call__(self, text: str) -> tuple[str, float]:
        # Ask for all labels: p(is) is meaningful even when another language wins.
        # This also avoids fasttext 0.9.x's predict/NumPy-2 copy=False failure.
        words = text.replace("\n", " ").strip()
        predictions = self.model.f.predict(words + "\n", -1, 0.0, "strict")
        probabilities = {label.removeprefix("__label__"): float(prob) for prob, label in predictions}
        return max(probabilities, key=probabilities.get), probabilities.get("is", 0.0)


def word_shingles(text: str, n: int = 5) -> set[bytes]:
    words = WORD_RE.findall(text.casefold())
    if not words:
        return set()
    width = min(n, len(words))
    return {hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(len(words) - width + 1)}


class DiskDeduplicator:
    """Fixed 128-permutation MinHash; 32x4 bands have high recall around J=.8."""
    def __init__(self, connection: sqlite3.Connection, seed: int, threshold: float):
        try:
            from datasketch import MinHash
        except ImportError as exc:
            raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
        self.MinHash = MinHash
        self.db = connection
        self.seed = seed
        self.threshold = threshold
        self.confirmed_edges = 0
        self.candidate_pairs = 0

    def root(self, idx: int) -> int:
        trail = []
        while True:
            parent = self.db.execute("SELECT parent FROM documents WHERE idx=?", (idx,)).fetchone()[0]
            if parent == idx:
                break
            trail.append(idx)
            idx = parent
        self.db.executemany("UPDATE documents SET parent=? WHERE idx=?", ((idx, item) for item in trail))
        return idx

    def add(self, idx: int, text: str) -> None:
        shingles = word_shingles(text)
        mh = self.MinHash(num_perm=128, seed=self.seed)
        batch = []
        for shingle in sorted(shingles):
            batch.append(shingle)
            if len(batch) == 512:
                mh.update_batch(batch)
                batch.clear()
        if batch:
            mh.update_batch(batch)
        bands = [(band, struct.pack("<4Q", *(int(x) for x in mh.hashvalues[band * 4:band * 4 + 4])))
                 for band in range(32)]
        candidates = set()
        for band, signature in bands:
            candidates.update(row[0] for row in self.db.execute(
                "SELECT doc FROM buckets WHERE band=? AND signature=?", (band, signature)))
        for other in sorted(candidates):
            other_text = self.db.execute("SELECT text FROM documents WHERE idx=?", (other,)).fetchone()[0]
            other_shingles = word_shingles(other_text)
            union = len(shingles | other_shingles)
            similarity = len(shingles & other_shingles) / union if union else 1.0
            self.candidate_pairs += 1
            if similarity >= self.threshold:
                a, b = self.root(idx), self.root(other)
                self.db.execute("UPDATE documents SET parent=? WHERE idx=?", (min(a, b), max(a, b)))
                self.confirmed_edges += 1
        self.db.executemany("INSERT INTO buckets(band,signature,doc) VALUES (?,?,?)",
                            ((band, signature, idx) for band, signature in bands))


class Reservoir:
    def __init__(self, seed: int, size: int = 100):
        self.rng, self.size, self.seen, self.items = random.Random(seed), size, 0, []

    def add(self, record: dict):
        self.seen += 1
        # Audit samples are excerpts, not an extra corpus copy.
        record = {**record, "text": record.get("text", "")[:2000]}
        if len(self.items) < self.size:
            self.items.append(record)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.size:
                self.items[index] = record


def _record_stats(target: dict, source: str, text: str) -> None:
    target["documents"] += 1
    target["bytes"] += len(text.encode("utf-8"))
    target["characters"] += len(text)
    target["sources"][source] += 1
    target["length_histogram"][str(2 ** max(0, len(text).bit_length() - 1))] += 1


def _new_stats():
    return {"documents": 0, "bytes": 0, "characters": 0,
            "sources": Counter(), "length_histogram": Counter()}


def _word_summary(db: sqlite3.Connection, phase: str) -> dict:
    row = db.execute("SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(n=1),0) "
                     "FROM words WHERE phase=?", (phase,)).fetchone()
    top = db.execute("SELECT word,n FROM words WHERE phase=? ORDER BY n DESC,word LIMIT 100",
                     (phase,)).fetchall()
    return {"types": row[0], "tokens": row[1], "hapax_types": row[2],
            "top_100": [{"word": word, "count": count} for word, count in top]}


def _flush_words(db: sqlite3.Connection, phase: str, counts: Counter) -> None:
    db.executemany("INSERT INTO words(phase,word,n) VALUES (?,?,?) ON CONFLICT(phase,word) "
                   "DO UPDATE SET n=n+excluded.n",
                   ((phase, word, count) for word, count in counts.items()))
    counts.clear()


def _split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Round by source; reserve a holdout for each enabled split when possible."""
    val, test = int(round(n * ratios[1])), int(round(n * ratios[2]))
    if n >= 3:
        val, test = max(int(ratios[1] > 0), val), max(int(ratios[2] > 0), test)
    while val + test >= n and val + test > 0:
        if val >= test:
            val -= 1
        else:
            test -= 1
    return n - val - test, val, test


def _source_budgets(available: dict[str, int], budget: int, weights: dict[str, float]) -> dict[str, float]:
    """Redistribute exhausted quotas without repeating documents."""
    if set(available) - set(weights):
        raise ValueError(f"Missing source weights for {sorted(set(available) - set(weights))}")
    result = {source: 0.0 for source in available}
    remaining = float(min(budget, sum(available.values())))
    active = {source for source in available if weights[source] > 0}
    while active and remaining > 0:
        total_weight = sum(weights[source] for source in active)
        proposal = {source: remaining * weights[source] / total_weight for source in active}
        capped = {source for source in active if proposal[source] >= available[source] - result[source]}
        if not capped:
            for source, amount in proposal.items():
                result[source] += amount
            break
        for source in capped:
            amount = available[source] - result[source]
            result[source] += amount
            remaining -= amount
        active -= capped
    return result


def prepare_corpus(
    input_dir: str | Path, output_dir: str | Path, *, lid_model: str | Path | None = None,
    skip_lid: bool = False, seed: int = 42, min_chars: int = 200, min_letters_ratio: float = .60,
    min_is_probability: float = .80, dedup_threshold: float = .80,
    split_ratios: tuple[float, float, float] = (.98, .01, .01),
    max_train_bytes: int | None = None, source_weights: dict[str, float] | None = None,
    language_identifier=None, keep_spool: bool = False,
) -> dict:
    """Prepare immutable, deduplicated, source-stratified document splits.

    language_identifier is an injectable (text -> top_language, p_is) function
    used by offline tests. Production requires the fastText LID model.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output must not be inside input (would ingest earlier outputs).")
    if min_chars < 1 or not 0 <= min_letters_ratio <= 1 or not 0 <= min_is_probability <= 1:
        raise ValueError("Invalid text filtering thresholds")
    if not 0 < dedup_threshold <= 1 or len(split_ratios) != 3 or any(x < 0 for x in split_ratios):
        raise ValueError("Invalid dedup threshold or split ratios")
    if not math.isclose(sum(split_ratios), 1.0) or split_ratios[0] <= 0:
        raise ValueError("Split ratios must sum to 1 with positive train fraction")
    if max_train_bytes is not None and max_train_bytes <= 0:
        raise ValueError("max_train_bytes must be positive")
    if source_weights and max_train_bytes is None:
        raise ValueError("--source-weights requires --max-train-bytes")
    if source_weights and (any(not math.isfinite(w) or w < 0 for w in source_weights.values())
                           or not sum(source_weights.values()) > 0):
        raise ValueError("Source weights must be finite, nonnegative and have a positive sum")
    if skip_lid and (lid_model or language_identifier):
        raise ValueError("Choose either language identification or explicit --skip-lid")
    if not skip_lid and language_identifier is None:
        if not lid_model:
            raise ValueError("Supply --lid-model lid.176.bin; --skip-lid is for fixtures/development only")
        language_identifier = LanguageIdentifier(lid_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [output_dir / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "stats.json")]
    if any(path.exists() for path in protected):
        raise FileExistsError("Prepared outputs already exist; choose a new output directory.")
    stats = {
        "schema_version": 1, "seed": seed, "dev_only_skip_lid": skip_lid,
        "filter": {"min_chars": min_chars, "min_letters_ratio": min_letters_ratio,
                   "min_is_probability": min_is_probability},
        "split_ratios": split_ratios, "split_policy": "hash-order stratified by representative source",
        "raw": _new_stats(), "filtered_unique": _new_stats(), "final": _new_stats(),
        "rejected": Counter(), "splits": {}, "dedup": {
            "method": "MinHash LSH candidates, exact word-5-shingle Jaccard confirmation",
            "threshold": dedup_threshold, "permutations": 128, "bands": 32,
            "rows_per_band": 4, "candidate_recall_is_approximate": True,
            "cluster_policy": "one representative (minimum content SHA-256) per connected component",
        },
        "train_sampling": {"max_bytes": max_train_bytes, "source_weights": source_weights,
                           "unit": "UTF-8 bytes", "replacement": False},
    }
    manifest = input_dir / "download_manifest.json"
    if manifest.exists():
        stats["input_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    if lid_model:
        stats["lid_model_sha256"] = sha256_file(lid_model)
    stats["actual_input_files"] = inspect_inputs(input_dir)
    accepted, rejected = Reservoir(seed), Reservoir(seed + 1)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as work:
        db_path = Path(work) / "corpus.sqlite3"
        db = sqlite3.connect(db_path)
        db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-65536;
            CREATE TABLE documents(idx INTEGER PRIMARY KEY, id TEXT UNIQUE, text TEXT,
                source TEXT, provenance TEXT, parent INTEGER, split TEXT, ordering TEXT, bytes INTEGER);
            CREATE TABLE provenance(doc INTEGER, source TEXT, origin TEXT,
                UNIQUE(doc,source,origin));
            CREATE TABLE buckets(band INTEGER, signature BLOB, doc INTEGER);
            CREATE INDEX bucket_lookup ON buckets(band,signature);
            CREATE TABLE words(phase TEXT,word TEXT,n INTEGER,PRIMARY KEY(phase,word)) WITHOUT ROWID;
        """)
        dedup = DiskDeduplicator(db, seed, dedup_threshold)
        raw_words = Counter()
        try:
            for number, record in enumerate(iter_raw_documents(input_dir), start=1):
                original = record["text"]
                _record_stats(stats["raw"], record["source"], original)
                raw_words.update(WORD_RE.findall(original.casefold()))
                if len(raw_words) >= 100_000:
                    _flush_words(db, "raw", raw_words)
                text = normalize_text(original)
                reason = None
                language, p_is = None, None
                nonspace = sum(not c.isspace() for c in text)
                if len(text) < min_chars:
                    reason = "too_short"
                elif sum(c.isalpha() for c in text) / max(nonspace, 1) < min_letters_ratio:
                    reason = "low_letter_ratio"
                elif "<|endoftext|>" in text:
                    reason = "reserved_token_literal"
                elif not skip_lid:
                    language, p_is = language_identifier(text)
                    if language != "is" or p_is < min_is_probability:
                        reason = "language"
                audit = {**record, "text": text, "top_language": language, "p_is": p_is}
                if reason:
                    stats["rejected"][reason] += 1
                    rejected.add({**audit, "reason": reason})
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                existing = db.execute("SELECT idx FROM documents WHERE id=?", (digest,)).fetchone()
                if existing:
                    stats["rejected"]["exact_duplicate"] += 1
                    db.execute("INSERT OR IGNORE INTO provenance VALUES (?,?,?)",
                               (existing[0], record["source"], record["origin"]))
                    rejected.add({**audit, "reason": "exact_duplicate"})
                    continue
                provenance = {key: value for key, value in record.items() if key != "text"}
                provenance.update({"p_is": p_is, "top_language": language})
                cursor = db.execute(
                    "INSERT INTO documents(id,text,source,provenance,bytes) VALUES (?,?,?,?,?)",
                    (digest, text, record["source"], json.dumps(provenance, ensure_ascii=False),
                     len(text.encode("utf-8"))))
                idx = cursor.lastrowid
                db.execute("UPDATE documents SET parent=? WHERE idx=?", (idx, idx))
                db.execute("INSERT INTO provenance VALUES (?,?,?)", (idx, record["source"], record["origin"]))
                dedup.add(idx, text)
                _record_stats(stats["filtered_unique"], record["source"], text)
                accepted.add(audit)
                if number % 1000 == 0:
                    db.commit()
                    print(f"prepare: {number:,} raw documents; "
                          f"{stats['filtered_unique']['documents']:,} filtered unique", file=sys.stderr)
            _flush_words(db, "raw", raw_words)
            db.commit()
            # Resolve all graph components before selecting representatives/splits.
            for (idx,) in db.execute("SELECT idx FROM documents"):
                dedup.root(idx)
            db.execute("CREATE INDEX cluster_members ON documents(parent)")
            db.execute("CREATE TEMP TABLE representatives AS SELECT MIN(id) AS id,parent "
                       "FROM documents GROUP BY parent")
            for digest, in db.execute("SELECT id FROM representatives"):
                ordering = hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()
                db.execute("UPDATE documents SET ordering=? WHERE id=?", (ordering, digest))
            db.execute("CREATE INDEX source_order ON documents(source,ordering)")
            source_counts = dict(db.execute(
                "SELECT source,COUNT(*) FROM documents WHERE ordering IS NOT NULL GROUP BY source"))
            if not source_counts:
                raise ValueError("No documents remain after cleaning")
            for source, count in source_counts.items():
                n_train, n_val, _ = _split_counts(count, split_ratios)
                rows = db.execute("SELECT idx FROM documents WHERE source=? AND ordering IS NOT NULL "
                                  "ORDER BY ordering", (source,))
                for position, (idx,) in enumerate(rows):
                    split = "train" if position < n_train else "val" if position < n_train + n_val else "test"
                    db.execute("UPDATE documents SET split=? WHERE idx=?", (split, idx))
            available = dict(db.execute(
                "SELECT source,SUM(bytes) FROM documents WHERE split='train' GROUP BY source"))
            budgets = None
            if max_train_bytes is not None:
                weights = source_weights or {source: amount for source, amount in available.items()}
                budgets = _source_budgets(available, max_train_bytes, weights)
            stats["train_sampling"]["source_byte_budgets"] = budgets
            stats["dedup"]["candidate_pairs"] = dedup.candidate_pairs
            stats["dedup"]["confirmed_edges"] = dedup.confirmed_edges
            representatives = sum(source_counts.values())
            stats["rejected"]["near_duplicate"] = stats["filtered_unique"]["documents"] - representatives
            final_words, consumed = Counter(), Counter()
            with ExitStack() as stack:
                handles = {split: stack.enter_context((output_dir / f"{split}.jsonl").open("w", encoding="utf-8"))
                           for split in ("train", "val", "test")}
                split_stats = {split: _new_stats() for split in handles}
                for digest, text, source, provenance, parent, split, n_bytes in db.execute(
                    "SELECT id,text,source,provenance,parent,split,bytes FROM documents "
                    "WHERE split IS NOT NULL ORDER BY ordering"
                ):
                    if split == "train" and budgets is not None:
                        # Whole documents only; shortfall is reported, never oversampled.
                        if consumed[source] + n_bytes > budgets[source]:
                            stats["rejected"]["train_byte_budget"] += 1
                            continue
                        consumed[source] += n_bytes
                    provenance_sources = [row[0] for row in db.execute(
                        "SELECT DISTINCT p.source FROM provenance p JOIN documents d ON p.doc=d.idx "
                        "WHERE d.parent=? ORDER BY p.source", (parent,))]
                    item = {"id": digest, "text": text, "source": source,
                            "provenance": json.loads(provenance), "cluster_sources": provenance_sources}
                    handles[split].write(json.dumps(item, ensure_ascii=False) + "\n")
                    _record_stats(stats["final"], source, text)
                    _record_stats(split_stats[split], source, text)
                    final_words.update(WORD_RE.findall(text.casefold()))
                    if len(final_words) >= 100_000:
                        _flush_words(db, "final", final_words)
            _flush_words(db, "final", final_words)
            db.commit()
            stats["words"] = {phase: _word_summary(db, phase) for phase in ("raw", "final")}
            for split, value in split_stats.items():
                stats["splits"][split] = {**value, "sha256": sha256_file(output_dir / f"{split}.jsonl")}
            for name, reservoir in (("accepted_sample", accepted), ("rejected_sample", rejected)):
                with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in reservoir.items:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not stats["splits"]["train"]["documents"]:
                raise ValueError("Training split is empty; increase data/budget")
            write_json(output_dir / "stats.json", stats)
        finally:
            db.close()
        if keep_spool:
            db_path.replace(output_dir / "corpus.sqlite3")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch and verify four pinned IGC archives")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--sources", nargs="+", choices=list(SOURCES))
    prepare = commands.add_parser("prepare", help="Clean, deduplicate and split complete documents")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--lid-model", type=Path)
    prepare.add_argument("--skip-lid", action="store_true", help="Explicit development/fixture mode only")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--min-chars", type=int, default=200)
    prepare.add_argument("--min-letters-ratio", type=float, default=.60)
    prepare.add_argument("--min-is-probability", type=float, default=.80)
    prepare.add_argument("--dedup-threshold", type=float, default=.80)
    prepare.add_argument("--split-ratios", nargs=3, type=float, default=(.98, .01, .01))
    prepare.add_argument("--max-train-bytes", type=int)
    prepare.add_argument("--source-weights", help="e.g. visir=.6,ruv=.25,frettabladid=.1,wiki=.05")
    prepare.add_argument("--keep-spool", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        result = download_corpus(args.output, args.sources)
        summary = {"revision": result["revision"], "archives": len(result["files"]),
                   "manifest": str(args.output / "download_manifest.json")}
    else:
        if args.skip_lid:
            print("WARNING: --skip-lid is development/fixture mode. These data are not "
                  "a language-filtered production corpus.", file=sys.stderr)
        weights = None
        if args.source_weights:
            try:
                weights = {name: float(value) for name, value in
                           (part.split("=", 1) for part in args.source_weights.split(","))}
            except ValueError as exc:
                parser.error(f"Invalid --source-weights: {exc}")
        result = prepare_corpus(
            args.input, args.output, lid_model=args.lid_model, skip_lid=args.skip_lid,
            seed=args.seed, min_chars=args.min_chars, min_letters_ratio=args.min_letters_ratio,
            min_is_probability=args.min_is_probability, dedup_threshold=args.dedup_threshold,
            split_ratios=tuple(args.split_ratios), max_train_bytes=args.max_train_bytes,
            source_weights=weights, keep_spool=args.keep_spool,
        )
        summary = {
            "raw_documents": result["raw"]["documents"],
            "final_documents": result["final"]["documents"], "rejected": result["rejected"],
            "splits": {name: {"documents": split["documents"], "bytes": split["bytes"]}
                       for name, split in result["splits"].items()},
            "dev_only_skip_lid": result["dev_only_skip_lid"], "report": str(args.output / "stats.json"),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
