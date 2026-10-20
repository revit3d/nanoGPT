import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from icelandic_gpt.data import (
    LanguageIdentifier, _source_budgets, inspect_inputs, iter_raw_documents,
    normalize_text, prepare_corpus, word_shingles,
)


def write_raw(path: Path, records: list[dict]) -> None:
    path.mkdir()
    with (path / "fixture.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_records(path: Path) -> dict[str, list[dict]]:
    return {split: [json.loads(line) for line in (path / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "val", "test")}


def test_normalization_preserves_paragraphs_and_icelandic():
    assert normalize_text("  A\u0301r\tog  dagur.\r\n\r\n\r\nÞetta\x00 er gott. ") == "Ár og dagur.\n\nÞetta er gott."


def test_raw_zip_reader_uses_document_schema_and_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "igc_news1_ruv.zip", "w") as archive:
        archive.writestr("news.jsonl", json.dumps({
            "document": "Þetta er íslensk frétt.", "metadata": {"source": "RÚV", "xml_id": "a1"}
        }))
    records = list(iter_raw_documents(raw))
    assert records[0]["source"] == "ruv"
    assert records[0]["original_id"] == "a1"
    assert records[0]["text"] == "Þetta er íslensk frétt."


def test_filters_dedup_before_split_and_reproducibility(tmp_path):
    pytest.importorskip("datasketch")
    base = " ".join(f"íslenska{chr(0x61 + i // 26)}{chr(0x61 + i % 26)}" for i in range(70))
    variant = base + " nýjasta fréttin"
    assert len(word_shingles(base) & word_shingles(variant)) / len(word_shingles(base) | word_shingles(variant)) > .8
    records = [
        {"document": base, "source": "ruv"},
        {"document": base, "source": "visir"},
        {"document": variant, "source": "visir"},
        {"document": "short", "source": "ruv"},
        {"document": "12345 " * 30, "source": "ruv"},
        {"document": "This is a sentence in English and is not Icelandic. " * 4, "source": "ruv"},
    ]
    # Distinct whole-word vocabularies prevent accidental near-duplication.
    for i in range(36):
        prefix = chr(0x61 + i // 26) + chr(0x61 + i % 26)
        records.append({"document": " ".join(f"{prefix}orð{chr(0x61 + j)}" for j in range(26)),
                        "source": "ruv" if i % 2 else "visir"})
    raw = tmp_path / "raw"
    write_raw(raw, records)
    classifier = lambda text: ("en", .01) if text.startswith("This") else ("is", .99)
    stats = prepare_corpus(raw, tmp_path / "out", min_chars=20,
                           split_ratios=(.6, .2, .2), language_identifier=classifier)
    assert stats["rejected"] == {"too_short": 1, "low_letter_ratio": 1, "language": 1,
                                 "exact_duplicate": 1, "near_duplicate": 1}
    splits = clean_records(tmp_path / "out")
    digests = [{r["id"] for r in values} for values in splits.values()]
    assert all(digests)
    assert not (digests[0] & digests[1] or digests[0] & digests[2] or digests[1] & digests[2])
    assert sum(map(len, digests)) == 37
    kept = [r for rows in splits.values() for r in rows if r["text"] in {base, variant}]
    assert len(kept) == 1
    assert kept[0]["cluster_sources"] == ["ruv", "visir"]
    assert kept[0]["id"] == min(hashlib.sha256(text.encode()).hexdigest() for text in (base, variant))
    prepare_corpus(raw, tmp_path / "again", min_chars=20,
                   split_ratios=(.6, .2, .2), language_identifier=classifier)
    assert splits == clean_records(tmp_path / "again")


def test_requires_explicit_language_filter_choice(tmp_path):
    with pytest.raises(ValueError, match="lid-model"):
        prepare_corpus(tmp_path / "raw", tmp_path / "output")


def test_quota_redistribution_does_not_oversample():
    budget = _source_budgets({"visir": 1000, "wiki": 10}, 500, {"visir": .6, "wiki": .4})
    assert budget == {"visir": 490, "wiki": 10}
    with pytest.raises(ValueError, match="Missing source weights"):
        _source_budgets({"visir": 10}, 10, {"wiki": 1})


def test_input_manifest_records_actual_files_and_checks_pinned_archives(tmp_path):
    path = tmp_path / "custom.jsonl"
    path.write_text('{"document":"Halló!"}\n', encoding="utf-8")
    manifest = inspect_inputs(tmp_path)
    assert manifest == [{"path": "custom.jsonl", "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "pinned_igc_archive": False}]
    (tmp_path / "igc_news1_ruv.zip").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        inspect_inputs(tmp_path)


def test_fasttext_binding_works_without_numpy_copy_false(tmp_path):
    fasttext = pytest.importorskip("fasttext")
    training = tmp_path / "tiny-lid.txt"
    training.write_text(
        ("__label__is þetta er íslenska og fallegt veður\n"
         "__label__en this is english and beautiful weather\n") * 20, encoding="utf-8"
    )
    model = fasttext.train_supervised(
        input=str(training), dim=8, epoch=2, thread=1, minCount=1, bucket=100, verbose=0,
    )
    path = tmp_path / "tiny-lid.bin"
    model.save_model(str(path))
    language, probability = LanguageIdentifier(path)("þetta er íslenska\nog fallegt veður")
    assert language in {"is", "en"}
    assert 0 <= probability <= 1
