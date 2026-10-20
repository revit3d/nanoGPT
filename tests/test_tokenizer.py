import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from icelandic_gpt.tokenizer import EOS_TOKEN, encode_corpus, train_tokenizers


def write_splits(root: Path, train: list[str], val: list[str], test: list[str]) -> None:
    root.mkdir()
    for split, texts in (("train", train), ("val", val), ("test", test)):
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({
                    "id": hashlib.sha256(text.encode()).hexdigest(), "text": text, "source": "fixture"
                }, ensure_ascii=False) + "\n")


def test_tokenizer_roundtrip_unseen_unicode_and_encoded_boundaries(tmp_path):
    texts = ["Þetta er íslenskur texti.\n\nÁ Íslandi er kalt.", "Hesturinn hleypur yfir græna túnið."]
    clean = tmp_path / "clean"
    write_splits(clean, texts, ["Óþekkt orð og tölur: 12345."], ["Annað dæmi."])
    result = train_tokenizers(clean, tmp_path / "tokenizers", vocab_sizes=[512], sample_mb=.01)
    path = tmp_path / "tokenizers" / "512" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    for text in ["🙂 日本語 Þ æ ö e\u0301 \n\t🌋", *texts]:
        assert tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) == text
    assert result["metrics"][0]["train_sample"]["roundtrip_failures"] == 0
    assert result["metrics"][0]["actual_vocab_size"] >= 257
    metadata = encode_corpus(clean, path, tmp_path / "tokens")
    eos = tokenizer.token_to_id(EOS_TOKEN)
    expected = [eos]
    for text in texts:
        expected += tokenizer.encode(text).ids + [eos]
    actual = np.fromfile(tmp_path / "tokens" / "train.bin", dtype="<u2")
    assert actual.tolist() == expected
    assert metadata["splits"]["train"]["tokens"] == len(expected)
    assert metadata["splits"]["train"]["sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    assert metadata["tokenizer_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    indexes = [json.loads(line) for line in (tmp_path / "tokens" / "train.index.jsonl").read_text().splitlines()]
    assert indexes[0]["start"] == 1
    assert indexes[1]["start"] == indexes[0]["eos_offset"] + 1


def test_validation_never_changes_training_sample_or_vocabulary(tmp_path):
    train = ["Íslenskur texti með mörgum fallegum orðum. " * 10,
             "Veðrið á morgun verður gott og bjart. " * 10]
    a, b = tmp_path / "a", tmp_path / "b"
    write_splits(a, train, ["Eðlileg staðfesting."], ["Próf."])
    write_splits(b, train, ["zzzyyyxxxyyyzzz " * 100], ["Different test data."])
    train_tokenizers(a, tmp_path / "ta", vocab_sizes=[300], sample_mb=.01)
    train_tokenizers(b, tmp_path / "tb", vocab_sizes=[300], sample_mb=.01)
    first = Tokenizer.from_file(str(tmp_path / "ta" / "300" / "tokenizer.json"))
    second = Tokenizer.from_file(str(tmp_path / "tb" / "300" / "tokenizer.json"))
    assert first.get_vocab() == second.get_vocab()
    for text in train:
        assert first.encode(text).ids == second.encode(text).ids


def test_reject_vocab_that_cannot_represent_all_bytes(tmp_path):
    with pytest.raises(ValueError, match="257"):
        train_tokenizers(tmp_path, tmp_path / "out", vocab_sizes=[256])
