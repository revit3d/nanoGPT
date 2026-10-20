"""Checks for evaluation accounting, Unicode boundaries, and benchmark balance."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from icelandic_gpt.evaluate import (
    build_benchmark, evaluate_benchmark, evaluate_lm, score_token_ids, truncate_text,
)
from icelandic_gpt.generate import EOS_TOKEN, generate_text, load_bundle


class MarkovModel(torch.nn.Module):
    """Known local next-token distribution, invariant to evaluation windowing."""

    def __init__(self, vocab_size=11, block_size=7):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(block_size=block_size, vocab_size=vocab_size)
        self.window_lengths = []

    def forward(self, indices, targets=None):
        self.window_lengths.append(indices.shape[1])
        logits = torch.full(
            (*indices.shape, self.config.vocab_size), -1.0, device=indices.device,
        )
        logits.scatter_(-1, ((indices + 1) % self.config.vocab_size).unsqueeze(-1), 3)
        return logits, None


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [
            "Ísland er eyja í Norður-Atlantshafi. Reykjavík er höfuðborg Íslands.",
            "Þjóðin talar íslensku. Börn lesa bækur og læra ný orð í skólanum.",
        ] * 20,
        trainer=trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=[EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tokenizer


def direct_markov_nll(model, ids, eos_id=0, score_start=0):
    inputs = torch.tensor([[eos_id] + ids[:-1]])
    logits, _ = model(inputs)
    return float(F.cross_entropy(
        logits[0, score_start:], torch.tensor(ids[score_start:]), reduction="sum",
    ))


@pytest.mark.parametrize("stride", [1, 3, 7])
@pytest.mark.parametrize("score_start", [0, 4, 21])
def test_windowing_counts_each_shifted_target_once(stride, score_start):
    model = MarkovModel()
    ids = [(index * 3) % 10 + 1 for index in range(29)]
    expected = direct_markov_nll(model, ids, score_start=score_start)
    model.window_lengths.clear()
    nll, count = score_token_ids(model, ids, eos_id=0, stride=stride, score_start=score_start)
    assert count == len(ids) - score_start
    assert nll == pytest.approx(expected, abs=5e-5)
    assert max(model.window_lengths) <= model.config.block_size


def test_empty_and_invalid_windows():
    model = MarkovModel()
    assert score_token_ids(model, [], eos_id=0, stride=3) == (0.0, 0)
    assert score_token_ids(model, [1, 2], eos_id=0, score_start=2, stride=3) == (0.0, 0)
    with pytest.raises(ValueError, match="stride"):
        score_token_ids(model, [1, 2], eos_id=0, stride=8)
    with pytest.raises(ValueError, match="score_start"):
        score_token_ids(model, [1, 2], eos_id=0, score_start=3, stride=3)


def test_truncation_preserves_exact_unicode_bytes(tokenizer):
    text = "Ísland er fagurt. Þjóðin talar íslensku. 🌋"
    total = len(tokenizer.encode(text).ids)
    for budget in range(1, total + 1):
        prefix, ids = truncate_text(text, tokenizer, budget)
        assert text.startswith(prefix)
        assert len(ids) <= budget
        assert ids == tokenizer.encode(prefix, add_special_tokens=False).ids
        assert tokenizer.decode(ids) == prefix
        assert b"\xef\xbf\xbd" not in prefix.encode("utf-8")


def test_lm_bytes_exclude_special_tokens_and_count_documents(tmp_path, tokenizer):
    texts = ["Ísland er fagurt.", "Þjóðin talar íslensku."]
    records = [{"id": str(i), "source": "test", "text": text} for i, text in enumerate(texts)]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    model = MarkovModel(vocab_size=tokenizer.get_vocab_size(), block_size=8)
    result = evaluate_lm(model, tokenizer, data, stride=4, max_tokens=None)
    assert result["tokens"] == sum(len(tokenizer.encode(text).ids) for text in texts)
    assert result["utf8_bytes"] == sum(len(text.encode("utf-8")) for text in texts)
    assert result["documents"] == 2
    assert result["partial_documents"] == 0
    assert result["sources"]["test"]["tokens"] == result["tokens"]
    prefix, ids = truncate_text(texts[0], tokenizer, 7)
    limited_data = tmp_path / "limited.jsonl"
    limited_data.write_text(json.dumps(records[0]), encoding="utf-8")
    limited = evaluate_lm(model, tokenizer, limited_data, stride=4, max_tokens=7)
    assert limited["tokens"] == len(ids)
    assert limited["utf8_bytes"] == len(prefix.encode("utf-8"))
    assert limited["partial_documents"] == 1


def test_budgeted_lm_selection_does_not_depend_on_source_file_order(tmp_path, tokenizer):
    records = [
        {"id": "alpha", "source": "news", "text": "Ísland er fagurt. " * 10},
        {"id": "beta", "source": "wiki", "text": "Þjóðin talar íslensku. " * 10},
    ]
    model = MarkovModel(vocab_size=tokenizer.get_vocab_size(), block_size=8)
    results = []
    for index, order in enumerate([records, list(reversed(records))]):
        data = tmp_path / f"order-{index}.jsonl"
        data.write_text("\n".join(json.dumps(item) for item in order), encoding="utf-8")
        results.append(evaluate_lm(model, tokenizer, data, stride=4, max_tokens=20))
    assert results[0] == results[1]


def test_benchmark_without_context_is_counterbalanced(tmp_path, tokenizer):
    tasks = []
    candidates = [" Ísland", " Þjóðin", " Reykjavík", " Börn"]
    for answer in range(4):
        tasks.append({
            "id": str(answer), "group_id": "g0", "answer": answer,
            "prefix": "Ísland er", "candidates": candidates,
        })
    data = tmp_path / "benchmark.jsonl"
    data.write_text("\n".join(json.dumps(task) for task in tasks), encoding="utf-8")
    model = MarkovModel(vocab_size=tokenizer.get_vocab_size(), block_size=16)
    result = evaluate_benchmark(model, tokenizer, data, stride=8)
    assert result["without_context_accuracy"] == 0.25
    assert result["tasks"] == 4
    assert len(result["predictions"]) == 4
    assert result["groups"] == 1


def test_benchmark_rejects_unbalanced_answers(tmp_path, tokenizer):
    task = {"id": "0", "group_id": "g0", "answer": 0, "prefix": "Ísland",
            "candidates": [" er", " var", " hefur", " verður"]}
    data = tmp_path / "benchmark.jsonl"
    data.write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(ValueError, match="counterbalanced"):
        evaluate_benchmark(MarkovModel(), tokenizer, data, stride=3)


def test_benchmark_builder_freezes_balanced_groups(tmp_path, tokenizer):
    records = [
        {"id": str(index), "source": "news", "text": (f"Ísland {index} " * 100)}
        for index in range(8)
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    tasks = build_benchmark(path, tokenizer, groups=2, seed=7)
    assert tasks == build_benchmark(path, tokenizer, groups=2, seed=7)
    assert len(tasks) == 8
    assert len({task["document_id"] for task in tasks}) == 8
    for group_id in {task["group_id"] for task in tasks}:
        group = [task for task in tasks if task["group_id"] == group_id]
        assert sorted(task["answer"] for task in group) == [0, 1, 2, 3]
        assert all(task["candidates"] == group[0]["candidates"] for task in group)
        assert all(task["source"] == "news" for task in group)


def test_wrong_tokenizer_rejected(tmp_path, tokenizer, monkeypatch):
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    model = MarkovModel(vocab_size=tokenizer.get_vocab_size())
    monkeypatch.setattr(
        "icelandic_gpt.generate.load_model",
        lambda *args, **kwargs: (model, {"tokenizer_sha256": "incorrect"}),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle("unused.pt", path, "cpu")


def test_generation_with_real_tiny_model(tokenizer):
    from icelandic_gpt.model import GPT, GPTConfig

    torch.manual_seed(7)
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16,
                          vocab_size=tokenizer.get_vocab_size(), dropout=0.0, bias=False))
    first = generate_text(model, tokenizer, "Ísland", max_new_tokens=5, seed=42)
    second = generate_text(model, tokenizer, "Ísland", max_new_tokens=5, seed=42)
    assert first["generated_token_ids"] == second["generated_token_ids"]
    assert 1 <= first["generated_tokens"] <= 5
    assert first["prompt"] == "Ísland"
    assert isinstance(first["text"], str)
    greedy_first = generate_text(model, tokenizer, "Ísland", max_new_tokens=5,
                                 temperature=0, seed=1)
    greedy_second = generate_text(model, tokenizer, "Ísland", max_new_tokens=5,
                                  temperature=0, seed=2)
    assert greedy_first["generated_token_ids"] == greedy_second["generated_token_ids"]
