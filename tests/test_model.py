import pytest
import torch
import torch.nn.functional as F

from icelandic_gpt.inspect import memory_report
from icelandic_gpt.model import GPT, GPTConfig


def tiny_model(**kwargs):
    return GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=12, vocab_size=31, **kwargs))


def test_attention_cannot_see_future():
    torch.manual_seed(42)
    model = tiny_model().eval()
    first = torch.randint(31, (2, 12))
    changed = first.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 31
    with torch.no_grad():
        a, _ = model(first)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :6], b[:, :6], rtol=0, atol=0)
    assert not torch.equal(a[:, 6:], b[:, 6:])


def test_targets_are_shifted_once_by_caller_and_masked():
    model = tiny_model()
    sequence = torch.randint(31, (2, 13))
    x, y = sequence[:, :-1], sequence[:, 1:].clone()
    y[:, :4] = -100
    logits, loss = model(x, y)
    expected = F.cross_entropy(logits[:, 4:].reshape(-1, 31), y[:, 4:].reshape(-1))
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match="block_size"):
        model(sequence)


def test_weight_tying_and_measured_adam_memory():
    model = tiny_model()
    assert model.lm_head.weight is model.token_embedding.weight
    optimizer = model.configure_optimizer()
    x = torch.randint(31, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    optimizer.step()
    report = memory_report(model, optimizer)
    block_params = 12 * 16 * 16 + 2 * 16
    assert report["blocks"]["2"]["parameters"] == block_params
    assert report["blocks"]["2"]["gradient_bytes"] == block_params * 4
    assert report["blocks"]["2"]["adam_moment_bytes"] == block_params * 8
    assert report["blocks"]["2"]["optimizer_scalar_bytes"] == 6 * 4
    assert report["total"]["parameters"] == 31 * 16 + 12 * 16 + 2 * block_params + 16
    optimizer.zero_grad(set_to_none=True)
    assert memory_report(model, optimizer)["total"]["gradient_bytes"] == 0


def test_can_overfit_a_tiny_fixed_batch():
    torch.manual_seed(1)
    model = tiny_model()
    optimizer = model.configure_optimizer(learning_rate=0.02, weight_decay=0)
    x = torch.arange(12).repeat(2, 1)
    y = (x + 1) % 31
    initial = float(model(x, y)[1].detach())
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    assert float(model(x, y)[1].detach()) < initial * 0.1


def test_generation_restores_mode_and_respects_context():
    model = tiny_model().train()
    prompt = torch.arange(20).remainder(31).view(1, -1)
    result = model.generate(prompt, max_new_tokens=8, temperature=0)
    assert result.shape == (1, 28)
    assert model.training


@pytest.mark.parametrize("layers,expected", [(12, 46_412_288), (6, 27_531_776)])
def test_production_parameter_counts(layers, expected):
    # Meta tensors verify the real module structure without allocating model storage.
    with torch.device("meta"):
        model = GPT(GPTConfig(n_layer=layers))
    assert sum(p.numel() for p in model.parameters()) == expected
