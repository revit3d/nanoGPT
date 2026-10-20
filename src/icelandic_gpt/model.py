"""A small, conventional pre-LN GPT. No downloaded weights or hidden model API."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 16384
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self):
        if min(self.block_size, self.vocab_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)


class LayerNorm(nn.Module):
    def __init__(self, width: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q, k, v = (
            t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """targets are ALREADY shifted by the loader. Never shift them here again."""
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape [batch, nonempty time]")
        if idx.size(1) > self.config.block_size:
            raise ValueError("Input is longer than block_size")
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must match idx shape")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def configure_optimizer(self, learning_rate=6e-4, weight_decay=0.1, betas=(0.9, 0.95)):
        # named_parameters deduplicates the tied embedding/head.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=tuple(betas),
            fused=next(self.parameters()).device.type == "cuda",
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.95,
                 top_k=None, eos_id=None, generator=None):
        if temperature < 0 or not 0 < top_p <= 1 or max_new_tokens < 0:
            raise ValueError("Invalid sampling parameters")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                logits = logits[:, -1, :].float()
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                        logits = logits.masked_fill(logits < cutoff, float("-inf"))
                    if top_p < 1:
                        values, indices = logits.sort(descending=True)
                        remove = values.softmax(-1).cumsum(-1) > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        values = values.masked_fill(remove, float("-inf"))
                        logits = torch.full_like(logits, float("-inf")).scatter(1, indices, values)
                    next_token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
                if eos_id is not None:
                    next_token = next_token.masked_fill(finished[:, None], eos_id)
                    finished |= next_token[:, 0] == eos_id
                idx = torch.cat((idx, next_token), dim=1)
                if eos_id is not None and finished.all():
                    break
        finally:
            self.train(was_training)
        return idx
