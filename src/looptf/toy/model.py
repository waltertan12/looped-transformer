"""Tiny decoder-only transformer with optional looped core, in plain PyTorch.

One class covers three variants:

* shallow: small ``n_core``, ``n_loops=1``
* deep:    large ``n_core``, ``n_loops=1`` (more unique layers)
* looped:  small ``n_core`` reused for ``n_loops`` passes (shared weights)

Forward pass::

    e = prelude(embed(x))
    h = e
    for _ in range(n_loops):          # same core weights every pass
        u = adapter(cat([h, e]))      # inject="concat"; or h + e / h
        h = core(u)
    return lm_head(final_norm(coda(h)))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_INJECT_MODES = ("concat", "add", "none")

# Token ids from looptf.toy.data.VOCAB (kept literal so the model has no data dependency;
# tests check they match): '0'..'9' -> 1..10, '=' -> 16, '?' -> 44.
_DIGIT_LO, _DIGIT_HI, _EQ_ID, _Q_ID = 1, 10, 16, 44


def _run_index(is_digit: torch.Tensor) -> torch.Tensor:
    """For each position, how many digits precede it within the same run of digits."""
    B, T = is_digit.shape
    t = torch.arange(T, device=is_digit.device).expand(B, T)
    last_non = torch.where(is_digit, torch.full_like(t, -1), t).cummax(dim=1).values
    return t - last_non - 1


def answer_mask(x: torch.Tensor) -> torch.Tensor:
    """True for tokens strictly after the terminal ('?' if the sequence has one, else '=')."""
    has_q = (x == _Q_ID).any(dim=1, keepdim=True)
    is_term = torch.where(has_q, x == _Q_ID, x == _EQ_ID)
    c = is_term.long().cumsum(dim=1)
    return (c - is_term.long()) > 0


def place_ids(x: torch.Tensor, max_places: int, offset: torch.Tensor | None = None) -> torch.Tensor:
    """Place index per token: 0 for non-digits, else 1 + place.

    Question digits (before the terminal '=' / '?'): place counted from the END of the number,
    i.e. true place value (ones = 0, tens = 1, ...); the question is given in full, so this
    uses no hidden information. Answer digits (after the terminal): counted from the START of
    the answer, so the tag never reveals how long the answer will be (causal). With reversed
    answers, counting from the start equals place value, so question and answer line up.
    """
    is_digit = (x >= _DIGIT_LO) & (x <= _DIGIT_HI)
    from_start = _run_index(is_digit)
    from_end = _run_index(is_digit.flip(1)).flip(1)
    # terminal: '?' if the sequence has one (chain task, where '=' appears inside statements),
    # else '='; answer region = strictly after its first occurrence
    after = answer_mask(x)
    if offset is not None:  # H18: shift question-digit places (offset: [B, 1] ints)
        from_end = from_end + offset
    place = torch.where(after, from_start, from_end).clamp(max=max_places - 1)
    return torch.where(is_digit, place + 1, torch.zeros_like(place))


@dataclass
class ToyConfig:
    vocab_size: int = 17
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    max_len: int = 128
    n_prelude: int = 1  # unique layers before the core
    n_core: int = 2  # unique layers in the core block
    n_coda: int = 1  # unique layers after the core
    inject: str = "concat"  # "concat" | "add" | "none"
    tag_offset: int = 0  # H18: train-only random shift U[0, tag_offset] on question-digit tags
    sep_tags: bool = False  # H18: separate tag table for answer digits
    dropout: float = 0.0
    # Place-value embeddings (McLeish et al. 2024 call them "Abacus"): each digit token gets an
    # extra learned embedding for its place within its number. Off by default.
    place_emb: bool = False
    max_places: int = 24


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def rope_cache(max_len: int, head_dim: int, base: float = 10000.0):
    """Return cos, sin of shape (max_len, head_dim // 2), in float32."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE. x: (B, H, T, head_dim); cos/sin: (T, head_dim // 2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm transformer block: RoPE causal attention + SwiGLU, both residual."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ToyLM(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        if cfg.inject not in _INJECT_MODES:
            raise ValueError(f"inject must be one of {_INJECT_MODES}, got {cfg.inject!r}")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = cfg.d_model // cfg.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if cfg.n_core < 1:
            raise ValueError("n_core must be >= 1")
        if cfg.n_prelude < 0 or cfg.n_coda < 0:
            raise ValueError("n_prelude and n_coda must be >= 0")

        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.place = nn.Embedding(cfg.max_places + 1, cfg.d_model, padding_idx=0) if cfg.place_emb else None
        self.place_ans = (nn.Embedding(cfg.max_places + 1, cfg.d_model, padding_idx=0)
                          if cfg.place_emb and cfg.sep_tags else None)
        self.prelude = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
        self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
        self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
        # Concat adapter only exists when it is used, so num_params stays honest.
        self.adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False) if cfg.inject == "concat" else None
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = rope_cache(cfg.max_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        cfg = self.cfg
        n_layers = cfg.n_prelude + cfg.n_core + cfg.n_coda
        res_std = 0.02 / math.sqrt(2 * n_layers)  # GPT-2 style residual scaling

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for blk in [*self.prelude, *self.core, *self.coda]:
            nn.init.normal_(blk.attn.out.weight, mean=0.0, std=res_std)
            nn.init.normal_(blk.mlp.w_down.weight, mean=0.0, std=res_std)
        if self.adapter is not None:
            # Fan-in scaling so the concatenated input (two streams) maps to
            # roughly the same per-feature magnitude as h, not 0.02-shrunk.
            nn.init.normal_(self.adapter.weight, mean=0.0, std=1.0 / math.sqrt(2 * cfg.d_model))
        if self.place is not None:
            with torch.no_grad():
                self.place.weight[0].zero_()  # non-digit tokens get no place signal
                if self.place_ans is not None:
                    self.place_ans.weight[0].zero_()

    def forward(self, x: torch.Tensor, n_loops: int = 1) -> torch.Tensor:
        if n_loops < 0:
            raise ValueError("n_loops must be >= 0")
        B, T = x.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.cfg.max_len}")
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        e = self.embed(x)
        if self.place is not None:
            off = None
            if self.training and self.cfg.tag_offset > 0:
                off = torch.randint(0, self.cfg.tag_offset + 1, (B, 1), device=x.device)
            pid = place_ids(x, self.cfg.max_places, off)
            if self.place_ans is None:
                e = e + self.place(pid)
            else:
                after = answer_mask(x)
                zero = torch.zeros_like(pid)
                e = e + self.place(torch.where(after, zero, pid)) + self.place_ans(torch.where(after, pid, zero))
        for blk in self.prelude:
            e = blk(e, cos, sin)

        h = e
        for _ in range(n_loops):
            if self.cfg.inject == "concat":
                u = self.adapter(torch.cat([h, e], dim=-1))
            elif self.cfg.inject == "add":
                u = h + e
            else:
                u = h
            for blk in self.core:
                u = blk(u, cos, sin)
            h = u

        for blk in self.coda:
            h = blk(h, cos, sin)
        return self.lm_head(self.final_norm(h))

    def num_params(self) -> int:
        """Number of unique parameters (independent of n_loops)."""
        return sum(p.numel() for p in self.parameters())
