"""Load Qwen models and rewire their decoder stack into a looped transformer.

A loop layout is a list of original layer indices to execute in order. Looped
layers appear multiple times and share weights, e.g. for a 28-layer model:

    prelude 0-3, loop 4-13 x 3, coda 24-27  ->  0..3, (4..13)*3, 24..27

Layers 14-23 are dropped; the loop block re-applies 4-13 to recover depth.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoopSpec:
    """prelude=[0, loop_start), loop=[loop_start, loop_end) x n_loops, coda=[coda_start, L)."""

    loop_start: int
    loop_end: int
    n_loops: int = 1
    coda_start: int | None = None  # defaults to loop_end (no layers dropped)

    @classmethod
    def parse(cls, s: str) -> "LoopSpec":
        """Parse 'start:end:n_loops[:coda_start]', e.g. '4:14:3:24'."""
        parts = [int(p) for p in s.split(":")]
        return cls(*parts)

    def layout(self, num_layers: int) -> list[int]:
        coda_start = self.loop_end if self.coda_start is None else self.coda_start
        if not (0 <= self.loop_start < self.loop_end <= coda_start <= num_layers):
            raise ValueError(f"bad loop spec {self} for {num_layers} layers")
        loop = list(range(self.loop_start, self.loop_end))
        return list(range(self.loop_start)) + loop * self.n_loops + list(range(coda_start, num_layers))


class LoopSlot(nn.Module):
    """One position in the unrolled stack. Points at a (possibly shared) decoder layer
    and gives it a unique KV-cache index for this position."""

    def __init__(self, layer: nn.Module, slot_idx: int, source_idx: int):
        super().__init__()
        self.layer = layer
        self.slot_idx = slot_idx
        self.source_idx = source_idx

    def forward(self, hidden_states, *args, **kwargs):
        self.layer.self_attn.layer_idx = self.slot_idx
        return self.layer(hidden_states, *args, **kwargs)


def load_model(name: str, dtype=torch.bfloat16, device: str = "cuda", attn: str = "sdpa"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn).to(device)
    return model, tok


def apply_loop(model, spec: LoopSpec | list[int]):
    """Rewire model in place into a looped stack. Returns the model."""
    inner = model.model
    original = list(inner.layers)
    n = len(original)
    layout = spec if isinstance(spec, list) else spec.layout(n)
    orig_types = list(getattr(model.config, "layer_types", None) or ["full_attention"] * n)

    inner.layers = nn.ModuleList(LoopSlot(original[src], i, src) for i, src in enumerate(layout))
    cfg = model.config
    cfg.num_hidden_layers = len(layout)
    cfg.layer_types = [orig_types[src] for src in layout]
    cfg.loop_layout = layout
    return model


def describe(model) -> str:
    layout = getattr(model.config, "loop_layout", None)
    unique = sum(p.numel() for p in model.parameters())
    depth = model.config.num_hidden_layers
    return f"depth={depth} unique_params={unique/1e6:.1f}M layout={layout}"


def clone_model(model):
    return copy.deepcopy(model)


def save_looped(model, base_name: str, path: str):
    """Save a looped model as {base model name, layout, state_dict}. Shared layers are stored once."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"base": base_name, "layout": model.config.loop_layout, "state_dict": model.state_dict()},
        path,
    )


def load_looped(path: str, device: str = "cuda", dtype=torch.bfloat16):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, tok = load_model(ckpt["base"], dtype=dtype, device=device)
    apply_loop(model, ckpt["layout"])
    model.load_state_dict(ckpt["state_dict"])
    return model, tok
