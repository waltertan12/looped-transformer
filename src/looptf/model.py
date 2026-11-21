"""Load Qwen models and rewire their decoder stack into a looped transformer.

A loop layout is a list of original layer indices to execute in order. Looped
layers appear multiple times and share weights, e.g. for a 28-layer model:

    prelude 0-3, loop 4-13 x 3, coda 24-27  ->  0..3, (4..13)*3, 24..27

Layers 14-23 are dropped; the loop block re-applies 4-13 to recover depth.

Gated loops (Track A, "don't break the model"): with `gated=True`, every pass of
the loop block after the first is wrapped as  h + g * (layer(h) - h),  with one
learnable scalar g per (extra pass, core layer), initialized to 0. At g = 0 the
extra passes are no-ops, so the looped model is exactly the original model.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

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
    gated: bool = False
    gate_init: float = 0.0  # starting value of the dials (0 = extra passes are no-ops)

    @classmethod
    def parse(cls, s: str, gated: bool = False) -> "LoopSpec":
        """Parse 'start:end:n_loops[:coda_start]', e.g. '4:14:3:24'."""
        parts = [int(p) for p in s.split(":")]
        return cls(*parts, gated=gated)

    def layout(self, num_layers: int) -> list[int]:
        coda_start = self.loop_end if self.coda_start is None else self.coda_start
        if not (0 <= self.loop_start < self.loop_end <= coda_start <= num_layers):
            raise ValueError(f"bad loop spec {self} for {num_layers} layers")
        loop = list(range(self.loop_start, self.loop_end))
        return list(range(self.loop_start)) + loop * self.n_loops + list(range(coda_start, num_layers))

    def tag(self) -> str:
        s = f"{self.loop_start}-{self.loop_end}-x{self.n_loops}"
        if self.coda_start is not None and self.coda_start != self.loop_end:
            s += f"-coda{self.coda_start}"
        if self.gated:
            s += f"-gated{self.gate_init:g}"
        return s


class LoopGates(nn.Module):
    """Gate scalars, one row per extra pass (pass 2, 3, ...), one column per core layer."""

    def __init__(self, n_core: int, n_extra: int, init: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((max(n_extra, 1), n_core), float(init), dtype=torch.float32))

    def ensure(self, n_extra: int):
        """Grow to n_extra rows. New rows copy the last trained row (for eval at larger K)."""
        rows = self.weight.shape[0]
        if n_extra > rows:
            extra = self.weight.data[-1:].expand(n_extra - rows, -1)
            self.weight = nn.Parameter(torch.cat([self.weight.data, extra]))


class LoopSlot(nn.Module):
    """One position in the unrolled stack. Points at a (possibly shared) decoder layer,
    gives it a unique KV-cache index for this position, and optionally gates its update."""

    def __init__(self, layer: nn.Module, slot_idx: int, source_idx: int, gate=None):
        super().__init__()
        self.layer = layer
        self.slot_idx = slot_idx
        self.source_idx = source_idx
        # (LoopGates, row, col) kept as a plain attribute so the gates aren't registered twice
        object.__setattr__(self, "_gate", gate)

    def forward(self, hidden_states, *args, **kwargs):
        self.layer.self_attn.layer_idx = self.slot_idx
        out = self.layer(hidden_states, *args, **kwargs)
        if self._gate is not None:
            gates, row, col = self._gate
            g = gates.weight[row, col].to(out.dtype)
            out = hidden_states + g * (out - hidden_states)
        return out


def load_model(name: str, dtype=torch.bfloat16, device: str = "cuda", attn: str = "sdpa"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn).to(device)
    return model, tok


def apply_loop(model, spec: LoopSpec | list[int]):
    """Rewire model in place into a looped stack. Can be called again to change n_loops."""
    inner = model.model
    if not hasattr(model, "_loop_base"):
        # remember the original layers and attention types the first time
        object.__setattr__(model, "_loop_base", list(inner.layers))
        n0 = len(inner.layers)
        object.__setattr__(
            model, "_loop_types", list(getattr(model.config, "layer_types", None) or ["full_attention"] * n0)
        )
    base, types = model._loop_base, model._loop_types
    n = len(base)

    if isinstance(spec, list):
        layout, gated = spec, False
    else:
        layout, gated = spec.layout(n), spec.gated

    gates = None
    if gated:
        n_core = spec.loop_end - spec.loop_start
        if getattr(inner, "loop_gates", None) is None:
            inner.loop_gates = LoopGates(n_core, spec.n_loops - 1, spec.gate_init).to(next(model.parameters()).device)
        inner.loop_gates.ensure(spec.n_loops - 1)
        gates = inner.loop_gates

    slots = []
    seen: dict[int, int] = {}
    for i, src in enumerate(layout):
        gate = None
        if gated and spec.loop_start <= src < spec.loop_end:
            pass_idx = seen.get(src, 0)  # 0 = first pass through this core layer
            seen[src] = pass_idx + 1
            if pass_idx > 0:
                gate = (gates, pass_idx - 1, src - spec.loop_start)
        slots.append(LoopSlot(base[src], i, src, gate))

    inner.layers = nn.ModuleList(slots)
    cfg = model.config
    cfg.num_hidden_layers = len(layout)
    cfg.layer_types = [types[src] for src in layout]
    cfg.loop_layout = layout
    cfg.loop_spec = asdict(spec) if isinstance(spec, LoopSpec) else None
    return model


def set_n_loops(model, n_loops: int):
    """Change the number of loop passes on an already-looped model (keeps weights and gates)."""
    spec = LoopSpec(**model.config.loop_spec)
    spec.n_loops = n_loops
    return apply_loop(model, spec)


def weight_bytes(model) -> int:
    """Bytes of unique parameters (shared layers counted once)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def describe(model) -> str:
    unique = sum(p.numel() for p in model.parameters())
    depth = model.config.num_hidden_layers
    spec = getattr(model.config, "loop_spec", None)
    return (
        f"depth={depth} unique_params={unique/1e6:.1f}M weights={weight_bytes(model)/2**30:.2f}GB "
        f"spec={spec} layout={getattr(model.config, 'loop_layout', None)}"
    )


def clone_model(model):
    return copy.deepcopy(model)


def save_looped(model, base_name: str, path: str):
    """Save a looped model as {base model name, loop spec/layout, state_dict}."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base": base_name,
            "spec": model.config.loop_spec,
            "layout": model.config.loop_layout,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_looped(path: str, device: str = "cuda", dtype=torch.bfloat16, n_loops: int | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, tok = load_model(ckpt["base"], dtype=dtype, device=device)
    spec = LoopSpec(**ckpt["spec"]) if ckpt.get("spec") else ckpt["layout"]
    apply_loop(model, spec)
    model.load_state_dict(ckpt["state_dict"])
    if n_loops is not None:
        set_n_loops(model, n_loops)
    return model, tok
