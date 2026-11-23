#!/usr/bin/env python
"""Export attention weights for the attention viewer (docs/visual/attention.html).

For each model and example problem: greedy-decode the model's answer, then run prompt + answer once
and record every attention call in order (prelude, each core block on each pass, coda).

    scripts/attn_export.py                       # looped, deep, shallow (seed 0) -> docs/visual/attn/*.json

Weights are stored compactly: each [T, T] map is one string of T*T base-36 digits, value
round(w * 35) (so 0 = under ~1.4%, z = 100%). Row i = token i looking back at tokens 0..i.
CPU only, a few seconds per model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ckpt import load_checkpoint, train_k  # noqa: E402
from looptf.toy.data import decode, encode  # noqa: E402
from looptf.toy.model import apply_rope  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = ["3+4*5=", "2*(3+4)=", "1+2+3+4+5+6=", "7*8-6*5=", "8-2-3-1+6=", "(8-3)*(4-6)+7=",
            "(5+2)*(9-6)-(4+1)=", "9-(7-9-(1*4+3*4))=", "(1-1*(9+6*(5-(6-(1-6)))))*2="]
MODELS = {"looped": "long/c2k8-s0", "deep": "long/c16k1-s0", "shallow": "long/c2k1-s0"}
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


@torch.no_grad()
def greedy(model, k: int, prompt: str, maxn: int = 12) -> str:
    x, out = encode(prompt), ""
    for _ in range(maxn):
        t = int(model(torch.tensor([x]), n_loops=k)[0, -1].argmax())
        x.append(t)
        c = decode([t])
        if c == "#":
            break
        out += c
    return out


@torch.no_grad()
def record(model, k: int, seq: str) -> list[tuple[str, torch.Tensor]]:
    """Run seq once; return (layer label, [heads, T, T] weights) for every attention call, in order."""
    names = {}
    for i, b in enumerate(model.prelude):
        names[id(b.attn)] = "prelude"
    for i, b in enumerate(model.core):
        names[id(b.attn)] = f"core{i + 1}"
    for i, b in enumerate(getattr(model, "coda", [])):
        names[id(b.attn)] = "coda"
    calls = []

    def hook(mod, args):
        x, cos, sin = args
        a = mod  # x arrives already normed (pre-hook on Attention), so recompute the weights here
        B, T, _ = x.shape
        q, kk, _v = a.qkv(x).view(B, T, 3, a.n_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, kk = apply_rope(q, cos, sin), apply_rope(kk, cos, sin)
        s = (q @ kk.transpose(-1, -2)) / a.head_dim ** 0.5
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        calls.append((names[id(mod)], torch.softmax(s.masked_fill(~mask, float("-inf")), -1)[0]))

    hs = [b.attn.register_forward_pre_hook(hook) for b in [*model.prelude, *model.core, *getattr(model, "coda", [])]]
    try:
        model(torch.tensor([encode(seq)]), n_loops=k)
    finally:
        for h in hs:
            h.remove()
    return calls


def label_layers(calls, k: int, n_core: int) -> list[str]:
    out, core_seen = [], 0
    for name, _ in calls:
        if name.startswith("core"):
            p = core_seen // n_core + 1
            out.append(f"pass {p} · block {name[4:]}" if k > 1 else f"layer {core_seen + 2}")
            core_seen += 1
        else:
            out.append(name)
    return out


def pack(w: torch.Tensor) -> str:
    return "".join(B36[v] for v in (w.clamp(0, 1) * 35).round().int().flatten().tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/visual/attn")
    a = ap.parse_args()
    torch.set_num_threads(4)
    outdir = REPO / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for mname, ckpt in MODELS.items():
        model, args = load_checkpoint(REPO / f"checkpoints/toy/{ckpt}.pt", "cpu")
        model.eval()
        k = train_k(args)
        res = {"model": mname, "ckpt": ckpt, "loops": k, "n_core": len(model.core), "examples": []}
        for p in EXAMPLES:
            ans = greedy(model, k, p)
            truth = str(eval(p[:-1]))
            calls = record(model, k, p + ans)
            res["layers"] = label_layers(calls, k, len(model.core))
            res["examples"].append({"prompt": p, "answer": ans, "truth": truth, "tokens": list(p + ans),
                                    "maps": [[pack(w[h]) for h in range(w.shape[0])] for _, w in calls]})
        path = outdir / f"{mname}.json"
        path.write_text(json.dumps(res, separators=(",", ":")))
        ok = sum(e["answer"] == e["truth"] for e in res["examples"])
        print(f"{mname}: {len(res['layers'])} attention layers, {ok}/{len(EXAMPLES)} correct, "
              f"{path.stat().st_size / 1e3:.0f} KB -> {path}")


if __name__ == "__main__":
    main()
