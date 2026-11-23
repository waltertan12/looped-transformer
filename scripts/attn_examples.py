#!/usr/bin/env python
"""H19 illustration: attention weights per pass, per core block, per head, for a few problems.

    scripts/attn_examples.py --ckpt long/c2k8-s0 --out results/toy/probe/attn_c2k8-s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ckpt import load_checkpoint, train_k  # noqa: E402
from looptf.toy.data import encode  # noqa: E402
from probe_levels import pass_states, subexpr_nodes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = ["(8-3)*(4-6)+7=", "9-(7-9-(1*4+3*4))=", "(1-1*(9+6*(5-(6-(1-6)))))*2="]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="long/c2k8-s0")
    ap.add_argument("--out", default="results/toy/probe/attn_c2k8-s0.json")
    a = ap.parse_args()
    model, args = load_checkpoint(REPO / f"checkpoints/toy/{a.ckpt}.pt", "cpu")
    k = train_k(args)
    out = {"ckpt": a.ckpt, "loops": k, "examples": []}
    for p in EXAMPLES:
        x = torch.tensor([encode(p)])
        _, attn = pass_states(model, x, k, per_layer=(k == 1))
        w = torch.stack(attn)[:, :, 0]  # [pass, core block, head, T, T]
        out["examples"].append({"prompt": p, "nodes": subexpr_nodes(p),
                                "attn": [[[[round(v, 3) for v in row] for row in hd] for hd in blk] for blk in w.tolist()]})
    path = REPO / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"{a.ckpt}: {len(EXAMPLES)} examples, attention shape {tuple(w.shape)} -> {path}")


if __name__ == "__main__":
    main()
