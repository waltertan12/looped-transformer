#!/usr/bin/env python
"""Score a saved toy checkpoint on any frozen dataset at the run's training loop count.

    scripts/eval_ckpt.py --ckpt checkpoints/toy/long/c2k8-s0.pt --data data/nested_add \
        --out results/toy/ops/c2k8-s0-add/per_example.npz

Respects the run's --reverse_answer. Output has the same layout as training's per_example.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from looptf.toy.dataset import FrozenData
from looptf.toy.model import ToyConfig, ToyLM
from looptf.toy.train import dump_per_example

REPO = Path(__file__).resolve().parent.parent


def load_checkpoint(path: Path, device: str) -> tuple[ToyLM, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ToyLM(ToyConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["args"]


def train_k(args: dict) -> int:
    v = str(args["train_loops"])
    if not v.isdigit():
        raise ValueError(f"train_loops {v!r} is not a single loop count")
    return int(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem_fraction", type=float, default=None)
    a = ap.parse_args()
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    res = lambda p: Path(p) if Path(p).is_absolute() else REPO / p  # noqa: E731
    model, args = load_checkpoint(res(a.ckpt), "cuda")
    fd = FrozenData(res(a.data), device="cuda", reverse_answer=bool(args.get("reverse_answer", False)))
    out = res(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    k = train_k(args)
    dump_per_example(model, fd.test_sets(), [k], out)
    print(f"{a.ckpt} on {a.data} at K={k} (reverse={args.get('reverse_answer', False)}) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
