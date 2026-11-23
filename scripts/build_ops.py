#!/usr/bin/env python
"""Build test-only single-operator datasets (H13): data/nested_add (only '+') and data/nested_sub (only '-').

    scripts/build_ops.py --kind add --out data/nested_add
    scripts/build_ops.py --kind sub --out data/nested_sub

Level = depth. Add: depth 1-15 (= terms - 1; flat sums). Sub: depth 1-8. Inputs exactly k digits,
k in 1..4; up to --n_per_cell distinct prompts per (depth, k); full row (prompt + answer + '#') <= 80.
Same frozen format as nested_bal (test split only). Check with scripts/verify_data.py --root <out>.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np

from looptf.toy.data import BOUND, EOS, VOCAB, answer_tokens, num_digits
from looptf.toy.dataset import encode_rows, full_sequence, slice_paths, write_meta, write_slice
from looptf.toy.ops_data import feasible_n_ops, gen_single_op

REPO = Path(__file__).resolve().parent.parent
WIDTH = 80
COLUMNS = ["answer", "n_ops", "depth", "length", "n_tgt", "in_digits", "ans_digits"]
MAX_DEPTH = {"add": 15, "sub": 8}
IN_DIGITS = (1, 2, 3, 4)


def seed_for(seed: int, name: str, split: str, level: int, chunk: int) -> int:
    key = f"{seed}|{name}|{split}|{level}|{chunk}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _gen_cell(kind: str, depth: int, k: int, n: int, rng: random.Random) -> list[tuple[str, int, int]]:
    lo, hi = feasible_n_ops(kind, depth)
    seen: dict[str, tuple[int, int]] = {}
    for _ in range(50 * n):
        if len(seen) >= n:
            break
        n_ops = rng.randint(lo, hi)
        expr, ans = gen_single_op(kind, n_ops, depth, rng, k)
        prompt = expr + "="
        if len(prompt) + len(answer_tokens(ans)) > WIDTH:
            continue
        seen.setdefault(prompt, (ans, n_ops))
    return [(p, a, o) for p, (a, o) in seen.items()]


def _train_prompts(root: Path, depth: int) -> set[bytes]:
    tp, mp = slice_paths(root, "train", depth)
    if not tp.exists():
        return set()
    tok, lab = np.load(tp, mmap_mode="r"), np.load(mp, mmap_mode="r")
    import json
    cols = json.loads((root / "meta.json").read_text())["columns"]
    L = np.asarray(lab[:, cols.index("length")])
    out = set()
    step = 200_000
    for s in range(0, len(tok), step):
        t = np.asarray(tok[s:s + step])
        for row, n in zip(t, L[s:s + step]):
            out.add(row[:n].tobytes())
    return out


def build_ops(kind: str, root: Path, n_per_cell: int = 1000, seed: int = 0, max_depth: int | None = None,
              overlap_root: Path | None = None) -> dict:
    name = f"nested_{kind}"
    root = Path(root)
    slices = {}
    for depth in range(1, (max_depth or MAX_DEPTH[kind]) + 1):
        rows, cells = [], {}
        for k in IN_DIGITS:
            cell = _gen_cell(kind, depth, k, n_per_cell, random.Random(seed_for(seed, name, "test", depth, k)))
            cells[f"in{k}"] = len(cell)
            for prompt, ans, n_ops in cell:
                rows.append((prompt, ans, [ans, n_ops, depth, len(prompt), len(answer_tokens(ans)), k,
                                           num_digits(ans)]))
        if not rows:
            break
        random.Random(seed_for(seed, name, "shuffle", depth, 0)).shuffle(rows)
        tok = encode_rows([full_sequence(p, a) for p, a, _ in rows], WIDTH)
        lab = np.array([r for _, _, r in rows], dtype=np.int64)
        write_slice(root, "test", depth, tok, lab)
        sl = {"train": 0, "test": len(rows), "cells": cells}
        if overlap_root is not None:
            tr = _train_prompts(Path(overlap_root), depth)
            sl["overlap_with_nested_bal_train"] = float(np.mean([tok[i, :lab[i, 3]].tobytes() in tr
                                                                 for i in range(len(tok))]))
        slices[str(depth)] = sl
        print(f"  {name} depth {depth:2d}: {len(rows):5d} rows  {cells}"
              + (f"  overlap {sl['overlap_with_nested_bal_train']:.4f}" if overlap_root else ""), flush=True)
    meta = {"name": name, "vocab": VOCAB, "width": WIDTH, "columns": COLUMNS, "level_column": "depth",
            "bound": BOUND, "eos": EOS, "seed": seed, "test_only": True, "kind": kind,
            "n_per_cell": n_per_cell, "in_digits": list(IN_DIGITS), "slices": slices,
            "generator": "looptf.toy.ops_data.gen_single_op"}
    write_meta(root, meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["add", "sub"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_cell", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap_root", default="data/nested_bal", help="'' to skip the overlap check")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    ov = (REPO / a.overlap_root) if a.overlap_root else None
    t0 = time.time()
    build_ops(a.kind, out, a.n_per_cell, a.seed, overlap_root=ov if ov and ov.exists() else None)
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
