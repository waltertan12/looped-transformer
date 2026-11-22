#!/usr/bin/env python
"""Pre-generate the frozen toy datasets (nested expressions, intermediate-value chains).

Writes <out>/nested and <out>/chain in the layout described in looptf/toy/dataset.py.
Output is bit-identical across reruns with the same arguments: every chunk gets its own
random.Random seeded from sha256(seed, dataset, split, level, chunk).

Examples:
  .venv/bin/python scripts/build_data.py --out data
  .venv/bin/python scripts/build_data.py --out /tmp/x --scale 0.002 --test_n 200 --which chain
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

from looptf.toy.chain import gen_chain
from looptf.toy.data import MOD, VOCAB, gen_expr
from looptf.toy.dataset import encode_rows, write_meta, write_slice

REPO = Path(__file__).resolve().parent.parent
CHUNK = 50_000          # examples per worker task
MAX_REFILL = 50         # extra test-generation attempts after the first one

SPECS = {
    "nested": dict(
        width=64, train_n=1_250_000, level_column="depth",
        columns=["answer", "n_ops", "depth", "length"],
        generator=("nested: random PEMDAS expression trees with n_ops in [depth, min(16, 2**depth-1)] "
                   "operators at exactly the given depth, leaves 1-6, answer = value mod 7; prompt ends '='"),
    ),
    "chain": dict(
        width=256, train_n=440_000, level_column="chain_len",
        columns=["answer", "chain_len", "n_distractors", "n_ops_total", "length"],
        generator=("chain: shuffled 'name=expr' statements with chain_len linked variables and 0-4 "
                   "distractors; answer = value of the query variable mod 7; prompt ends '?'"),
    ),
}


def seed_for(seed: int, name: str, split: str, level: int, chunk: int) -> int:
    key = f"{seed}|{name}|{split}|{level}|{chunk}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def gen_chunk(task):
    """Worker: returns (prompts, label rows) for one chunk. Deterministic in the task."""
    name, split, level, chunk, n, seed = task
    rng = random.Random(seed_for(seed, name, split, level, chunk))
    prompts, rows = [], []
    if name == "nested":
        for _ in range(n):
            n_ops = rng.randint(level, min(16, 2**level - 1))
            expr, ans = gen_expr(n_ops, level, rng)
            p = expr + "="
            prompts.append(p)
            rows.append((ans, n_ops, level, len(p)))
    else:
        for _ in range(n):
            nd = rng.randint(0, 4)
            p, ans, m = gen_chain(level, nd, rng)
            prompts.append(p)
            rows.append((ans, m["chain_len"], m["n_distractors"], m["n_ops_total"], m["length"]))
    return prompts, rows


def chunk_sizes(total: int) -> list[int]:
    sizes = [CHUNK] * (total // CHUNK)
    if total % CHUNK:
        sizes.append(total % CHUNK)
    return sizes


def gen_train(pool, name, level, n, seed):
    tasks = [(name, "train", level, i, size, seed) for i, size in enumerate(chunk_sizes(n))]
    prompts, rows = [], []
    for p, r in pool.map(gen_chunk, tasks, chunksize=1):
        prompts.extend(p)
        rows.extend(r)
    return prompts, rows


def gen_test(pool, name, level, test_n, train_set, seed):
    """Unique, train-free test prompts; tops up with more chunks, then falls back to overlap.

    Returns (prompts, rows, n_refill_attempts, shortfall) where shortfall is the number of
    rows that could not be filled with overlap-free unique prompts.
    """
    cand_size = min(CHUNK, max(2000, 2 * test_n))
    test_seen: set[str] = set()
    test_p: list[str] = []
    test_r: list = []
    overlap_seen: set[str] = set()
    overlap_p: list[str] = []
    overlap_r: list = []
    attempts = 0
    while len(test_p) < test_n and attempts <= MAX_REFILL:
        task = (name, "test", level, attempts, cand_size, seed)
        prompts, rows = pool.apply(gen_chunk, (task,))
        attempts += 1
        for p, r in zip(prompts, rows):
            if p in train_set:
                if p not in overlap_seen:
                    overlap_seen.add(p)
                    overlap_p.append(p)
                    overlap_r.append(r)
            elif p not in test_seen:
                test_seen.add(p)
                test_p.append(p)
                test_r.append(r)
                if len(test_p) >= test_n:
                    break
    shortfall = test_n - len(test_p)
    if shortfall > 0:
        # Fill with overlapping prompts: unique ones first, duplicates only if unavoidable.
        pool_p, pool_r = (overlap_p, overlap_r) if overlap_p else (test_p, test_r)
        if not pool_p:
            raise RuntimeError(f"{name} level {level}: no candidate prompts at all")
        for i in range(shortfall):
            test_p.append(pool_p[i % len(pool_p)])
            test_r.append(pool_r[i % len(pool_r)])
    return test_p, test_r, attempts - 1, shortfall


def build_dataset(pool, name: str, root: Path, args) -> None:
    spec = SPECS[name]
    levels = list(range(1, 17))
    train_n = max(1, int(round(spec["train_n"] * args.scale)))
    t_start = time.time()
    slices = {}
    print(f"== {name}: width {spec['width']}, levels {levels[0]}-{levels[-1]}, "
          f"train {train_n}/level, test {args.test_n}/level -> {root}", flush=True)
    for level in levels:
        t0 = time.time()
        tr_p, tr_r = gen_train(pool, name, level, train_n, args.seed)
        train_set = set(tr_p)
        te_p, te_r, refills, shortfall = gen_test(pool, name, level, args.test_n, train_set, args.seed)
        overlap = sum(p in train_set for p in te_p) / len(te_p) if te_p else 0.0

        write_slice(root, "train", level, encode_rows(tr_p, spec["width"]),
                    np.array(tr_r, dtype=np.int16).reshape(len(tr_r), len(spec["columns"])))
        write_slice(root, "test", level, encode_rows(te_p, spec["width"]),
                    np.array(te_r, dtype=np.int16).reshape(len(te_r), len(spec["columns"])))
        slices[str(level)] = {"train": len(tr_p), "test": len(te_p),
                              "test_overlap_with_train": float(overlap)}
        print(f"  level {level:2d}: train {len(tr_p):>9,d}  test {len(te_p):>5d}  "
              f"overlap {overlap:.4f}  refills {refills}  shortfall {shortfall}  "
              f"{time.time() - t0:6.1f}s", flush=True)
        del tr_p, tr_r, train_set, te_p, te_r

    build_seconds = time.time() - t_start
    meta = {
        "name": name,
        "vocab": VOCAB,
        "width": spec["width"],
        "columns": spec["columns"],
        "level_column": spec["level_column"],
        "mod": MOD,
        "seed": args.seed,
        "scale": args.scale,
        "slices": slices,
        "generator": spec["generator"],
        "build_seconds": round(build_seconds, 1),
    }
    write_meta(root, meta)
    size = sum(f.stat().st_size for f in root.rglob("*.npy"))
    print(f"== {name}: done in {build_seconds:.1f}s, size {size / 1e9:.3f} GB "
          f"({size / 1e9:.3f} GB of .npy files in {root})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory (relative to repo root)")
    ap.add_argument("--which", default="nested,chain", help="comma-separated subset of nested,chain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies train sizes")
    ap.add_argument("--test_n", type=int, default=1000, help="test prompts per level")
    args = ap.parse_args()

    names = [w.strip() for w in args.which.split(",") if w.strip()]
    for n in names:
        if n not in SPECS:
            ap.error(f"unknown dataset {n!r}; choose from {sorted(SPECS)}")
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for name in names:
            build_dataset(pool, name, out / name, args)
    print(f"total build time {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
