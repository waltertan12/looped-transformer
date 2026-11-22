#!/usr/bin/env python
"""Pre-generate the frozen toy datasets (nested expressions, intermediate-value chains).

Writes <out>/nested and <out>/chain in the layout described in looptf/toy/dataset.py.
Tokens are the full sequence ``prompt + answer digits + '#'``; labels are int64 columns.
Output is bit-identical across reruns with the same arguments: every chunk gets its own
random.Random seeded from sha256(seed, dataset, split, level, chunk).

Test/train de-overlap is judged on the PROMPT (the part before the answer).

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
from looptf.toy.data import BOUND, EOS, VOCAB, answer_tokens, gen_expr
from looptf.toy.dataset import encode_rows, full_sequence, write_meta, write_slice

REPO = Path(__file__).resolve().parent.parent
CHUNK = 50_000          # examples per worker task
MAX_REFILL = 50         # extra test-generation attempts after the first one

SPECS = {
    "nested": dict(
        width=80, train_n=1_250_000, level_column="depth",
        columns=["answer", "n_ops", "depth", "length", "n_tgt"],
        generator=("nested: random PEMDAS expression trees with n_ops in [depth, min(16, 2**depth-1)] "
                   "operators at exactly the given depth, leaves 1-9, answer = exact integer value "
                   "(unbounded); prompt ends '=', answer digits then '#'"),
    ),
    "chain": dict(
        width=256, train_n=440_000, level_column="chain_len",
        columns=["answer", "chain_len", "n_distractors", "n_ops_total", "length", "n_tgt"],
        generator=("chain: shuffled 'name=expr' statements with chain_len linked variables and 0-4 "
                   "distractors; answer = exact integer value of the query variable (unbounded); "
                   "prompt ends '?', answer digits then '#'"),
    ),
}


def seed_for(seed: int, name: str, split: str, level: int, chunk: int) -> int:
    key = f"{seed}|{name}|{split}|{level}|{chunk}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def gen_chunk(task):
    """Worker: returns [(prompt, row)] for one chunk. Deterministic in the task.

    row[0] is the answer (the full token sequence is full_sequence(prompt, row[0]))."""
    name, split, level, chunk, n, seed = task
    rng = random.Random(seed_for(seed, name, split, level, chunk))
    items = []
    if name == "nested":
        for _ in range(n):
            n_ops = rng.randint(level, min(16, 2**level - 1))
            expr, ans = gen_expr(n_ops, level, rng)
            p = expr + "="
            items.append((p, (ans, n_ops, level, len(p), len(answer_tokens(ans)))))
    else:
        for _ in range(n):
            nd = rng.randint(0, 4)
            p, ans, m = gen_chain(level, nd, rng)
            items.append((p, (ans, m["chain_len"], m["n_distractors"], m["n_ops_total"],
                              m["length"], len(answer_tokens(ans)))))
    return items


def chunk_sizes(total: int) -> list[int]:
    sizes = [CHUNK] * (total // CHUNK)
    if total % CHUNK:
        sizes.append(total % CHUNK)
    return sizes


def gen_train(pool, name, level, n, seed):
    tasks = [(name, "train", level, i, size, seed) for i, size in enumerate(chunk_sizes(n))]
    items = []
    for chunk_items in pool.map(gen_chunk, tasks, chunksize=1):
        items.extend(chunk_items)
    return items


def gen_test(pool, name, level, test_n, train_set, seed):
    """Unique, train-free test items (judged on prompt); tops up with more chunks, then falls
    back to overlap.

    Returns (items, n_refill_attempts, shortfall) where shortfall is the number of items that
    could not be filled with overlap-free unique prompts.
    """
    cand_size = min(CHUNK, max(2000, 2 * test_n))
    test_seen: set[str] = set()
    test_items: list = []
    overlap_seen: set[str] = set()
    overlap_items: list = []
    attempts = 0
    while len(test_items) < test_n and attempts <= MAX_REFILL:
        task = (name, "test", level, attempts, cand_size, seed)
        items = pool.apply(gen_chunk, (task,))
        attempts += 1
        for it in items:
            p = it[0]
            if p in train_set:
                if p not in overlap_seen:
                    overlap_seen.add(p)
                    overlap_items.append(it)
            elif p not in test_seen:
                test_seen.add(p)
                test_items.append(it)
                if len(test_items) >= test_n:
                    break
    shortfall = test_n - len(test_items)
    if shortfall > 0:
        # Fill with overlapping prompts: unique ones first, duplicates only if unavoidable.
        pool_items = overlap_items if overlap_items else test_items
        if not pool_items:
            raise RuntimeError(f"{name} level {level}: no candidate prompts at all")
        for i in range(shortfall):
            test_items.append(pool_items[i % len(pool_items)])
    return test_items, attempts - 1, shortfall


def gen_holdout(pool, name, level, train_n, test_n, frac, seed):
    """For levels with only a few distinct problems (nested depth 1-2), every test prompt would
    also be in train. Instead: enumerate the distinct prompts, hold out `frac` of them (seeded),
    train only on the rest, and test only on the held-out ones (each once, up to test_n).
    Returns (train_items, test_items, n_distinct)."""
    distinct: dict[str, tuple] = {}
    for chunk_items in pool.map(gen_chunk, [(name, "enum", level, i, CHUNK, seed) for i in range(20)]):
        for p_, row in chunk_items:
            distinct.setdefault(p_, row)
    keys = sorted(distinct)
    rng = random.Random(seed_for(seed, name, "holdout", level, 0))
    rng.shuffle(keys)
    held = set(keys[: max(1, int(round(frac * len(keys))))])
    test_items = [(k, distinct[k]) for k in sorted(held)]
    random.Random(seed_for(seed, name, "holdout-order", level, 0)).shuffle(test_items)
    test_items = test_items[:test_n]
    train_items, chunk = [], 0
    while len(train_items) < train_n:
        for it in gen_chunk((name, "train", level, chunk, CHUNK, seed)):
            if it[0] not in held:
                train_items.append(it)
        chunk += 1
    return train_items[:train_n], test_items, len(keys)


def to_arrays(items, width: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = [full_sequence(p, row[0]) for p, row in items]
    tokens = encode_rows(seqs, width)
    labels = np.array([row for _, row in items], dtype=np.int64).reshape(len(items), n_cols)
    return tokens, labels


def build_dataset(pool, name: str, root: Path, args) -> None:
    spec = SPECS[name]
    levels = list(range(1, 17))
    n_cols = len(spec["columns"])
    train_n = max(1, int(round(spec["train_n"] * args.scale)))
    t_start = time.time()
    slices = {}
    print(f"== {name}: width {spec['width']}, levels {levels[0]}-{levels[-1]}, "
          f"train {train_n}/level, test {args.test_n}/level -> {root}", flush=True)
    for level in levels:
        t0 = time.time()
        holdout = name == "nested" and level in args.holdout_levels
        if holdout:
            tr, te, n_distinct = gen_holdout(pool, name, level, train_n, args.test_n,
                                             args.holdout_frac, args.seed)
            train_set = {p for p, _ in tr}
            refills, shortfall = 0, 0
        else:
            tr = gen_train(pool, name, level, train_n, args.seed)
            train_set = {p for p, _ in tr}
            te, refills, shortfall = gen_test(pool, name, level, args.test_n, train_set, args.seed)
        overlap = sum(p in train_set for p, _ in te) / len(te) if te else 0.0

        tr_tok, tr_lab = to_arrays(tr, spec["width"], n_cols)
        te_tok, te_lab = to_arrays(te, spec["width"], n_cols)
        write_slice(root, "train", level, tr_tok, tr_lab)
        write_slice(root, "test", level, te_tok, te_lab)
        slices[str(level)] = {"train": len(tr), "test": len(te),
                              "test_overlap_with_train": float(overlap)}
        if holdout:
            slices[str(level)].update({"holdout": True, "distinct_prompts": n_distinct,
                                       "holdout_frac": args.holdout_frac})
        print(f"  level {level:2d}: train {len(tr):>9,d}  test {len(te):>5d}  "
              f"overlap {overlap:.4f}  refills {refills}  shortfall {shortfall}  "
              f"{time.time() - t0:6.1f}s", flush=True)
        del tr, tr_tok, tr_lab, train_set, te, te_tok, te_lab

    build_seconds = time.time() - t_start
    meta = {
        "name": name,
        "vocab": VOCAB,
        "width": spec["width"],
        "columns": spec["columns"],
        "level_column": spec["level_column"],
        "bound": BOUND,
        "eos": EOS,
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
    ap.add_argument("--holdout_levels", default="1,2",
                    help="nested levels with few distinct problems: hold out a fraction of the "
                         "distinct prompts from train and test only on those")
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    args = ap.parse_args()
    args.holdout_levels = {int(v) for v in args.holdout_levels.split(",") if v.strip()}

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
