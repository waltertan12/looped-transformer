"""Operator-mix labels for frozen test rows (H14), aligned with a run's per_example.npz rows.

The final per-example dump is the full test set in level order, repeated once per loop count K,
so labels are tiled over the K blocks. `answer` and `depth` are checked row by row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from looptf.toy.data import STOI
from looptf.toy.dataset import slice_paths


def operator_counts(tokens: np.ndarray, length: np.ndarray) -> dict[str, np.ndarray]:
    """Counts over the prompt only (tokens[i, :length[i]]), so an answer's '-' sign is ignored."""
    T = tokens.shape[1]
    prompt = np.arange(T)[None, :] < np.asarray(length)[:, None]
    cnt = {name: ((tokens == STOI[ch]) & prompt).sum(1)
           for name, ch in (("n_mul", "*"), ("n_add", "+"), ("n_sub", "-"), ("n_paren", "("))}
    total = cnt["n_mul"] + cnt["n_add"] + cnt["n_sub"]
    cnt["mul_frac"] = cnt["n_mul"] / np.maximum(total, 1)
    return cnt


def labels_for_run(data_root: Path, per_example: dict) -> dict[str, np.ndarray]:
    root = Path(data_root)
    meta = json.loads((root / "meta.json").read_text())
    cols = meta["columns"]
    toks, labs = [], []
    for lv in sorted(int(k) for k in meta["slices"]):
        tp, mp = slice_paths(root, "test", lv)
        if tp.exists():
            toks.append(np.load(tp)); labs.append(np.load(mp))
    tok, lab = np.concatenate(toks), np.concatenate(labs)
    n = len(tok)
    reps, rem = divmod(len(per_example["answer"]), n)
    assert rem == 0, f"per_example rows {len(per_example['answer'])} not a multiple of test rows {n}"
    assert (np.tile(lab[:, cols.index("answer")], reps) == per_example["answer"]).all(), "answer mismatch"
    assert (np.tile(lab[:, cols.index("depth")], reps) == per_example["depth"]).all(), "depth mismatch"
    c = operator_counts(tok, lab[:, cols.index("length")])
    return {k: np.tile(v, reps) for k, v in c.items()}
