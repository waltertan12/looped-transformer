"""Frozen, pre-generated arithmetic datasets: storage format, writer and loader.

Layout (one directory per dataset, e.g. data/nested, data/chain)::

    <root>/meta.json
    <root>/<split>/<level:02d>.tokens.npy   uint8 [N, width]  prompt token ids, PAD (0) padded
    <root>/<split>/<level:02d>.meta.npy     int16 [N, F]      label columns, see meta.json

`split` is "train" or "test". `level` is the difficulty slice: depth for the nested set,
chain length for the chain set. The prompt includes its terminal token ('=' or '?'); the
model reads its answer at position length-1. The answer is NOT in the tokens.

meta.json keys: name, vocab, width, columns (label column names; always starts with
"answer" and contains "length"), level_column, mod, seed, slices
({level: {"train": n, "test": n, "test_overlap_with_train": fraction}}), plus anything the
builder adds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from looptf.toy.data import VOCAB, STOI, PAD_ID, encode


def slice_paths(root: Path, split: str, level: int) -> tuple[Path, Path]:
    d = Path(root) / split
    return d / f"{level:02d}.tokens.npy", d / f"{level:02d}.meta.npy"


def encode_rows(prompts: list[str], width: int) -> np.ndarray:
    out = np.full((len(prompts), width), PAD_ID, dtype=np.uint8)
    for i, p in enumerate(prompts):
        ids = encode(p)
        if len(ids) > width:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds width {width}: {p!r}")
        out[i, : len(ids)] = ids
    return out


def write_slice(root, split: str, level: int, tokens: np.ndarray, labels: np.ndarray) -> None:
    tp, mp = slice_paths(root, split, level)
    tp.parent.mkdir(parents=True, exist_ok=True)
    assert tokens.dtype == np.uint8 and labels.dtype == np.int16 and len(tokens) == len(labels)
    np.save(tp, tokens)
    np.save(mp, labels)


def write_meta(root, meta: dict) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "meta.json").write_text(json.dumps(meta, indent=2))


class FrozenData:
    """Loads a frozen dataset and serves batches shaped like `data.make_batch`."""

    def __init__(self, root, device="cuda"):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if self.meta["vocab"] != VOCAB[: len(self.meta["vocab"])]:
            raise ValueError("dataset vocab does not match looptf.toy.data.VOCAB")
        self.cols = {c: i for i, c in enumerate(self.meta["columns"])}
        self.levels = sorted(int(k) for k in self.meta["slices"])
        self.device = device
        self._train = {}
        for lv in self.levels:
            tp, mp = slice_paths(self.root, "train", lv)
            if tp.exists():
                self._train[lv] = (np.load(tp, mmap_mode="r"), np.load(mp, mmap_mode="r"))

    def _to_batch(self, tokens: np.ndarray, labels: np.ndarray) -> dict:
        length = labels[:, self.cols["length"]].astype(np.int64)
        T = int(length.max())
        x = torch.from_numpy(np.ascontiguousarray(tokens[:, :T]).astype(np.int64))
        b = {
            "x": x.to(self.device, non_blocking=True),
            "pos": torch.from_numpy(length - 1).to(self.device),
            "y": torch.tensor([STOI[str(int(a))] for a in labels[:, self.cols["answer"]]],
                              dtype=torch.long, device=self.device),
        }
        for c, i in self.cols.items():
            if c != "answer":
                b[c] = torch.from_numpy(labels[:, i].astype(np.int64)).to(self.device)
        return b

    def batch(self, rng: np.random.Generator, batch_size: int, level_range: tuple[int, int]) -> dict:
        """Uniform over levels in [lo, hi] (inclusive), then uniform within each level."""
        lvls = [lv for lv in self.levels if level_range[0] <= lv <= level_range[1] and lv in self._train]
        if not lvls:
            raise ValueError(f"no training slices in range {level_range}")
        pick = rng.integers(0, len(lvls), batch_size)
        toks, labs = [], []
        for j, lv in enumerate(lvls):
            n = int((pick == j).sum())
            if n == 0:
                continue
            t, m = self._train[lv]
            idx = np.sort(rng.integers(0, len(t), n))
            toks.append(t[idx]); labs.append(m[idx])
        return self._to_batch(np.concatenate(toks), np.concatenate(labs))

    def test_sets(self, levels=None, max_per_level: int | None = None) -> dict[int, dict]:
        out = {}
        for lv in levels or self.levels:
            tp, mp = slice_paths(self.root, "test", lv)
            if not tp.exists():
                continue
            t, m = np.load(tp), np.load(mp)
            if max_per_level:
                t, m = t[:max_per_level], m[:max_per_level]
            out[lv] = self._to_batch(t, m)
        return out
