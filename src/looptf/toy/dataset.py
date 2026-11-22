"""Frozen, pre-generated arithmetic datasets: storage format, writer and loader.

Layout (one directory per dataset, e.g. data/nested, data/chain)::

    <root>/meta.json
    <root>/<split>/<level:02d>.tokens.npy   uint8 [N, width]  full sequence, PAD (0) padded
    <root>/<split>/<level:02d>.meta.npy     int64 [N, F]      label columns, see meta.json

Each row's tokens are ``prompt + answer digits + EOS``, e.g. ``(8-3)*(4-6)+7=-3#``; the prompt
ends with '=' (nested) or '?' (chain). `split` is "train" or "test"; `level` is the difficulty
slice (depth for nested, chain length for chain).

meta.json keys: name, vocab, width, columns (label column names; always contains "answer",
"length" = prompt length including its terminal token, and "n_tgt" = answer tokens including
EOS), level_column, seed, slices ({level: {"train": n, "test": n,
"test_overlap_with_train": fraction}}), plus anything the builder adds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from looptf.toy.data import VOCAB, STOI, PAD_ID, encode, answer_tokens


def slice_paths(root: Path, split: str, level: int) -> tuple[Path, Path]:
    d = Path(root) / split
    return d / f"{level:02d}.tokens.npy", d / f"{level:02d}.meta.npy"


def full_sequence(prompt: str, answer: int) -> str:
    return prompt + answer_tokens(answer)


def encode_rows(seqs: list[str], width: int) -> np.ndarray:
    """Encode full sequences (prompt + answer + EOS) into a PAD-padded uint8 array."""
    out = np.full((len(seqs), width), PAD_ID, dtype=np.uint8)
    for i, p in enumerate(seqs):
        ids = encode(p)
        if len(ids) > width:
            raise ValueError(f"sequence of {len(ids)} tokens exceeds width {width}: {p!r}")
        out[i, : len(ids)] = ids
    return out


def write_slice(root, split: str, level: int, tokens: np.ndarray, labels: np.ndarray) -> None:
    tp, mp = slice_paths(root, split, level)
    tp.parent.mkdir(parents=True, exist_ok=True)
    assert tokens.dtype == np.uint8 and labels.dtype == np.int64 and len(tokens) == len(labels)
    np.save(tp, tokens)
    np.save(mp, labels)


def write_meta(root, meta: dict) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "meta.json").write_text(json.dumps(meta, indent=2))


class FrozenData:
    """Loads a frozen dataset and serves batches shaped like `data.make_batch`.

    reverse_answer: write the answer's digits smallest first (sign stays first, '#' last),
    e.g. stored `=-258#` is served as `=-852#`. Applied on the fly; the files are unchanged.
    """

    def __init__(self, root, device="cuda", reverse_answer: bool = False):
        self.reverse_answer = reverse_answer
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
        length = labels[:, self.cols["length"]]
        n_tgt = labels[:, self.cols["n_tgt"]]
        T = int((length + n_tgt).max())
        tok = np.array(tokens[:, :T], dtype=np.int64)
        if self.reverse_answer:
            minus = STOI["-"]
            for i in range(len(tok)):
                s0, e = int(length[i]), int(length[i] + n_tgt[i] - 1)  # e = index of '#'
                s0 += int(tok[i, s0] == minus)
                tok[i, s0:e] = tok[i, s0:e][::-1].copy()
        x = torch.from_numpy(tok)
        b = {"x": x.to(self.device, non_blocking=True),
             "pos": torch.from_numpy(length - 1).to(self.device)}
        for c, i in self.cols.items():
            b[c] = torch.from_numpy(np.ascontiguousarray(labels[:, i])).to(self.device)
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
