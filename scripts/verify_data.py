#!/usr/bin/env python
"""Independently re-verify a frozen dataset written by scripts/build_data.py.

Does not import the generators: answers and labels are recomputed from the stored prompt
strings (eval / ast for nested, a small reference resolver for chain).

Examples:
  .venv/bin/python scripts/verify_data.py --root data/nested --sample 20000
  .venv/bin/python scripts/verify_data.py --root data/chain --sample 0
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

from looptf.toy.data import MOD, VOCAB, decode
from looptf.toy.dataset import slice_paths

REPO = Path(__file__).resolve().parent.parent
BATCH = 200_000
OPS = "+-*"
NESTED_EOS = VOCAB.index("=")
CHAIN_EOS = VOCAB.index("?")

fails: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


# ---------------------------------------------------------------------------
# Per-prompt reference checks
# ---------------------------------------------------------------------------

def _bin_depth(node) -> int:
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            raise ValueError("unexpected operator")
        return 1 + max(_bin_depth(node.left), _bin_depth(node.right))
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return 0
    raise ValueError(f"unexpected node {type(node).__name__}")


def check_nested(prompt: str, answer: int, n_ops: int, depth: int, length: int, errs: list[str]) -> None:
    if not prompt.endswith("="):
        errs.append("prompt does not end with '='")
        return
    expr = prompt[:-1]
    try:
        value = eval(expr, {"__builtins__": {}}) % MOD
        tree = ast.parse(expr, mode="eval").body
        true_depth = _bin_depth(tree)
    except Exception as e:  # noqa: BLE001
        errs.append(f"cannot evaluate {expr!r}: {e}")
        return
    if value != answer:
        errs.append(f"answer {answer} != recomputed {value} for {expr!r}")
    true_ops = sum(expr.count(c) for c in OPS)
    if true_ops != n_ops:
        errs.append(f"n_ops {n_ops} != {true_ops} for {expr!r}")
    if true_depth != depth:
        errs.append(f"depth {depth} != AST depth {true_depth} for {expr!r}")
    if len(prompt) != length:
        errs.append(f"length {length} != {len(prompt)}")


def check_chain(prompt: str, answer: int, chain_len: int, n_dist: int, n_ops_total: int,
                length: int, level: int, errs: list[str]) -> None:
    if not prompt.endswith("?"):
        errs.append("prompt does not end with '?'")
        return
    parts = prompt[:-1].split(";")
    query = parts[-1]
    defs: dict[str, str] = {}
    for st in parts[:-1]:
        name, sep, expr = st.partition("=")
        if not sep or not re.fullmatch("[a-z]", name) or name in defs:
            errs.append(f"bad or duplicate statement {st!r}")
            return
        defs[name] = expr
    if not re.fullmatch("[a-z]", query) or query not in defs:
        errs.append(f"query {query!r} not defined")
        return

    # Chain: follow references from the query down to a digits-only statement.
    visited: list[str] = []
    cur = query
    while True:
        if cur in visited or cur not in defs:
            errs.append(f"broken reference chain at {cur!r}")
            return
        visited.append(cur)
        refs = set(re.findall("[a-z]", defs[cur]))
        if not refs:
            break
        if len(refs) > 1:
            errs.append(f"statement {cur!r} references several variables")
            return
        cur = refs.pop()
    true_chain = len(visited)
    if true_chain != chain_len or true_chain != level:
        errs.append(f"chain_len {chain_len} (file level {level}) != traversed {true_chain}")
    true_dist = len(defs) - true_chain
    if true_dist != n_dist:
        errs.append(f"n_distractors {n_dist} != {true_dist}")
    true_ops = sum(sum(defs[v].count(c) for c in OPS) for v in visited)
    if true_ops != n_ops_total:
        errs.append(f"n_ops_total {n_ops_total} != {true_ops}")

    # Value: substitute already-reduced variable values into each expression, memoized.
    memo: dict[str, int] = {}

    def value(name: str, stack: tuple = ()) -> int:
        if name in memo:
            return memo[name]
        if name in stack:
            raise ValueError("cycle")
        sub = re.sub("[a-z]", lambda m: f"({value(m.group(0), stack + (name,))})", defs[name])
        v = eval(sub, {"__builtins__": {}}) % MOD
        memo[name] = v
        return v

    try:
        got = value(query)
    except Exception as e:  # noqa: BLE001
        errs.append(f"cannot evaluate query {query!r}: {e}")
        return
    if got != answer:
        errs.append(f"answer {answer} != recomputed {got}")
    if len(prompt) != length:
        errs.append(f"length {length} != {len(prompt)}")


# ---------------------------------------------------------------------------
# Per-slice checks
# ---------------------------------------------------------------------------

def check_slice(root: Path, name: str, split: str, level: int, meta: dict, expected_rows: int,
                sample: int, is_nested: bool, width: int, cols: dict) -> tuple[np.ndarray, np.ndarray] | None:
    tp, mp_ = slice_paths(root, split, level)
    tag = f"{name} {split} level {level}"
    if not tp.exists() or not mp_.exists():
        fail(f"{tag}: missing file(s) {tp} / {mp_}")
        return None
    tok = np.load(tp, mmap_mode="r")
    lab = np.load(mp_, mmap_mode="r")
    if tok.dtype != np.uint8 or tok.ndim != 2 or tok.shape[1] != width:
        fail(f"{tag}: tokens dtype/shape {tok.dtype} {tok.shape}, expected uint8 [N, {width}]")
    if lab.dtype != np.int16 or lab.ndim != 2 or lab.shape[1] != len(cols):
        fail(f"{tag}: labels dtype/shape {lab.dtype} {lab.shape}, expected int16 [N, {len(cols)}]")
    if tok.shape[0] != lab.shape[0]:
        fail(f"{tag}: token rows {tok.shape[0]} != label rows {lab.shape[0]}")
    N = tok.shape[0]
    if N != expected_rows:
        fail(f"{tag}: rows {N} != meta {expected_rows}")
    if N == 0:
        return tok, lab

    eos = NESTED_EOS if is_nested else CHAIN_EOS
    length_all = np.asarray(lab[:, cols["length"]]).astype(np.int64)
    if length_all.min() < 1 or length_all.max() > width:
        fail(f"{tag}: length out of range [{length_all.min()}, {length_all.max()}]")
    ans = np.asarray(lab[:, cols["answer"]]).astype(np.int64)
    if ans.min() < 0 or ans.max() >= MOD:
        fail(f"{tag}: answer out of range [0, {MOD})")

    # Vectorised structural checks over every row: PAD tail, non-PAD body, terminal token.
    bad_tail = bad_body = bad_eos = 0
    first_bad = None
    pos = np.arange(width)[None, :]
    for s in range(0, N, BATCH):
        tok_b = np.asarray(tok[s:s + BATCH])
        ln = length_all[s:s + BATCH]
        inside = pos < ln[:, None]
        row_tail = ((tok_b != 0) & ~inside).any(axis=1)      # non-PAD after length
        row_body = ((tok_b == 0) & inside).any(axis=1)       # PAD inside the prompt
        last = tok_b[np.arange(len(tok_b)), np.clip(ln - 1, 0, width - 1)]
        row_eos = last != eos                                # wrong terminal token
        bad_tail += int(row_tail.sum())
        bad_body += int(row_body.sum())
        bad_eos += int(row_eos.sum())
        any_bad = row_tail | row_body | row_eos
        if first_bad is None and any_bad.any():
            first_bad = s + int(np.argmax(any_bad))
    if bad_tail or bad_body:
        fail(f"{tag}: {bad_tail} rows with non-PAD after length, {bad_body} rows with PAD inside length "
             f"(first bad row {first_bad})")
    if bad_eos:
        fail(f"{tag}: {bad_eos} rows whose token at length-1 is not '{VOCAB[eos]}'")

    # Per-row semantic checks on a sample (or all rows).
    if sample and sample < N:
        idx = np.sort(np.random.default_rng(0).choice(N, size=sample, replace=False))
    else:
        idx = np.arange(N)
    n_err = 0
    for i in idx:
        i = int(i)
        ln = int(length_all[i])
        prompt = decode(np.asarray(tok[i, :ln]).tolist())
        labels = np.asarray(lab[i]).astype(np.int64)
        errs: list[str] = []
        if is_nested:
            check_nested(prompt, int(labels[cols["answer"]]), int(labels[cols["n_ops"]]),
                         int(labels[cols["depth"]]), ln, errs)
            if int(labels[cols["depth"]]) != level:
                errs.append(f"depth {int(labels[cols['depth']])} != file level {level}")
        else:
            check_chain(prompt, int(labels[cols["answer"]]), int(labels[cols["chain_len"]]),
                        int(labels[cols["n_distractors"]]), int(labels[cols["n_ops_total"]]),
                        ln, level, errs)
        if errs:
            n_err += 1
            if n_err <= 5:
                fail(f"{tag} row {i}: {errs[0]} (+{len(errs) - 1} more)" if len(errs) > 1
                     else f"{tag} row {i}: {errs[0]}")
    if n_err > 5:
        fail(f"{tag}: {n_err} rows failed semantic checks in total (first 5 listed)")
    return tok, lab


def key_set(tok, lab, length_col: int) -> list[bytes]:
    ln = np.asarray(lab[:, length_col]).astype(np.int64)
    return [np.asarray(tok[i, :ln[i]]).tobytes() for i in range(len(ln))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset dir, e.g. data/nested (relative to repo root)")
    ap.add_argument("--sample", type=int, default=20000, help="rows checked per (split, level); 0 = all")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root
    meta_path = root / "meta.json"
    if not meta_path.exists():
        print(f"FAILED: {meta_path} does not exist")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    name = meta.get("name", root.name)
    is_nested = name == "nested"
    cols = {c: i for i, c in enumerate(meta["columns"])}
    width = meta["width"]

    if meta["vocab"] != VOCAB[: len(meta["vocab"])]:
        fail("meta vocab is not a prefix of looptf.toy.data.VOCAB")
    if meta["mod"] != MOD:
        fail(f"meta mod {meta['mod']} != {MOD}")
    if "answer" not in cols or "length" not in cols or meta["level_column"] not in cols:
        fail("meta columns must include answer, length and level_column")
    if meta["columns"][0] != "answer":
        fail("meta columns must start with 'answer'")

    levels = sorted(int(k) for k in meta["slices"])
    print(f"Verifying {root} ({name}): {len(levels)} levels, width {width}, "
          f"sample {args.sample or 'all'} rows per (split, level)")
    hdr = (f"{'lvl':>3} {'split':>5} {'rows':>9} {'len_mean':>8} {'len_max':>7}  "
           + " ".join(f"{'a' + str(k):>6}" for k in range(MOD)) + f"  {'overlap':>8}")
    print(hdr)
    print("-" * len(hdr))

    for level in levels:
        sl = meta["slices"][str(level)]
        train = check_slice(root, name, "train", level, meta, sl["train"], args.sample,
                            is_nested, width, cols)
        test = check_slice(root, name, "test", level, meta, sl["test"], args.sample,
                           is_nested, width, cols)
        if train is None or test is None:
            continue
        tr_tok, tr_lab = train
        te_tok, te_lab = test
        lk = cols["length"]
        if tr_lab.shape[0] > 0:
            tr_len = np.asarray(tr_lab[:, lk]).astype(np.int64)
            dist = np.bincount(np.asarray(tr_lab[:, cols["answer"]]).astype(np.int64),
                               minlength=MOD) / tr_lab.shape[0]
            lvl_col = np.asarray(tr_lab[:, cols[meta["level_column"]]]).astype(np.int64)
            if np.any(lvl_col != level):
                fail(f"{name} train level {level}: {int(np.sum(lvl_col != level))} rows with "
                     f"{meta['level_column']} != file level")
            print(f"{level:>3} {'train':>5} {tr_lab.shape[0]:>9,d} {tr_len.mean():>8.2f} {tr_len.max():>7d}  "
                  + " ".join(f"{d:>6.3f}" for d in dist) + f"  {'':>8}")
        if te_lab.shape[0] > 0:
            te_len = np.asarray(te_lab[:, lk]).astype(np.int64)
            tr_keys = set(key_set(tr_tok, tr_lab, lk))
            te_keys = key_set(te_tok, te_lab, lk)
            overlap = sum(k in tr_keys for k in te_keys) / len(te_keys)
            if abs(overlap - sl["test_overlap_with_train"]) > 1e-9:
                fail(f"{name} level {level}: recomputed test overlap {overlap:.6f} != meta "
                     f"{sl['test_overlap_with_train']:.6f}")
            print(f"{level:>3} {'test':>5} {te_lab.shape[0]:>9,d} {te_len.mean():>8.2f} {te_len.max():>7d}  "
                  + " ".join(f"{d:>6.3f}" for d in np.bincount(np.asarray(te_lab[:, cols['answer']]).astype(np.int64),
                                                                 minlength=MOD) / te_lab.shape[0])
                  + f"  {overlap:>8.4f}")
        else:
            print(f"{level:>3} {'test':>5} {0:>9,d}")

    print()
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for f_ in fails[:200]:
            print("  - " + f_)
        if len(fails) > 200:
            print(f"  ... and {len(fails) - 200} more")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
