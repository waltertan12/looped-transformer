#!/usr/bin/env python
"""Independently re-verify a frozen dataset written by scripts/build_data.py.

Does not import the generators: answers and labels are recomputed from the stored prompt
strings (eval / ast for nested, a small reference resolver for chain). Checks, per row:
  tokens[:length]                 is the prompt, ending in '=' (nested) or '?' (chain)
  tokens[length:length+n_tgt]     is the answer digits followed by '#'
  everything after                is PAD
  answer                          equals the exact value of the prompt

For nested_bal it additionally checks op_digits (max leaf digit count, re-derived from the AST,
must be in 1..op_digits column and the column must be allowed for the depth), ans_digits (recomputed
from the answer), and recounts the per-cell test counts against meta, then prints the per-cell table.

Examples:
  .venv/bin/python scripts/verify_data.py --root data/nested --sample 20000
  .venv/bin/python scripts/verify_data.py --root data/chain --sample 0
  .venv/bin/python scripts/verify_data.py --root data/nested_bal --sample 20000
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

from looptf.toy.data import BOUND, EOS, PAD_ID, VOCAB, answer_tokens, decode
from looptf.toy.dataset import slice_paths

REPO = Path(__file__).resolve().parent.parent
BATCH = 200_000
OPS = "+-*"
NESTED_TERMINAL = VOCAB.index("=")   # last prompt token, nested
CHAIN_TERMINAL = VOCAB.index("?")    # last prompt token, chain
EOS_ID = VOCAB.index(EOS)

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


def check_answer_text(answer: int, n_tgt: int, errs: list[str]) -> None:
    if n_tgt != len(answer_tokens(answer)):
        errs.append(f"n_tgt {n_tgt} != {len(answer_tokens(answer))} for answer {answer}")


def check_nested(prompt: str, answer: int, n_ops: int, depth: int, length: int, n_tgt: int,
                 errs: list[str]) -> None:
    if not prompt.endswith("="):
        errs.append("prompt does not end with '='")
        return
    expr = prompt[:-1]
    try:
        value = eval(expr, {"__builtins__": {}})
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
    check_answer_text(answer, n_tgt, errs)


def check_bal(prompt: str, answer: int, op_digits: int, ans_digits: int, depth: int,
              allowed: dict, errs: list[str]) -> None:
    """nested_bal extras. The leaves are re-read from the expression text; the op_digits column is an
    upper bound on the digit count of every leaf, and must be one of the depth's allowed values."""
    expr = prompt[:-1]
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        errs.append(f"cannot parse {expr!r}: {e}")
        return
    leaves = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    if not leaves or any(not isinstance(v, int) or v < 0 for v in leaves):
        errs.append(f"non-integer or negative leaf in {expr!r}")
        return
    max_digits = max(len(str(v)) for v in leaves)
    if not (1 <= max_digits <= op_digits):
        errs.append(f"op_digits column {op_digits} but max leaf digits {max_digits} in {expr!r}")
    if op_digits not in allowed.get(str(depth), []):
        errs.append(f"op_digits {op_digits} not allowed at depth {depth}")
    if ans_digits != len(str(abs(answer))):
        errs.append(f"ans_digits {ans_digits} != {len(str(abs(answer)))} for answer {answer}")


def check_chain(prompt: str, answer: int, chain_len: int, n_dist: int, n_ops_total: int,
                length: int, n_tgt: int, level: int, errs: list[str]) -> None:
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

    # Value: substitute already-reduced variable values into each expression, memoized, no modulus.
    memo: dict[str, int] = {}

    def value(name: str, stack: tuple = ()) -> int:
        if name in memo:
            return memo[name]
        if name in stack:
            raise ValueError("cycle")
        sub = re.sub("[a-z]", lambda m: f"({value(m.group(0), stack + (name,))})", defs[name])
        v = eval(sub, {"__builtins__": {}})
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
    check_answer_text(answer, n_tgt, errs)


# ---------------------------------------------------------------------------
# Per-slice checks
# ---------------------------------------------------------------------------

def check_ops(root: Path, meta: dict) -> None:
    """Test-only single-operator sets (nested_add / nested_sub): re-check every row from its text."""
    cols = {c: i for i, c in enumerate(meta["columns"])}
    op = {"add": "+", "sub": "-", "mul": "*"}[meta["kind"]]
    width = meta["width"]
    print(f"Verifying {root} ({meta['name']}, test only)")
    for lv in sorted(int(k) for k in meta["slices"]):
        sl = meta["slices"][str(lv)]
        tp, mp_ = slice_paths(root, "test", lv)
        if not tp.exists() or not mp_.exists():
            fail(f"level {lv}: missing files")
            continue
        tok, lab = np.load(tp), np.load(mp_)
        if tok.dtype != np.uint8 or tok.shape[1] != width or lab.shape != (len(tok), len(cols)):
            fail(f"level {lv}: bad dtype/shape {tok.dtype} {tok.shape} {lab.shape}")
            continue
        if len(tok) != sl["test"] or sum(sl["cells"].values()) != len(tok):
            fail(f"level {lv}: row count {len(tok)} != meta {sl['test']} / cells {sl['cells']}")
        seen, cells = set(), {}
        for i in range(len(tok)):
            r = lab[i]
            L = int(r[cols["length"]])
            prompt = decode(tok[i, :L])
            full = decode(tok[i])
            if prompt in seen:
                fail(f"level {lv} row {i}: duplicate prompt {prompt!r}")
            seen.add(prompt)
            if not prompt.endswith("="):
                fail(f"level {lv} row {i}: prompt does not end with '=': {prompt!r}")
                continue
            expr = prompt[:-1]
            if set(expr) - set("0123456789()" + op):
                fail(f"level {lv} row {i}: unexpected characters in {expr!r}")
            k = int(r[cols["in_digits"]])
            cells[f"in{k}"] = cells.get(f"in{k}", 0) + 1
            if any(len(n) != k for n in re.findall(r"\d+", expr)):
                fail(f"level {lv} row {i}: numbers not all {k} digits: {expr!r}")
            try:
                tree = ast.parse(expr, mode="eval").body
                val, dep = eval(expr), _bin_depth(tree)
            except Exception as e:  # noqa: BLE001
                fail(f"level {lv} row {i}: cannot parse {expr!r}: {e}")
                continue
            ans = int(r[cols["answer"]])
            if val != ans:
                fail(f"level {lv} row {i}: answer {ans} != {val} for {expr!r}")
            if dep != lv or int(r[cols["depth"]]) != lv:
                fail(f"level {lv} row {i}: depth {dep} / label {int(r[cols['depth']])} != level")
            if int(r[cols["n_ops"]]) != expr.count(op):
                fail(f"level {lv} row {i}: n_ops label != operator count")
            if int(r[cols["ans_digits"]]) != len(str(abs(ans))):
                fail(f"level {lv} row {i}: ans_digits label wrong")
            errs: list[str] = []
            check_answer_text(ans, int(r[cols["n_tgt"]]), errs)
            for e in errs:
                fail(f"level {lv} row {i}: {e}")
            if full != prompt + answer_tokens(ans) or len(full) > width:
                fail(f"level {lv} row {i}: stored row {full!r} != prompt + answer")
        if cells != {c: n for c, n in sl["cells"].items() if n}:
            fail(f"level {lv}: recounted cells {cells} != meta {sl['cells']}")
        print(f"  depth {lv:2d}: {len(tok):5d} rows  {sl['cells']}"
              + (f"  overlap {sl['overlap_with_nested_bal_train']:.4f}" if "overlap_with_nested_bal_train" in sl else ""))


def check_slice(root: Path, name: str, split: str, level: int, expected_rows: int, sample: int,
                is_nested: bool, width: int, cols: dict, allowed: dict | None = None
                ) -> tuple[np.ndarray, np.ndarray] | None:
    tp, mp_ = slice_paths(root, split, level)
    tag = f"{name} {split} level {level}"
    if not tp.exists() or not mp_.exists():
        fail(f"{tag}: missing file(s) {tp} / {mp_}")
        return None
    tok = np.load(tp, mmap_mode="r")
    lab = np.load(mp_, mmap_mode="r")
    if tok.dtype != np.uint8 or tok.ndim != 2 or tok.shape[1] != width:
        fail(f"{tag}: tokens dtype/shape {tok.dtype} {tok.shape}, expected uint8 [N, {width}]")
    if lab.dtype != np.int64 or lab.ndim != 2 or lab.shape[1] != len(cols):
        fail(f"{tag}: labels dtype/shape {lab.dtype} {lab.shape}, expected int64 [N, {len(cols)}]")
    if tok.shape[0] != lab.shape[0]:
        fail(f"{tag}: token rows {tok.shape[0]} != label rows {lab.shape[0]}")
    N = tok.shape[0]
    if N != expected_rows:
        fail(f"{tag}: rows {N} != meta {expected_rows}")
    if N == 0:
        return tok, lab

    terminal = NESTED_TERMINAL if is_nested else CHAIN_TERMINAL
    length_all = np.asarray(lab[:, cols["length"]]).astype(np.int64)
    n_tgt_all = np.asarray(lab[:, cols["n_tgt"]]).astype(np.int64)
    end_all = length_all + n_tgt_all  # index one past the answer's EOS
    if length_all.min() < 1 or end_all.max() > width:
        fail(f"{tag}: prompt+answer out of range (length [{length_all.min()}, {length_all.max()}], "
             f"end max {end_all.max()}, width {width})")
    if n_tgt_all.min() < 2:
        fail(f"{tag}: n_tgt below 2 (answer needs at least one digit and EOS)")

    # Vectorised structural checks over every row.
    bad_tail = bad_prompt = bad_answer = bad_eos = 0
    first_bad = None
    pos = np.arange(width)[None, :]
    for s in range(0, N, BATCH):
        tok_b = np.asarray(tok[s:s + BATCH])
        ln = length_all[s:s + BATCH]
        en = end_all[s:s + BATCH]
        in_prompt = pos < ln[:, None]
        in_answer = (pos >= ln[:, None]) & (pos < en[:, None])
        row_tail = ((tok_b != PAD_ID) & (pos >= en[:, None])).any(axis=1)   # non-PAD after the answer
        row_prompt = ((tok_b == PAD_ID) & in_prompt).any(axis=1)            # PAD inside the prompt
        row_answer = ((tok_b == PAD_ID) & in_answer).any(axis=1)            # PAD inside the answer
        last_prompt = tok_b[np.arange(len(tok_b)), np.clip(ln - 1, 0, width - 1)]
        row_term = last_prompt != terminal                                  # prompt must end '=' / '?'
        last_ans = tok_b[np.arange(len(tok_b)), np.clip(en - 1, 0, width - 1)]
        row_eos = last_ans != EOS_ID                                        # answer must end '#'
        bad_tail += int(row_tail.sum())
        bad_prompt += int((row_prompt | row_term).sum())
        bad_answer += int(row_answer.sum())
        bad_eos += int(row_eos.sum())
        any_bad = row_tail | row_prompt | row_answer | row_term | row_eos
        if first_bad is None and any_bad.any():
            first_bad = s + int(np.argmax(any_bad))
    if bad_tail:
        fail(f"{tag}: {bad_tail} rows with non-PAD after the answer (first bad row {first_bad})")
    if bad_prompt:
        fail(f"{tag}: {bad_prompt} rows with PAD inside the prompt or wrong terminal "
             f"'{VOCAB[terminal]}' (first bad row {first_bad})")
    if bad_answer:
        fail(f"{tag}: {bad_answer} rows with PAD inside the answer span")
    if bad_eos:
        fail(f"{tag}: {bad_eos} rows whose last answer token is not '{EOS}'")

    # Per-row semantic checks on a sample (or all rows).
    if sample and sample < N:
        idx = np.sort(np.random.default_rng(0).choice(N, size=sample, replace=False))
    else:
        idx = np.arange(N)
    n_err = 0
    for i in idx:
        i = int(i)
        ln = int(length_all[i])
        en = int(end_all[i])
        prompt = decode(np.asarray(tok[i, :ln]).tolist())
        answer_txt = decode(np.asarray(tok[i, ln:en]).tolist())
        labels = np.asarray(lab[i]).astype(np.int64)
        answer = int(labels[cols["answer"]])
        errs: list[str] = []
        if answer_txt != answer_tokens(answer):
            errs.append(f"answer tokens {answer_txt!r} != {answer_tokens(answer)!r}")
        if is_nested:
            check_nested(prompt, answer, int(labels[cols["n_ops"]]), int(labels[cols["depth"]]),
                         ln, n_tgt_all[i].item(), errs)
            if int(labels[cols["depth"]]) != level:
                errs.append(f"depth {int(labels[cols['depth']])} != file level {level}")
            if allowed is not None:
                check_bal(prompt, answer, int(labels[cols["op_digits"]]), int(labels[cols["ans_digits"]]),
                          int(labels[cols["depth"]]), allowed, errs)
        else:
            check_chain(prompt, answer, int(labels[cols["chain_len"]]),
                        int(labels[cols["n_distractors"]]), int(labels[cols["n_ops_total"]]),
                        ln, n_tgt_all[i].item(), level, errs)
        if errs:
            n_err += 1
            if n_err <= 5:
                fail(f"{tag} row {i}: {errs[0]} (+{len(errs) - 1} more)" if len(errs) > 1
                     else f"{tag} row {i}: {errs[0]}")
    if n_err > 5:
        fail(f"{tag}: {n_err} rows failed semantic checks in total (first 5 listed)")
    return tok, lab


def bal_level_report(level: int, sl: dict, tr_lab, te_lab, cols, fail_fn, name: str) -> list[str]:
    """Recount test cells from the op_digits / ans_digits columns, compare with meta, and describe
    the train answer-digit shares for the level."""
    out = []
    op_c, ad_c = cols["op_digits"], cols["ans_digits"]
    te_ops = np.asarray(te_lab[:, op_c]).astype(np.int64)
    te_ad = np.asarray(te_lab[:, ad_c]).astype(np.int64)
    counted = {}
    for o, a in zip(te_ops.tolist(), te_ad.tolist()):
        counted[f"op{o}_ans{a}"] = counted.get(f"op{o}_ans{a}", 0) + 1
    meta_cells = sl.get("test_cells", {})
    for k in set(meta_cells) | set(counted):
        if meta_cells.get(k, 0) != counted.get(k, 0):
            fail_fn(f"{name} level {level}: test cell {k} meta {meta_cells.get(k, 0)} != rows {counted.get(k, 0)}")
    if sum(counted.values()) != sl["test"]:
        fail_fn(f"{name} level {level}: test cell total {sum(counted.values())} != {sl['test']}")
    cells = " ".join(f"{k}={counted.get(k, 0)}" for k in sorted(meta_cells, key=lambda k: (int(k.split('_')[0][2:]), int(k.split('ans')[1]))))
    out.append(f"  lvl {level:>2} test: {cells}")
    if tr_lab.shape[0]:
        tr_ad = np.asarray(tr_lab[:, ad_c]).astype(np.int64)
        share = np.bincount(tr_ad, minlength=13)[1:13] / tr_ad.size
        out.append(f"  lvl {level:>2} train ans_digits share 1..12: "
                   + " ".join(f"{x * 100:.1f}" for x in share) + "  (% of train rows)")
    return out


def key_set(tok, lab, length_col: int) -> list[bytes]:
    """Prompt keys (tokens[:length]); overlap is judged on the prompt, not the answer."""
    ln = np.asarray(lab[:, length_col]).astype(np.int64)
    return [np.asarray(tok[i, :ln[i]]).tobytes() for i in range(len(ln))]


def answer_stats(lab, cols) -> tuple:
    ans = np.asarray(lab[:, cols["answer"]]).astype(np.int64)
    n_tgt = np.asarray(lab[:, cols["n_tgt"]]).astype(np.int64)
    return (int(ans.min()), int(np.median(ans)), int(ans.max()),
            float((n_tgt - 1).mean()), float((ans < 0).mean()))


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
    if meta.get("generator") == "looptf.toy.ops_data.gen_single_op":
        if meta["vocab"] != VOCAB[: len(meta["vocab"])]:
            fail("meta vocab is not a prefix of looptf.toy.data.VOCAB")
        check_ops(root, meta)
        if fails:
            print(f"FAILURES ({len(fails)}):")
            for f_ in fails[:200]:
                print("  - " + f_)
            sys.exit(1)
        print("ALL CHECKS PASSED")
        return
    is_nested = name in ("nested", "nested_bal")
    is_bal = name == "nested_bal"
    cols = {c: i for i, c in enumerate(meta["columns"])}
    width = meta["width"]

    if meta["vocab"] != VOCAB[: len(meta["vocab"])]:
        fail("meta vocab is not a prefix of looptf.toy.data.VOCAB")
    if meta.get("eos") != EOS:
        fail(f"meta eos {meta.get('eos')!r} != {EOS!r}")
    if meta.get("bound") != BOUND:
        fail(f"meta bound {meta.get('bound')!r} != looptf.toy.data.BOUND {BOUND!r}")
    if "mod" in meta:
        fail("meta still has a 'mod' key (obsolete mod-7 format)")
    if is_bal:
        for need in ("op_digits", "ans_digits"):
            if need not in cols:
                fail(f"nested_bal meta columns must include {need!r}")
    for need in ("answer", "length", "n_tgt", meta["level_column"]):
        if need not in cols:
            fail(f"meta columns must include {need!r}")
    if meta["columns"][0] != "answer":
        fail("meta columns must start with 'answer'")

    levels = sorted(int(k) for k in meta["slices"])
    print(f"Verifying {root} ({name}): {len(levels)} levels, width {width}, "
          f"sample {args.sample or 'all'} rows per (split, level)")
    hdr = (f"{'lvl':>3} {'split':>5} {'rows':>9} {'len_mean':>8} {'len_max':>7}  "
           f"{'ans_min':>12} {'ans_med':>10} {'ans_max':>12} {'ans_chr':>7} {'neg':>6}  {'overlap':>8}")
    print(hdr)
    print("-" * len(hdr))

    allowed = meta.get("op_digits_allowed") if is_bal else None
    bal_lines: list[str] = []
    for level in levels:
        sl = meta["slices"][str(level)]
        train = check_slice(root, name, "train", level, sl["train"], args.sample, is_nested, width, cols, allowed)
        test = check_slice(root, name, "test", level, sl["test"], args.sample, is_nested, width, cols, allowed)
        if train is None or test is None:
            continue
        tr_tok, tr_lab = train
        te_tok, te_lab = test
        if is_bal:
            bal_lines.extend(bal_level_report(level, sl, tr_lab, te_lab, cols, fail, name))
        lk = cols["length"]
        if tr_lab.shape[0] > 0:
            tr_len = np.asarray(tr_lab[:, lk]).astype(np.int64)
            lvl_col = np.asarray(tr_lab[:, cols[meta["level_column"]]]).astype(np.int64)
            if np.any(lvl_col != level):
                fail(f"{name} train level {level}: {int(np.sum(lvl_col != level))} rows with "
                     f"{meta['level_column']} != file level")
            mn, md, mx, chr_mean, neg = answer_stats(tr_lab, cols)
            print(f"{level:>3} {'train':>5} {tr_lab.shape[0]:>9,d} {tr_len.mean():>8.2f} {tr_len.max():>7d}  "
                  f"{mn:>12,d} {md:>10,d} {mx:>12,d} {chr_mean:>7.2f} {neg:>6.3f}  {'':>8}")
        if te_lab.shape[0] > 0:
            te_len = np.asarray(te_lab[:, lk]).astype(np.int64)
            tr_keys = set(key_set(tr_tok, tr_lab, lk))
            te_keys = key_set(te_tok, te_lab, lk)
            overlap = sum(k in tr_keys for k in te_keys) / len(te_keys)
            if abs(overlap - sl["test_overlap_with_train"]) > 1e-9:
                fail(f"{name} level {level}: recomputed test overlap {overlap:.6f} != meta "
                     f"{sl['test_overlap_with_train']:.6f}")
            mn, md, mx, chr_mean, neg = answer_stats(te_lab, cols)
            print(f"{level:>3} {'test':>5} {te_lab.shape[0]:>9,d} {te_len.mean():>8.2f} {te_len.max():>7d}  "
                  f"{mn:>12,d} {md:>10,d} {mx:>12,d} {chr_mean:>7.2f} {neg:>6.3f}  {overlap:>8.4f}")
        else:
            print(f"{level:>3} {'test':>5} {0:>9,d}")

    if is_bal and bal_lines:
        print()
        print("nested_bal per-level cells (op_digits x ans_digits; test counts recounted from the rows)")
        print("\n".join(bal_lines))

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
