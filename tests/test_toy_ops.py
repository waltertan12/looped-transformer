import ast
import random
import re

import pytest

from looptf.toy.ops_data import exact_leaf, feasible_n_ops, gen_single_op


def _bin_depth(node) -> int:
    if isinstance(node, ast.BinOp):
        return 1 + max(_bin_depth(node.left), _bin_depth(node.right))
    return 0


def test_exact_leaf_ranges():
    rng = random.Random(0)
    assert all(1 <= exact_leaf(rng, 1) <= 9 for _ in range(500))
    assert all(1000 <= exact_leaf(rng, 4) <= 9999 for _ in range(500))


def test_add_is_flat_chain():
    rng = random.Random(1)
    for _ in range(50):
        expr, ans = gen_single_op("add", 3, 3, rng, 2)
        assert expr.count("+") == 3
        assert not set(expr) & set("()-*")
        assert all(len(n) == 2 for n in re.findall(r"\d+", expr))
        assert eval(expr) == ans
        assert _bin_depth(ast.parse(expr, mode="eval").body) == 3


def test_add_requires_depth_eq_nops():
    with pytest.raises(ValueError):
        gen_single_op("add", 4, 3, random.Random(0), 1)
    assert feasible_n_ops("add", 5) == (5, 5)
    assert feasible_n_ops("sub", 3) == (3, 7)


def test_sub_depth_and_value():
    rng = random.Random(2)
    for _ in range(200):
        depth = rng.randint(1, 6)
        lo, hi = feasible_n_ops("sub", depth)
        n_ops = rng.randint(lo, min(hi, 12))
        k = rng.randint(1, 4)
        expr, ans = gen_single_op("sub", n_ops, depth, rng, k)
        assert set(expr) <= set("0123456789-()")
        assert expr.count("-") == n_ops
        assert eval(expr) == ans
        assert _bin_depth(ast.parse(expr, mode="eval").body) == depth
        assert all(len(n) == k for n in re.findall(r"\d+", expr))


# ---- Task 2: build + verify ----
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def test_build_ops_small(tmp_path):
    from build_ops import build_ops
    from looptf.toy.data import decode
    from looptf.toy.dataset import FrozenData

    meta = build_ops("sub", tmp_path, n_per_cell=20, max_depth=3)
    fd = FrozenData(tmp_path, device="cpu")
    sets = fd.test_sets()
    assert sorted(sets) == [1, 2, 3]
    assert meta["columns"] == ["answer", "n_ops", "depth", "length", "n_tgt", "in_digits", "ans_digits", "in_train"]
    for lv, b in sets.items():
        prompts = []
        for i in range(len(b["pos"])):
            L = int(b["length"][i])
            row = decode(b["x"][i])
            assert len(row) <= 80
            p = row[:L]
            prompts.append(p)
            assert p.endswith("=") and set(p[:-1]) <= set("0123456789-()")
            k = int(b["in_digits"][i])
            assert all(len(n) == k for n in re.findall(r"\d+", p))
            assert int(b["depth"][i]) == lv
        assert len(set(prompts)) == len(prompts)


def test_build_ops_thin_cell(tmp_path):
    from build_ops import build_ops

    meta = build_ops("add", tmp_path, n_per_cell=1000, max_depth=1)
    assert meta["slices"]["1"]["cells"]["in1"] == 81


def test_verify_accepts_ops_build(tmp_path):
    from build_ops import build_ops

    build_ops("add", tmp_path / "nested_add", n_per_cell=30, max_depth=4)
    r = subprocess.run([sys.executable, str(REPO / "scripts/verify_data.py"), "--root", str(tmp_path / "nested_add")],
                       capture_output=True, text=True, cwd=REPO)
    assert "ALL CHECKS PASSED" in r.stdout, r.stdout[-2000:]


# ---- Task 3: checkpoint scorer ----
import torch


def test_load_checkpoint_roundtrip(tmp_path):
    from eval_ckpt import load_checkpoint, train_k
    from looptf.toy.model import ToyConfig, ToyLM

    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=46, d_model=32, n_heads=2, d_ff=64, max_len=128, place_emb=True)
    m = ToyLM(cfg)
    p = tmp_path / "m.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(),
                "args": {"train_loops": "8", "reverse_answer": True}}, p)
    m2, args = load_checkpoint(p, "cpu")
    x = torch.randint(1, 17, (3, 20))
    m.eval()
    assert torch.allclose(m(x, n_loops=2), m2(x, n_loops=2))
    assert not m2.training
    assert train_k(args) == 8
    assert args["reverse_answer"] is True


def test_train_k_rejects_range():
    from eval_ckpt import train_k

    with pytest.raises(ValueError):
        train_k({"train_loops": "1-8"})


def test_dump_per_example_keeps_in_digits():
    import inspect
    from looptf.toy import train

    assert '"in_digits"' in inspect.getsource(train.dump_per_example)


# ---- Task 4: operator labels ----
import numpy as np


def _rows(seqs):
    from looptf.toy.dataset import encode_rows
    tok = encode_rows(seqs, 80)
    length = np.array([s.index("=") + 1 for s in seqs])
    return tok, length


def test_operator_counts():
    from ops_labels import operator_counts

    tok, length = _rows(["(1+2)*3-4=5#", "7*8=56#", "2-9=-7#"])
    c = operator_counts(tok, length)
    assert c["n_mul"].tolist() == [1, 1, 0]
    assert c["n_add"].tolist() == [1, 0, 0]
    assert c["n_sub"].tolist() == [1, 0, 1]  # the answer's '-' is not counted
    assert c["n_paren"].tolist() == [1, 0, 0]
    assert np.allclose(c["mul_frac"], [1 / 3, 1.0, 0.0])


def test_labels_join_asserts(tmp_path):
    from build_ops import build_ops
    from ops_labels import labels_for_run

    build_ops("sub", tmp_path, n_per_cell=10, max_depth=2)
    from looptf.toy.dataset import FrozenData
    sets = FrozenData(tmp_path, device="cpu").test_sets()
    ans = np.concatenate([b["answer"].numpy() for b in sets.values()])
    dep = np.concatenate([b["depth"].numpy() for b in sets.values()])
    pe = {"K": np.concatenate([np.full(len(ans), 1), np.full(len(ans), 8)]),
          "answer": np.concatenate([ans, ans]), "depth": np.concatenate([dep, dep])}
    lab = labels_for_run(tmp_path, pe)
    assert len(lab["n_sub"]) == 2 * len(ans) and (lab["n_mul"] == 0).all()
    bad = dict(pe, answer=pe["answer"][::-1].copy())
    with pytest.raises(AssertionError):
        labels_for_run(tmp_path, bad)


# ---- Task 5: report logic ----
def test_mul_tercile_gap():
    from ops_report import mul_tercile_gap

    mf = np.linspace(0, 1, 90)
    correct = mf < 0.5  # low-mul rows right, high-mul rows wrong
    gap, rows = mul_tercile_gap(correct, mf, np.zeros(90, dtype=int))
    assert gap == pytest.approx(1.0) and rows == 60
    gap, rows = mul_tercile_gap(correct[:30], mf[:30], np.zeros(30, dtype=int))
    assert rows == 0


def test_h13_verdict():
    from ops_report import h13_verdict

    assert h13_verdict([0.01, 0.02], [0.0, 0.03]) == "supported"
    assert h13_verdict([0.01], [0.06]) == "rejected"
    assert h13_verdict([0.04], [0.0]) == "unclear"


# ---- Final-review fixes ----
def _fake_train_root(src: Path, dst: Path):
    """A dataset whose train split is src's test split (so every src prompt is 'seen')."""
    import json, shutil
    meta = json.loads((src / "meta.json").read_text())
    (dst / "train").mkdir(parents=True)
    for f in (src / "test").iterdir():
        shutil.copy(f, dst / "train" / f.name)
    (dst / "meta.json").write_text(json.dumps(meta))


def test_build_ops_overlap_per_cell_and_row_flag(tmp_path):
    from build_ops import build_ops
    from looptf.toy.dataset import FrozenData

    build_ops("sub", tmp_path / "a", n_per_cell=20, max_depth=2)
    _fake_train_root(tmp_path / "a", tmp_path / "b")
    meta = build_ops("sub", tmp_path / "c", n_per_cell=20, max_depth=2, overlap_root=tmp_path / "b")
    assert meta["columns"][-1] == "in_train"
    for lv in ("1", "2"):
        ov = meta["slices"][lv]["overlap_by_cell"]
        assert set(ov) == {"in1", "in2", "in3", "in4"} and all(v == 1.0 for v in ov.values())
    for b in FrozenData(tmp_path / "c", device="cpu").test_sets().values():
        assert (b["in_train"] == 1).all()
    no_ov = build_ops("sub", tmp_path / "d", n_per_cell=5, max_depth=1)
    assert "overlap_by_cell" not in no_ov["slices"]["1"]


def test_region_acc_excludes_seen():
    from ops_report import region_acc

    d = {"depth": np.array([1, 1, 2, 4]), "in_digits": np.array([2, 2, 3, 2]),
         "in_train": np.array([1, 0, 0, 0]), "correct": np.array([1.0, 0.0, 1.0, 1.0])}
    assert region_acc(d) == pytest.approx(0.5)  # seen row and depth-4 row excluded


# ---- H18: longer-number test sets ----
def test_mul_is_flat_chain():
    rng = random.Random(3)
    expr, ans = gen_single_op("mul", 2, 2, rng, 3)
    assert expr.count("*") == 2 and "(" not in expr and eval(expr) == ans
    assert _bin_depth(ast.parse(expr, mode="eval").body) == 2


def test_build_ops_custom_digits(tmp_path):
    from build_ops import build_ops
    meta = build_ops("add", tmp_path, n_per_cell=10, max_depth=1, in_digits=(4, 5))
    assert set(meta["slices"]["1"]["cells"]) == {"in4", "in5"}
    assert meta["in_digits"] == [4, 5]
