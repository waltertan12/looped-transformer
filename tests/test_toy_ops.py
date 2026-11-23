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
