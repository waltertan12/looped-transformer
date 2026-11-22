import ast
import random

import pytest
import torch

from looptf.toy.data import (
    MOD,
    PAD_ID,
    STOI,
    VOCAB,
    decode,
    encode,
    gen_expr,
    make_batch,
    render,
    sample_example,
)


def _to_tree(node):
    """Convert a Python ast expression into the (op, left, right) tree form."""
    if isinstance(node, ast.BinOp):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(node.op)]
        return (op, _to_tree(node.left), _to_tree(node.right))
    assert isinstance(node, ast.Constant) and isinstance(node.value, int)
    return node.value


def _ast_depth(node):
    if isinstance(node, ast.BinOp):
        return 1 + max(_ast_depth(node.left), _ast_depth(node.right))
    return 0


def _parse(expr):
    return ast.parse(expr, mode="eval").body


def _combos():
    for d in range(1, 17):
        for n in range(d, min(16, 2**d - 1) + 1):
            yield n, d


def test_vocab_and_codec():
    assert VOCAB[0] == "_" and STOI["_"] == PAD_ID == 0
    assert VOCAB[:17] == ["_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "(", ")", "="]
    assert VOCAB[17:] == [chr(c) for c in range(ord("a"), ord("z") + 1)] + [";", "?"]
    assert len(set(VOCAB)) == len(VOCAB) == 45
    ids = encode("3+4*(2-5)=")
    assert decode(ids + [PAD_ID, PAD_ID]) == "3+4*(2-5)="


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gen_expr_op_count_depth_and_answer(seed):
    rng = random.Random(seed)
    for n, d in _combos():
        expr, ans = gen_expr(n, d, rng)
        assert sum(expr.count(c) for c in "+-*") == n, expr
        assert set(expr) <= set("0123456789+-*()"), expr
        tree = _parse(expr)
        assert _ast_depth(tree) == d, (expr, n, d)
        assert ans == eval(expr) % MOD, expr
        assert 0 <= ans <= 9


@pytest.mark.parametrize("seed", [0, 1])
def test_minimal_parens_ast_round_trip(seed):
    rng = random.Random(seed)
    for n, d in _combos():
        expr, _ = gen_expr(n, d, rng)
        assert render(_to_tree(_parse(expr))) == expr, expr


def test_minimal_parens_examples():
    assert render(("*", ("+", 3, 4), ("-", 2, 5))) == "(3+4)*(2-5)"
    assert render(("+", 3, ("*", 4, ("-", 2, 5)))) == "3+4*(2-5)"
    assert render(("-", 1, ("-", 2, 3))) == "1-(2-3)"
    assert render(("-", 1, ("+", 2, 3))) == "1-(2+3)"
    assert render(("+", 1, ("+", 2, 3))) == "1+2+3"
    assert render(("-", ("+", 1, 2), 3)) == "1+2-3"


def test_gen_expr_rejects_infeasible():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        gen_expr(2, 3, rng)  # n_ops < depth
    with pytest.raises(ValueError):
        gen_expr(8, 3, rng)  # n_ops > 2**3-1
    with pytest.raises(ValueError):
        gen_expr(1, 0, rng)  # depth < 1


def test_sample_example_ranges():
    rng = random.Random(0)
    for _ in range(300):
        expr, ans, n, d = sample_example(rng, (2, 10), (1, 4))
        assert 1 <= d <= 4
        assert d <= n <= min(10, 2**d - 1)
        assert sum(expr.count(c) for c in "+-*") == n
        assert _ast_depth(_parse(expr)) == d
        assert ans == eval(expr) % MOD


def test_make_batch_shapes_and_contents():
    rng = random.Random(123)
    B = 32
    batch = make_batch(B, rng, (1, 16), (1, 16), max_len=128)
    x, pos, y, n_ops, depth = batch["x"], batch["pos"], batch["y"], batch["n_ops"], batch["depth"]
    assert x.dtype == pos.dtype == y.dtype == torch.long
    assert x.shape[0] == B and x.shape[1] <= 128
    assert pos.shape == y.shape == n_ops.shape == depth.shape == (B,)
    T = x.shape[1]
    assert int(pos.max()) == T - 1  # longest example ends its '=' at the last column
    for b in range(B):
        p = int(pos[b])
        assert int(x[b, p]) == STOI["="]
        assert (x[b, p + 1 :] == PAD_ID).all()
        assert (x[b, :p] != PAD_ID).all()
        expr = decode(x[b, :p].tolist())
        assert sum(expr.count(c) for c in "+-*") == int(n_ops[b])
        assert _ast_depth(_parse(expr)) == int(depth[b])
        assert int(y[b]) == STOI[str(eval(expr) % MOD)]


def test_make_batch_deterministic_given_seed():
    a = make_batch(16, random.Random(7), (1, 8), (1, 4))
    b = make_batch(16, random.Random(7), (1, 8), (1, 4))
    for k in a:
        assert torch.equal(a[k], b[k])


def test_longest_example_fits_in_128_tokens():
    rng = random.Random(0)
    longest = 0
    for _ in range(50):
        expr, _ = gen_expr(16, 16, rng)
        longest = max(longest, len(encode(expr + "=")))
    assert longest <= 128
    # worst case by construction: a pure chain with parens everywhere still fits
    expr, _ = gen_expr(16, 16, random.Random(1))
    assert len(expr) + 1 <= 128


def test_leaves_exclude_zero_and_answers_below_mod():
    rng = random.Random(7)
    for _ in range(500):
        expr, ans, _, _ = sample_example(rng, (1, 16), (1, 16))
        assert "0" not in expr and "7" not in expr and "8" not in expr and "9" not in expr
        assert 0 <= ans < MOD
