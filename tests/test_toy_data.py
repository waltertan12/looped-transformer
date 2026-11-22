import ast
import random

import pytest
import torch

import looptf.toy.data as data
from looptf.toy.data import (
    EOS,
    PAD_ID,
    STOI,
    VOCAB,
    answer_tokens,
    decode,
    encode,
    gen_expr,
    make_batch,
    render,
    sample_example,
    tree_max_abs,
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
    assert VOCAB[17:] == [chr(c) for c in range(ord("a"), ord("z") + 1)] + [";", "?", "#"]
    assert len(set(VOCAB)) == len(VOCAB) == 46
    assert VOCAB[45] == "#" == EOS and STOI["#"] == 45
    ids = encode("3+4*(2-5)=")
    assert decode(ids + [PAD_ID, PAD_ID]) == "3+4*(2-5)="


def test_answer_tokens():
    assert answer_tokens(-3) == "-3#"
    assert answer_tokens(0) == "0#"
    assert answer_tokens(1234567890) == "1234567890#"
    assert decode(encode(answer_tokens(-120))) == "-120#"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gen_expr_op_count_depth_and_answer(seed):
    rng = random.Random(seed)
    for n, d in _combos():
        expr, ans = gen_expr(n, d, rng)
        assert sum(expr.count(c) for c in "+-*") == n, expr
        assert set(expr) <= set("0123456789+-*()"), expr
        tree = _parse(expr)
        assert _ast_depth(tree) == d, (expr, n, d)
        assert isinstance(ans, int)
        assert ans == eval(expr), expr


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
        assert ans == eval(expr)


def test_make_batch_shapes_and_contents():
    rng = random.Random(123)
    B = 32
    batch = make_batch(B, rng, (1, 16), (1, 16), max_len=128)
    assert set(batch) == {"x", "pos", "n_tgt", "answer", "n_ops", "depth"}
    x, pos, n_tgt = batch["x"], batch["pos"], batch["n_tgt"]
    answer, n_ops, depth = batch["answer"], batch["n_ops"], batch["depth"]
    assert all(v.dtype == torch.long for v in batch.values())
    assert x.shape[0] == B and x.shape[1] <= 128
    assert pos.shape == n_tgt.shape == answer.shape == n_ops.shape == depth.shape == (B,)
    T = x.shape[1]
    assert int((pos + 1 + n_tgt).max()) == T  # longest full sequence fills the batch width
    for b in range(B):
        p = int(pos[b])
        assert int(x[b, p]) == STOI["="]
        assert (x[b, : p + 1 + int(n_tgt[b])] != PAD_ID).all()
        assert (x[b, p + 1 + int(n_tgt[b]) :] == PAD_ID).all()
        expr = decode(x[b, :p].tolist())
        assert sum(expr.count(c) for c in "+-*") == int(n_ops[b])
        assert _ast_depth(_parse(expr)) == int(depth[b])
        assert eval(expr) == int(answer[b])
        assert decode(x[b, p + 1 :].tolist()) == answer_tokens(int(answer[b]))
        assert int(n_tgt[b]) == len(answer_tokens(int(answer[b])))


def test_make_batch_deterministic_given_seed():
    a = make_batch(16, random.Random(7), (1, 8), (1, 4))
    b = make_batch(16, random.Random(7), (1, 8), (1, 4))
    for k in a:
        assert torch.equal(a[k], b[k])


def test_longest_example_fits_in_128_tokens():
    rng = random.Random(0)
    longest = 0
    for _ in range(50):
        expr, ans = gen_expr(16, 16, rng)
        longest = max(longest, len(encode(expr + "=" + answer_tokens(ans))))
    assert longest <= 128
    # worst case by construction: a pure chain with parens everywhere still fits
    expr, _ = gen_expr(16, 16, random.Random(1))
    assert len(expr) + 1 <= 128


def test_leaves_are_1_to_9_and_never_zero():
    rng = random.Random(7)
    seen = set()
    for _ in range(500):
        expr, ans, _, _ = sample_example(rng, (1, 16), (1, 16))
        assert "0" not in expr
        seen |= set(expr) & set("123456789")
        assert ans == eval(expr)
    assert seen == set("123456789")  # every nonzero digit is produced as a leaf


def test_answers_are_unbounded_and_can_be_negative():
    rng = random.Random(8)
    answers = [sample_example(rng, (1, 16), (1, 16))[1] for _ in range(2000)]
    assert min(answers) < 0 < max(answers)
    assert max(abs(a) for a in answers) > 10**4  # not confined to a small range


@pytest.mark.parametrize("seed", [0, 1])
def test_bound_keeps_every_intermediate_within_1000(monkeypatch, seed):
    monkeypatch.setattr(data, "BOUND", 1000)
    rng = random.Random(seed)
    for n, d in _combos():
        expr, ans = gen_expr(n, d, rng)
        assert ans == eval(expr)
        assert _ast_depth(_parse(expr)) == d and sum(expr.count(c) for c in "+-*") == n
        assert tree_max_abs(_to_tree(_parse(expr))) <= 1000, expr
        # independent check: evaluate every BinOp subtree on its own
        for node in ast.walk(_parse(expr)):
            if isinstance(node, ast.BinOp):
                assert abs(eval(ast.unparse(node))) <= 1000, (expr, ast.unparse(node))


def test_bound_none_leaves_values_unbounded():
    assert data.BOUND is None
    rng = random.Random(3)
    assert any(abs(sample_example(rng, (10, 16), (4, 16))[1]) > 1000 for _ in range(200))
