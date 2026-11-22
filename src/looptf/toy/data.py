"""Synthetic arithmetic expressions with PEMDAS precedence and their real integer value.

Leaves are digits 1-9 (no 0). Values are unbounded by default (answers up to ~10 chars);
if BOUND is set, samples whose working leaves [-BOUND, BOUND] are regenerated. The answer is written digit by digit
after '=' and ends with the stop token EOS, e.g. ``(8-3)*(4-6)+7=-3#``.

Difficulty knobs: ``n_ops`` (total binary operators) and ``depth`` (operators on
the longest root-to-leaf path of the expression tree), set independently.

Expression trees are represented as nested tuples ``(op, left, right)`` with
leaves being ints LEAF_LO..LEAF_HI (1-6).

Rendering note: the minimal-parenthesis rules alone would let some trees
re-parse to a different shape (e.g. ``a+(b+c)`` renders as ``a+b+c``, which
parses left-deep, changing the AST depth). To keep the rendered string's depth
equal to the tree depth, the generator never places a ``+``/``-`` child on the
right of ``+`` or a ``*`` child on the right of ``*``. Every tree shape is still
reachable, so the depth and operator-count ranges are unchanged.
"""

from __future__ import annotations

import random

import torch

# Shared by every arithmetic dataset. Append only: existing ids must never change.
# 0-16: nested expressions. 17-44: chain task (variable names, ';', '?'). 45: '#' = EOS.
VOCAB: list[str] = (["_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "(", ")", "="]
                    + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [";", "?", "#"])
PAD_ID: int = 0
STOI: dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
ITOS: dict[int, str] = {i: t for i, t in enumerate(VOCAB)}

_OPS = ("+", "-", "*")
LEAF_LO, LEAF_HI = 1, 9
BOUND: int | None = None  # None = no bound (real values). Else every intermediate stays in [-BOUND, BOUND]
EOS: str = "#"


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids if int(i) != PAD_ID)


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def render(t) -> str:
    """Render a tree with minimal parentheses (PEMDAS, left-associative)."""
    if isinstance(t, int):
        return str(t)
    op, left, right = t
    ls, rs = render(left), render(right)
    if op == "*":
        if isinstance(left, tuple) and left[0] in ("+", "-"):
            ls = f"({ls})"
        if isinstance(right, tuple) and right[0] in ("+", "-"):
            rs = f"({rs})"
    elif op == "-":
        if isinstance(right, tuple) and right[0] in ("+", "-"):
            rs = f"({rs})"
    return f"{ls}{op}{rs}"


def tree_value(t) -> int:
    if isinstance(t, int):
        return t
    op, left, right = t
    a, b = tree_value(left), tree_value(right)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def tree_max_abs(t) -> int:
    """Largest |value| over every node (leaves and all intermediate results)."""
    if isinstance(t, int):
        return abs(t)
    return max(abs(tree_value(t)), tree_max_abs(t[1]), tree_max_abs(t[2]))


def answer_tokens(answer: int) -> str:
    """How an answer is written after '=' / '?': its digits (with '-') then EOS."""
    return f"{answer}{EOS}"


def tree_depth(t) -> int:
    """Number of operator nodes on the longest root-to-leaf path."""
    if isinstance(t, int):
        return 0
    return 1 + max(tree_depth(t[1]), tree_depth(t[2]))


def _new_op(parent_op: str | None, side: int, rng: random.Random) -> str:
    # See module docstring: keep rendered strings parse-consistent.
    if parent_op == "+" and side == 1:
        return "*"
    if parent_op == "*" and side == 1:
        return rng.choice("+-")
    return rng.choice(_OPS)


def _build(n_ops: int, depth: int, rng: random.Random):
    """Mutable tree [op, left, right]; None marks a leaf slot until filled."""
    root = [_new_op(None, 0, rng), None, None]
    # Spine of `depth` operator nodes guarantees max depth == depth.
    # Each slot is (parent_node, side, child_level); child_level is the
    # operator-level a node placed in that slot would occupy.
    slots: list[tuple[list, int, int]] = []
    cur, level = root, 1
    while level < depth:
        side = rng.randint(0, 1)
        child = [_new_op(cur[0], side, rng), None, None]
        cur[1 + side] = child
        slots.append((cur, 1 - side, level + 1))
        cur, level = child, level + 1

    for _ in range(n_ops - depth):
        i = rng.randrange(len(slots))
        parent, side, child_level = slots[i]
        slots[i] = slots[-1]
        slots.pop()
        node = [_new_op(parent[0], side, rng), None, None]
        parent[1 + side] = node
        if child_level + 1 <= depth:
            slots.append((node, 0, child_level + 1))
            slots.append((node, 1, child_level + 1))
    return root


def _leaf(rng: random.Random, op_digits: int) -> int:
    """One leaf. op_digits=1 is the classic 1-9 draw (identical RNG use to before). For op_digits>1
    the leaf gets a digit count k uniform in 1..op_digits; k=1 gives 1-9, k>=2 gives [10**(k-1), 10**k-1]."""
    if op_digits == 1:
        return rng.randint(LEAF_LO, LEAF_HI)
    k = rng.randint(1, op_digits)
    if k == 1:
        return rng.randint(LEAF_LO, LEAF_HI)
    return rng.randint(10 ** (k - 1), 10**k - 1)


def _freeze(node, rng: random.Random, op_digits: int = 1):
    if node is None:
        return _leaf(rng, op_digits)
    return (node[0], _freeze(node[1], rng, op_digits), _freeze(node[2], rng, op_digits))


def num_digits(n: int) -> int:
    """Number of decimal digits of abs(n); 0 has one digit."""
    return len(str(abs(n)))


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def gen_expr(n_ops: int, depth: int, rng: random.Random, op_digits: int = 1) -> tuple[str, int]:
    """Random expression with EXACTLY n_ops binary operators (each from + - *)
    and EXACTLY the given depth. Leaves are digits LEAF_LO..LEAF_HI (intermediates bounded only if BOUND is set).

    op_digits=k>1: each leaf independently has 1..k digits (uniform digit count, then uniform value
    with that many digits; never 0). op_digits=1 (default) is the classic 1-9 leaves.

    Requires 1 <= depth <= n_ops <= 2**depth - 1, else ValueError.
    Returns (expr_string, answer) with answer = the exact integer value.
    """
    if not (1 <= depth <= n_ops <= 2**depth - 1):
        raise ValueError(
            f"infeasible: need 1 <= depth <= n_ops <= 2**depth-1, got n_ops={n_ops}, depth={depth}"
        )
    if op_digits < 1:
        raise ValueError(f"op_digits must be >= 1, got {op_digits}")
    while True:
        shape = _build(n_ops, depth, rng)
        for _ in range(50):  # resample digits; rebuild the shape if they keep overflowing
            tree = _freeze(shape, rng, op_digits)
            if BOUND is None or tree_max_abs(tree) <= BOUND:
                return render(tree), tree_value(tree)


def sample_example(rng, n_ops_range: tuple[int, int], depth_range: tuple[int, int],
                   op_digits: int = 1) -> tuple[str, int, int, int]:
    """Pick depth uniformly in depth_range (inclusive), then n_ops uniformly in
    [max(depth, n_ops_lo), min(n_ops_hi, 2**depth-1)]; retry if that is empty.
    Returns (expr, answer, n_ops, depth)."""
    n_lo, n_hi = n_ops_range
    d_lo, d_hi = depth_range
    feasible = []
    for d in range(d_lo, d_hi + 1):
        lo, hi = max(d, n_lo), min(n_hi, 2**d - 1)
        if d >= 1 and lo <= hi:
            feasible.append((d, lo, hi))
    if not feasible:
        raise ValueError(f"no feasible (n_ops, depth) in ranges {n_ops_range}, {depth_range}")
    while True:
        d = rng.randint(d_lo, d_hi)
        lo, hi = max(d, n_lo), min(n_hi, 2**d - 1)
        if d < 1 or lo > hi:
            continue  # retry another depth
        n = rng.randint(lo, hi)
        expr, ans = gen_expr(n, d, rng, op_digits)
        return expr, ans, n, d


def make_batch(batch_size: int, rng, n_ops_range, depth_range, max_len: int = 128, device="cpu",
               op_digits: int = 1):
    """Batch of ``expr=<answer>#`` sequences, right-padded with PAD_ID.

    Returns dict with:
      x: LongTensor [B, T] full sequence (prompt, answer digits, EOS), T = longest in batch
      pos: LongTensor [B] index of the '=' token (prediction of the answer starts there)
      n_tgt: LongTensor [B] number of answer tokens including EOS
      answer, n_ops, depth: LongTensor [B]
    """
    items = [sample_example(rng, n_ops_range, depth_range, op_digits) for _ in range(batch_size)]
    seqs = [encode(expr + "=" + answer_tokens(ans)) for expr, ans, _, _ in items]
    T = max(len(s) for s in seqs)
    assert T <= max_len, f"batch needs {T} tokens > max_len={max_len}"
    x = torch.full((batch_size, T), PAD_ID, dtype=torch.long)
    for b, s in enumerate(seqs):
        x[b, : len(s)] = torch.tensor(s, dtype=torch.long)
    t = lambda vals: torch.tensor(vals, dtype=torch.long, device=device)
    return {
        "x": x.to(device),
        "pos": t([len(expr) for expr, _, _, _ in items]),
        "n_tgt": t([len(answer_tokens(ans)) for _, ans, _, _ in items]),
        "answer": t([ans for _, ans, _, _ in items]),
        "n_ops": t([n for _, _, n, _ in items]),
        "depth": t([d for _, _, _, d in items]),
    }
