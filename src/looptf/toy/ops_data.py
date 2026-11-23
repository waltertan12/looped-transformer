"""Single-operator expressions (only '+', only '-' or only '*') with exact-size inputs, for test-only sets (H13).

Separate from data.py so the nested generators' RNG use (and so the frozen datasets) never change.
- add: the nested generator never puts '+' on the right of '+', so an add-only tree is a left-deep
  chain: depth == n_ops (== terms - 1), rendered flat ("12+34+56"), and it parses that way too.
- mul: like add (the renderer drops brackets in a*(b*c)), so a flat left-deep chain.
- sub: any tree shape; data.render parenthesizes a '-' right child, so the printed depth is the tree depth.
"""

from __future__ import annotations

import random

from looptf.toy.data import render, tree_value

_KIND_OP = {"add": "+", "sub": "-", "mul": "*"}


def exact_leaf(rng: random.Random, k: int) -> int:
    """A number with exactly k digits (k=1: 1-9, never 0)."""
    if k == 1:
        return rng.randint(1, 9)
    return rng.randint(10 ** (k - 1), 10**k - 1)


def feasible_n_ops(kind: str, depth: int) -> tuple[int, int]:
    if kind in ("add", "mul"):  # flat chains: depth == n_ops
        return depth, depth
    if kind == "sub":
        return depth, 2**depth - 1
    raise ValueError(f"unknown kind {kind!r}")


def _shape(n_ops: int, depth: int, rng: random.Random, op: str):
    """Mutable tree [op, left, right] with exactly n_ops operators and max depth `depth`."""
    root = [op, None, None]
    slots: list[tuple[list, int, int]] = []
    cur, level = root, 1
    while level < depth:
        side = rng.randint(0, 1)
        child = [op, None, None]
        cur[1 + side] = child
        slots.append((cur, 1 - side, level + 1))
        cur, level = child, level + 1
    for _ in range(n_ops - depth):
        i = rng.randrange(len(slots))
        parent, side, child_level = slots[i]
        slots[i] = slots[-1]
        slots.pop()
        node = [op, None, None]
        parent[1 + side] = node
        if child_level + 1 <= depth:
            slots.append((node, 0, child_level + 1))
            slots.append((node, 1, child_level + 1))
    return root


def _left_chain(n_ops: int, op: str):
    node = [op, None, None]
    for _ in range(n_ops - 1):
        node = [op, node, None]
    return node


def _freeze(node, rng: random.Random, k: int):
    if node is None:
        return exact_leaf(rng, k)
    return (node[0], _freeze(node[1], rng, k), _freeze(node[2], rng, k))


def gen_single_op(kind: str, n_ops: int, depth: int, rng: random.Random, in_digits: int) -> tuple[str, int]:
    """Expression using only one operator kind, every number exactly `in_digits` digits.
    Returns (expr_string, exact integer value)."""
    lo, hi = feasible_n_ops(kind, depth)
    if not (1 <= depth and lo <= n_ops <= hi):
        raise ValueError(f"infeasible {kind}: n_ops={n_ops}, depth={depth} (need {lo}..{hi})")
    if in_digits < 1:
        raise ValueError("in_digits must be >= 1")
    op = _KIND_OP[kind]
    shape = _left_chain(n_ops, op) if kind in ("add", "mul") else _shape(n_ops, depth, rng, op)
    tree = _freeze(shape, rng, in_digits)
    return render(tree), tree_value(tree)
