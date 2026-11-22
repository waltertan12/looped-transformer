"""Intermediate-values chain task: follow variable references through a shuffled program.

A prompt is a list of assignments ``name=expr`` separated by ';', followed by ``;<query>?``.
The answer is the value of the query variable mod MOD (7), with PEMDAS precedence.

Only the statements on the reference chain from the query matter; distractor statements
are never referenced by the chain or the query. All statements are shuffled, so a
referenced variable may be defined before or after its use.
"""

from __future__ import annotations

import random
import string

from looptf.toy.data import LEAF_HI, LEAF_LO, MOD

MAX_CHAIN_LEN = 16
MAX_DISTRACTORS = 6

# A tree is an int (digit leaf), a str (variable-name leaf) or a tuple (op, left, right).
_BINOPS = "+-*"


def _random_tree(leaves: list, parent_op: str | None, side: int, rng: random.Random):
    """Random binary tree over ``leaves`` (in order) respecting the rendering constraints.

    A right child of '+' is never +/- and a right child of '*' is never '*', so the
    rendered string re-parses to the same tree (same rule as data.py).
    """
    if len(leaves) == 1:
        return leaves[0]
    if side == 1 and parent_op == "+":
        op = "*"
    elif side == 1 and parent_op == "*":
        op = rng.choice("+-")
    else:
        op = rng.choice(_BINOPS)
    cut = rng.randint(1, len(leaves) - 1)
    left = _random_tree(leaves[:cut], op, 0, rng)
    right = _random_tree(leaves[cut:], op, 1, rng)
    return (op, left, right)


def _render(t) -> str:
    """Minimal-parenthesis rendering (PEMDAS, left-associative)."""
    if not isinstance(t, tuple):
        return str(t)
    op, left, right = t
    ls, rs = _render(left), _render(right)
    if op == "*":
        if isinstance(left, tuple) and left[0] in ("+", "-"):
            ls = f"({ls})"
        if isinstance(right, tuple) and right[0] in ("+", "-"):
            rs = f"({rs})"
    elif op == "-":
        if isinstance(right, tuple) and right[0] in ("+", "-"):
            rs = f"({rs})"
    return f"{ls}{op}{rs}"


def _eval(t, env: dict[str, int]) -> int:
    if isinstance(t, int):
        return t
    if isinstance(t, str):
        return env[t]
    op, left, right = t
    a, b = _eval(left, env), _eval(right, env)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def _n_ops(t) -> int:
    if not isinstance(t, tuple):
        return 0
    return 1 + _n_ops(t[1]) + _n_ops(t[2])


def _expr_leaves(n_leaves: int, ref: str | None, rng: random.Random) -> list:
    """Digit leaves, with one variable reference placed at a random position if given."""
    leaves: list = [rng.randint(LEAF_LO, LEAF_HI) for _ in range(n_leaves)]
    if ref is not None:
        leaves[rng.randrange(n_leaves)] = ref
    return leaves


def _chain_answer(chain_names: list[str], chain_trees: list, forced_first: int | None = None) -> int:
    """Value of the last chain variable; values are reduced mod MOD at every step.

    If ``forced_first`` is given, v1's value is overridden with it (used by the
    dependency check).
    """
    env: dict[str, int] = {}
    for k, (name, tree) in enumerate(zip(chain_names, chain_trees)):
        if k == 0 and forced_first is not None:
            env[name] = forced_first
        else:
            env[name] = _eval(tree, env) % MOD
    return env[chain_names[-1]]


def _try_chain(chain_len: int, n_distractors: int, rng: random.Random):
    """One sampling attempt. Returns (prompt, answer, meta), or None if the dependency
    check rejects the sample."""
    names = rng.sample(string.ascii_lowercase, chain_len + n_distractors)
    chain_names = names[:chain_len]
    dist_names = names[chain_len:]

    chain_trees = []
    for k in range(chain_len):
        ref = chain_names[k - 1] if k > 0 else None
        chain_trees.append(_random_tree(_expr_leaves(rng.choice((2, 3)), ref, rng), None, 0, rng))

    dist_trees = []
    for i in range(n_distractors):
        kinds = ["digits", "chain"] + (["distractor"] if i > 0 else [])
        kind = rng.choice(kinds)
        if kind == "digits":
            ref = None
        elif kind == "chain":
            ref = rng.choice(chain_names)
        else:
            ref = rng.choice(dist_names[:i])  # earlier distractors only: no cycles
        dist_trees.append(_random_tree(_expr_leaves(rng.choice((2, 3)), ref, rng), None, 0, rng))

    answer = _chain_answer(chain_names, chain_trees)
    # Dependency guarantee: the answer must change when v1's value changes.
    if len({_chain_answer(chain_names, chain_trees, forced_first=x) for x in range(MOD)}) == 1:
        return None

    stmts = [f"{n}={_render(t)}" for n, t in zip(chain_names, chain_trees)]
    stmts += [f"{n}={_render(t)}" for n, t in zip(dist_names, dist_trees)]
    rng.shuffle(stmts)
    prompt = ";".join(stmts) + f";{chain_names[-1]}?"
    meta = {
        "chain_len": chain_len,
        "n_distractors": n_distractors,
        "n_ops_total": sum(_n_ops(t) for t in chain_trees),
        "length": len(prompt),
    }
    return prompt, answer, meta


def gen_chain(chain_len: int, n_distractors: int, rng: random.Random) -> tuple[str, int, dict]:
    """Returns (prompt, answer, meta).

    prompt: statements joined by ';' then ';' + query_var + '?'. No spaces.
    answer: int in 0..MOD-1.
    meta: {"chain_len", "n_distractors", "n_ops_total" (operators in chain statements only),
           "length": len(prompt)}.
    Requires 1 <= chain_len <= 16 and 0 <= n_distractors <= 6, else ValueError.
    """
    if not isinstance(chain_len, int) or isinstance(chain_len, bool) or not 1 <= chain_len <= MAX_CHAIN_LEN:
        raise ValueError(f"chain_len must be an int in [1, {MAX_CHAIN_LEN}], got {chain_len!r}")
    if (not isinstance(n_distractors, int) or isinstance(n_distractors, bool)
            or not 0 <= n_distractors <= MAX_DISTRACTORS):
        raise ValueError(f"n_distractors must be an int in [0, {MAX_DISTRACTORS}], got {n_distractors!r}")
    while True:
        out = _try_chain(chain_len, n_distractors, rng)
        if out is not None:
            return out
