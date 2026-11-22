import random
import re

import pytest

from looptf.toy.chain import gen_chain
from looptf.toy.data import MOD, STOI, VOCAB, encode


def parse(prompt):
    """Independent parser: returns (defs: name -> expr string, query name)."""
    parts = prompt.split(";")
    assert parts[-1].endswith("?")
    query = parts[-1][:-1]
    defs = {}
    for stmt in parts[:-1]:
        name, expr = stmt.split("=", 1)
        assert name not in defs
        defs[name] = expr
    return defs, query


def evaluate(defs, query, forced=None):
    """Independent evaluator: substitutes variable values into the expression string and
    uses Python eval. ``forced=(name, value)`` overrides one variable's value."""
    cache = {}

    def val(v):
        if forced is not None and v == forced[0]:
            return forced[1]
        if v not in cache:
            sub = re.sub(r"[a-z]", lambda m: str(val(m.group(0))), defs[v])
            cache[v] = eval(sub) % MOD
        return cache[v]

    return val(query)


def chain_path(defs, query):
    """Follow references from the query until a digits-only statement. Returns the
    list of variable names, query first, v1 last."""
    path = [query]
    cur = query
    while True:
        refs = re.findall(r"[a-z]", defs[cur])
        if not refs:
            return path
        assert len(set(refs)) == 1, f"chain statement {cur} has refs {refs}"
        cur = refs[0]
        assert cur not in path, "cycle"
        path.append(cur)


def n_ops(expr):
    return sum(expr.count(c) for c in "+-*")


SWEEP = [(L, D) for L in range(1, 17) for D in range(0, 7)]


def test_answer_matches_independent_eval():
    rng = random.Random(0)
    for L, D in SWEEP:
        for _ in range(20):
            prompt, ans, meta = gen_chain(L, D, rng)
            defs, q = parse(prompt)
            assert isinstance(ans, int)
            assert 0 <= ans < MOD
            assert evaluate(defs, q) == ans, prompt


def test_answer_depends_on_v1():
    rng = random.Random(1)
    for L, D in SWEEP:
        for _ in range(20):
            prompt, _, _ = gen_chain(L, D, rng)
            defs, q = parse(prompt)
            v1 = chain_path(defs, q)[-1]
            answers = {evaluate(defs, q, forced=(v1, x)) for x in range(MOD)}
            assert len(answers) > 1, prompt


def test_chain_has_stated_length_and_distractors_unreachable():
    rng = random.Random(2)
    for L, D in SWEEP:
        for _ in range(20):
            prompt, _, meta = gen_chain(L, D, rng)
            defs, q = parse(prompt)
            path = chain_path(defs, q)
            assert len(path) == L == meta["chain_len"]
            assert len(defs) == L + D == meta["chain_len"] + meta["n_distractors"]
            assert meta["n_distractors"] == D
            distractors = set(defs) - set(path)
            assert len(distractors) == D
            assert not (distractors & set(path))
            # v1 uses digits only; every later chain statement references exactly its predecessor.
            v1 = path[-1]
            assert re.findall(r"[a-z]", defs[v1]) == []
            for cur, nxt in zip(path[:-1], path[1:]):
                assert re.findall(r"[a-z]", defs[cur]) == [nxt]
            # Each statement has 1 or 2 binary operators.
            for expr in defs.values():
                assert 1 <= n_ops(expr) <= 2, expr
            # n_ops_total counts chain statements only.
            assert meta["n_ops_total"] == sum(n_ops(defs[v]) for v in path)


def test_meta_length_and_format():
    rng = random.Random(3)
    for L, D in SWEEP:
        prompt, _, meta = gen_chain(L, D, rng)
        assert meta["length"] == len(prompt)
        assert prompt.endswith("?")
        assert prompt.count("?") == 1
        assert " " not in prompt
        assert all(ch in STOI for ch in prompt)
        assert len(encode(prompt)) == len(prompt)


def test_names_distinct_and_random_order():
    rng = random.Random(4)
    unsorted = 0
    for _ in range(300):
        prompt, _, _ = gen_chain(rng.randint(1, 16), rng.randint(0, 6), rng)
        names = [s.split("=", 1)[0] for s in prompt.split(";")[:-1]]
        assert len(set(names)) == len(names)
        assert all(len(n) == 1 and "a" <= n <= "z" for n in names)
        if names != sorted(names):
            unsorted += 1
    assert unsorted > 0


def test_length_bound_max_case():
    rng = random.Random(5)
    lengths = [gen_chain(16, 6, rng)[2]["length"] for _ in range(500)]
    assert max(lengths) <= 256


def test_statement_order_is_shuffled():
    rng = random.Random(6)
    not_last = 0
    n = 300
    for _ in range(n):
        prompt, _, _ = gen_chain(4, 3, rng)
        defs, q = parse(prompt)
        stmts = prompt.split(";")[:-1]
        if not stmts[-1].startswith(q + "="):
            not_last += 1
    assert 0 < not_last < n


def test_determinism_with_seed():
    a, b = random.Random(7), random.Random(7)
    out_a = [gen_chain(1 + i % 16, i % 7, a) for i in range(50)]
    out_b = [gen_chain(1 + i % 16, i % 7, b) for i in range(50)]
    assert out_a == out_b


@pytest.mark.parametrize("chain_len, n_distractors", [
    (0, 0), (17, 0), (-1, 0), (1, -1), (1, 7), (True, 0), (1, True),
])
def test_invalid_args_raise(chain_len, n_distractors):
    with pytest.raises(ValueError):
        gen_chain(chain_len, n_distractors, random.Random(0))
