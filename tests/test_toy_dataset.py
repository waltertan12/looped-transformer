import random

import numpy as np

from looptf.toy.data import VOCAB, ITOS, decode, gen_expr, answer_tokens
from looptf.toy.dataset import FrozenData, encode_rows, full_sequence, write_meta, write_slice

COLS = ["answer", "n_ops", "depth", "length", "n_tgt"]


def _build(root, levels=(1, 2, 3), n=50):
    rng = random.Random(0)
    slices = {}
    for split in ("train", "test"):
        for d in levels:
            seqs, labels = [], []
            for _ in range(n):
                k = rng.randint(d, min(16, 2**d - 1))
                expr, ans = gen_expr(k, d, rng)
                prompt = expr + "="
                seqs.append(full_sequence(prompt, ans))
                labels.append([ans, k, d, len(prompt), len(answer_tokens(ans))])
            write_slice(root, split, d, encode_rows(seqs, 64), np.array(labels, dtype=np.int64))
            slices.setdefault(str(d), {})[split] = n
    write_meta(root, {"name": "nested", "vocab": VOCAB, "width": 64, "columns": COLS,
                      "level_column": "depth", "seed": 0, "slices": slices})


def test_full_sequence_and_encoding_roundtrip():
    assert full_sequence("1+2=", -3) == "1+2=-3#"
    tok = encode_rows([full_sequence("1+2=", -3)], 16)
    assert tok.dtype == np.uint8 and tok.shape == (1, 16)
    assert decode(tok[0].tolist()) == "1+2=-3#"
    assert (tok[0, 7:] == 0).all()


def test_roundtrip_batch(tmp_path):
    _build(tmp_path)
    fd = FrozenData(tmp_path, device="cpu")
    b = fd.batch(np.random.default_rng(0), 64, (2, 3))
    assert set(b["depth"].tolist()) <= {2, 3}
    assert set(["answer", "n_ops", "depth", "length", "n_tgt", "x", "pos"]) <= set(b)
    for i in range(64):
        p = b["pos"][i].item()
        n_tgt = b["n_tgt"][i].item()
        assert ITOS[b["x"][i, p].item()] == "="
        prompt = decode(b["x"][i, : p + 1].tolist())
        assert eval(prompt[:-1]) == b["answer"][i].item()
        assert decode(b["x"][i, p + 1 : p + 1 + n_tgt].tolist()) == answer_tokens(b["answer"][i].item())
        assert n_tgt == len(answer_tokens(b["answer"][i].item()))
        assert (b["x"][i, p + 1 + n_tgt :] == 0).all()


def test_test_sets(tmp_path):
    _build(tmp_path)
    fd = FrozenData(tmp_path, device="cpu")
    ts = fd.test_sets([1, 3], max_per_level=10)
    assert sorted(ts) == [1, 3] and ts[3]["x"].shape[0] == 10
    assert (ts[3]["depth"] == 3).all()


def test_reverse_answer(tmp_path):
    _build(tmp_path)
    fwd = FrozenData(tmp_path, device="cpu")
    rev = FrozenData(tmp_path, device="cpu", reverse_answer=True)
    ts_f, ts_r = fwd.test_sets([3]), rev.test_sets([3])
    for i in range(len(ts_f[3]["x"])):
        p, n = ts_f[3]["pos"][i].item(), ts_f[3]["n_tgt"][i].item()
        a = decode(ts_f[3]["x"][i, p + 1: p + 1 + n].tolist())
        b = decode(ts_r[3]["x"][i, p + 1: p + 1 + n].tolist())
        assert a.endswith("#") and b.endswith("#")
        sign = "-" if a.startswith("-") else ""
        assert b == sign + a[len(sign):-1][::-1] + "#"
        assert decode(ts_f[3]["x"][i, : p + 1].tolist()) == decode(ts_r[3]["x"][i, : p + 1].tolist())
