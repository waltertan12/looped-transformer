import random

import numpy as np

from looptf.toy.data import VOCAB, MOD, ITOS, decode, gen_expr
from looptf.toy.dataset import FrozenData, encode_rows, write_meta, write_slice

COLS = ["answer", "n_ops", "depth", "length"]


def _build(root, levels=(1, 2, 3), n=50):
    rng = random.Random(0)
    slices = {}
    for split in ("train", "test"):
        for d in levels:
            prompts, labels = [], []
            for _ in range(n):
                k = rng.randint(d, min(16, 2**d - 1))
                expr, ans = gen_expr(k, d, rng)
                prompts.append(expr + "=")
                labels.append([ans, k, d, len(expr) + 1])
            write_slice(root, split, d, encode_rows(prompts, 64), np.array(labels, dtype=np.int16))
            slices.setdefault(str(d), {})[split] = n
    write_meta(root, {"name": "nested", "vocab": VOCAB, "width": 64, "columns": COLS,
                      "level_column": "depth", "mod": MOD, "seed": 0, "slices": slices})


def test_roundtrip_batch(tmp_path):
    _build(tmp_path)
    fd = FrozenData(tmp_path, device="cpu")
    b = fd.batch(np.random.default_rng(0), 64, (2, 3))
    assert set(b["depth"].tolist()) <= {2, 3}
    for i in range(64):
        p = b["pos"][i].item()
        assert ITOS[b["x"][i, p].item()] == "="
        expr = decode(b["x"][i, :p].tolist())
        assert eval(expr) % MOD == int(ITOS[b["y"][i].item()])
        assert (b["x"][i, p + 1:] == 0).all()


def test_test_sets(tmp_path):
    _build(tmp_path)
    fd = FrozenData(tmp_path, device="cpu")
    ts = fd.test_sets([1, 3], max_per_level=10)
    assert sorted(ts) == [1, 3] and ts[3]["x"].shape[0] == 10
    assert (ts[3]["depth"] == 3).all()
