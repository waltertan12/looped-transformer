"""Build and verify a small nested_bal dataset end to end (scripts/build_data.py + verify_data.py)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bal_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("nbal") / "data"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/build_data.py"), "--out", str(out), "--which", "nested_bal",
         "--scale", "0.001", "--workers", "8"],
        check=True, cwd=REPO, capture_output=True, text=True,
    )
    return out / "nested_bal"


def test_build_then_verify_passes(bal_root):
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts/verify_data.py"), "--root", str(bal_root), "--sample", "0"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ALL CHECKS PASSED" in res.stdout


def test_meta_shape_and_zero_overlap(bal_root):
    meta = json.loads((bal_root / "meta.json").read_text())
    assert meta["name"] == "nested_bal" and meta["width"] == 80
    assert meta["columns"] == ["answer", "n_ops", "depth", "length", "n_tgt", "op_digits", "ans_digits"]
    assert meta["op_digits_allowed"]["1"] == [1, 2, 3]
    assert meta["op_digits_allowed"]["3"] == [1, 2]
    assert meta["op_digits_allowed"]["4"] == [1]
    assert meta["op_digits_allowed"]["16"] == [1]
    assert sorted(int(k) for k in meta["slices"]) == list(range(1, 17))
    for lv, sl in meta["slices"].items():
        assert sl["test_overlap_with_train"] == 0.0, lv
        assert sum(sl["test_cells"].values()) == sl["test"], lv
        assert sl["test"] > 0, lv


def test_train_answer_digits_balanced_over_feasible_buckets(bal_root):
    meta = json.loads((bal_root / "meta.json").read_text())
    for lv, sl in meta["slices"].items():
        cells = sl["train_cells"]
        for op, buckets in sl["feasible_cells"].items():
            counts = [cells[f"{op}_ans{b}"] for b in buckets]
            assert all(c > 0 for c in counts), (lv, op, counts)
            mean = sum(counts) / len(counts)
            for c in counts:
                assert abs(c - mean) <= 0.2 * mean, (lv, op, counts)


def test_test_cells_only_feasible(bal_root):
    meta = json.loads((bal_root / "meta.json").read_text())
    for lv, sl in meta["slices"].items():
        feas = {f"op{op[2:]}_ans{b}" for op, bs in sl["feasible_cells"].items() for b in bs}
        assert set(sl["test_cells"]) == feas, lv
