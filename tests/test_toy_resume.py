"""Pause at step N then resume must give the same model as an unbroken run (H16 40k look)."""
import shutil

import pytest
import torch

from looptf.toy import train

ROOT = train.ROOT
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BASE = ["--d_model", "32", "--n_heads", "2", "--d_ff", "64", "--n_core", "1", "--train_loops", "2",
        "--train_depth", "1-2", "--n_ops", "1-3", "--steps", "400", "--eval_every", "1000",
        "--eval_depths", "1-2", "--eval_loops", "2", "--eval_n", "20", "--bs", "16", "--warmup", "50"]


def _cleanup(*names):
    for n in names:
        shutil.rmtree(ROOT / "results/toy" / n, ignore_errors=True)
        for f in (ROOT / "checkpoints/toy").glob(n + "*"):
            f.unlink()


def test_pause_resume_matches_unbroken_run():
    a, b = "zz_test_full", "zz_test_pause"
    _cleanup(a, b)
    try:
        train.main(["--name", a, *BASE])
        train.main(["--name", b, *BASE, "--pause_at", "200"])
        assert (ROOT / "checkpoints/toy" / f"{b}@0.2k.pt").exists()
        assert (ROOT / "results/toy" / b / "per_example@0.2k.npz").exists()
        assert not (ROOT / "checkpoints/toy" / f"{b}.pt").exists()
        train.main(["--name", b, *BASE, "--resume"])
        sa = torch.load(ROOT / "checkpoints/toy" / f"{a}.pt", weights_only=False)["state_dict"]
        sb = torch.load(ROOT / "checkpoints/toy" / f"{b}.pt", weights_only=False)["state_dict"]
        for k in sa:
            assert torch.allclose(sa[k], sb[k], atol=1e-5), k
    finally:
        _cleanup(a, b)
