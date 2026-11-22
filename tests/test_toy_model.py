import pytest
import torch

from looptf.toy.model import ToyConfig, ToyLM

SMALL = dict(vocab_size=17, d_model=64, n_heads=4, d_ff=128, max_len=32)


def make(**kw) -> ToyLM:
    torch.manual_seed(0)
    return ToyLM(ToyConfig(**{**SMALL, **kw}))


def tokens(B=2, T=12, vocab=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


@pytest.mark.parametrize("n_loops", [1, 2, 5])
def test_output_shape(n_loops):
    m = make(n_core=2)
    x = tokens()
    out = m(x, n_loops=n_loops)
    assert out.shape == (2, 12, 17)


def test_eval_loops_differ_from_train_loops():
    m = make(n_core=2).eval()
    x = tokens()
    with torch.no_grad():
        for n in range(1, 17):
            assert m(x, n_loops=n).shape == (2, 12, 17)


def test_num_params_independent_of_loops():
    m = make(n_core=2)
    n = m.num_params()
    assert n == sum(p.numel() for p in m.parameters())
    # num_params must not depend on how many loops are run
    with torch.no_grad():
        m(tokens(), n_loops=1)
        m(tokens(), n_loops=8)
    assert m.num_params() == n


def test_looped_has_far_fewer_params_than_deep():
    looped = ToyLM(ToyConfig(n_core=2))  # default d_model=256
    deep = ToyLM(ToyConfig(n_core=16))
    assert looped.num_params() < deep.num_params() / 4


def test_causality():
    m = make(n_core=2, n_prelude=1, n_coda=1).eval()
    x = tokens(B=1, T=10)
    t = 6
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 5) % 17
    with torch.no_grad():
        for n in (1, 3):
            a = m(x, n_loops=n)
            b = m(x2, n_loops=n)
            assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)
            assert not torch.allclose(a[:, t:], b[:, t:])


@pytest.mark.parametrize("inject", ["concat", "add", "none"])
def test_inject_variants_run(inject):
    m = make(n_core=2, inject=inject)
    out = m(tokens(), n_loops=3)
    assert out.shape == (2, 12, 17)
    assert torch.isfinite(out).all()
    assert (m.adapter is not None) == (inject == "concat")


def test_inject_none_one_loop_is_standard_stack():
    m = make(n_prelude=1, n_core=2, n_coda=1, inject="none").eval()
    x = tokens()
    with torch.no_grad():
        h = m.embed(x)
        for blk in [*m.prelude, *m.core, *m.coda]:
            h = blk(h, m.rope_cos[: x.shape[1]], m.rope_sin[: x.shape[1]])
        expected = m.lm_head(m.final_norm(h))
        got = m(x, n_loops=1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_gradients_reach_adapter_and_core():
    m = make(n_core=2, inject="concat")
    out = m(tokens(), n_loops=3)
    loss = out.float().logsumexp(-1).mean()
    loss.backward()
    assert m.adapter is not None
    for p in [m.adapter.weight, *m.core[0].parameters(), *m.core[1].parameters()]:
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_autocast_cuda():
    m = make(n_core=2, inject="concat").cuda()
    assert all(p.dtype == torch.float32 for p in m.parameters())
    x = tokens().cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x, n_loops=3)
        loss = out.float().logsumexp(-1).mean()
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    loss.backward()
    assert m.core[0].attn.qkv.weight.grad is not None
    assert m.core[0].attn.qkv.weight.grad.abs().sum().item() > 0
