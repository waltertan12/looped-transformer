import torch

from looptf.model import LoopSpec, apply_loop, load_model

NAME = "Qwen/Qwen3-0.6B-Base"


def test_layout():
    assert LoopSpec(2, 4, 3, 6).layout(8) == [0, 1, 2, 3, 2, 3, 2, 3, 6, 7]
    assert LoopSpec(0, 8, 1).layout(8) == list(range(8))


@torch.no_grad()
def test_identity_loop_matches_base():
    model, tok = load_model(NAME, dtype=torch.float32)
    x = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.cuda()
    ref = model(x).logits
    n = model.config.num_hidden_layers
    apply_loop(model, LoopSpec(0, n, 1))
    torch.testing.assert_close(model(x).logits, ref)


@torch.no_grad()
def test_looped_generate_with_cache():
    model, tok = load_model(NAME)
    apply_loop(model, LoopSpec(4, 16, 2, 28))
    assert model.config.num_hidden_layers == 28
    x = tok("def fib(n):", return_tensors="pt").to("cuda")
    out = model.generate(**x, max_new_tokens=8, do_sample=False)
    # cached decoding must match a full recompute of the final token
    full = model(out[:, :-1]).logits[:, -1].argmax(-1)
    assert full.item() == out[0, -1].item()


@torch.no_grad()
def test_gated_identity_at_init():
    """Gated extra passes start at g=0, so a looped model equals the original exactly."""
    model, tok = load_model(NAME, dtype=torch.float32)
    x = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.cuda()
    ref = model(x).logits
    apply_loop(model, LoopSpec(8, 20, 3, gated=True))
    assert model.config.num_hidden_layers == 28 + 2 * 12
    torch.testing.assert_close(model(x).logits, ref)


@torch.no_grad()
def test_gated_nonzero_changes_output_and_set_n_loops():
    from looptf.model import set_n_loops

    model, tok = load_model(NAME, dtype=torch.float32)
    x = tok("Hello world", return_tensors="pt").input_ids.cuda()
    apply_loop(model, LoopSpec(8, 20, 2, gated=True))
    ref = model(x).logits
    model.model.loop_gates.weight.fill_(0.5)
    assert not torch.allclose(model(x).logits, ref)
    set_n_loops(model, 4)
    assert model.config.num_hidden_layers == 28 + 3 * 12
    assert model.model.loop_gates.weight.shape[0] == 3
    set_n_loops(model, 1)
    model.model.loop_gates.weight.fill_(0.5)  # K=1 has no gated passes
    assert model.config.num_hidden_layers == 28


@torch.no_grad()
def test_kv_cache_parity_multitoken():
    """Cached generation must equal no-cache generation over a multi-token rollout."""
    model, tok = load_model(NAME, dtype=torch.float32)
    apply_loop(model, LoopSpec(8, 20, 3, gated=True))
    model.model.loop_gates.weight.fill_(0.3)
    x = tok("def fibonacci(n):", return_tensors="pt").to("cuda")
    a = model.generate(**x, max_new_tokens=24, do_sample=False, use_cache=True)
    b = model.generate(**x, max_new_tokens=24, do_sample=False, use_cache=False)
    assert torch.equal(a, b)
