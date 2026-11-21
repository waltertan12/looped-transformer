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
