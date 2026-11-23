"""H19 probes: sub-expression extraction and per-pass hidden states."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_subexpr_nodes_positions_heights_values():
    from probe_levels import subexpr_nodes
    p = "(8-3)*(4-6)+7="
    nodes = subexpr_nodes(p)
    got = {(p[pos], h, v) for pos, h, v in nodes}
    # (8-3) ends at ')' index 4, (4-6) at ')' index 10, (..)*(..) finishes at '+', root at '='
    assert (")", 1, 5) in got and (")", 1, -2) in got
    assert ("+", 2, -10) in got and ("=", 3, -3) in got
    assert len(nodes) == 4


def test_subexpr_nodes_left_chain():
    from probe_levels import subexpr_nodes
    p = "6+5+2="
    assert sorted(subexpr_nodes(p)) == [(3, 1, 11), (5, 2, 13)]


def test_pass_states_match_model_output():
    from looptf.toy.data import VOCAB, encode
    from looptf.toy.model import ToyConfig, ToyLM
    from probe_levels import pass_states
    torch.manual_seed(0)
    m = ToyLM(ToyConfig(vocab_size=len(VOCAB), d_model=32, n_heads=2, d_ff=64, n_core=2, place_emb=True)).eval()
    x = torch.tensor([encode("(8-3)*(4-6)+7=")])
    states, attn = pass_states(m, x, n_loops=3, per_layer=False)
    assert len(states) == 4  # before the core + after each of 3 passes
    h = states[-1]
    cos, sin = m.rope_cos[:x.shape[1]], m.rope_sin[:x.shape[1]]
    for blk in m.coda:
        h = blk(h, cos, sin)
    assert torch.allclose(m.lm_head(m.final_norm(h)), m(x, n_loops=3), atol=1e-5)
    assert len(attn) == 3 and attn[0].shape == (2, 1, 2, x.shape[1], x.shape[1])  # [core blk, B, heads, T, T]
    s2, _ = pass_states(m, x, n_loops=1, per_layer=True)
    assert len(s2) == 3  # before the core + after each of the 2 core blocks
