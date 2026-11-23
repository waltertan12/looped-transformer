"""H18 tag fixes: random offset on question tags (train only) and separate question/answer tag tables."""
import torch

from looptf.toy.data import VOCAB, encode
from looptf.toy.model import ToyConfig, ToyLM, answer_mask, place_ids


def ids(s):
    return torch.tensor([encode(s)])


def test_place_ids_offset_shifts_question_digits_only():
    x = ids("347*28=9716#")
    base = place_ids(x, 24)
    shifted = place_ids(x, 24, offset=torch.tensor([[5]]))
    after = answer_mask(x)
    digit = base > 0
    assert (shifted[digit & ~after] == base[digit & ~after] + 5).all()
    assert (shifted[after] == base[after]).all()
    assert (shifted[~digit] == 0).all()


def test_answer_mask():
    assert answer_mask(ids("12+3=15#")).tolist() == [[False] * 5 + [True] * 3]


def _cfg(**kw):
    return ToyConfig(vocab_size=len(VOCAB), d_model=32, n_heads=2, d_ff=64, place_emb=True, **kw)


def test_offset_only_in_training():
    torch.manual_seed(0)
    m = ToyLM(_cfg(tag_offset=8))
    x = ids("347*28=9716#")
    m.eval()
    a, b = m(x), m(x)
    assert torch.equal(a, b)  # no randomness at eval
    ref = ToyLM(_cfg())
    ref.load_state_dict(m.state_dict())
    ref.eval()
    assert torch.allclose(a, ref(x))  # eval with offset model == plain tags
    m.train()
    outs = {tuple(m(x)[0, 0, :3].tolist()) for _ in range(20)}
    assert len(outs) > 1  # training draws offsets


def test_sep_tags_uses_answer_table_for_answer_digits():
    torch.manual_seed(0)
    m = ToyLM(_cfg(sep_tags=True))
    assert m.place_ans is not None
    x = ids("12+3=15#")
    m.eval()
    out1 = m(x)
    with torch.no_grad():
        m.place_ans.weight[1:].add_(1.0)  # row 0 = "no tag", used by non-answer tokens
    out2 = m(x)
    q = answer_mask(x)[0]
    # changing the answer table moves outputs at answer positions only (causal: positions before stay)
    first_ans = int(q.nonzero()[0])
    assert torch.allclose(out1[0, :first_ans], out2[0, :first_ans])
    assert not torch.allclose(out1[0, first_ans:], out2[0, first_ans:])


def test_old_checkpoint_config_loads():
    cfg = ToyConfig(**{k: v for k, v in _cfg().__dict__.items() if k not in ("tag_offset", "sep_tags")})
    assert cfg.tag_offset == 0 and cfg.sep_tags is False
