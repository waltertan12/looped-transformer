import torch

from looptf.toy.data import STOI, VOCAB, encode
from looptf.toy.model import ToyConfig, ToyLM, place_ids, _DIGIT_LO, _DIGIT_HI, _EQ_ID, _Q_ID


def ids(s):
    return torch.tensor([encode(s)])


def test_token_ids_match_vocab():
    assert (_DIGIT_LO, _DIGIT_HI) == (STOI["0"], STOI["9"])
    assert _EQ_ID == STOI["="] and _Q_ID == STOI["?"]


def test_place_ids_nested_normal_and_reversed():
    # question digits: place from the end (+1); answer digits: index from the start (+1)
    assert place_ids(ids("347*28=9716#"), 24).tolist() == [[3, 2, 1, 0, 2, 1, 0, 1, 2, 3, 4, 0]]
    # reversed answer 6179: first answer digit is the ones place -> lines up with the question
    assert place_ids(ids("347*28=6179#"), 24).tolist() == [[3, 2, 1, 0, 2, 1, 0, 1, 2, 3, 4, 0]]
    assert place_ids(ids("(12-5)*3=-21#"), 24).tolist() == [[0, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 2, 0]]


def test_place_ids_chain_uses_question_mark():
    s = "a=12+3;b=a*45;b?540#"
    got = place_ids(ids(s), 24)[0].tolist()
    # '=' inside statements must not start the answer region
    assert got[s.index("12")] == 2 and got[s.index("12") + 1] == 1
    assert got[s.index("45")] == 2 and got[s.index("45") + 1] == 1
    q = s.index("?")
    assert got[q + 1: q + 4] == [1, 2, 3]


def test_clamp_and_padding():
    long = "1" * 30 + "+1="
    p = place_ids(ids(long), 24)[0]
    assert p.max().item() == 24 and p[-1].item() == 0


def test_answer_region_is_causal():
    """Changing an answer token must not change logits at earlier positions."""
    torch.manual_seed(0)
    m = ToyLM(ToyConfig(vocab_size=len(VOCAB), place_emb=True)).eval()
    a, b = ids("347*28=9716#"), ids("347*28=9755#")  # differ from answer position 9 on
    with torch.no_grad():
        la, lb = m(a, n_loops=2), m(b, n_loops=2)
    first_diff = 9
    assert torch.allclose(la[0, :first_diff], lb[0, :first_diff], atol=1e-5)


def test_off_by_default_and_param_count():
    base = ToyLM(ToyConfig(vocab_size=len(VOCAB)))
    pe = ToyLM(ToyConfig(vocab_size=len(VOCAB), place_emb=True))
    assert base.place is None
    assert pe.num_params() - base.num_params() == (24 + 1) * 256
    assert pe.place.weight[0].abs().sum().item() == 0.0
