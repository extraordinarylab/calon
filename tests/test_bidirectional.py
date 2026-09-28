"""SPEC 6 test 6: the Qwen encoder is really bidirectional, padding-isolated, and
SDPA / eager agree. Skipped when the weights are not on disk."""
from pathlib import Path

import pytest
import torch

QWEN = Path("/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B")
pytestmark = pytest.mark.skipif(not (QWEN / "config.json").exists(), reason="Qwen2.5-0.5B not on disk")


def build(impl):
    from transformers import AutoTokenizer
    from ierlcd.collator import add_control_tokens
    from ierlcd.model import SpanLabelModel
    tok = AutoTokenizer.from_pretrained(QWEN)
    add_control_tokens(tok)
    m = SpanLabelModel(str(QWEN), len(tok), attn_implementation=impl).eval()
    return tok, m


@pytest.fixture(scope="module")
def sdpa():
    return build("sdpa")


def hidden(model, ids, mask):
    with torch.no_grad():
        return model.encode(torch.tensor(ids), torch.tensor(mask)).float()


def test_right_context_changes_left_states(sdpa):
    tok, m = sdpa
    a = tok("Paris is lovely", add_special_tokens=False)["input_ids"]
    b = tok("Paris is awful", add_special_tokens=False)["input_ids"]
    assert len(a) == len(b) and a[0] == b[0]
    ha = hidden(m, [a], [[1] * len(a)])
    hb = hidden(m, [b], [[1] * len(b)])
    assert (ha[0, 0] - hb[0, 0]).abs().max() > 1e-3          # position 0 sees the last word


def test_padding_does_not_change_real_tokens(sdpa):
    tok, m = sdpa
    a = tok("Acme hired Maya yesterday", add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    h1 = hidden(m, [a], [[1] * len(a)])
    h2 = hidden(m, [a + [pad] * 5], [[1] * len(a) + [0] * 5])
    assert torch.allclose(h1[0], h2[0, :len(a)], atol=1e-4)


def test_sdpa_matches_eager(sdpa):
    tok, m = sdpa
    _, me = build("eager")
    me.load_state_dict(m.state_dict())
    a = tok("Taylor Swift released Midnights in 2022", add_special_tokens=False)["input_ids"]
    ids, mask = [a + [0, 0]], [[1] * len(a) + [0, 0]]
    hs, he = hidden(m, ids, mask), hidden(me, ids, mask)
    diff = (hs[0, :len(a)] - he[0, :len(a)]).abs().max().item()
    scale = hs[0, :len(a)].abs().max().item()
    assert diff <= 1e-4 * scale + 1e-4, (diff, scale)
    # eager must be bidirectional too, not silently causal
    b = tok("Paris is lovely", add_special_tokens=False)["input_ids"]
    c = tok("Paris is awful", add_special_tokens=False)["input_ids"]
    assert (hidden(me, [b], [[1] * len(b)])[0, 0] - hidden(me, [c], [[1] * len(c)])[0, 0]).abs().max() > 1e-3
