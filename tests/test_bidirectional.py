"""SPEC 6 test 6: the Qwen encoder is really bidirectional, padding-isolated, and
SDPA / eager agree. Skipped when the weights are not on disk."""
import os
from pathlib import Path

import pytest
import torch

QWEN = Path(os.environ.get("QWEN_TEST_PATH",
                           "/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B"))
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
    diff = (h1[0] - h2[0, :len(a)]).abs().max().item()
    scale = h1[0].abs().max().item()
    assert diff <= 1e-4 * scale + 1e-4, (diff, scale)
    # and padded keys really are ignored: changing the pad token id changes nothing
    h3 = hidden(m, [a + [pad + 1] * 5], [[1] * len(a) + [0] * 5])
    assert (h2[0, :len(a)] - h3[0, :len(a)]).abs().max().item() <= 1e-4 * scale + 1e-4


def test_head_layers_ignore_padding(sdpa):
    """The extra transformer layers see padding only through the key mask."""
    tok, m = sdpa
    from ierlcd.model import SpanLabelModel
    torch.manual_seed(0)
    post = SpanLabelModel.__new__(SpanLabelModel)
    torch.nn.Module.__init__(post)
    d = m.encoder.config.hidden_size
    layer = torch.nn.TransformerEncoderLayer(d, 16, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True)
    post.post = torch.nn.TransformerEncoder(layer, 2, norm=torch.nn.LayerNorm(d), enable_nested_tensor=False).eval()
    a = tok("Acme hired Maya yesterday", add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    m1, m2 = torch.tensor([[1] * len(a)]), torch.tensor([[1] * len(a) + [0] * 5])
    with torch.no_grad():
        h1 = post.contextualize(hidden(m, [a], m1.tolist()), m1)
        h2 = post.contextualize(hidden(m, [a + [pad] * 5], m2.tolist()), m2)
    assert (h1[0] - h2[0, :len(a)]).abs().max().item() <= 1e-3


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
