"""Relation extraction: ordered entity pairs, gold pairs, collation shapes, relation head
forward / loss / gradient. Needs the Qwen tokenizer and weights (skipped otherwise)."""
import os
from pathlib import Path

import pytest
import torch

from calon.data.roots import MODELS

QWEN = Path(os.environ.get("QWEN_TEST_PATH", str(MODELS / "Qwen/Qwen3-0.6B")))
pytestmark = pytest.mark.skipif(not (QWEN / "config.json").exists(), reason="Qwen weights not on disk")

TEXT = "Taylor Swift released Fearless in Nashville ."


def row():
    q = lambda l, s="positive": {"label": l, "description": "", "status": s}
    return {"id": "r", "text": TEXT, "queries": [q("person"), q("album"), q("city")],
            "entities": [{"start": 0, "end": 12, "label": "person"}, {"start": 22, "end": 30, "label": "album"},
                         {"start": 34, "end": 43, "label": "city"}],
            "rel_queries": [q("artifact"), q("physical"), q("social", "explicit_absent")],
            "relations": [{"head": [22, 30], "tail": [0, 12], "label": "artifact"}]}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    from calon.data.collator import add_control_tokens
    t = AutoTokenizer.from_pretrained(QWEN)
    add_control_tokens(t)
    return t


def test_pairs_and_gold(tok):
    from calon.data.collator import encode_row
    e = encode_row(row(), tok, 32)
    assert len(e["pairs"]) == 6                                  # 3 entities, ordered, no self pairs
    k = e["pair_char"].index(((22, 30), (0, 12)))
    assert e["rel_gold"][0] == [k] and e["rel_gold"][1] == [] and e["rel_gold"][2] == []
    assert e["gold_triples"] == {((22, 30), (0, 12), "artifact")}
    assert ((0, 12), (22, 30)) in e["pair_char"]                 # direction matters: both orders exist


def test_end_to_end_pair_spans(tok):
    from calon.data.collator import encode_row
    e = encode_row(row(), tok, 32, pair_spans=[(0, 12), (34, 43)])
    assert len(e["pairs"]) == 2 and all(p != ((22, 30), (0, 12)) for p in e["pair_char"])
    assert e["gold_triples"] == {((22, 30), (0, 12), "artifact")}   # still counted as gold (missed)


def test_forward_loss_and_gradient(tok):
    from calon.data.collator import Collator, add_control_tokens, encode_row
    from calon.model.losses import rlcd_loss
    from calon.model.model import SpanLabelModel
    lab, txt = add_control_tokens(tok)
    collate = Collator(tok, lab, txt)
    plain = dict(row())
    del plain["rel_queries"], plain["relations"]
    inputs, tgt = collate([encode_row(row(), tok, 32), encode_row(plain, tok, 32)])
    assert inputs["rel_mask"].shape[:2] == (2, 3) and inputs["pair_h"].shape == (2, 6)
    assert not tgt["rel"]["valid"][1].any()                     # the NER-only row has no relation decisions
    torch.manual_seed(0)
    m = SpanLabelModel(str(QWEN), len(tok), head="rlcd", relations=True)
    z, zr = m(**inputs)
    assert zr.shape == (2, 4, 6)                                 # [none + 3 relations, 6 pairs]
    rt = tgt["rel"]
    loss = rlcd_loss(zr, rt["y"], rt["valid"], rt["query_mask"], rt["cand_mask"], none_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.none_rel.grad is not None and m.pair_mlp[0].weight.grad.abs().sum() > 0
