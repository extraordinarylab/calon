"""SPEC 6 correctness tests 1-3 for offsets, collation and splits."""
import json
import os

import pytest
import torch

from ierlcd.data.alignment import build_candidates, coverage
from ierlcd.data.collator import Collator, add_control_tokens, encode_row
from ierlcd.data.prepare_data import adapt_finerweb, group_rows, split_groups

TEXTS = [
    ("Café Zoë met Zoë in Zürich .", [(0, 8, "org"), (13, 16, "person"), (20, 26, "city")]),
    ("東京 and 東京 again", [(0, 2, "city"), (7, 9, "city")]),
    ("  spaced   out\ttext  ", [(2, 8, "x"), (11, 14, "x")]),
    ("emoji 👩‍🔬 scientist Ada", [(6, 9, "x"), (20, 23, "person")]),
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    add_control_tokens(t)
    return t


def row(text, ents):
    labels = sorted({e[2] for e in ents})
    return {"id": "t", "text": text, "entities": [{"start": s, "end": e, "label": l} for s, e, l in ents],
            "queries": [{"label": l, "description": "", "status": "positive"} for l in labels]}


@pytest.mark.parametrize("text,ents", TEXTS)
def test_candidate_offsets_round_trip(tok, text, ents):
    enc = encode_row(row(text, ents), tok, W=32)
    for s, e, i, j in enc["cands"]:
        span = text[s:e]
        assert span and span == span.strip()
    got = {(enc["cands"][c][0], enc["cands"][c][1]) for g in enc["gold"] for c in g}
    for s, e, _ in ents:
        if text[s:e] == text[s:e].strip():
            assert (s, e) in got, (text[s:e], enc["cands"])
    assert enc["unrepresented"] == len(ents) - len(got)


def test_repeated_mentions_keep_distinct_offsets(tok):
    enc = encode_row(row(*TEXTS[1]), tok, W=32)
    spans = sorted((enc["cands"][c][0], enc["cands"][c][1]) for c in enc["gold"][0])
    assert spans == [(0, 2), (7, 9)]


def test_duplicate_char_spans_keep_fewest_tokens():
    text = "a  b"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]      # two whitespace tokens
    cands = build_candidates(text, offsets, W=4)
    keys = [(s, e) for s, e, _, _ in cands]
    assert len(keys) == len(set(keys))
    assert (0, 1, 0, 0) in cands                    # "a" from 1 token, not "a " (2 tokens)


def test_coverage_counts_width():
    text = "one two three"
    offsets = [(0, 3), (3, 7), (7, 13)]
    ents = [{"start": 0, "end": 13, "label": "x"}]
    assert coverage(text, offsets, ents, W=2) == {"gold": 1, "boundary": 1, "width": 0}
    assert coverage(text, offsets, ents, W=3)["width"] == 1


def test_collate_padding_and_masks(tok):
    lab, txt = tok.convert_tokens_to_ids(["<LABEL>", "<TEXT>"])
    r1 = row(*TEXTS[0])
    r2 = row(*TEXTS[1])
    r2["queries"].append({"label": "ghost", "description": "", "status": "unknown"})
    e1, e2 = encode_row(r1, tok, 32), encode_row(r2, tok, 32)
    inputs, tgt = Collator(tok, lab, txt)([e1, e2])
    assert tgt["valid"][1, 1].sum() == 0            # unknown query fully masked
    # gold targets sit only on valid cells
    assert torch.all(tgt["valid"][tgt["y"].bool()])
    # padded candidates are never valid
    C1 = len(e1["cands"])
    assert not tgt["valid"][0, :, C1:].any()
    # candidate indices point at text tokens
    ids = inputs["input_ids"][0]
    assert ids[inputs["cand_i"][0, 0] - 1] == txt


def test_finerweb_adapter_and_split(tmp_path):
    recs = [
        {"input": "Acme hired Maya .", "output": {"entities": {"person": [{"text": "Maya", "start": 11, "end": 15}]}},
         "schema": {"entities": [{"label": "person", "description": "a human"}]}},
        {"input": "Acme hired Maya .", "output": {"entities": {"org": [{"text": "Acme", "start": 0, "end": 4}]}},
         "schema": [{"label": "org", "description": ""}, {"label": "city", "description": ""}]},
        {"input": "the quick brown fox jumps over the lazy dog in the park near the old river bank today", "output": {"entities": {}},
         "schema": {"entities": []}},
        {"input": "The quick brown fox jumps over the lazy dog in the park near the old river bank today!", "output": {"entities": {}},
         "schema": {"entities": []}},
    ]
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    rows, audit = adapt_finerweb(p)
    assert audit["offset_mismatch"] == 0
    city = [q for q in rows[1]["queries"] if q["label"] == "city"][0]
    assert city["status"] == "unknown"             # label absence is not a negative
    groups, _ = group_rows(rows)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3]]
    assign = split_groups(groups, len(rows), 0)
    assert assign[0] == assign[1] and assign[2] == assign[3]
