"""Mixture adapters, query sampling, dedup and the vectorized candidate builder."""
import collections
import random

import pytest

from ierlcd.data.alignment import build_candidates, build_candidates_np, tokenize_text
from ierlcd.data.prepare_mix import contamination_filter, fewnerd_name, find_all, parse_massive, sample_queries


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def test_np_candidates_match_reference(tok):
    rng = random.Random(0)
    alphabet = list("abc XYZ,.'-\t\n  é東👩‍🔬") + ["  ", "\n\n"]
    texts = ["Acme hired Maya.", "  spaced   out\ttext  ", "a  b", ""]
    texts += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 120))) for _ in range(200)]
    for text in texts:
        _, offsets = tokenize_text(tok, text)
        for W in (1, 3, 32):
            assert build_candidates_np(text, offsets, W) == build_candidates(text, offsets, W), text


def q(label, status):
    return {"label": label, "status": status}


def test_sample_queries_rule():
    queries = ([q(f"p{i}", "positive") for i in range(3)] + [q(f"e{i}", "explicit_absent") for i in range(2)]
               + [q(f"s{i}", "scope_assumed_absent") for i in range(10)] + [q("u", "unknown")])
    for seed in range(20):
        out = sample_queries(queries, random.Random(seed))
        c = collections.Counter(x["status"] for x in out)
        assert c["positive"] == 3 and c["unknown"] == 1
        assert c["explicit_absent"] == 2 and c["scope_assumed_absent"] == 1   # cap = 3, explicit first
    only_absent = [q(f"s{i}", "scope_assumed_absent") for i in range(20)]
    assert len(sample_queries(only_absent, random.Random(0))) == 8
    one_pos = [q("p", "positive")] + only_absent
    assert len(sample_queries(one_pos, random.Random(0))) == 3               # cap = max(2, 1)


def test_parse_massive():
    text, ents = parse_massive("wake me up at [time : five am] [date : this week]",
                               "wake me up at five am this week")
    assert text == "wake me up at five am this week"
    assert [(text[e["start"]:e["end"]], e["slot"]) for e in ents] == [("five am", "time"),
                                                                      ("this week", "date")]
    assert parse_massive("[time : five]", "six") == (None, None)


def test_find_all_respects_word_boundaries():
    text = "women and men; men-only (BI) BI."
    assert find_all(text, "men") == [(10, 13), (15, 18)]
    assert find_all(text, "(BI)") == [(24, 28)]
    assert find_all(text, "BI") == [(25, 27), (29, 31)]


def test_contamination_filter_containment():
    eval_texts = ["the quick brown fox jumps over the lazy dog near the river bank"]
    rows = [{"text": "Intro. The quick brown fox jumps over the lazy dog near the river bank. "
                     "Then many other words follow here to make the passage much longer than before."},
            {"text": "a completely different sentence about astronomy and telescopes at night"},
            {"text": "The quick brown fox jumps over the lazy dog near the river bank"}]
    stats = collections.Counter()
    kept = contamination_filter(rows, eval_texts, stats, "t")
    assert kept == [rows[1]]
    assert stats["t_drop_exact"] == 1 and stats["t_drop_near"] == 1


def test_fewnerd_names():
    assert fewnerd_name("person-politician") == "politician"
    assert fewnerd_name("person-other") == "other person"
    assert fewnerd_name("person-artist/author") == "artist / author"
    assert fewnerd_name("event-attack/battle/war/militaryconflict") == "attack / battle / war / military conflict"
    assert fewnerd_name("other") == "other entity"
