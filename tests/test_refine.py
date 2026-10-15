"""Refinement helpers: second-opinion alignment and judge prompt."""
import collections

from calon.data.prepare_mix import find_all
from calon.synth.refine import jd_prompt, second_opinion


def row():
    q = lambda l, s: {"label": l, "description": "", "status": s}
    return {"text": "Obama visited Paris and Paris again.",
            "queries": [q("person", "positive"), q("city", "positive"), q("band", "unknown")],
            "entities": [{"start": 0, "end": 5, "label": "person"}]}


def test_second_opinion_aligns_every_occurrence_and_skips_unasked_types():
    stats = collections.Counter()
    got = second_opinion(row(), '{"person": ["Obama"], "city": ["Paris"], "band": ["Obama"]}', stats, find_all)
    assert got == {(0, 5, "person"), (14, 19, "city"), (24, 29, "city")}


def test_second_opinion_unparseable():
    assert second_opinion(row(), "no json here", collections.Counter(), find_all) is None


def test_jd_prompt_numbers_candidates():
    r = {"text": "Obama visited Paris.", "types": [("city", "a city")],
         "cands": [[14, 19, "city", "Paris"]]}
    p = jd_prompt(r)
    assert '1. "Paris" as city' in p and '- "city": a city' in p
