"""Answer parsing of the decoder-LLM baselines."""
from calon.baselines.llm_ner_eval import gner_parse, json_list


def test_gner_parse_offsets_and_bio():
    text = "did  george clooney make a musical in the 1980s"
    ans = "did(O) george(B-actor) clooney(I-actor) make(O) a(O) musical(B-genre) in(O) the(O) 1980s(B-year)"
    got = gner_parse(ans, text, ["actor", "genre", "year"])
    spans = {(text[s:e], l) for s, e, l in got}
    assert spans == {("george clooney", "actor"), ("musical", "genre"), ("1980s", "year")}


def test_gner_parse_multiword_label_and_skipped_word():
    text = "The Clash played London"
    ans = "The(B-band) Clash(I-band) London(B-musical place)"      # "played" dropped
    got = gner_parse(ans, text, ["band", "musical place"])
    assert {(text[s:e], l) for s, e, l in got} == {("The Clash", "band"), ("London", "musical place")}


def test_gner_punctuation_split():
    text = "in Poland, Latvia."
    ans = "in(O) Poland(B-country) ,(O) Latvia(B-country) .(O)"
    got = gner_parse(ans, text, ["country"])
    assert {text[s:e] for s, e, _ in got} == {"Poland", "Latvia"}


def test_json_list():
    assert json_list('["a", "b c"]') == ["a", "b c"]
    assert json_list("[]") == []
    assert json_list("no list") == []
