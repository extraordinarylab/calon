"""Answer parsing of the InstructUIE relation / event baselines, the gpt-oss NER baseline and the
GLiNER2 gold-trigger proxy."""
from calon.baselines.ee_baselines import to_gold_triggers
from calon.baselines.instructuie_ie import pairs, parse_re
from calon.baselines.llm_ner_eval import prompt_preds


def test_instructuie_re():
    items = parse_re(" part-of: Gaussian mixture model, generative model; usage: none; Foo: a, b", ["part-of", "usage"])
    assert items == [{"head": "Gaussian mixture model", "relation": "part-of", "tail": "generative model"}]
    assert parse_re(" None", ["part-of"]) == []


def test_instructuie_eea_last_colon():
    assert pairs(" 10:30 a.m.: time; UConn Health: victim", sep_last=True) == [("10:30 a.m.", "time"),
                                                                              ("UConn Health", "victim")]


def test_gpt_oss_ner():
    r = {"text": "Maya met Maya in Paris.", "queries": [{"label": "person"}, {"label": "city"}]}
    ans = 'final: [{"text": "Maya", "type": "person"}, {"text": "Paris", "type": "country"}]'
    assert prompt_preds(ans, r) == {(0, 4, "person"), (9, 13, "person")}


def test_gold_trigger_proxy():
    text = "Hackers breached the server and breached it again."
    r = {"text": text, "events": [{"type": "attack", "trigger": [8, 16], "args": []},
                                  {"type": "attack", "trigger": [32, 40], "args": []}]}
    pt = {(0, 16, "attack"), (32, 40, "attack"), (21, 27, "other")}
    pa = {(0, 16, "attack", 21, 27, "target"), (21, 27, "other", 0, 7, "x")}
    gt, args = to_gold_triggers(pt, pa, r)
    assert gt == {(8, 16, "attack"), (32, 40, "attack")}
    assert args == {(8, 16, "attack", 21, 27, "target")}
