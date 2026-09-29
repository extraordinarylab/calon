"""JSON extraction from LLM answers."""
from ierlcd.eval.newtest import final_text, last_json


def test_last_json_returns_outer_object():
    assert last_json('{"a": ["x"], "b": []}') == {"a": ["x"], "b": []}


def test_last_json_after_reasoning():
    t = final_text('analysis: maybe {"a": 1}?assistantfinal{"a": ["y"], "c": []}')
    assert last_json(t) == {"a": ["y"], "c": []}


def test_last_json_list_and_garbage():
    assert last_json('here: ["one", "two"] done') == ["one", "two"]
    assert last_json("nothing") is None
