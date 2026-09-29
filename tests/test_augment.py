"""Label anonymization keeps queries and gold entities consistent."""
import random

from calon.data.mixdata import anonymize_labels


def test_anonymize_labels():
    q = lambda l, d: {"label": l, "description": d, "status": "positive"}
    row = {"text": "Obama in Paris", "queries": [q("person", "a human"), q("city", "")],
           "entities": [{"start": 0, "end": 5, "label": "person"}, {"start": 9, "end": 14, "label": "city"}]}
    out = anonymize_labels(row, random.Random(0), 1.0)
    new = out["queries"][0]["label"]
    assert new.startswith("entity ") and out["queries"][0]["description"] == "a human"
    assert out["queries"][1]["label"] == "city"                  # no description: never renamed
    assert [e["label"] for e in out["entities"]] == [new, "city"]
    assert row["queries"][0]["label"] == "person"                # input row untouched
