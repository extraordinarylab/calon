"""Few-NERD precedence mode: most specific type wins, "other X" becomes the coarse type."""
import json

from calon.data.mixdata import PRECEDENCE_NOTE, fewnerd_precedence


def test_precedence(tmp_path):
    inv = {"fewnerd": [{"raw": "person", "name": "person", "parent": None},
                       {"raw": "person-politician", "name": "politician", "parent": "person"},
                       {"raw": "person-other", "name": "other person", "parent": "person"}]}
    (tmp_path / "inventory.json").write_text(json.dumps(inv))
    q = lambda l, s: {"label": l, "description": "d", "status": s}
    row = {"id": "r", "text": "Obama met Bob .",
           "queries": [q("person", "positive"), q("politician", "positive"), q("other person", "positive")],
           "entities": [{"start": 0, "end": 5, "label": "politician"}, {"start": 0, "end": 5, "label": "person"},
                        {"start": 10, "end": 13, "label": "other person"}, {"start": 10, "end": 13, "label": "person"}]}
    (out,) = fewnerd_precedence([row], tmp_path)
    assert sorted((e["start"], e["label"]) for e in out["entities"]) == [(0, "politician"), (10, "person")]
    labels = {x["label"]: x for x in out["queries"]}
    assert set(labels) == {"person", "politician"}
    assert labels["person"]["description"].endswith(PRECEDENCE_NOTE)
    assert labels["person"]["status"] == "positive"
