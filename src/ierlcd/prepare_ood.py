"""OOD test sets (CrossNER x5, MIT-Movie, MIT-Restaurant) -> unified rows.

Every row carries its domain's full schema. Labels with no mention in a row
are `scope_assumed_absent` under scope_policy=benchmark_closed_world: the
standard complete-annotation assumption of these benchmarks, not a verified
negative.
"""
import argparse
import json
import os
from pathlib import Path

from .prepare_data import schema_labels

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
SOURCES = {
    "crossner_ai": ("mneb/cross-ner", "64e9a4a56dbe9457c7ad8fe68a710b81ed63599f", "data/ai/test.jsonl"),
    "crossner_literature": ("mneb/cross-ner", "64e9a4a56dbe9457c7ad8fe68a710b81ed63599f", "data/literature/test.jsonl"),
    "crossner_music": ("mneb/cross-ner", "64e9a4a56dbe9457c7ad8fe68a710b81ed63599f", "data/music/test.jsonl"),
    "crossner_politics": ("mneb/cross-ner", "64e9a4a56dbe9457c7ad8fe68a710b81ed63599f", "data/politics/test.jsonl"),
    "crossner_science": ("mneb/cross-ner", "64e9a4a56dbe9457c7ad8fe68a710b81ed63599f", "data/science/test.jsonl"),
    "mit_movie": ("mneb/mit-movie", "7713e53a7e6b4b181ebdb984803d024a825b83c6", "data/mit_movie_complex_test.jsonl"),
    "mit_restaurant": ("mneb/mit-restaurant", "9dc75d3666eea550722ae4cc3bb9dfc9d3b7ceed", "data/mit_restaurant_test.jsonl"),
}


def label_key(label):
    """Seen/unseen bookkeeping only; the model always gets the raw string."""
    return label.lower().replace("_", " ").strip()


def adapt(name, repo, rev, rel):
    path = HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev / rel
    rows, mismatch = [], 0
    with open(path) as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            text = rec["input"]
            ents = rec["output"].get("entities", {}) or {}
            entities = []
            for label, mentions in ents.items():
                for m in mentions:
                    if text[m["start"]:m["end"]] != m["text"]:
                        mismatch += 1
                        continue
                    entities.append({"start": m["start"], "end": m["end"], "label": label})
            pos = {e["label"] for e in entities}
            queries = [{
                "label": label, "description": desc,
                "status": "positive" if label in pos else "scope_assumed_absent",
                "description_origin": "source" if desc else "name_only",
                "supervision_origin": "source" if label in pos else "benchmark_closed_world"}
                for label, desc in schema_labels(rec["schema"])]
            assert pos <= {q["label"] for q in queries}, (name, idx)
            rows.append({"id": f"{name}:test:{idx}", "source": name, "text": text,
                         "queries": queries, "entities": entities})
    return rows, mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="FiNERWeb train.jsonl, for seen-label bookkeeping")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.train) as f:
        train_labels = {label_key(q["label"]) for line in f for q in json.loads(line)["queries"]}
    manifest = {"scope_policy": "benchmark_closed_world", "splits_used": "test only",
                "threshold_policy": "zero-shot: FiNERWeb dev_calib T and threshold", "sets": {}}
    for name, (repo, rev, rel) in SOURCES.items():
        rows, mismatch = adapt(name, repo, rev, rel)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        labels = [q["label"] for q in rows[0]["queries"]]
        n_q = sum(len(r["queries"]) for r in rows)
        n_abs = sum(q["status"] == "scope_assumed_absent" for r in rows for q in r["queries"])
        manifest["sets"][name] = {
            "repo": repo, "revision": rev, "file": rel, "rows": len(rows),
            "entities": sum(len(r["entities"]) for r in rows), "offset_mismatch": mismatch,
            "queries": n_q, "scope_assumed_absent": n_abs,
            "labels": labels,
            "seen_labels": [l for l in labels if label_key(l) in train_labels],
            "unseen_labels": [l for l in labels if label_key(l) not in train_labels]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
