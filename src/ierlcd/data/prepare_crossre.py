"""DFKI-SLT/cross_re -> mneb-style jsonl (input / output / schema), data only.

output = {"entities": {type: [{text, start, end}]},
          "relations": {rel: [{head: {text, start, end}, tail: {...},
                               explanation, uncertain, syntax_ambiguity}]}}
schema = [{"label", "description", "task": "entities" | "relations"}] per domain
(all types seen in that domain's train split). Entity descriptions reuse
mneb/cross-ner's where the label string matches; relation descriptions are empty.
Token spans (inclusive) become right-open character spans over the tokens
joined by single spaces. Also reports overlap with the CrossNER test sets,
which CrossRE re-annotates.
"""
import collections
import json
import os
from pathlib import Path

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
OUT = ROOT / "data/cross_re"
CROSSNER = Path(os.path.expanduser("~/.cache/huggingface/hub")) / \
    "datasets--mneb--cross-ner/snapshots/64e9a4a56dbe9457c7ad8fe68a710b81ed63599f/data"
DOMAINS = ["ai", "literature", "music", "news", "politics", "science"]


def span(tokens, starts, i, j):
    s, e = starts[i], starts[j] + len(tokens[j])
    return {"text": " ".join(tokens[i:j + 1]), "start": s, "end": e}


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download
    from .prepare_data import norm_text, sha, word_ngrams
    api = HfApi()
    info = api.dataset_info("DFKI-SLT/cross_re")
    local = snapshot_download("DFKI-SLT/cross_re", repo_type="dataset", revision="refs/convert/parquet",
                              local_dir=str(ROOT / "raw/cross_re"), max_workers=2)
    report = {"repo": "DFKI-SLT/cross_re", "revision": info.sha, "domains": {}}
    for dom in DOMAINS:
        desc = {}
        cn_test = []
        if (CROSSNER / dom).exists():
            first = json.loads(open(CROSSNER / dom / "test.jsonl").readline())
            items = first["schema"].get("entities", []) if isinstance(first["schema"], dict) else first["schema"]
            desc = {it["label"]: it.get("description") or "" for it in items}
            cn_test = [json.loads(l)["input"] for l in open(CROSSNER / dom / "test.jsonl")]
        cn_hash = {sha(norm_text(t)) for t in cn_test}
        cn_grams = [word_ngrams(t) for t in cn_test]
        splits = {}
        for split in ("train", "validation", "test"):
            files = sorted(Path(local).glob(f"{dom}/{split}/*.parquet"))
            splits[split] = [r for f in files for r in pq.read_table(f).to_pylist()]
        ent_types = sorted({n["entity-type"] for r in splits["train"] for n in r["ner"]})
        rel_types = sorted({x["relation-type"] for r in splits["train"] for x in r["relations"]})
        schema = ([{"label": t, "description": desc.get(t, ""), "task": "entities"} for t in ent_types] +
                  [{"label": t, "description": "", "task": "relations"} for t in rel_types])
        dinfo = {"entity_types": len(ent_types), "relation_types": len(rel_types),
                 "entity_descriptions_from_crossner": sum(bool(desc.get(t)) for t in ent_types)}
        for split, rows in splits.items():
            stats = collections.Counter()
            (OUT / dom).mkdir(parents=True, exist_ok=True)
            with open(OUT / dom / f"{split}.jsonl", "w") as f:
                for r in rows:
                    toks = r["sentence"]
                    starts, pos = [], 0
                    for t in toks:
                        starts.append(pos)
                        pos += len(t) + 1
                    text = " ".join(toks)
                    ents = collections.defaultdict(list)
                    for n in r["ner"]:
                        ents[n["entity-type"]].append(span(toks, starts, n["id-start"], n["id-end"]))
                        stats["entities"] += 1
                        if n["entity-type"] not in ent_types:
                            stats["entity_type_not_in_train_schema"] += 1
                    rels = collections.defaultdict(list)
                    for x in r["relations"]:
                        rels[x["relation-type"]].append({
                            "head": span(toks, starts, x["id_1-start"], x["id_1-end"]),
                            "tail": span(toks, starts, x["id_2-start"], x["id_2-end"]),
                            "explanation": x.get("Exp", ""), "uncertain": bool(x.get("Un")),
                            "syntax_ambiguity": bool(x.get("SA"))})
                        stats["relations"] += 1
                    for lab, items in ents.items():            # offsets round-trip
                        stats["offset_mismatch"] += sum(text[i["start"]:i["end"]] != i["text"] for i in items)
                    h = sha(norm_text(text))
                    if h in cn_hash:
                        stats["exact_in_crossner_test"] += 1
                    else:
                        g = word_ngrams(text)
                        if g and any(cg and len(g & cg) / len(cg) >= 0.8 for cg in cn_grams):
                            stats["near_in_crossner_test"] += 1
                    f.write(json.dumps({"id": f"cross_re:{dom}:{split}:{r['doc_key']}", "input": text,
                                        "output": {"entities": dict(ents), "relations": dict(rels)},
                                        "schema": schema}, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
            dinfo[split] = dict(stats)
        report["domains"][dom] = dinfo
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    for dom, d in report["domains"].items():
        print(dom, {k: d[k] for k in ("entity_types", "relation_types", "entity_descriptions_from_crossner")},
              {s: {k: d[s].get(k, 0) for k in ("rows", "entities", "relations", "offset_mismatch",
                                               "exact_in_crossner_test", "near_in_crossner_test")}
               for s in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
