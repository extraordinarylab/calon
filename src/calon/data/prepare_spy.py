"""Evaluation sets from mneb/spy: personal identifiers in English legal questions
(spy_legal) and medical consultations (spy_medical). Evaluation only, never trained on.

Documents run to ~8,500 characters, longer than our 1,024-token window and GNER's
2,048-token context, so each document is cut into chunks of at most MAX_CHARS
characters at whitespace that falls outside every gold entity; every model sees
the same chunks. Each chunk asks the full label set; labels not mentioned in it
are scope_assumed_absent (benchmark_closed_world), as for the other OOD sets.
Label names are the dataset's own codes made readable; descriptions are authored
by label_bank.py from the name alone.
Stages: build (download, convert, chunk, write inventory) -> label_bank -> finish.
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
OUT = ROOT / "data/ood_spy"
REPO = "mneb/spy"
MAX_CHARS = 1200
SET_OF = {"legal_questions": "spy_legal", "medical_consultations": "spy_medical"}
NAMES = {"name": "name", "email": "email address", "phone_num": "phone number", "address": "address",
         "url": "URL", "username": "username", "id_num": "identification number"}


def readable(code):
    return NAMES.get(code, code.replace("_num", " number").replace("_", " "))


def entities_of(text, output, stats):
    """output: {label: [{text, start, end}, ...]} (lists may be None)."""
    ents = []
    for code, items in (output or {}).items():
        for it in items or []:
            s, e = int(it["start"]), int(it["end"])
            if text[s:e] != it.get("text", text[s:e]):
                stats["offset_mismatch"] += 1
                continue
            ents.append({"start": s, "end": e, "code": code})
    return ents


def chunks(text, ents, max_chars):
    """Cut points at whitespace outside all entities, each piece <= max_chars when possible."""
    inside = bytearray(len(text) + 1)
    for e in ents:
        for k in range(e["start"], e["end"]):
            inside[k] = 1
    out, start = [], 0
    while len(text) - start > max_chars:
        cut = -1
        for k in range(start + max_chars, start, -1):
            if text[k].isspace() and not inside[k]:
                cut = k
                break
        if cut <= start:                      # no clean cut: take the next clean one after the limit
            cut = next((k for k in range(start + max_chars, len(text))
                        if text[k].isspace() and not inside[k]), len(text))
        out.append((start, cut))
        start = cut
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        out.append((start, len(text)))
    return out


def build(args):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info(REPO)
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset", revision=info.sha)
             if Path(f).stem in SET_OF]
    stats, docs = collections.Counter(), []
    for f in files:
        path = hf_hub_download(REPO, f, repo_type="dataset", revision=info.sha, cache_dir=str(ROOT / "raw/hf"))
        if f.endswith(".parquet"):
            recs = pd.read_parquet(path).to_dict("records")
        elif f.endswith(".jsonl"):
            recs = [json.loads(l) for l in open(path)]
        else:
            v = json.load(open(path))
            recs = v if isinstance(v, list) else v.get("data", [])
        stats[f"file:{f}"] = len(recs)
        for k, rec in enumerate(recs):
            out = rec["output"]
            if isinstance(out, str):
                out = json.loads(out)
            out = dict(out).get("entities", out)            # {"entities": {label: [spans]}}
            out = {c: (list(v) if v is not None else []) for c, v in dict(out).items()}
            docs.append({"set": SET_OF[Path(f).stem], "doc": f"{Path(f).stem}:{k}", "text": rec["input"],
                         "ents": entities_of(rec["input"], out, stats)})
    rows = []
    for d in docs:
        for a, b in chunks(d["text"], d["ents"], MAX_CHARS):
            ents = [dict(e, start=e["start"] - a, end=e["end"] - a) for e in d["ents"]
                    if e["start"] >= a and e["end"] <= b]
            stats["entities_kept"] += len(ents)
            rows.append({"set": d["set"], "id": f"{d['doc']}:{a}", "text": d["text"][a:b], "ents": ents})
    stats["entities_total"] = sum(len(d["ents"]) for d in docs)
    stats["docs"], stats["chunks"] = len(docs), len(rows)
    stats["chunks_over_limit"] = sum(len(r["text"]) > MAX_CHARS for r in rows)
    codes = sorted({e["code"] for d in docs for e in d["ents"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_rows.json").write_text(json.dumps(rows))
    (OUT / "inventory.json").write_text(json.dumps(
        {"spy": [{"raw": c, "name": readable(c), "parent": None} for c in codes]}, indent=1))
    (OUT / "build_meta.json").write_text(json.dumps({"repo": REPO, "revision": info.sha, "files": files,
                                                     "codes": codes, "stats": dict(stats)}, indent=1))
    print(json.dumps({"labels": [readable(c) for c in codes], "stats": dict(stats)}))


def finish(args):
    bank = json.loads((OUT / "label_bank.json").read_text())["bank"]["spy"]
    rows = json.loads((OUT / "raw_rows.json").read_text())
    train_labels = set()
    for line in open(ROOT / "data/mix/finerweb/train.jsonl"):
        train_labels |= {q["label"].lower().replace("_", " ") for q in json.loads(line)["queries"]}
    by_set = collections.defaultdict(list)
    for r in rows:
        ents = [{"start": e["start"], "end": e["end"], "label": bank[e["code"]]["name"]} for e in r["ents"]]
        present = {e["label"] for e in ents}
        queries = [{"label": v["name"], "description": v["description"],
                    "status": "positive" if v["name"] in present else "scope_assumed_absent",
                    "description_origin": v["description_origin"],
                    "supervision_origin": "source" if v["name"] in present else "benchmark_closed_world"}
                   for v in bank.values()]
        by_set[r["set"]].append({"id": f"{r['set']}:test:{r['id']}", "source": r["set"], "text": r["text"],
                                 "queries": queries, "entities": ents})
    labels = [v["name"] for v in bank.values()]
    manifest = {"scope_policy": "benchmark_closed_world",
                "splits_used": f"all rows, documents cut into <= {MAX_CHARS}-character chunks",
                "threshold_policy": "zero-shot: training sources' dev_calib T and threshold",
                "descriptions": "authored by label_bank.py from the label name only", "sets": {}}
    for name, out in sorted(by_set.items()):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {
            "repo": REPO, "rows": len(out), "entities": sum(len(r["entities"]) for r in out),
            "labels": labels, "seen_labels": [l for l in labels if l in train_labels],
            "unseen_labels": [l for l in labels if l not in train_labels]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {"rows": v["rows"], "entities": v["entities"]} for k, v in manifest["sets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
