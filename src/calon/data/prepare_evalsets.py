"""Newer zero-shot evaluation sets (evaluation only, never trained on).

  dynamicner_en  DynamicNER English BASE test (300 sentences, 133 fine types). Each
                 sentence asks its gold fine types plus sampled absent ones up to
                 N_QUERY: siblings under the same mid-level category first, then
                 random fine types (fixed seed), so every model sees the same list.
  buster         BUSTER gold fold 5 (M&A filings), documents cut into <= 1200-character
                 chunks at whitespace outside gold entities; all 6 types asked.
  scier          SciER LLM-format test and test_ood (entities: method / task / dataset;
                 9 relation types). NER rows go to data/ood_x, NER+RE rows to data/ood_re_x.
Re-DocRED is document-level (entity clusters) and is prepared separately.

Absent types are scope_assumed_absent (benchmark_closed_world). Descriptions come
from label_bank.py (name, parent and a fixed dataset context only).
Stages: build (convert, write raw rows and inventory) -> label_bank -> finish.
"""
import argparse
import collections
import json
import random

from calon.data.roots import ROOT

from .prepare_mix import find_all
from .prepare_spy import chunks

RAW = ROOT / "raw/evalsets"
OUT = ROOT / "data/ood_x"
OUT_RE = ROOT / "data/ood_re_x"
N_QUERY = 10
BUSTER_NAMES = {"Parties.BUYING_COMPANY": "buying company", "Parties.ACQUIRED_COMPANY": "acquired company",
                "Parties.SELLING_COMPANY": "selling company",
                "Advisors.GENERIC_CONSULTING_COMPANY": "consulting company",
                "Advisors.LEGAL_CONSULTING_COMPANY": "legal consulting company",
                "Generic_Info.ANNUAL_REVENUES": "annual revenues"}


def nth_occurrences(text, s, n, used):
    """First n occurrences of s not already used by another mention."""
    out = []
    for occ in find_all(text, s):
        if occ not in used:
            out.append(occ)
            if len(out) == n:
                break
    return out


def dynamicner(stats, split="test"):
    d = RAW / "DynamicNER"
    tree = json.load(open(d / "DynamicNER_process/category.json"))
    parent = {}
    for top, mids in tree.items():
        for mid, fines in mids.items():
            for f in fines:
                parent[f.lower()] = mid.lower()
    js = json.load(open(d / f"data/DynamicNER/base/en/BASE/{split}.json"))
    rows = []
    for key, v in js.items():
        text, used, ents = v["sentence"], set(), []
        for s, lab in zip(v["entity"], v["category"]):
            occ = nth_occurrences(text, s, 1, used)
            if not occ:
                stats["dynamicner_unaligned"] += 1
                continue
            used |= set(occ)
            ents.append({"start": occ[0][0], "end": occ[0][1], "code": lab.lower()})
        rows.append({"set": "dynamicner_en", "id": f"dynamicner_en:{split}:{key}", "text": text, "ents": ents})
    all_types = sorted({e["code"] for r in rows for e in r["ents"]} | set(parent))
    for r in rows:
        rng = random.Random(r["id"])
        gold = sorted({e["code"] for e in r["ents"]})
        sib = sorted({t for t in all_types if t not in gold and parent.get(t) in {parent.get(g) for g in gold}})
        rng.shuffle(sib)
        ask = gold + sib[:max(0, min(4, N_QUERY - len(gold)))]
        rest = [t for t in all_types if t not in ask]
        rng.shuffle(rest)
        r["ask"] = ask + rest[:max(0, N_QUERY - len(ask))]
    stats["dynamicner_rows"] = len(rows)
    stats["dynamicner_not_in_hierarchy"] = len({e["code"] for r in rows for e in r["ents"]} - set(parent))
    inv = [{"raw": t, "name": t, "parent": parent.get(t)} for t in all_types]
    return rows, inv


def buster(stats, folds=(5,)):
    """Gold folds (1-5; fold 5 is the evaluation set)."""
    import numpy as np
    import pandas as pd
    recs = [dict(r, fold=k) for k in folds
            for r in pd.read_parquet(RAW / f"BUSTER/data/FOLD_{k}-00000-of-00001.parquet").to_dict("records")]
    rows = []
    for rec in recs:
        text = rec["text"]
        toks, labs = list(np.asarray(rec["tokens"])), list(np.asarray(rec["labels"]))
        pos, spans = 0, []
        for t in toks:                                 # tokens in order inside the text
            k = text.find(t, pos)
            if k < 0:
                spans.append(None)
                stats["buster_token_unaligned"] += 1
                continue
            spans.append((k, k + len(t)))
            pos = k + len(t)
        ents, cur = [], None
        for sp, lab in zip(spans, labs):
            if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur["code"] != lab[2:])):
                if cur:
                    ents.append(cur)
                cur = {"start": sp[0], "end": sp[1], "code": lab[2:]} if sp else None
            elif lab.startswith("I-") and sp:
                cur["end"] = sp[1]
            else:
                if cur:
                    ents.append(cur)
                cur = None
        if cur:
            ents.append(cur)
        for a, b in chunks(text, ents, 1200):
            ce = [dict(e, start=e["start"] - a, end=e["end"] - a) for e in ents if e["start"] >= a and e["end"] <= b]
            rows.append({"set": "buster", "id": f"buster:fold{rec['fold']}:{rec['document_id']}:{a}", "text": text[a:b],
                         "ents": ce, "ask": sorted(BUSTER_NAMES)})
    stats["buster_rows"] = len(rows)
    stats["buster_entities"] = sum(len(r["ents"]) for r in rows)
    return rows, [{"raw": c, "name": n, "parent": None} for c, n in BUSTER_NAMES.items()]


def scier(stats, splits=(("test", "scier"), ("test_ood", "scier_ood"))):
    rows = []
    for split, name in splits:
        for k, line in enumerate(open(RAW / f"SciER/SciER/LLM/{split}.jsonl")):
            rec = json.loads(line)
            text, used, ents = rec["sentence"], set(), []
            want = collections.Counter((s, t.lower()) for s, t in rec["ner"])
            for (s, t), n in want.items():
                occ = nth_occurrences(text, s, n, used)
                stats["scier_unaligned"] += n - len(occ)
                used |= set(occ)
                ents += [{"start": a, "end": b, "code": t} for a, b in occ]
            by_str = collections.defaultdict(list)
            for e in ents:
                by_str[text[e["start"]:e["end"]]].append((e["start"], e["end"]))
            rels = set()
            for h, lab, t in rec["rel"]:
                hs, ts = by_str.get(h, []), by_str.get(t, [])
                pairs = [(x, y) for x in hs for y in ts if x != y]
                if not pairs:
                    stats["scier_rel_unaligned"] += 1
                    continue
                x, y = min(pairs, key=lambda p: abs(p[0][0] - p[1][0]))
                rels.add((x, y, lab.lower()))
            rows.append({"set": name, "id": f"{name}:{rec['doc_id']}:{k}", "text": text, "ents": ents,
                         "ask": ["method", "task", "dataset"],
                         "rels": [{"head": list(x), "tail": list(y), "label": l} for x, y, l in sorted(rels)]})
    rel_types = sorted({r["label"] for x in rows for r in x["rels"]})
    stats["scier_rows"] = len(rows)
    stats["scier_relations"] = sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in ("method", "task", "dataset")], rel_types


def build(args):
    stats = collections.Counter()
    dn, dn_inv = dynamicner(stats)
    bu, bu_inv = buster(stats)
    sc, sc_inv, rel_types = scier(stats)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_rows.json").write_text(json.dumps(dn + bu + sc))
    (OUT / "inventory.json").write_text(json.dumps(
        {"dynamicner": dn_inv, "buster": bu_inv, "scier": sc_inv,
         "scier_rel": [{"raw": t, "name": t, "parent": None, "kind": "relation"} for t in rel_types]}, indent=1))
    (OUT / "build_meta.json").write_text(json.dumps({"stats": dict(stats), "rel_types": rel_types}, indent=1))
    print(json.dumps({"stats": dict(stats), "rel_types": rel_types}))


def finish(args):
    bank = json.loads((OUT / "label_bank.json").read_text())["bank"]
    src_of = {"dynamicner_en": "dynamicner", "buster": "buster", "scier": "scier", "scier_ood": "scier"}
    rows = json.loads((OUT / "raw_rows.json").read_text())
    train_labels = set()           # as prepare_spy: labels of a training source, for seen / unseen
    for line in open(ROOT / "data/mix/finerweb/train.jsonl"):
        train_labels |= {q["label"].lower().replace("_", " ") for q in json.loads(line)["queries"]}
    by_set, re_sets = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        b = bank[src_of[r["set"]]]
        ents = [{"start": e["start"], "end": e["end"], "label": b[e["code"]]["name"]} for e in r["ents"]]
        present = {e["label"] for e in ents}
        queries = [{"label": b[c]["name"], "description": b[c]["description"],
                    "status": "positive" if b[c]["name"] in present else "scope_assumed_absent",
                    "description_origin": b[c]["description_origin"],
                    "supervision_origin": "source" if b[c]["name"] in present else "benchmark_closed_world"}
                   for c in r["ask"]]
        row = {"id": r["id"], "source": r["set"], "text": r["text"], "queries": queries, "entities": ents}
        by_set[r["set"]].append(row)
        if "rels" in r:
            rb = bank["scier_rel"]
            used = {x["label"] for x in r["rels"]}
            re_sets[r["set"]].append(dict(row, relations=r["rels"], rel_queries=[
                {"label": v["name"], "description": v["description"],
                 "status": "positive" if v["name"] in used else "scope_assumed_absent"} for v in rb.values()]))
    for out_dir, sets, kind in ((OUT, by_set, "ner"), (OUT_RE, re_sets, "re")):
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"scope_policy": "benchmark_closed_world",
                    "threshold_policy": "zero-shot: training sources' dev_calib T and threshold",
                    "descriptions": "authored by label_bank.py from the label name, parent and dataset context",
                    "sets": {}}
        for name, out in sorted(sets.items()):
            with open(out_dir / f"{name}.jsonl", "w") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            labels = sorted({q["label"] for r in out for q in r["queries"]})
            manifest["sets"][name] = {"rows": len(out), "entities": sum(len(r["entities"]) for r in out),
                                      "labels": labels, "seen_labels": [x for x in labels if x in train_labels],
                                      "unseen_labels": [x for x in labels if x not in train_labels]}
            if kind == "re":
                manifest["sets"][name]["relations"] = sum(len(r["relations"]) for r in out)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(kind, json.dumps({k: {x: v[x] for x in v if x != "labels"} for k, v in manifest["sets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
