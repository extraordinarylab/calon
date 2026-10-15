"""Relation extraction data (branch exp/re).

rebel    Qwen3.8-annotated mneb/rebel (synth_rebel/) -> data/re/rebel/{train,dev_select,dev_calib}.jsonl
         Row = NER row (queries, entities) plus
           rel_queries: [{label, description, status}]  positive = has a relation in the
                        row; explicit_absent = Qwen's "not stated in the text" types
           relations:   [{head: [s, e], tail: [s, e], label}]  over entity spans
         Entity types come from Qwen; a relation is kept only when its head and tail
         strings are entity mentions found in the text (nearest pair of occurrences).
         Rows near any evaluation text (CrossRE all splits, CrossNER / MIT / PII / spy
         tests; word 5-gram containment >= 0.8, or exact) are dropped.
crossre  data/cross_re/<domain>/test.jsonl -> data/ood_re/crossre_<domain>.jsonl, every
         row asking the domain's entity types and relation types; relation
         descriptions are the CrossRE paper's definitions (appendix B).
"""
import argparse
import collections
import json
import random

from calon.data.roots import ROOT

from ..synth.llm_annotate import clean_type, parse_json, read_jsonl, write_jsonl
from .prepare_mix import contamination_filter, find_all

REB = ROOT / "synth_rebel"
OUT = ROOT / "data/re"
CROSSRE = ROOT / "data/cross_re"
CR_DOMAINS = ["ai", "literature", "music", "news", "politics", "science"]
CROSSRE_DEFS = {
    "part-of": "Something that is part of something else, for example a song that is part of an album.",
    "physical": "Answers the question where: location, nearness, destination, located in, based in.",
    "usage": "Something which makes use of something else in order to accomplish its scope.",
    "role": "Two entities linked by a business related role, for example management or founder.",
    "social": "Two entities linked by a non-business related role, for example parent, sibling or spouse.",
    "general-affiliation": "Religion, ethnicity or genre, for example a book genre or a music genre.",
    "compare": "Something that is compared with something else.",
    "temporal": "Something that happens or exists during an event.",
    "artifact": "Something concrete which is the result of the work of someone, for example written by.",
    "origin": "Something abstract which is originated by something else, for example invented or idea.",
    "topic": "The topic or focus of something.",
    "opposite": "Something that is physically or idealistically opposite, contrary or against something.",
    "cause-effect": "An event or object which leads to an effect.",
    "win-defeat": "Someone or something who has won or lost a competition or award.",
    "type-of": "The type, property, feature or characteristic of something.",
    "named": "Two spans which refer to the same entity, for example a nickname or an acronym.",
    "related-to": "Two semantically connected entities which do not fall in any other relation type.",
}


def label_text(t):
    return clean_type(t).replace("_", " ")


def eval_texts():
    texts = []
    for d in CR_DOMAINS:
        for sp in ("train", "validation", "test"):
            p = CROSSRE / d / f"{sp}.jsonl"
            if p.exists():
                texts += [json.loads(l)["input"] for l in open(p)]
    for sub in ("ood", "ood_pii", "ood_spy", "ood_new", "ood_x", "ood_ee", "ood_re_x", "ood_x2", "ood_re_x2",
                "ood_ee2"):
        for f in (ROOT / "data" / sub).glob("*.jsonl"):
            texts += [json.loads(l)["text"] for l in open(f)]
    return texts


def nearest(text, a, b):
    """Occurrences of head string a and tail string b closest to each other."""
    oa, ob = find_all(text, a), find_all(text, b)
    if not oa or not ob:
        return None
    return min(((x, y) for x in oa for y in ob if x != y), key=lambda p: abs(p[0][0] - p[1][0]), default=None)


def rebel_row(req, output, stats):
    js = parse_json(output)
    if not isinstance(js, dict):
        stats["parse_fail"] += 1
        return None
    text = req["text"]
    etypes = {label_text(k): str(v) for k, v in (js.get("entity_types") or {}).items()}
    ents = set()
    mention_type = {}
    for e in js.get("entities") or []:
        if not isinstance(e, dict) or not e.get("text"):
            continue
        t = label_text(e.get("type", ""))
        if t not in etypes:
            etypes[t] = ""
        s = str(e["text"]).strip()
        occ = find_all(text, s)
        if not occ:
            stats["entity_unaligned"] += 1
            continue
        mention_type.setdefault(s, t)
        ents |= {(a, b, t) for a, b in occ}
    rtypes = {label_text(k): str(v) for k, v in (js.get("relation_types") or {}).items()}
    rels = set()
    for r in js.get("relations") or []:
        if not isinstance(r, dict):
            continue
        h, t, lab = str(r.get("head", "")).strip(), str(r.get("tail", "")).strip(), label_text(r.get("relation", ""))
        if not h or not t or not lab or h not in mention_type or t not in mention_type:
            stats["relation_not_between_mentions"] += 1
            continue
        pair = nearest(text, h, t)
        if pair is None:
            stats["relation_unaligned"] += 1
            continue
        rels.add((pair[0][0], pair[0][1], pair[1][0], pair[1][1], lab))
        rtypes.setdefault(lab, "")
    if not ents:
        stats["no_entities"] += 1
        return None
    absent = {label_text(k): str(v) for k, v in (js.get("absent_relations") or {}).items()}
    used = {r[4] for r in rels}
    present_types = {t for _, _, t in ents}
    queries = [{"label": t, "description": etypes.get(t, ""), "status": "positive",
                "description_origin": "llm_authored" if etypes.get(t) else "name_only",
                "supervision_origin": "llm_rebel"} for t in sorted(present_types)]
    rel_queries = [{"label": l, "description": rtypes.get(l, ""), "status": "positive"} for l in sorted(used)]
    rel_queries += [{"label": l, "description": d, "status": "explicit_absent"}
                    for l, d in sorted(absent.items()) if l not in used]
    stats["relations"] += len(rels)
    stats["rows_with_relations"] += bool(rels)
    return {"id": req["id"], "source": "rebel", "text": text, "queries": queries,
            "entities": [{"start": a, "end": b, "label": t} for a, b, t in sorted(ents)],
            "rel_queries": rel_queries,
            "relations": [{"head": [a, b], "tail": [c, d], "label": l} for a, b, c, d, l in sorted(rels)]}


def rebel(args):
    stats = collections.Counter()
    by_split = collections.defaultdict(list)
    for k in range(256):
        reqs = {r["id"]: r for r in read_jsonl(REB / f"req-{k:05d}.jsonl")}
        for o in read_jsonl(REB / "out" / f"out-{k:05d}.jsonl"):
            req = reqs.get(o["id"])
            if req is None:
                continue
            if req["split"] == "train" and len(by_split["train"]) >= args.max_train:
                continue
            row = rebel_row(req, o["output"], stats)
            if row is not None:
                by_split[req["split"]].append(row)
        print("shard", k, dict(stats)["relations"] if "relations" in stats else 0, flush=True)
    texts = eval_texts()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train = contamination_filter(by_split["train"], texts, stats, "train")
    heldout = contamination_filter(by_split["validation"], texts, stats, "validation")
    rng.shuffle(heldout)
    rebel_dir = OUT / "rebel"
    rebel_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rebel_dir / "train.jsonl", train)
    write_jsonl(rebel_dir / "dev_select.jsonl", heldout[:args.n_dev])
    write_jsonl(rebel_dir / "dev_calib.jsonl", heldout[args.n_dev:2 * args.n_dev])
    # in-domain test (REBEL test split): NER calibration reports every source's test split
    test = contamination_filter(by_split["test"], texts, stats, "test")
    rng.shuffle(test)
    write_jsonl(rebel_dir / "test.jsonl", test[:args.n_dev])
    stats["train_rows"], stats["dev_rows"] = len(train), min(len(heldout), args.n_dev)
    (OUT / "rebel_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def crossre(args):
    out_dir = ROOT / "data/ood_re"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"scope_policy": "benchmark_closed_world", "splits_used": "CrossRE test",
                "relation_descriptions": "CrossRE paper, appendix B", "sets": {}}
    for d in CR_DOMAINS:
        rows = []
        for rec in read_jsonl(CROSSRE / d / "test.jsonl"):
            ent_labels = [(s["label"], s.get("description", "")) for s in rec["schema"] if s["task"] == "entities"]
            rel_labels = sorted({s["label"] for s in rec["schema"] if s["task"] == "relations"} | set(CROSSRE_DEFS))
            ents = [{"start": m["start"], "end": m["end"], "label": lab}
                    for lab, ms in rec["output"].get("entities", {}).items() for m in ms]
            rels = [{"head": [m["head"]["start"], m["head"]["end"]], "tail": [m["tail"]["start"], m["tail"]["end"]],
                     "label": lab} for lab, ms in rec["output"].get("relations", {}).items() for m in ms]
            pe, pr = {e["label"] for e in ents}, {r["label"] for r in rels}
            rows.append({"id": rec["id"], "source": f"crossre_{d}", "text": rec["input"],
                         "queries": [{"label": l, "description": ds,
                                      "status": "positive" if l in pe else "scope_assumed_absent"}
                                     for l, ds in ent_labels],
                         "entities": ents,
                         "rel_queries": [{"label": l, "description": CROSSRE_DEFS.get(l, ""),
                                          "status": "positive" if l in pr else "scope_assumed_absent"}
                                         for l in rel_labels],
                         "relations": rels})
        write_jsonl(out_dir / f"crossre_{d}.jsonl", rows)
        manifest["sets"][f"crossre_{d}"] = {"rows": len(rows), "entities": sum(len(r["entities"]) for r in rows),
                                            "relations": sum(len(r["relations"]) for r in rows)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sets"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rebel", "crossre"])
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"rebel": rebel, "crossre": crossre}[args.cmd](args)


if __name__ == "__main__":
    main()
