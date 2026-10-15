"""Blind label descriptions for training rows (for ee5, from the DynamicNER diagnosis).

Training descriptions of generated sources are written by the generator together with the text;
evaluation descriptions are written by label_bank.py from the label name and a fixed context only.
With the latter the model doubts every candidate on fine-grained schemas (DynamicNER: recall 0.63
-> 0.39 with descriptions). Here every label that occurs at least MIN_COUNT times in a generated
source gets a blind description in the label_bank way (Qwen3.8-27B, name and a generic context
only, never the text), and FRACTION of each generated source's train rows (fixed by row id) take
these descriptions in place of the generator's; labels without one keep theirs. dev / test files
are unchanged.

  inventory  label names per kind (entity type, relation type, event type, argument role)
  requests   synth2-style requests (task lbd, run by llm_gen2 / gen_node.sh)
  apply      <out>/<source>/train.jsonl with the rewritten rows, other files linked
"""
import argparse
import collections
import json
import os
import random
from pathlib import Path

from .label_bank import EVENT_TEMPLATE, REL_TEMPLATE, ROLE_TEMPLATE, SYSTEM, TEMPLATE
from .llm_annotate import write_jsonl
from .paths import S2

EE = {"synth_gee", "synth_cee", "synth_nest", "ccnews_ee"}
MIN_COUNT, FRACTION = 3, 0.3
CONTEXT = {"entity": "entity mentions in English texts of many genres and domains (news, encyclopedia articles, web "
                     "pages, reviews, messages, reports and other documents)",
           "relation": "relations between entities in English texts of many genres and domains",
           "event": "events in English texts of many genres and domains",
           "role": "argument roles of events in English texts of many genres and domains"}


def kinds_of(src, row):
    """[(kind, label)] of the queries of one row."""
    if src in EE:
        if row["id"].endswith(":trig"):
            return [("event", q["label"]) for q in row.get("queries", [])]
        rels = {q["label"] for q in row.get("rel_queries", [])}
        return [("role" if q["label"] in rels else "event", q["label"]) for q in row.get("queries", [])] + \
               [("role", q["label"]) for q in row.get("rel_queries", [])]
    out = [("entity", q["label"]) for q in row.get("queries", [])]
    return out + [("relation", q["label"]) for q in row.get("rel_queries", [])]


def chosen(row_id):
    return random.Random(f"blind:{row_id}").random() < FRACTION


def inventory(args):
    count = collections.Counter()
    for src in args.sources.split(","):
        with open(Path(args.mix) / src / "train.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if chosen(r["id"]):
                    count.update(set(kinds_of(src, r)))
    keep = [(k, lab) for (k, lab), n in count.items() if n >= MIN_COUNT]
    reqs = [{"id": f"lbd:{k}:{lab}", "kind": k, "label": lab} for k, lab in sorted(keep)]
    write_jsonl(S2 / "lbd_requests.jsonl", reqs)
    print(json.dumps({"labels_seen": len(count), "requests": len(reqs),
                      "by_kind": dict(collections.Counter(k for k, _ in keep))}))


def prompt_for(task, r):
    tpl = {"relation": REL_TEMPLATE, "event": EVENT_TEMPLATE, "role": ROLE_TEMPLATE}.get(r["kind"], TEMPLATE)
    return SYSTEM + "\n\n" + tpl.format(context=CONTEXT[r["kind"]], name=r["label"], parent="")


def apply(args):
    from .ee_gen import load_outputs, parse
    bank, stats = {}, collections.Counter()
    for rid, o in load_outputs("lbd").items():
        js = parse(o["output"])
        d = str(js.get("description", "")).strip() if isinstance(js, dict) else ""
        _, kind, label = rid.split(":", 2)
        if d:
            bank[(kind, label)] = d
        else:
            stats["no_description"] += 1
    stats["descriptions"] = len(bank)
    for src in args.sources.split(","):
        d, o = Path(args.mix) / src, Path(args.out) / src
        o.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl") as f, open(o / "train.jsonl", "w") as g:
            for line in f:
                r = json.loads(line)
                if chosen(r["id"]):
                    stats[f"{src}_rows_rewritten"] += 1
                    rels = {q["label"] for q in r.get("rel_queries", [])}
                    for key in ("queries", "rel_queries"):
                        for q in r.get(key, []):
                            if src in EE:
                                kind = "event" if r["id"].endswith(":trig") or (key == "queries" and q["label"] not in rels) else "role"
                            else:
                                kind = "entity" if key == "queries" else "relation"
                            new = bank.get((kind, q["label"]))
                            if new:
                                q["description"], q["description_origin"] = new, "authored_blind"
                                stats[f"{src}_descriptions_replaced"] += 1
                            else:
                                stats[f"{src}_descriptions_kept"] += 1
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        for p in d.iterdir():
            if p.name != "train.jsonl" and not (o / p.name).exists():
                os.symlink(p.resolve(), o / p.name)
    (Path(args.out) / "summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inventory", "apply"])
    ap.add_argument("--mix", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"inventory": inventory, "apply": apply}[args.stage](args)


if __name__ == "__main__":
    main()
