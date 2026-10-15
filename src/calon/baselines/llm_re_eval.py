"""Prompted LLM baseline for relation extraction on data/ood_re (CrossRE test), scored
like re_eval (strict head span, tail span, relation label).

  given    the gold entities are listed with ids; the model names relations between ids
  end2end  the model is given the entity and relation types and returns
           (head text, relation, tail text); strings map to the nearest occurrences
Relation types come with the CrossRE definitions unless --name_only.
"""
import argparse
import json
from pathlib import Path

from calon.data.roots import ROOT

from ..eval.newtest import chat, last_json, llm
from .re_baselines import f1, fold, triples

GIVEN = """Text:
{text}

Entities (id: text [type]):
{ents}

Relation types:
{rels}

List every relation that the text expresses between two of the entities above, as directed pairs (head, tail) with one relation type each. Use only the listed types.
Return JSON only: a list of {{"head": id, "tail": id, "relation": "type"}}; return [] if there is none."""

E2E = """Text:
{text}

Entity types: {etypes}

Relation types:
{rels}

List every relation that the text expresses between two entity mentions of the entity types above, as directed pairs with one relation type each. Copy the head and tail exactly as they appear in the text.
Return JSON only: a list of {{"head": "text", "relation": "type", "tail": "text"}}; return [] if there is none."""


def rel_lines(r, name_only):
    return "\n".join(f'- "{q["label"]}"' + ("" if name_only or not q["description"] else f': {q["description"]}')
                     for q in r["rel_queries"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["given", "end2end"])
    ap.add_argument("--ood", default=str(ROOT / "data/ood_re"))
    ap.add_argument("--name_only", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024, help="answer budget (dense sets: MAVEN-ERE, LitBank)")
    ap.add_argument("--max_len", type=int, default=8192)
    args = ap.parse_args()
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    engine = llm(args.model, max_len=args.max_len, tp=args.tp)
    res = {"model": args.model, "mode": args.mode, "name_only": args.name_only, "sets": {}, "samples": []}
    for name, info in manifest["sets"].items():
        if args.mode not in info.get("settings", [args.mode]):
            continue
        rows = [json.loads(line) for line in open(Path(args.ood) / f"{name}.jsonl")]
        prompts = []
        for r in rows:
            if args.mode == "given":
                ents = "\n".join(f'{k}: {r["text"][e["start"]:e["end"]]} [{e["label"]}]'
                                 for k, e in enumerate(r["entities"]))
                prompts.append(GIVEN.format(text=r["text"], ents=ents, rels=rel_lines(r, args.name_only)))
            else:
                etypes = ", ".join(q["label"] for q in r["queries"])
                prompts.append(E2E.format(text=r["text"], etypes=etypes, rels=rel_lines(r, args.name_only)))
        outs = chat(engine, prompts, args.max_tokens)
        tp = fp = fn = 0
        per, confusion = {}, {}
        labels_all = {q["label"] for r in rows for q in r["rel_queries"]}
        for r, o in zip(rows, outs):
            js = last_json(o)
            items = js if isinstance(js, list) else []
            if args.mode == "given":
                pred = set()
                for it in items:
                    try:
                        h, t = r["entities"][int(it["head"])], r["entities"][int(it["tail"])]
                    except (KeyError, ValueError, IndexError, TypeError):
                        continue
                    if it.get("relation") in labels_all and (h["start"], h["end"]) != (t["start"], t["end"]):
                        pred.add(((h["start"], h["end"]), (t["start"], t["end"]), it["relation"]))
            else:
                pred = triples(items, r["text"], {q["label"] for q in r["rel_queries"]})
            sym = set(r.get("symmetric_labels", []))
            pred = fold(pred, sym)
            gold = fold({(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}, sym)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            for x in pred | gold:
                c = per.setdefault(x[2], [0, 0, 0])
                c[0 if x in pred and x in gold else 1 if x in pred else 2] += 1
            pmap = {(h, t): lab for h, t, lab in pred}
            for h, t, lab in gold - pred:                 # gold pair predicted with another label
                k = f"{lab} -> {pmap.get((h, t), 'none')}"
                confusion[k] = confusion.get(k, 0) + 1
            if len(res["samples"]) < 3:
                res["samples"].append({"text": r["text"][:200], "answer": o[:400]})
        res["sets"][name] = {args.mode: dict(f1(tp, fp, fn), per_label={
            k: dict(f1(*v), gold=v[0] + v[2]) for k, v in sorted(per.items())},
            confusion=dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:30]))}
        print(name, round(res["sets"][name][args.mode]["f1"], 4), flush=True)
    res[f"macro_f1_{args.mode}"] = sum(v[args.mode]["f1"] for v in res["sets"].values()) / max(1, len(res["sets"]))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({f"macro_f1_{args.mode}": res[f"macro_f1_{args.mode}"]}))


if __name__ == "__main__":
    main()
