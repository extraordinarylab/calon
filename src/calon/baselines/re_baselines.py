"""GLiNER2 / gliformer relation extraction on data/ood_re (CrossRE test), end to end,
scored like our re_eval (strict: head span, tail span, relation label).

Each row asks the domain's relation types (with the CrossRE definitions unless
--name_only). Answers are read generically: every dict carrying a head and a tail
(keys head/source/subject and tail/target/object) and a relation name (relation /
label / type, or the key it is listed under). Spans come from start/end when given,
else from the nearest pair of string occurrences. The first raw answers are saved.
"""
import argparse
import json
from pathlib import Path

from calon.data.roots import ROOT

HEAD_KEYS, TAIL_KEYS, REL_KEYS = ("head", "source", "subject"), ("tail", "target", "object"), \
    ("relation", "label", "type", "predicate")


def find_all(text, s):
    out, k = [], text.find(s)
    while s and k >= 0:
        out.append((k, k + len(s)))
        k = text.find(s, k + 1)
    return out


def span_of(x, text):
    if isinstance(x, dict):
        if "start" in x and "end" in x:
            return [(int(x["start"]), int(x["end"]))]
        x = x.get("text", "")
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        x = x[0]
    return find_all(text, str(x).strip())


def triples(obj, text, labels, under=None):
    """Walk the answer; yield (head span, tail span, label)."""
    out = set()
    if isinstance(obj, dict):
        h = next((obj[k] for k in HEAD_KEYS if k in obj), None)
        t = next((obj[k] for k in TAIL_KEYS if k in obj), None)
        if h is not None and t is not None:
            lab = next((obj[k] for k in REL_KEYS if k in obj and isinstance(obj[k], str)), under)
            if lab in labels:
                hs, ts = span_of(h, text), span_of(t, text)
                pairs = [(a, b) for a in hs for b in ts if a != b]
                if pairs:
                    a, b = min(pairs, key=lambda p: abs(p[0][0] - p[1][0]))
                    out.add((a, b, lab))
            return out
        for k, v in obj.items():
            # GLiNER2 keys a described relation as "name: description"
            name = k.split(": ", 1)[0] if isinstance(k, str) else k
            out |= triples(v, text, labels, name if name in labels else under)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(x, (str, dict)) for x in obj) and under in labels:
            hs, ts = span_of(obj[0], text), span_of(obj[1], text)
            pairs = [(a, b) for a in hs for b in ts if a != b]
            if pairs:
                a, b = min(pairs, key=lambda p: abs(p[0][0] - p[1][0]))
                out.add((a, b, under))
            return out
        for v in obj:
            out |= triples(v, text, labels, under)
    return out


def to_gold_entities(pred, r):
    """Closest stand-in for the gold-entity setting for models without one (GLiNER2,
    gliformer): each predicted head and tail is moved to the gold entity with the same span,
    else to the nearest gold entity with the same text (case-insensitive; concept-level sets
    keep one mention per concept); triples with an end that matches no gold entity are dropped."""
    ents = [(e["start"], e["end"]) for e in r["entities"]]
    exact = set(ents)

    def snap(s):
        if s in exact:
            return s
        txt = r["text"][s[0]:s[1]].casefold()
        same = [e for e in ents if r["text"][e[0]:e[1]].casefold() == txt]
        return min(same, key=lambda e: abs(e[0] - s[0])) if same else None
    out = set()
    for h, t, lab in pred:
        h, t = snap(tuple(h)), snap(tuple(t))
        if h and t and h != t:
            out.add((h, t, lab))
    return out


def fold(triples, symmetric):
    """Undirected labels (BioRED): (head, tail) and (tail, head) count as one relation."""
    return {(min(h, t), max(h, t), lab) if lab in symmetric else (h, t, lab) for h, t, lab in triples}


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gliner2", "gliformer", "glirel"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ood", default=str(ROOT / "data/ood_re"))
    ap.add_argument("--name_only", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--label_words", action="store_true", help="glirel: 'part-of' asked as 'part of'")
    ap.add_argument("--sanity", action="store_true", help="glirel: run the README example first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--given_filter", action="store_true",
                    help="every set: end-to-end answers kept only between gold entities (see to_gold_entities)")
    args = ap.parse_args()
    mode = "end2end"
    if args.backend == "glirel":
        # GLiREL classifies relations between given entities: the gold-entity setting.
        # CrossRE text is tokens joined by single spaces, so tokens map to exact offsets.
        from glirel import GLiREL
        model = GLiREL.from_pretrained(args.model).to("cuda")
        mode = "given"
        if args.sanity:      # GLiREL loads with strict=False: report keys that did not match
            import torch
            from huggingface_hub import hf_hub_download
            sd = torch.load(hf_hub_download(args.model, "pytorch_model.bin"), map_location="cpu")
            res = model.load_state_dict(sd, strict=False)
            print("missing", len(res.missing_keys), res.missing_keys[:8], "unexpected", len(res.unexpected_keys),
                  res.unexpected_keys[:8], flush=True)
        if args.sanity:      # the GLiREL README example (expected: characters, score 0.99)
            toks = ('Derren Nesbitt had a history of being cast in " Doctor Who " , having played villainous '
                    'warlord Tegana in the 1964 First Doctor serial " Marco Polo " .').split(" ")
            ex = model.predict_relations(toks, ["country of origin", "licensed to broadcast to", "father",
                                                "followed by", "characters"], threshold=0.0, top_k=1,
                                         ner=[[26, 27, "PERSON", "Marco Polo"], [22, 23, "Q2989412", "First Doctor"]])
            print("sanity:", json.dumps(ex, default=str)[:600], flush=True)

        def run(text, rels, ents, gold_ents):
            starts, pos = [], 0
            toks = text.split(" ")
            for t in toks:
                starts.append(pos)
                pos += len(t) + 1
            ends = [s + len(t) for s, t in zip(starts, toks)]
            tok_of_start = {s: k for k, s in enumerate(starts)}
            tok_of_end = {e: k for k, e in enumerate(ends)}
            ner, back = [], {}
            for e in gold_ents:
                i, j = tok_of_start.get(e["start"]), tok_of_end.get(e["end"])
                if i is None or j is None:
                    continue
                ner.append([i, j, e["label"], text[e["start"]:e["end"]]])
                back[(i, j)] = (e["start"], e["end"])
            if len(ner) < 2:
                return []
            # CrossRE names are hyphenated ("part-of"); --label_words asks GLiREL in words
            asked = {(lab.replace("-", " ") if args.label_words else lab): lab for lab in rels}
            out = model.predict_relations(toks, list(asked), threshold=args.threshold, ner=ner, top_k=1)
            res = []
            for o in out:
                # the output end index is exclusive (README: ner [26, 27] -> head_pos [26, 28])
                h = back.get((o["head_pos"][0], o["head_pos"][1] - 1))
                t = back.get((o["tail_pos"][0], o["tail_pos"][1] - 1))
                if h and t and h != t and o["label"] in asked:
                    res.append({"head": {"start": h[0], "end": h[1]}, "tail": {"start": t[0], "end": t[1]},
                                "relation": asked[o["label"]]})
            return res
    elif args.backend == "gliformer":
        from gliformer import GLiFormer
        model = GLiFormer.from_pretrained(args.model, load_tokenizer=True).to("cuda").eval()

        def run(text, rels, ents, gold_ents):
            # the released checkpoints have the joint NER + relation head only
            groups = {"schema": {"entities": list(ents), "relations": list(rels)}}
            return model.inference(text, joint_relations=groups, threshold=args.threshold)
    else:
        try:
            from gliner2 import AutoExtractor as Loader
        except ImportError:
            from gliner2 import GLiNER2 as Loader
        model = Loader.from_pretrained(args.model)
        try:
            model = model.to("cuda")
        except Exception:
            pass

        def run(text, rels, ents, gold_ents):
            spec = list(rels) if args.name_only else rels
            try:
                return model.extract_relations(text, spec, threshold=args.threshold, include_spans=True)
            except (TypeError, ValueError):
                return model.extract_relations(text, list(rels), threshold=args.threshold)
    if args.given_filter:
        mode = "given_filter"
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    res = {"model": args.model, "name_only": args.name_only, "sets": {}, "samples": []}
    for name, info in manifest["sets"].items():
        if mode not in info.get("settings", [mode]) and mode != "given_filter":
            continue
        tp = fp = fn = 0
        for line in open(Path(args.ood) / f"{name}.jsonl"):
            r = json.loads(line)
            sym = set(r.get("symmetric_labels", []))
            rels = {q["label"]: q["description"] for q in r["rel_queries"]}
            ans = run(r["text"], rels, [q["label"] for q in r["queries"]], r["entities"])
            if len(res["samples"]) < 4:
                res["samples"].append({"text": r["text"][:200], "answer": json.loads(json.dumps(ans, default=str))})
            pred = triples(ans, r["text"], set(rels))
            if mode == "given_filter":
                pred = to_gold_entities(pred, r)
            pred = fold(pred, sym)
            gold =fold({(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}, sym)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        res["sets"][name] = {mode: f1(tp, fp, fn)}
        print(name, round(res["sets"][name][mode]["f1"], 4), flush=True)
    res[f"macro_f1_{mode}"] = sum(v[mode]["f1"] for v in res["sets"].values()) / max(1, len(res["sets"]))
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({f"macro_f1_{mode}": res[f"macro_f1_{mode}"]}))


if __name__ == "__main__":
    main()
