"""Zero-shot GLiNER2 baseline on the OOD test sets, scored like our models.

Runs in its own env (gliner2 package). Each row gets its domain's full label set,
with descriptions (default) or names only; predictions are the model's own
spans at its default threshold. Metrics: strict typed character-span F1 per set,
macro averages, and the absent-query false-positive rate.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def find_all(text, s):
    out, k = [], text.find(s)
    while k >= 0:
        out.append((k, k + len(s)))
        k = text.find(s, k + 1)
    return out


def spans_of(result, text):
    """Normalize GLiNER2 output to {(start, end, label)}."""
    ents = result.get("entities", result) if isinstance(result, dict) else {}
    out = set()
    for lab, items in (ents or {}).items():
        for it in items or []:
            if isinstance(it, dict) and "start" in it and "end" in it:
                out.add((int(it["start"]), int(it["end"]), lab))
            else:
                s = it["text"] if isinstance(it, dict) else str(it)
                out |= {(a, b, lab) for a, b in find_all(text, s)}
    return out


def _group(items):
    """[{label, start, end, ...}] -> {label: [items]}"""
    out = {}
    for d in items:
        out.setdefault(d["label"], []).append(d)
    return out


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name_only", action="store_true")
    ap.add_argument("--backend", default="gliner2", choices=["gliner2", "gliformer"])
    args = ap.parse_args()
    if args.backend == "gliformer":
        import torch
        from gliformer import GLiFormer
        model = GLiFormer.from_pretrained(args.model, load_tokenizer=True).to("cuda").eval()
        desc_ok = [None]

        def extract(text, labels):
            """GLiFormer returns [{text, label, start, end, score}]; descriptions are
            tried as a {label: description} mapping and, if refused, run() stops."""
            if isinstance(labels, dict):
                try:
                    out = model.predict_entities(text, labels, threshold=0.5)
                    desc_ok[0] = True
                except Exception as exc:
                    desc_ok[0] = False
                    raise SystemExit(f"descriptions not supported by gliformer: {exc!r}")
            else:
                with torch.no_grad():
                    out = model.predict_entities(text, labels, threshold=0.5)
            return {"entities": _group(out or [])}
    else:
        try:                                   # newer releases expose AutoExtractor
            from gliner2 import AutoExtractor as Loader
        except ImportError:
            from gliner2 import GLiNER2 as Loader
        model = Loader.from_pretrained(args.model)
        try:
            model = model.to("cuda")
        except Exception:
            pass

        def extract(text, labels):
            try:
                return model.extract_entities(text, labels, include_spans=True, include_confidence=True)
            except TypeError:
                return model.extract_entities(text, labels)
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    res = {"model": args.model, "name_only": args.name_only, "sets": {}}
    for name in manifest["sets"]:
        rows = read_jsonl(Path(args.ood) / f"{name}.jsonl")
        tp = fp = fn = absent_q = absent_fp = 0
        for r in rows:
            labels = ([q["label"] for q in r["queries"]] if args.name_only else
                      {q["label"]: q["description"] or q["label"] for q in r["queries"]})
            pred = spans_of(extract(r["text"], labels), r["text"])
            gold = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            for q in r["queries"]:
                if not any(g[2] == q["label"] for g in gold):
                    absent_q += 1
                    absent_fp += any(p[2] == q["label"] for p in pred)
        m = f1(tp, fp, fn)
        m["absent_query_fpr"] = absent_fp / absent_q if absent_q else float("nan")
        res["sets"][name] = m
        print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)
    sets = res["sets"]
    res["macro_f1_crossner"] = sum(v["f1"] for k, v in sets.items() if k.startswith("crossner")) / 5
    res["macro_f1_all"] = sum(v["f1"] for v in sets.values()) / len(sets)
    res["mean_absent_fpr"] = sum(v["absent_query_fpr"] for v in sets.values()) / len(sets)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "sets"}))


if __name__ == "__main__":
    main()
