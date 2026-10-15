"""GLiREL (zero-shot relation classification between given entities) on the relation evaluation sets, gold entities.

GLiREL takes word tokens, the entity spans as inclusive token ranges, and the relation names; it scores every ordered
pair of the given entities and keeps the best label per pair (top_k=1). Texts are split into words and punctuation,
with extra cuts at every gold entity boundary, so each entity maps to whole tokens. Predictions are kept with their
scores and F1 is computed for several thresholds in one pass; the threshold is chosen on a calibration file with gold
entities from the training side (REBEL dev_calib, as for CalOn's relation threshold), never on an evaluation set.
Scoring as in re_baselines: head span, tail span and label must match; symmetric labels are folded.

Run: python -m calon.baselines.glirel_eval --model jackboyla/glirel-large-v0 --sets <ood dirs> --out <json>
"""
import argparse
import json
import re
import time
from pathlib import Path

from calon.baselines.re_baselines import f1, fold

THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ["no relation"] with --no_relation: a pair whose best label it is gets no prediction
NO_RELATION: list[str] = []
# [True] with --related_only: only the gold entity pairs that hold a relation are classified
RELATED_ONLY: list[bool] = []
TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(text, cuts):
    """Word / punctuation tokens as (start, end) character spans, also cut at every position in `cuts`."""
    out = []
    for m in TOKEN.finditer(text):
        s, e = m.span()
        inner = sorted(c for c in cuts if s < c < e)
        out += list(zip([s] + inner, inner + [e]))
    return out


def word_label(lab):
    return lab.replace("-", " ").replace("_", " ")


def predict(model, r, stats, max_ents):
    text = r["text"]
    ents = {(e["start"], e["end"]) for e in r["entities"]}
    pairs = {(tuple(x["head"]), tuple(x["tail"])) for x in r["relations"]}
    if RELATED_ONLY:       # GLiREL's own setting: classify the ordered pairs known to hold a relation
        ents = {e for p in pairs for e in p}
    toks = tokenize(text, {c for se in ents for c in se})
    first = {s: k for k, (s, _) in enumerate(toks)}
    last = {e: k for k, (_, e) in enumerate(toks)}
    ner, back = [], {}
    for s, e in sorted(ents):
        i, j = first.get(s), last.get(e)
        if i is None or j is None or j < i:
            stats["entity_unmapped"] += 1
            continue
        ner.append([i, j, "entity", text[s:e]])
        back[(i, j)] = (s, e)
    stats["entities"] += len(ents)
    if len(ner) > max_ents:                      # keep the pair count bounded on long documents
        stats["entity_dropped_cap"] += len(ner) - max_ents
        ner = ner[:max_ents]
    if len(ner) < 2:
        return []
    asked = {word_label(q["label"]): q["label"] for q in r["rel_queries"]}
    words = [text[s:e] for s, e in toks]
    out = model.predict_relations(words, list(asked) + NO_RELATION, threshold=0.0, ner=ner, top_k=1)
    res = []
    for o in out:
        # output end index is exclusive (README: ner [26, 27] -> head_pos [26, 28])
        h = back.get((o["head_pos"][0], o["head_pos"][1] - 1))
        t = back.get((o["tail_pos"][0], o["tail_pos"][1] - 1))
        if h and t and h != t and o["label"] in asked:
            if not RELATED_ONLY or (h, t) in pairs:
                res.append((h, t, asked[o["label"]], float(o["score"])))
        elif not (h and t):
            stats["pred_unmapped"] += 1
    return res


def score_file(model, path, limit, max_ents):
    stats = {"entities": 0, "entity_unmapped": 0, "entity_dropped_cap": 0, "pred_unmapped": 0, "rows": 0}
    counts = {t: [0, 0, 0] for t in THRESHOLDS}
    t0 = time.time()
    with open(path) as f:
        for k, line in enumerate(f):
            if limit and k >= limit:
                break
            r = json.loads(line)
            stats["rows"] += 1
            sym = set(r.get("symmetric_labels") or [])
            preds = predict(model, r, stats, max_ents)
            gold = fold({(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}, sym)
            for t in THRESHOLDS:
                pred = fold({(h, tl, lab) for h, tl, lab, s in preds if s >= t}, sym)
                c = counts[t]
                c[0] += len(pred & gold)
                c[1] += len(pred - gold)
                c[2] += len(gold - pred)
    stats["seconds"] = round(time.time() - t0, 1)
    return {str(t): f1(*c) for t, c in counts.items()}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jackboyla/glirel-large-v0")
    ap.add_argument("--sets", nargs="+", required=True, help="evaluation dirs (manifest.json + <set>.jsonl)")
    ap.add_argument("--only", nargs="*", default=None, help="set names to keep")
    ap.add_argument("--calib", default=None, help="jsonl with gold entities to choose the threshold on")
    ap.add_argument("--limit", type=int, default=0, help="rows per file (smoke test)")
    ap.add_argument("--max_ents", type=int, default=60)
    ap.add_argument("--no_relation", action="store_true", help="also ask \"no relation\" (in GLiREL's training data)")
    ap.add_argument("--related_only", action="store_true", help="classify only gold pairs that hold a relation")
    ap.add_argument("--table_out", default="", help="chosen threshold, table format")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.no_relation:
        NO_RELATION.append("no relation")
    if a.related_only:
        RELATED_ONLY.append(True)
    import torch
    from glirel import GLiREL
    model = GLiREL.from_pretrained(a.model).to("cuda").eval()
    # the GLiREL README example: expected "characters" with score ~0.99 both ways
    toks = ('Derren Nesbitt had a history of being cast in " Doctor Who " , having played villainous warlord Tegana '
            'in the 1964 First Doctor serial " Marco Polo " .').split(" ")
    ex = model.predict_relations(toks, ["country of origin", "licensed to broadcast to", "father", "followed by",
                                        "characters"], threshold=0.0, top_k=1,
                                 ner=[[26, 27, "PERSON", "Marco Polo"], [22, 23, "Q2989412", "First Doctor"]])
    sanity = [(o["label"], round(float(o["score"]), 4)) for o in ex]
    print("sanity:", sanity, flush=True)
    res = {"model": a.model, "sanity": sanity, "sets": {}, "stats": {}}
    if a.calib:
        res["calib"], res["stats"]["calib"] = score_file(model, a.calib, a.limit, a.max_ents)
        best = max(THRESHOLDS[1:], key=lambda t: res["calib"][str(t)]["f1"])    # 0.0: related-only scoring
        res["threshold"] = best
        print("calib", {t: round(v["f1"], 4) for t, v in res["calib"].items()}, "-> threshold", best, flush=True)
    torch.cuda.reset_peak_memory_stats()
    for d in a.sets:
        manifest = json.loads((Path(d) / "manifest.json").read_text())
        for name, info in manifest["sets"].items():
            if "given" not in info.get("settings", ["given"]) or (a.only and name not in a.only):
                continue
            res["sets"][name], res["stats"][name] = score_file(model, Path(d) / f"{name}.jsonl", a.limit, a.max_ents)
            t = str(res.get("threshold", 0.5))
            print(name, "F1@%s" % t, round(res["sets"][name][t]["f1"], 4), res["stats"][name], flush=True)
    if a.table_out:            # the threshold chosen on the calibration file, as round_summary reads RE baselines
        t = str(res.get("threshold", 0.5))
        Path(a.table_out).write_text(json.dumps({"model": a.model, "name_only": True, "threshold": float(t),
                                                 "sets": {s: {"given": v[t]} for s, v in res["sets"].items()}},
                                                indent=1))
    res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if "threshold" in res:
        t = str(res["threshold"])
        vals = [v[t]["f1"] for v in res["sets"].values()]
        print(json.dumps({"threshold": res["threshold"], "macro_f1": sum(vals) / max(1, len(vals))}))


if __name__ == "__main__":
    main()
