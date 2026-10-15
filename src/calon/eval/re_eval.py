"""Relation extraction evaluation on data/ood_re (CrossRE test domains).

Two settings, both strict (head span, tail span, relation label):
  given       ordered pairs of the gold entity spans; only the relation is predicted
  end2end     pairs of the entities our NER head predicts (entity decoding with the
              NER calibration's temperature and threshold)
The relation threshold on 1 - p(no relation) is chosen on REBEL dev_calib with gold
entities (training-source data only, zero-shot; --dev takes several files). Also: absent-
relation-query FPR, the share of relation types with no gold relation in a row that are
predicted there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from calon.data.roots import ROOT

from ..baselines.re_baselines import fold
from ..data.collator import Collator, add_control_tokens, encode_row, load_rows
from .evaluate import (
    NULL_SHIFT,
    best_re_threshold,
    decode,
    decode_relations,
    f1,
    fit_re_temperature,
    load_eval_model,
    predict,
    predict_re,
    re_temperature,
    set_re_temperature,
    with_temperature,
)

# REBEL dev_calib only: fitting on all relation sources (rebel, synth_cre, synth_gre) also
# chose 0.9 (once 0.95) and did not raise SciER / CrossRE recall, so the default stays
RE_DEV = [str(ROOT / "data/re/rebel/dev_calib.jsonl")]


def dev_rows(paths, limit=1000):
    """dev_calib rows of every relation source that exists, up to `limit` each."""
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += load_rows(p)[:limit]
    return rows


def encode_all(rows, tok, W, pair_spans=None):
    encs, kept = [], []
    for k, r in enumerate(rows):
        try:
            encs.append(encode_row(r, tok, W, None if pair_spans is None else pair_spans[k]))
            kept.append(r)
        except NotImplementedError:
            continue
    return encs, kept


def score(encs, outs, thr, symmetric=frozenset()):
    tp = fp = fn = absent = absent_fp = 0
    per, confusion = {}, {}
    for e, o in zip(encs, outs):
        pred = fold(decode_relations(e, o, thr), symmetric)
        gold = fold(e.get("gold_triples", set()), symmetric)
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
        gl, pl = {g[2] for g in gold}, {p[2] for p in pred}
        for lab, sup in zip(e["rel_labels"], e["rel_supervised"]):
            if sup and lab not in gl:
                absent += 1
                absent_fp += lab in pl
    m = f1(tp, fp, fn)
    m["absent_query_fpr"] = absent_fp / absent if absent else float("nan")
    m["gold"] = tp + fn
    m["rows_scored"] = len(encs)
    m["per_label"] = {k: dict(f1(*v), gold=v[0] + v[2]) for k, v in sorted(per.items())}
    m["confusion"] = dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:30])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", required=True, help="NER calibration.json (entity decoding)")
    ap.add_argument("--dev", nargs="+", default=RE_DEV, help="dev_calib files for the relation threshold")
    ap.add_argument("--ood", default=str(ROOT / "data/ood_re"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="", help="comma-separated subset of the manifest's sets (default: all)")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--head", default="rlcd", choices=["bce", "rlcd"], help="entity head (bce: ablation)")
    ap.add_argument("--name_only", action="store_true", help="drop all label and relation descriptions")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = load_eval_model(args.encoder, len(tok), args.W, args.head, args.ckpt, collate)
    cal = json.loads(Path(args.calibration).read_text())
    T, ner_thr = cal["temperature"], cal["dev_calib"]["calibrated@tuned"]["threshold"]
    dev_encs, _ = encode_all(dev_rows(args.dev), tok, args.W)
    dev_outs = predict_re(model, dev_encs, collate, "cuda")
    re_T = 1.0
    if NULL_SHIFT:         # corrected outputs: relation temperature on the same dev data, then the threshold
        re_T = fit_re_temperature(dev_encs, dev_outs)
        set_re_temperature(re_T)
        dev_outs = re_temperature(dev_outs, re_T)
    dev_f1, thr = best_re_threshold(dev_encs, dev_outs)
    res = {"ckpt": args.ckpt, "re_threshold": thr, "re_T": re_T, "null_shift": NULL_SHIFT,
           "rebel_dev_calib_f1": dev_f1, "ner_T": T,
           "ner_threshold": ner_thr, "name_only": args.name_only, "sets": {}}
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    for name, info in manifest["sets"].items():
        if args.sets and name not in args.sets.split(","):
            continue
        rows = load_rows(Path(args.ood) / f"{name}.jsonl")
        n_rows = len(rows)
        if args.name_only:
            for r in rows:
                for q in r["queries"] + r["rel_queries"]:
                    q["description"] = ""
        sym = frozenset(x for r in rows for x in r.get("symmetric_labels", []))
        encs, rows = encode_all(rows, tok, args.W)
        given = score(encs, predict_re(model, encs, collate, "cuda"), thr, sym)
        given["rows_total"] = n_rows
        res["sets"][name] = {"given": given}
        if "end2end" in info.get("settings", ["given", "end2end"]):
            # end to end: NER first, then relations over the predicted entity spans
            logit = float(np.log(ner_thr / (1 - ner_thr)))
            raw = predict(model, encs, collate, "cuda")
            if args.head == "rlcd":
                ner = with_temperature(raw, T, "entity")
                spans = [sorted({(s, t) for s, t, _ in decode(e, o["dz"], logit, o["choice"])})
                         for e, o in zip(encs, ner)]
            else:   # BCE: independent sigmoid per (type, span); logits divided by T
                spans = [sorted({(s, t) for s, t, _ in decode(e, o["z"], T * logit)}) for e, o in zip(encs, raw)]
            e2e_encs, _ = encode_all(rows, tok, args.W, spans)
            res["sets"][name]["end2end"] = score(e2e_encs, predict_re(model, e2e_encs, collate, "cuda"), thr, sym)
        print(name, {m: round(v["f1"], 4) for m, v in res["sets"][name].items()}, flush=True)
    for mode in ("given", "end2end"):
        res[f"macro_f1_{mode}"] = float(np.mean([v[mode]["f1"] for v in res["sets"].values() if mode in v]))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "sets"}))


if __name__ == "__main__":
    main()
