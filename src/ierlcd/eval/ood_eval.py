"""Zero-shot OOD evaluation of one checkpoint.

T and the threshold come from that checkpoint's FiNERWeb calibration.json
(fit on FiNERWeb dev_calib); no OOD data is used to choose them.

--name_only drops label descriptions (diagnostic). Every setting also gets an
error breakdown of false positives and false negatives.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ..data.collator import Collator, add_control_tokens, encode_row, load_rows
from ..model.model import load_model
from .evaluate import decode, f1, metrics, predict, with_temperature


def per_query(encs, outs, T, threshold):
    """Per-label counts and absent-query false-positive rate at sigmoid(z/T) >= threshold."""
    thr_logit = T * float(np.log(threshold / (1 - threshold)))
    counts = collections.defaultdict(lambda: [0, 0, 0])
    absent_q = absent_fp = present_q = present_miss = 0
    for e, o in zip(encs, outs):
        pred = decode(e, o.get("dz", o["z"]), thr_logit, o.get("choice"))
        for m, lab in enumerate(e["labels"]):
            p = {x for x in pred if x[2] == lab}
            g = {x for x in e["gold_spans"] if x[2] == lab}
            c = counts[lab]
            c[0] += len(p & g)
            c[1] += len(p - g)
            c[2] += len(g - p)
            if not g:
                absent_q += 1
                absent_fp += bool(p)
            else:                              # label present: did the model output it at all?
                present_q += 1
                present_miss += not p
    tp, fp, fn = (sum(c[i] for c in counts.values()) for i in range(3))
    return counts, {"absent_queries": absent_q,
                    "absent_query_fpr": absent_fp / absent_q if absent_q else float("nan"),
                    "present_queries": present_q,
                    "present_query_fnr": present_miss / present_q if present_q else float("nan"),
                    "entity_fn": fn, "entity_miss_rate": fn / (tp + fn) if tp + fn else float("nan")}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def error_breakdown(encs, outs, T, threshold):
    """FP: wrong_type (exact gold span, other label; `extra_label` if the gold
    label was also predicted there), boundary (overlaps same-label gold),
    boundary_and_type (overlaps other-label gold), spurious (no gold overlap).
    FN: found_as_other_type, boundary (overlapping same-label prediction), missed."""
    thr_logit = T * float(np.log(threshold / (1 - threshold)))
    fp, fn = collections.Counter(), collections.Counter()
    for e, o in zip(encs, outs):
        pred = decode(e, o.get("dz", o["z"]), thr_logit, o.get("choice"))
        gold = e["gold_spans"]
        gold_spans = {(s, t) for s, t, _ in gold}
        pred_spans = {(s, t) for s, t, _ in pred}
        for s, t, lab in pred - gold:
            if (s, t) in gold_spans:
                fp["wrong_type"] += 1
                if any((s, t, g) in pred for (gs, gt, g) in gold if (gs, gt) == (s, t)):
                    fp["wrong_type_extra_label"] += 1
            elif any(overlaps((s, t), (gs, gt)) for gs, gt, g in gold if g == lab):
                fp["boundary"] += 1
            elif any(overlaps((s, t), (gs, gt)) for gs, gt, _ in gold):
                fp["boundary_and_type"] += 1
            else:
                fp["spurious"] += 1
        for s, t, lab in gold - pred:
            if (s, t) in pred_spans:
                fn["found_as_other_type"] += 1
            elif any(overlaps((s, t), (ps, pt)) for ps, pt, p in pred if p == lab):
                fn["boundary"] += 1
            else:
                fn["missed"] += 1
    return {"fp": dict(fp), "fn": dict(fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", required=True, help="dir written by prepare_ood")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", required=True, help="FiNERWeb calibration.json of this ckpt")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name_only", action="store_true")
    ap.add_argument("--head", default="bce", choices=["bce", "rlcd"])
    ap.add_argument("--decode", default="label", choices=["label", "entity"],
                    help="RLCD: threshold the chosen label's probability or 1 - p(none)")
    args = ap.parse_args()
    cal = json.loads(Path(args.calibration).read_text())
    T = cal["temperature"]
    settings = {"raw@0.5": (1.0, 0.5),
                "raw@fw_tuned": (1.0, cal["dev_calib"]["raw@tuned"]["threshold"]),
                "calibrated@fw_tuned": (T, cal["dev_calib"]["calibrated@tuned"]["threshold"])}
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())

    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = load_model(args.encoder, len(tok), args.W, args.head, args.ckpt)

    res = {"ckpt": args.ckpt, "settings": settings, "name_only": args.name_only, "sets": {}}
    for name, info in manifest["sets"].items():
        rows = load_rows(Path(args.ood) / f"{name}.jsonl")
        if args.name_only:
            for r in rows:
                for q in r["queries"]:
                    q["description"], q["description_origin"] = "", "name_only"
        encs = [encode_row(r, tok, args.W) for r in rows]
        outs = predict(model, encs, collate, "cuda")
        seen = set(info["seen_labels"])
        r = {}
        for key, (t, thr) in settings.items():
            # RLCD rescales utilities before its softmax; BCE divides logits by t
            o_t, t = ((with_temperature(outs, t, args.decode), 1.0) if args.head == "rlcd"
                      else (outs, t))
            m = metrics(encs, o_t, t, thr)
            counts, absent = per_query(encs, o_t, t, thr)
            per_label = {lab: f1(*c) for lab, c in counts.items()}
            m.update(absent)
            m["macro_label_f1"] = float(np.mean([v["f1"] for v in per_label.values()]))
            for group, labs in (("seen", seen), ("unseen", set(counts) - seen)):
                tot = np.sum([counts[l] for l in labs], 0) if labs else [0, 0, 0]
                m[f"{group}_labels_micro"] = f1(*tot)
            m["per_label"] = per_label
            m["errors"] = error_breakdown(encs, o_t, t, thr)
            r[key] = m
        res["sets"][name] = r
        print(name, json.dumps({k: {x: round(v[x], 4) for x in
                                    ("f1", "precision", "recall", "absent_query_fpr", "ece_all",
                                     "ece_predicted", "nll_decision")}
                                for k, v in r.items()}), flush=True)
    for key in settings:
        cross = [res["sets"][n][key]["f1"] for n in res["sets"] if n.startswith("crossner")]
        res[f"macro_f1_crossner[{key}]"] = float(np.mean(cross))
        res[f"macro_f1_all[{key}]"] = float(np.mean([v[key]["f1"] for v in res["sets"].values()]))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("macro")}, indent=2))


if __name__ == "__main__":
    main()
