"""Zero-shot OOD evaluation of one checkpoint.

T and the threshold come from that checkpoint's FiNERWeb calibration.json
(fit on FiNERWeb dev_calib); no OOD data is used to choose them.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .collator import Collator, add_control_tokens
from .evaluate import decode, f1, metrics, predict
from .model import SpanLabelModel
from .train import encode_split


def per_query(encs, outs, T, threshold):
    """Per-label counts and absent-query false-positive rate at sigmoid(z/T) >= threshold."""
    thr_logit = T * float(np.log(threshold / (1 - threshold)))
    counts = collections.defaultdict(lambda: [0, 0, 0])
    absent_q = absent_fp = 0
    for e, o in zip(encs, outs):
        pred = decode(e, o["z"], thr_logit)
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
    return counts, {"absent_queries": absent_q,
                    "absent_query_fpr": absent_fp / absent_q if absent_q else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", required=True, help="dir written by prepare_ood")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", required=True, help="FiNERWeb calibration.json of this ckpt")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--out", required=True)
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
    model = SpanLabelModel(args.encoder, len(tok), W=args.W).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))

    res = {"ckpt": args.ckpt, "settings": settings, "sets": {}}
    for name, info in manifest["sets"].items():
        encs = encode_split(Path(args.ood) / f"{name}.jsonl", tok, args.W)
        outs = predict(model, encs, collate, "cuda")
        seen = set(info["seen_labels"])
        r = {}
        for key, (t, thr) in settings.items():
            m = metrics(encs, outs, t, thr)
            counts, absent = per_query(encs, outs, t, thr)
            per_label = {lab: f1(*c) for lab, c in counts.items()}
            m.update(absent)
            m["macro_label_f1"] = float(np.mean([v["f1"] for v in per_label.values()]))
            for group, labs in (("seen", seen), ("unseen", set(counts) - seen)):
                tot = np.sum([counts[l] for l in labs], 0) if labs else [0, 0, 0]
                m[f"{group}_labels_micro"] = f1(*tot)
            m["per_label"] = per_label
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
