"""Fit temperature T and a global F1 threshold on dev_calib; report test (SPEC 6).

F1 depends only on the ranking at a tuned threshold, so raw@tuned and
calibrated@tuned F1 coincide; T changes NLL / Brier / ECE only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .collator import Collator, add_control_tokens
from .evaluate import f1, f1_counts, metrics, predict, reliability, rlcd_brier, with_temperature
from .mixdata import fewnerd_transform
from .model import SpanLabelModel
from .train import encode_split


def fit_temperature(outs):
    z = torch.from_numpy(np.concatenate([o["z"][o["valid"]] for o in outs])).double()
    y = torch.from_numpy(np.concatenate([o["y"][o["valid"]] for o in outs])).double()
    log_t = torch.zeros((), dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def best_threshold(encs, outs, T):
    grid = np.concatenate([np.linspace(0.02, 0.98, 49)])
    scores = [(f1(*f1_counts(encs, outs, T * float(np.log(t / (1 - t)))))["f1"], t) for t in grid]
    return max(scores)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepare_mix directory")
    ap.add_argument("--sources", default="finerweb,fewnerd,pile,massive")
    ap.add_argument("--limit", type=int, default=1000, help="rows per source and split")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="bce", choices=["bce", "rlcd"])
    ap.add_argument("--fewnerd_fine_only", action="store_true")
    ap.add_argument("--fewnerd_mode", default="both", choices=["both", "fine", "precedence"])
    ap.add_argument("--decode", default="label", choices=["label", "entity"])
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = SpanLabelModel(args.encoder, len(tok), W=args.W, head=args.head).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    sources = args.sources.split(",")
    rlcd = args.head == "rlcd"
    if args.fewnerd_fine_only:
        args.fewnerd_mode = "fine"
    fewnerd_tf = fewnerd_transform(args.data, args.fewnerd_mode)

    # T and thresholds are fit on dev_calib pooled over sources (equal rows per source)
    enc, outs = {}, {}
    for split in ("dev_calib", "test"):
        for src in sources:
            tf = fewnerd_tf if src == "fewnerd" else None
            e = encode_split(Path(args.data) / src / f"{split}.jsonl", tok, args.W, args.limit, tf)
            enc[(split, src)] = e
            outs[(split, src)] = predict(model, e, collate, "cuda", batch_size=16)
    pool_e = [x for src in sources for x in enc[("dev_calib", src)]]
    pool_o = [x for src in sources for x in outs[("dev_calib", src)]]

    def view(o, T):
        """(rows, divisor) for metrics: RLCD rescales utilities before the softmax."""
        return (with_temperature(o, T, args.decode), 1.0) if rlcd else (o, T)

    if rlcd:   # RLCD: temperature minimizing multiway Brier on dev_calib (the blog's recipe)
        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 57))
        T = float(min(grid, key=lambda t: rlcd_brier(pool_o, t)))
    else:
        T = fit_temperature(pool_o)
    thr_raw = best_threshold(pool_e, *view(pool_o, 1.0))
    thr_cal = best_threshold(pool_e, *view(pool_o, T))
    res = {"ckpt": args.ckpt, "temperature": T, "sources": sources, "limit": args.limit,
           "head": args.head, "fewnerd_fine_only": args.fewnerd_fine_only}

    def three(e, o):
        return {"raw@0.5": metrics(e, *view(o, 1.0), 0.5),
                "raw@tuned": metrics(e, *view(o, 1.0), thr_raw),
                "calibrated@tuned": metrics(e, *view(o, T), thr_cal)}
    res["dev_calib"] = three(pool_e, pool_o)       # keys read by ood_eval
    res["test"] = {src: three(enc[("test", src)], outs[("test", src)]) for src in sources}
    for key in ("raw@0.5", "raw@tuned", "calibrated@tuned"):
        res[f"test_macro_f1[{key}]"] = float(np.mean([res["test"][s][key]["f1"] for s in sources]))
    test_o = [x for src in sources for x in outs[("test", src)]]
    for name, t in (("raw", 1.0), ("calibrated", T)):
        rows, div = view(test_o, t)
        zt = np.concatenate([o["z"][o["valid"]] for o in rows]) / div
        yt = np.concatenate([o["y"][o["valid"]] for o in rows])
        p = 1 / (1 + np.exp(-zt))
        res[f"test_reliability_{name}_all"] = reliability(p, yt)
        thr = thr_raw if name == "raw" else thr_cal
        sel = p >= thr
        if rlcd:
            dz = np.concatenate([o["dz"][o["valid"]] for o in rows])
            sel = (1 / (1 + np.exp(-dz)) >= thr) & np.concatenate([o["choice"][o["valid"]] for o in rows])
        res[f"test_reliability_{name}_predicted"] = reliability(p[sel], yt[sel])
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("T", round(T, 3), "thr_raw", thr_raw, "thr_cal", thr_cal)
    for src in sources:
        print(src, {k: round(v["f1"], 4) for k, v in res["test"][src].items()})


if __name__ == "__main__":
    main()
