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
from .evaluate import f1, f1_counts, metrics, predict, reliability
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = SpanLabelModel(args.encoder, len(tok), W=args.W).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))

    res = {"ckpt": args.ckpt}
    enc = {s: encode_split(Path(args.data) / f"{s}.jsonl", tok, args.W)
           for s in ("dev_calib", "test")}
    outs = {s: predict(model, e, collate, "cuda") for s, e in enc.items()}
    T = fit_temperature(outs["dev_calib"])
    thr_raw = best_threshold(enc["dev_calib"], outs["dev_calib"], 1.0)
    thr_cal = best_threshold(enc["dev_calib"], outs["dev_calib"], T)
    res["temperature"] = T
    for s in enc:
        res[s] = {
            "raw@0.5": metrics(enc[s], outs[s], 1.0, 0.5),
            "raw@tuned": metrics(enc[s], outs[s], 1.0, thr_raw),
            "calibrated@tuned": metrics(enc[s], outs[s], T, thr_cal),
        }
    zt = np.concatenate([o["z"][o["valid"]] for o in outs["test"]])
    yt = np.concatenate([o["y"][o["valid"]] for o in outs["test"]])
    for name, t in (("raw", 1.0), ("calibrated", T)):
        p = 1 / (1 + np.exp(-zt / t))
        res[f"test_reliability_{name}_all"] = reliability(p, yt)
        thr = thr_raw if name == "raw" else thr_cal
        res[f"test_reliability_{name}_predicted"] = reliability(p[p >= thr], yt[p >= thr])
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if "reliability" not in k}, indent=2))


if __name__ == "__main__":
    main()
