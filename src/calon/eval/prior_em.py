"""Prior-shift adaptation of the entity and relation decisions per evaluation set (no test labels used).

The share of candidate spans that are entities (and of candidate pairs that hold a relation) differs between
the training sources and each evaluation set, so p(entity | span) learned on the former is off on the latter.
  source   on the training sources' dev_calib (as calibrate.py, entity decoding, temperature T from the run's
           calibration.json): a bias b on the entity logit minimising the binary log loss (bias-corrected
           calibration, Alexandari et al., ICML 2020), and the source prior pi_s = mean p(entity) after it.
           Relations: pi_s = mean p(relation) over the REBEL dev_calib pairs (no bias: re_eval fits no T).
  target   per evaluation set, from its texts only: EM (Saerens et al., 2002) for the target prior pi_t,
           pi_t <- mean sigmoid(s + logit(pi_t) - logit(pi_s)) over its candidates; the decision logit is
           shifted by logit(pi_t) - logit(pi_s); entities are decoded at the threshold refitted on dev_calib
           after the bias, relations at re_eval's.
Reported per set: F1 / P / R / ECE (predicted) at the fitted threshold without adaptation, with the bias only,
and with bias + EM; pi_s, pi_t. Entities: descriptions, entity decoding; relations: gold entities given.
Output: <run>-ent/prior_em.json and a compact .txt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from calon.data.roots import ROOT

from ..data.collator import Collator, add_control_tokens, encode_row, load_rows
from ..data.mixdata import fewnerd_transform
from ..train.train import encode_split
from .evaluate import load_eval_model, metrics, predict, predict_re, with_temperature
from .re_eval import RE_DEV, dev_rows, encode_all, score

NER_GROUPS = ["ood", "ood_pii", "ood_spy", "ood_x", "ood_x2"]
RE_GROUPS = ["ood_re", "ood_re_x", "ood_re_x2"]
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p) - np.log1p(-p)


def cand_logits(outs):
    """Entity-decision logit per candidate (dz is the same for every label of a candidate) and gold."""
    s, y = [], []
    for o in outs:
        if not o["dz"].size:
            continue
        ok = o["valid"].any(0)
        if o.get("unknown_row"):
            ok &= ((o["y"] > 0) & o["valid"]).any(0)
        s.append(o["dz"][0, ok])
        y.append(((o["y"] > 0) & o["valid"]).any(0)[ok])
    return np.concatenate(s), np.concatenate(y).astype(float)


def solve(f, lo=-20.0, hi=20.0, it=80):
    """Root of an increasing function by bisection."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def em_prior(s, pi_s, it=200, tol=1e-9):
    pi = pi_s
    for _ in range(it):
        new = float(np.mean(sig(s + logit(pi) - logit(pi_s))))
        if abs(new - pi) < tol:
            break
        pi = min(max(new, 1e-7), 1 - 1e-7)
    return pi


def shift_ner(outs, delta):
    return [dict(o, dz=o["dz"] + delta) for o in outs]


def shift_re(outs, delta):
    res = []
    for o in outs:
        p = o["p"].copy()
        if p.shape[1] and p.shape[0] > 1:
            pr = np.clip(1 - p[0], 1e-12, 1 - 1e-12)
            pr2 = sig(logit(pr) + delta)
            p[1:] = p[1:] * (pr2 / pr)[None]               # labels keep their relative odds
            p[0] = 1 - pr2
        res.append(dict(o, p=p))
    return res


def re_pair_probs(outs):
    return np.concatenate([1 - o["p"][0] for o in outs if o["p"].shape[1] and o["p"].shape[0] > 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs2/<exp>-s<seed>/<name>")
    ap.add_argument("--data", required=True, help="the run's mix (dev_calib of its sources)")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--ner-groups", default=",".join(NER_GROUPS))
    ap.add_argument("--re-groups", default=",".join(RE_GROUPS))
    a = ap.parse_args()
    run = ROOT / a.run
    ent = run.parent / f"{run.name}-ent"
    cal = json.loads((ent / "calibration.json").read_text())
    T, thr = cal["temperature"], cal["dev_calib"]["calibrated@tuned"]["threshold"]
    re_thr = json.loads((ent / "re-desc.json").read_text())["re_threshold"]
    tok = AutoTokenizer.from_pretrained(a.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = load_eval_model(a.encoder, len(tok), 32, "rlcd", str(run / "best.pt"), collate)
    tf = fewnerd_transform(a.data, "fine")
    # ---- source: bias and prior on dev_calib (entities), REBEL dev_calib (relations)
    dev = []
    for src in a.sources.split(","):
        p = Path(a.data) / src / "dev_calib.jsonl"
        if p.exists() and p.stat().st_size:
            e = encode_split(p, tok, 32, a.limit, tf if src == "fewnerd" else None)
            dev.append((e, with_temperature(predict(model, e, collate, "cuda", batch_size=16), T, "entity")))
    s_dev = np.concatenate([cand_logits(o)[0] for _, o in dev])
    y_dev = np.concatenate([cand_logits(o)[1] for _, o in dev])
    b = solve(lambda x: float(np.mean(sig(s_dev + x) - y_dev)))
    pi_s = float(np.mean(sig(s_dev + b)))
    # the bias moves every probability, so the F1 threshold is fitted again on dev_calib after it
    from ..train.calibrate import best_threshold
    pool_e = [x for e, _ in dev for x in e]
    pool_o = [x for _, o in dev for x in o]
    thr_b = float(best_threshold(pool_e, shift_ner(pool_o, b), 1.0))
    re_dev_encs, _ = encode_all(dev_rows(RE_DEV), tok, 32)
    pi_s_re = float(np.mean(re_pair_probs(predict_re(model, re_dev_encs, collate, "cuda"))))
    res = {"run": a.run, "T": T, "threshold": thr, "bias": b, "threshold_after_bias": thr_b, "pi_s": pi_s,
           "dev_entity_rate": float(y_dev.mean()),
           "re_threshold": re_thr, "pi_s_re": pi_s_re, "ner": {}, "re": {}}
    lines = [f"{a.run}: T {T:.3f} threshold {thr:.2f} bias {b:+.3f} (threshold after it {thr_b:.2f}) "
             f"pi_s {pi_s:.5f} (dev entity rate {y_dev.mean():.5f});"
             f" relations: threshold {re_thr:.2f} pi_s {pi_s_re:.4f}"]
    # ---- entities
    for g in [x for x in a.ner_groups.split(",") if x]:
        man = json.loads((ROOT / "data" / g / "manifest.json").read_text())
        for name in man["sets"]:
            rows = load_rows(ROOT / "data" / g / f"{name}.jsonl")
            encs = [encode_row(r, tok, 32) for r in rows]
            outs = with_temperature(predict(model, encs, collate, "cuda", batch_size=16), T, "entity")
            s = np.concatenate([o["dz"][0, o["valid"].any(0)] for o in outs if o["dz"].size]) + b
            pi_t = em_prior(s, pi_s)
            delta = logit(pi_t) - logit(pi_s)
            r = {}
            for key, o2, t in (("fitted", outs, thr), ("bias", shift_ner(outs, b), thr_b),
                               ("bias+em", shift_ner(outs, b + delta), thr_b)):
                m = metrics(encs, o2, 1.0, t)
                r[key] = {k: m[k] for k in ("f1", "precision", "recall", "ece_predicted")}
            r.update(pi_t=pi_t, delta=delta)
            res["ner"][f"{g}/{name}"] = r
            lines.append(f"  NER {g + '/' + name:28s} pi_t {pi_t:.5f} | F1 fitted {r['fitted']['f1']:.3f} bias "
                         f"{r['bias']['f1']:.3f} bias+em {r['bias+em']['f1']:.3f} | P {r['fitted']['precision']:.3f}->"
                         f"{r['bias+em']['precision']:.3f} R {r['fitted']['recall']:.3f}->{r['bias+em']['recall']:.3f}"
                         f" | ECEpred {r['fitted']['ece_predicted']:.3f}->{r['bias+em']['ece_predicted']:.3f}")
    # ---- relations (gold entities given)
    for g in [x for x in a.re_groups.split(",") if x]:
        man = json.loads((ROOT / "data" / g / "manifest.json").read_text())
        for name in man["sets"]:
            rows = load_rows(ROOT / "data" / g / f"{name}.jsonl")
            sym = frozenset(x for row in rows for x in row.get("symmetric_labels", []))
            encs, _ = encode_all(rows, tok, 32)
            outs = predict_re(model, encs, collate, "cuda")
            pr = re_pair_probs(outs)
            if not pr.size:
                continue
            pi_t = em_prior(logit(np.clip(pr, 1e-12, 1 - 1e-12)), pi_s_re)
            delta = logit(pi_t) - logit(pi_s_re)
            fit, em = score(encs, outs, re_thr, sym), score(encs, shift_re(outs, delta), re_thr, sym)
            res["re"][f"{g}/{name}"] = {"fitted": {k: fit[k] for k in ("f1", "precision", "recall")},
                                        "em": {k: em[k] for k in ("f1", "precision", "recall")},
                                        "pi_t": pi_t, "delta": delta}
            lines.append(f"  RE  {g + '/' + name:28s} pi_t {pi_t:.4f} | F1 fitted {fit['f1']:.3f} em {em['f1']:.3f} |"
                         f" P {fit['precision']:.3f}->{em['precision']:.3f} R {fit['recall']:.3f}->{em['recall']:.3f}")
    for kind, key in (("ner", "bias+em"), ("re", "em")):
        if res[kind]:
            f0 = np.mean([v["fitted"]["f1"] for v in res[kind].values()])
            f1v = np.mean([v[key]["f1"] for v in res[kind].values()])
            res[f"{kind}_macro_f1"] = {"fitted": f0, key: f1v}
            lines.append(f"  {kind.upper()} macro F1 over {len(res[kind])} sets: fitted {f0:.3f} -> {key} {f1v:.3f}")
    (ent / "prior_em.json").write_text(json.dumps(res, indent=1))
    (ent / "prior_em.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
