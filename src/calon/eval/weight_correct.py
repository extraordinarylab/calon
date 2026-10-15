"""Undo the "none" down-weighting of the training loss analytically (no test labels used).

With "none" spans weighted w (none_weight, re_none_weight) in the loss, its minimiser reports the odds of
"none" against the labels w times too low (He et al., 2014: q = p / (p + (1 - p) / w) after negative
down-sampling). The correction adds log(1 / w) to the "none" utility before the softmax; the labels keep their
relative odds. Variants per evaluation set (entities: entity decoding, descriptions):
  fitted        the run's calibration.json temperature T and threshold (as reported)
  corrected     utilities corrected, T = 1, threshold refitted on dev_calib
  corrected+T   utilities corrected, then T refitted on dev_calib (minimum multiway Brier, calibrate.py's grid)
                and the threshold refitted
Relations (gold entities given): p(relation) corrected the same way, threshold refitted on REBEL dev_calib.
--fit fits the dev_calib settings once (weight_correct.fit.json); --shard i/n then evaluates every n-th set
into weight_correct.part<i>.json;
--merge n joins the parts into <run>-ent/weight_correct.json and a compact .txt.

Relation calibration (--fit-re, then --shard / --merge with --tag _re --ner-groups ""): the same three variants for
the relation choice of every candidate pair of gold entities, with T fitted on REBEL dev_calib (minimum multiway
Brier over the same grid, after the correction) and thresholds refitted there; per set F1, ECE over predicted pairs
(15 bins; confidence = probability of the chosen relation, hit = it is a gold relation of the pair) and the multiway
Brier score over all pairs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calon.data.roots import ROOT

from .evaluate import re_temperature
from .prior_em import NER_GROUPS, RE_GROUPS, shift_re

KEYS = ("f1", "precision", "recall", "ece_predicted", "ece_all", "brier_multiway")


def bins(outs, thr, nb=15):
    """Reliability-diagram sums over the predicted entities (as metrics' ece_predicted: the chosen label's
    probability against whether that label is gold): per bin the count, summed confidence, summed hits."""
    v = [o for o in outs if o["valid"].any()]
    z = np.concatenate([o["z"][o["valid"]] for o in v])
    dz = np.concatenate([o["dz"][o["valid"]] for o in v])
    ch = np.concatenate([o["choice"][o["valid"]] for o in v])
    y = np.concatenate([o["y"][o["valid"]] for o in v]) > 0
    pred = (dz >= np.log(thr / (1 - thr))) & ch
    p, t = 1 / (1 + np.exp(-z[pred])), y[pred]
    idx = np.clip(np.digitize(p, np.linspace(0, 1, nb + 1)[1:-1]), 0, nb - 1)
    return {"n": np.bincount(idx, minlength=nb).tolist(), "conf": np.bincount(idx, p, nb).tolist(),
            "hit": np.bincount(idx, t.astype(float), nb).tolist()}


def re_calib(encs, outs, thr, sym=frozenset(), nb=15):
    """F1 at thr, ECE over predicted pairs and multiway Brier over all candidate pairs, plus reliability bins."""
    from ..baselines.re_baselines import fold
    from .evaluate import decode_relations, f1
    tp = fp = fn = 0
    conf, hit, brier = [], [], []
    for e, o in zip(encs, outs):
        p = o["p"]
        gold = fold(e.get("gold_triples", set()), sym)
        if p.shape[1] and p.shape[0] > 1:
            labels = e["rel_labels"]
            best = p[1:].argmax(0)
            for k, (h, t) in enumerate(e["pair_char"]):
                g = {lab for lab in labels if fold({(h, t, lab)}, sym) & gold}
                y = np.zeros(p.shape[0])
                if g:
                    for lab in g:
                        y[1 + labels.index(lab)] = 1.0 / len(g)
                else:
                    y[0] = 1.0
                brier.append(float(((p[:, k] - y) ** 2).sum()))
                if 1.0 - p[0, k] >= thr:
                    conf.append(float(p[1 + best[k], k]))
                    hit.append(labels[best[k]] in g)
        pred = fold(decode_relations(e, o, thr), sym)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    m = f1(tp, fp, fn)
    c, h = np.array(conf), np.array(hit, float)
    idx = np.clip(np.digitize(c, np.linspace(0, 1, nb + 1)[1:-1]), 0, nb - 1) if c.size else np.zeros(0, int)
    n, sc, sh = np.bincount(idx, minlength=nb), np.bincount(idx, c, nb), np.bincount(idx, h, nb)
    m["ece_predicted"] = float(np.abs(sc - sh).sum() / max(1, c.size)) if c.size else float("nan")
    m["brier_multiway"] = float(np.mean(brier)) if brier else float("nan")
    m["n_pairs"], m["n_predicted"] = len(brier), int(c.size)
    m["bins"] = {"n": n.tolist(), "conf": sc.tolist(), "hit": sh.tolist()}
    return m


def correct(outs, w):
    res = []
    for o in outs:
        u = o["u"].copy()
        if u.size:
            u[0] += np.log(1 / w)
        res.append(dict(o, u=u))
    return res


def load(a, run):
    from transformers import AutoTokenizer

    from ..data.collator import Collator, add_control_tokens
    from .evaluate import load_eval_model

    tok = AutoTokenizer.from_pretrained(a.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    return tok, collate, load_eval_model(a.encoder, len(tok), 32, "rlcd", str(run / "best.pt"), collate)


def fit(a, ent):
    """dev_calib: temperature and thresholds of the three variants -> weight_correct.fit.json."""
    from ..data.mixdata import fewnerd_transform
    from ..train.calibrate import best_threshold
    from ..train.train import encode_split
    from .evaluate import best_re_threshold, metrics, predict, predict_re, rlcd_brier, with_temperature
    from .re_eval import RE_DEV, dev_rows, encode_all

    run = ROOT / a.run
    args = json.loads((run / "args.json").read_text())
    w, w_re = args["none_weight"], args["re_none_weight"]
    cal = json.loads((ent / "calibration.json").read_text())
    T0, thr0 = cal["temperature"], cal["dev_calib"]["calibrated@tuned"]["threshold"]
    re_thr0 = json.loads((ent / "re-desc.json").read_text())["re_threshold"]
    tok, collate, model = load(a, run)
    tf = fewnerd_transform(a.data, "fine")
    pool_e, pool_o = [], []
    for src in a.sources.split(","):
        p = Path(a.data) / src / "dev_calib.jsonl"
        if p.exists() and p.stat().st_size:
            e = encode_split(p, tok, 32, a.limit, tf if src == "fewnerd" else None)
            pool_e += e
            pool_o += predict(model, e, collate, "cuda", batch_size=16)
    pool_c = correct(pool_o, w)
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 57))
    T1 = float(min(grid, key=lambda t: rlcd_brier(pool_c, t)))
    dev = {}
    for k, (o, T, thr) in {"fitted": (pool_o, T0, thr0), "corrected": (pool_c, 1.0, None),
                           "corrected+T": (pool_c, T1, None)}.items():
        v = with_temperature(o, T, "entity")
        thr = thr if thr is not None else float(best_threshold(pool_e, v, 1.0))
        m = metrics(pool_e, v, 1.0, thr)
        dev[k] = {"T": T, "threshold": thr, **{x: m[x] for x in KEYS}}
    re_dev, _ = encode_all(dev_rows(RE_DEV), tok, 32)
    _, re_thr1 = best_re_threshold(re_dev, shift_re(predict_re(model, re_dev, collate, "cuda"), np.log(w_re)))
    res = {"run": a.run, "none_weight": w, "re_none_weight": w_re, "dev_calib": dev,
           "re_threshold": {"fitted": re_thr0, "corrected": re_thr1}, "ner": {}, "re": {}}
    (ent / "weight_correct.fit.json").write_text(json.dumps(res, indent=1))


def fit_re(a, ent):
    """REBEL dev_calib: relation temperature (after the correction) and thresholds -> "re_calib" in the fit file."""
    from .evaluate import best_re_threshold, predict_re
    from .re_eval import RE_DEV, dev_rows, encode_all

    path = ent / "weight_correct.fit.json"
    res = json.loads(path.read_text())
    w_re = res["re_none_weight"]
    tok, collate, model = load(a, ROOT / a.run)
    encs, _ = encode_all(dev_rows(RE_DEV), tok, 32)
    raw = predict_re(model, encs, collate, "cuda")
    cor = shift_re(raw, np.log(w_re))
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 57))
    T1 = float(min(grid, key=lambda t: re_calib(encs, re_temperature(cor, t), 2.0)["brier_multiway"]))
    _, thr_t = best_re_threshold(encs, re_temperature(cor, T1))
    rt = res["re_threshold"]
    variants = {"fitted": (raw, 1.0, rt["fitted"]), "corrected": (cor, 1.0, rt["corrected"]),
                "corrected+T": (cor, T1, thr_t)}
    res["re_calib"] = {}
    for k, (o, T, thr) in variants.items():
        m = re_calib(encs, re_temperature(o, T), thr)
        m.pop("bins")
        res["re_calib"][k] = {"T": T, "threshold": thr, **m}
        print("REBEL dev_calib", k, json.dumps(res["re_calib"][k]), flush=True)
    path.write_text(json.dumps(res, indent=1))


def compute(a, ent, i, n):
    from ..data.collator import encode_row, load_rows
    from .evaluate import metrics, predict, predict_re, with_temperature
    from .re_eval import encode_all, score

    run = ROOT / a.run
    res = json.loads((ent / "weight_correct.fit.json").read_text())
    w, w_re = res["none_weight"], res["re_none_weight"]
    setting = {k: (d["T"], d["threshold"]) for k, d in res["dev_calib"].items()}
    re_thr0, re_thr1 = res["re_threshold"]["fitted"], res["re_threshold"]["corrected"]
    tok, collate, model = load(a, run)
    tasks = [("ner", g, s) for g in a.ner_groups.split(",") if g
             for s in json.loads((ROOT / "data" / g / "manifest.json").read_text())["sets"]]
    tasks += [("re", g, s) for g in a.re_groups.split(",") if g
              for s in json.loads((ROOT / "data" / g / "manifest.json").read_text())["sets"]]
    for kind, g, name in tasks[i::n]:
        rows = load_rows(ROOT / "data" / g / f"{name}.jsonl")
        if kind == "ner":
            encs = [encode_row(r, tok, 32) for r in rows]
            raw = predict(model, encs, collate, "cuda", batch_size=16)
            r = {}
            for k, (T, thr) in setting.items():
                o = raw if k == "fitted" else correct(raw, w)
                v = with_temperature(o, T, "entity")
                m = metrics(encs, v, 1.0, thr)
                r[k] = {**{x: m[x] for x in KEYS}, "bins": bins(v, thr)}
            res["ner"][f"{g}/{name}"] = r
        else:
            sym = frozenset(x for row in rows for x in row.get("symmetric_labels", []))
            encs, _ = encode_all(rows, tok, 32)
            outs = predict_re(model, encs, collate, "cuda")
            if "re_calib" in res:
                cor_outs = shift_re(outs, np.log(w_re))
                res["re"][f"{g}/{name}"] = {
                    k: re_calib(encs, re_temperature(outs if k == "fitted" else cor_outs, d["T"]), d["threshold"], sym)
                    for k, d in res["re_calib"].items()}
                continue
            fit = score(encs, outs, re_thr0, sym)
            cor = score(encs, shift_re(outs, np.log(w_re)), re_thr1, sym)
            res["re"][f"{g}/{name}"] = {"fitted": {x: fit[x] for x in ("f1", "precision", "recall")},
                                        "corrected": {x: cor[x] for x in ("f1", "precision", "recall")}}
    return res


def summarize(res):
    w, w_re = res["none_weight"], res["re_none_weight"]
    lines = [f"{res['run']}: none_weight {w} (correction {np.log(1 / w):+.3f} on the none utility); "
             f"re_none_weight {w_re}"]
    for k, d in res["dev_calib"].items():
        lines.append(f"  dev_calib {k:12s} T {d['T']:.3f} thr {d['threshold']:.2f} | F1 {d['f1']:.3f} "
                     f"P {d['precision']:.3f}"
                     f" R {d['recall']:.3f} | ECEpred {d['ece_predicted']:.3f} ECEall {d['ece_all']:.4f}"
                     f" Brier {d['brier_multiway']:.4f}")
    rt = res["re_threshold"]
    lines.append(f"  relations: threshold fitted {rt['fitted']:.2f} -> corrected {rt['corrected']:.2f}")
    lines.append("  NER columns: fitted / corrected / corrected+T")
    for key, r in sorted(res["ner"].items()):
        f, c, ct = r["fitted"], r["corrected"], r["corrected+T"]
        lines.append(f"  NER {key:28s} F1 {f['f1']:.3f} {c['f1']:.3f} {ct['f1']:.3f} | P {f['precision']:.3f}"
                     f" {c['precision']:.3f} {ct['precision']:.3f} | ECEpred {f['ece_predicted']:.3f}"
                     f" {c['ece_predicted']:.3f} {ct['ece_predicted']:.3f}")
    if "re_calib" in res:
        lines.append("  RE calibration (REBEL dev_calib): " + "; ".join(
            f"{k} T {d['T']:.3f} thr {d['threshold']:.2f} F1 {d['f1']:.3f} ECEpred {d['ece_predicted']:.3f}"
            f" Brier {d['brier_multiway']:.4f}" for k, d in res["re_calib"].items()))
        lines.append("  RE columns: fitted / corrected / corrected+T")
        for key, r in sorted(res["re"].items()):
            f, c, ct = r["fitted"], r["corrected"], r["corrected+T"]
            lines.append(f"  RE  {key:28s} F1 {f['f1']:.3f} {c['f1']:.3f} {ct['f1']:.3f} | ECEpred "
                         f"{f['ece_predicted']:.3f} {c['ece_predicted']:.3f} {ct['ece_predicted']:.3f} | Brier "
                         f"{f['brier_multiway']:.4f} {c['brier_multiway']:.4f} {ct['brier_multiway']:.4f}")
        for k in res["re_calib"]:
            mm = res[f"re_macro[{k}]"] = {x: float(np.nanmean([v[k][x] for v in res["re"].values()]))
                                          for x in ("f1", "ece_predicted", "brier_multiway")}
            lines.append(f"  RE mean over {len(res['re'])} sets, {k:12s}: F1 {mm['f1']:.3f} ECEpred "
                         f"{mm['ece_predicted']:.3f} Brier {mm['brier_multiway']:.4f}")
        return lines
    for key, r in sorted(res["re"].items()):
        fit, cor = r["fitted"], r["corrected"]
        lines.append(f"  RE  {key:28s} F1 {fit['f1']:.3f} {cor['f1']:.3f} | P {fit['precision']:.3f}"
                     f" {cor['precision']:.3f} R {fit['recall']:.3f} {cor['recall']:.3f}")
    ner = list(res["ner"].values())
    for k in res["dev_calib"]:
        mm = res[f"ner_macro[{k}]"] = {x: float(np.nanmean([v[k][x] for v in ner])) for x in KEYS}
        lines.append(f"  NER mean over {len(ner)} sets, {k:12s}: F1 {mm['f1']:.3f} P {mm['precision']:.3f} "
                     f"R {mm['recall']:.3f}"
                     f" ECEpred {mm['ece_predicted']:.3f} ECEall {mm['ece_all']:.4f}")
    if res["re"]:
        for k in ("fitted", "corrected"):
            res[f"re_macro_f1[{k}]"] = float(np.mean([v[k]["f1"] for v in res["re"].values()]))
        lines.append(f"  RE mean F1 over {len(res['re'])} sets: fitted {res['re_macro_f1[fitted]']:.3f} corrected"
                     f" {res['re_macro_f1[corrected]']:.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs2/<exp>-s<seed>/<name>")
    ap.add_argument("--data", help="the run's mix (dev_calib of its sources)")
    ap.add_argument("--sources")
    ap.add_argument("--encoder")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--ner-groups", default=",".join(NER_GROUPS))
    ap.add_argument("--re-groups", default=",".join(RE_GROUPS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", type=int, default=0, help="join this many parts")
    ap.add_argument("--fit", action="store_true", help="dev_calib only: write weight_correct.fit.json")
    ap.add_argument("--fit-re", action="store_true", help="relation T, thresholds")
    ap.add_argument("--tag", default="", help="suffix of the part / merged files, e.g. _re")
    a = ap.parse_args()
    run = ROOT / a.run
    ent = run.parent / f"{run.name}-ent"
    if a.merge:
        parts = [json.loads((ent / f"weight_correct{a.tag}.part{i}.json").read_text()) for i in range(a.merge)]
        res = parts[0]
        for p in parts[1:]:
            res["ner"].update(p["ner"])
            res["re"].update(p["re"])
        lines = summarize(res)
        (ent / f"weight_correct{a.tag}.json").write_text(json.dumps(res, indent=1))
        (ent / f"weight_correct{a.tag}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    if a.fit_re:
        fit_re(a, ent)
        return
    if a.fit:
        fit(a, ent)
        return
    i, n = map(int, a.shard.split("/"))
    res = compute(a, ent, i, n)
    (ent / f"weight_correct{a.tag}.part{i}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
