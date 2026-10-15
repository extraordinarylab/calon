"""Event extraction evaluation on data/ood_ee (CASIE, WikiEvents test), zero-shot with the
joint NER + RE model: no event data in training.

Events are read with the two heads the model has:
  triggers    the entity head with the event types as labels (entity decoding, NER
              calibration temperature and threshold);
  arguments   the entity head with the role names as labels proposes argument spans,
              then the relation head scores ordered pairs (trigger -> span) with the
              roles as relation types; a pair keeps its most likely role among the
              roles of the trigger's event type if 1 - p(no relation) clears the
              relation threshold (REBEL dev_calib, as re_eval).
Settings: given (gold triggers and types, arguments predicted) and end2end (triggers
predicted, then arguments). Rows ask the roles of the event types they contain (given)
or were predicted (end2end).
Scores (strict offsets): trigger identification / classification (end2end), argument
identification (trigger span, event type, argument span) and classification (+ role);
also the absent-event-type FPR (queried types without a gold event that are predicted).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from calon.data.roots import ROOT

from ..data.collator import Collator, add_control_tokens, load_rows
from .evaluate import (
    NULL_SHIFT,
    best_re_threshold,
    decode,
    ece,
    f1,
    fit_re_temperature,
    load_eval_model,
    predict,
    predict_re,
    re_temperature,
    set_re_temperature,
    with_temperature,
)
from .re_eval import RE_DEV, dev_rows, encode_all


def gold_sets(r):
    trig = {(e["trigger"][0], e["trigger"][1], e["type"]) for e in r["events"]}
    args = {(e["trigger"][0], e["trigger"][1], e["type"], a["span"][0], a["span"][1], a["role"])
            for e in r["events"] for a in e["args"]}
    return trig, args


def head_match(gold, pred, with_role):
    """One-to-one matches where the gold argument span lies inside the predicted one (same
    trigger, event type and, if with_role, role): WikiEvents annotates head words."""
    used, tp = set(), 0
    for g in sorted(gold):
        for p in sorted(pred):
            if p in used or p[:3] != g[:3] or (with_role and p[5] != g[5]):
                continue
            if p[3] <= g[3] and g[4] <= p[4]:
                used.add(p)
                tp += 1
                break
    return tp


def score(rows, preds, event_types):
    """preds: per row (triggers {(s, e, type)}, args {(ts, te, type, as, ae, role)})."""
    c = {k: [0, 0, 0] for k in ("TI", "TC", "TC_head", "AI", "AC", "AI_head", "AC_head")}
    absent = absent_fp = 0
    for r, (pt, pa) in zip(rows, preds):
        gt, ga = gold_sets(r)
        pairs = {"TI": ({x[:2] for x in gt}, {x[:2] for x in pt}), "TC": (gt, pt),
                 "AI": ({x[:5] for x in ga}, {x[:5] for x in pa}), "AC": (ga, pa)}
        for k, (g, p) in pairs.items():
            c[k][0] += len(g & p)
            c[k][1] += len(p - g)
            c[k][2] += len(g - p)
        # triggers: the gold trigger word inside a predicted trigger phrase of the same type
        g = {(0, 0, t, s, e, "") for s, e, t in gt}
        p = {(0, 0, t, s, e, "") for s, e, t in pt}
        tp = head_match(g, p, False)
        c["TC_head"][0] += tp
        c["TC_head"][1] += len(p) - tp
        c["TC_head"][2] += len(g) - tp
        for k, role in (("AI_head", False), ("AC_head", True)):
            g = ga if role else {x[:5] + ("",) for x in ga}
            p = pa if role else {x[:5] + ("",) for x in pa}
            tp = head_match(g, p, role)
            c[k][0] += tp
            c[k][1] += len(p) - tp
            c[k][2] += len(g) - tp
        gtypes, ptypes = {x[2] for x in gt}, {x[2] for x in pt}
        for t in event_types:
            if t not in gtypes:
                absent += 1
                absent_fp += t in ptypes
    out = {k: dict(f1(*v), gold=v[0] + v[2]) for k, v in c.items()}
    for k in out:
        out[k]["miss_rate"] = 1 - out[k]["recall"]
    out["absent_event_type_fpr"] = absent_fp / absent if absent else float("nan")
    return out


def conf_ece(confs, golds, nb=15):
    """ECE (15 bins) of predicted items: confidence of the chosen type or role against whether the item is gold,
    plus the reliability-diagram sums."""
    c = np.array([p for d in confs for p in d.values()], float)
    y = np.array([x in g for d, g in zip(confs, golds) for x in d], float)
    idx = np.clip(np.digitize(c, np.linspace(0, 1, nb + 1)[1:-1]), 0, nb - 1) if c.size else np.zeros(0, int)
    return {"ece": ece(c, y, nb), "n": int(c.size), "accuracy": float(y.mean()) if y.size else float("nan"),
            "bins": {"n": np.bincount(idx, minlength=nb).tolist(), "conf": np.bincount(idx, c, nb).tolist(),
                     "hit": np.bincount(idx, y, nb).tolist()}}


def q(label, desc, positive):
    return {"label": label, "description": desc, "status": "positive" if positive else "scope_assumed_absent"}


class Reader:
    def __init__(self, args):
        self.tok = AutoTokenizer.from_pretrained(args.encoder)
        lab_id, txt_id = add_control_tokens(self.tok)
        self.collate = Collator(self.tok, lab_id, txt_id)
        self.head = getattr(args, "head", "rlcd")
        self.model = load_eval_model(args.encoder, len(self.tok), args.W, self.head, args.ckpt, self.collate)
        cal = json.loads(Path(args.calibration).read_text())
        self.T = cal["temperature"]
        thr = cal["dev_calib"]["calibrated@tuned"]["threshold"]
        self.logit = float(np.log(thr / (1 - thr)))
        self.W = args.W
        dev, _ = encode_all(dev_rows(args.dev), self.tok, self.W)
        dev_outs = predict_re(self.model, dev, self.collate, "cuda")
        self.re_T = 1.0
        if NULL_SHIFT:     # corrected outputs: relation temperature on the same dev data (as re_eval)
            self.re_T = fit_re_temperature(dev, dev_outs)
            set_re_temperature(self.re_T)
            dev_outs = re_temperature(dev_outs, self.re_T)
        self.dev_f1, self.re_thr = best_re_threshold(dev, dev_outs)

    def spans(self, rows):
        """Entity-head predictions {(s, e, label)} per row (None where a row cannot be encoded)."""
        encs, kept = encode_all(rows, self.tok, self.W)
        raw = predict(self.model, encs, self.collate, "cuda")
        if self.head == "rlcd":
            outs = with_temperature(raw, self.T, "entity")
            got = {r["id"]: decode(e, o["dz"], self.logit, o["choice"]) for r, e, o in zip(kept, encs, outs)}
            # confidence of each decoded (s, e, label): the chosen label's probability (for ECE)
            self.span_conf = {}
            for r, e, o in zip(kept, encs, outs):
                ms, cs = np.nonzero((o["dz"] >= self.logit) & o["choice"])
                self.span_conf[r["id"]] = {(e["cands"][c][0], e["cands"][c][1], e["labels"][m]):
                                           float(1 / (1 + np.exp(-o["z"][m, c]))) for m, c in zip(ms, cs)}
        else:   # BCE ablation: independent sigmoid per (type, span); logits divided by T
            got = {r["id"]: decode(e, o["z"], self.T * self.logit) for r, e, o in zip(kept, encs, raw)}
        return [got.get(r["id"]) for r in rows]

    def pairs(self, rows, pair_spans):
        """Relation-head choice probabilities per row: (pair_char, rel_labels, p) or None."""
        encs, kept = [], []
        for r, ps in zip(rows, pair_spans):
            e, _ = encode_all([r], self.tok, self.W, [ps])
            if e:
                encs += e
                kept.append(r["id"])
        outs = predict_re(self.model, encs, self.collate, "cuda")
        got = {k: (e["pair_char"], e["rel_labels"], o["p"]) for k, e, o in zip(kept, encs, outs)}
        return [got.get(r["id"]) for r in rows]


def arguments(reader, rows, triggers, meta, stats):
    """Arguments for given trigger sets: per row {(ts, te, type, as, ae, role)}."""
    # as the training argument rows: the triggers' event types and their roles as span
    # labels; argument candidates are the spans labelled with a role
    roles_of, rdesc, edesc = meta["event_roles"], meta["roles"], meta["event_types"]
    role_rows, asked = [], []
    for r, trig in zip(rows, triggers):
        types = sorted({t for _, _, t in trig})
        roles = sorted({x for t in types for x in roles_of.get(t, [])})
        asked.append(roles)
        gold_roles = {a["role"] for e in r["events"] for a in e["args"]}
        role_rows.append({"id": r["id"], "text": r["text"], "entities": [],
                          "queries": [q(t, edesc.get(t, ""), True) for t in types]
                          + [q(x, rdesc.get(x, ""), x in gold_roles) for x in roles if x not in types]})
    cand = reader.spans(role_rows)
    cand = [None if c is None else {x for x in c if x[2] in set(a)} for c, a in zip(cand, asked)]
    rel_rows, pspans, idx = [], [], []
    for k, (r, trig, roles, c) in enumerate(zip(rows, triggers, asked, cand)):
        if not trig or not roles:
            continue
        if c is None:
            stats["rows_not_encoded"] += 1
            continue
        tsp = list(dict.fromkeys((s, e) for s, e, _ in sorted(trig)))
        csp = [x for x in dict.fromkeys((s, e) for s, e, _ in sorted(c)) if x not in tsp]
        rq = [x for x in role_rows[k]["queries"] if x["label"] in set(roles)]
        rel_rows.append({"id": r["id"], "text": r["text"], "queries": role_rows[k]["queries"], "entities": [],
                         "rel_queries": rq, "relations": []})
        pspans.append(tsp + csp)
        idx.append(k)
        stats["argument_candidates"] += len(csp)
    out = [set() for _ in rows]
    conf = [{} for _ in rows]             # (ts, te, type, as, ae, role) -> probability of the chosen role
    reader.arg_conf = conf
    for k, res in zip(idx, reader.pairs(rel_rows, pspans)):
        if res is None:
            stats["rows_not_encoded"] += 1
            continue
        pair_char, labels, p = res
        types = {}
        for s, e, t in triggers[k]:
            types.setdefault((s, e), set()).add(t)
        for j, (h, t) in enumerate(pair_char):
            # the head is a trigger; the tail may be another event's trigger (MLEE: about 30%
            # of the arguments are events, e.g. the Theme of a regulation)
            if h not in types or t == h:
                continue
            if 1 - p[0, j] < reader.re_thr:
                continue
            for et in types[h]:
                valid = [m for m, lab in enumerate(labels) if lab in roles_of.get(et, [])]
                if valid:
                    m = max(valid, key=lambda m: p[1 + m, j])
                    out[k].add((h[0], h[1], et, t[0], t[1], labels[m]))
                    conf[k][(h[0], h[1], et, t[0], t[1], labels[m])] = float(p[1 + m, j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration", required=True, help="NER calibration.json (entity decoding)")
    ap.add_argument("--dev", nargs="+", default=RE_DEV, help="dev_calib files for the relation threshold")
    ap.add_argument("--ood", default=str(ROOT / "data/ood_ee"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="", help="comma-separated subset of the manifest's sets (default: all)")
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--head", default="rlcd", choices=["bce", "rlcd"], help="entity head (bce: ablation)")
    ap.add_argument("--name_only", action="store_true", help="drop event type and role descriptions")
    args = ap.parse_args()
    reader = Reader(args)
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    res = {"ckpt": args.ckpt, "re_threshold": reader.re_thr, "re_T": reader.re_T, "null_shift": NULL_SHIFT,
           "rebel_dev_calib_f1": reader.dev_f1,
           "name_only": args.name_only, "sets": {}}
    for name, meta in manifest["sets"].items():
        if args.sets and name not in args.sets.split(","):
            continue
        if args.name_only:
            meta = dict(meta, event_types={k: "" for k in meta["event_types"]}, roles={k: "" for k in meta["roles"]})
        rows = load_rows(Path(args.ood) / f"{name}.jsonl")
        stats = {"rows_not_encoded": 0, "argument_candidates": 0}
        # given: gold triggers
        gtrig = [gold_sets(r)[0] for r in rows]
        gold_args = [gold_sets(r)[1] for r in rows]
        gpred = list(zip(gtrig, arguments(reader, rows, gtrig, meta, stats)))
        given_conf = reader.arg_conf
        given = score(rows, gpred, list(meta["event_types"]))
        # end to end: triggers from the entity head over the event types
        types = list(meta["event_types"])
        trig_rows = [{"id": r["id"], "text": r["text"], "entities": [],
                      "queries": [q(t, meta["event_types"][t], t in {e["type"] for e in r["events"]}) for t in types]}
                     for r in rows]
        ptrig = [x or set() for x in reader.spans(trig_rows)]
        trig_conf = [getattr(reader, "span_conf", {}).get(r["id"], {}) for r in rows]
        epred = list(zip(ptrig, arguments(reader, rows, ptrig, meta, stats)))
        e2e = score(rows, epred, types)
        calib = {"arguments_given": conf_ece(given_conf, gold_args),
                 "triggers_end2end": conf_ece(trig_conf, gtrig),
                 "arguments_end2end": conf_ece(reader.arg_conf, gold_args)}
        res["sets"][name] = {"given": given, "end2end": e2e, "stats": stats, "calibration": calib,
                             "preds": {"given": [[sorted(t), sorted(a)] for t, a in gpred],
                                       "end2end": [[sorted(t), sorted(a)] for t, a in epred]}}
        print(name, "given AC", round(given["AC"]["f1"], 4), "end2end TC", round(e2e["TC"]["f1"], 4),
              "AC", round(e2e["AC"]["f1"], 4), stats, flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
