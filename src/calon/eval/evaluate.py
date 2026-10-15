"""Extraction F1, candidate recall and calibration metrics.

Null-class prior correction (CALON_NULL_CORRECT=<null weight of the training loss>, e.g. 0.1): predict() and
predict_re() add log(1 / w) to the null utility of every span and pair before the softmax, undoing the
down-weighting of null candidates in the loss (as weight_correct.py's analysis). Calibration then fits T and
thresholds on the corrected outputs; relations get their own temperature (set_re_temperature, fitted on REBEL
dev_calib by re_eval / ee_eval), applied by predict_re().
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

NULL_SHIFT = float(np.log(1.0 / float(os.environ["CALON_NULL_CORRECT"]))) if os.environ.get("CALON_NULL_CORRECT") \
    else 0.0
RE_T = [1.0]                 # relation temperature applied by predict_re (set_re_temperature)
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 57))


def set_re_temperature(T):
    RE_T[0] = float(T)


def re_temperature(outs, T):
    """Relation choice probabilities at temperature T: p_T proportional to p ** (1 / T) (masked labels stay 0)."""
    if T == 1.0:
        return outs
    res = []
    for o in outs:
        p = o["p"]
        if p.shape[1] and p.shape[0] > 1:
            lp = np.where(p > 0, np.log(np.clip(p, 1e-30, 1.0)) / T, -np.inf)
            lp = lp - lp.max(0, keepdims=True)
            q = np.exp(lp)
            p = q / q.sum(0, keepdims=True)
        res.append(dict(o, p=p))
    return res


def re_brier(encs, outs):
    """Mean multiway Brier over the candidate pairs (target: the pair's gold relations, else no relation)."""
    tot, n = 0.0, 0
    for e, o in zip(encs, outs):
        p = o["p"]
        if not (p.shape[1] and p.shape[0] > 1):
            continue
        gold = e.get("gold_triples", set())
        idx = {lab: k for k, lab in enumerate(e["rel_labels"])}
        for k, (h, t) in enumerate(e["pair_char"]):
            y = np.zeros(p.shape[0])
            g = [idx[lab] for (a, b, lab) in gold if (a, b) == (h, t) and lab in idx]
            if g:
                y[[1 + j for j in g]] = 1.0 / len(g)
            else:
                y[0] = 1.0
            tot += float(((p[:, k] - y) ** 2).sum())
            n += 1
    return tot / max(n, 1)


def fit_re_temperature(encs, outs):
    """Relation temperature with the lowest multiway Brier on (corrected) dev outputs, calibrate.py's grid."""
    if os.environ.get("CALON_FIX_T"):       # null-correction analysis without a refitted temperature
        return 1.0
    return float(min(T_GRID, key=lambda t: re_brier(encs, re_temperature(outs, t))))


def _replica(gpu, encoder, vocab_size, W, head, ckpt, collate, inq, outq):
    """One model copy on one GPU: answers (kind, rows, batch size) with predictions."""
    from ..model.model import load_model
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    model = load_model(encoder, vocab_size, W, head, ckpt, device=device)
    outq.put("ready")
    while (msg := inq.get()) is not None:
        kind, encs, batch_size = msg
        fn = predict if kind == "ner" else predict_re
        outq.put(fn(model, encs, collate, device, batch_size))


class ParallelModel:
    """Data-parallel inference over the visible GPUs: one replica process per GPU, the rows
    cut into contiguous slices and the predictions put back in order. predict() and
    predict_re() accept it in place of a model."""

    def __init__(self, encoder, vocab_size, W, head, ckpt, collate, n_gpus):
        from ..model.model import canonical_head
        ctx = torch.multiprocessing.get_context("spawn")
        self.head_type, self.workers = canonical_head(head), []
        for g in range(n_gpus):
            inq, outq = ctx.Queue(), ctx.Queue()
            p = ctx.Process(target=_replica, args=(g, encoder, vocab_size, W, head, ckpt, collate, inq, outq),
                            daemon=True)
            p.start()
            self.workers.append((p, inq, outq))
        for _, _, outq in self.workers:
            assert outq.get() == "ready"

    def run(self, kind, encs, batch_size):
        n = len(self.workers)
        cuts = [len(encs) * k // n for k in range(n + 1)]
        for k, (_, inq, _) in enumerate(self.workers):
            inq.put((kind, encs[cuts[k]:cuts[k + 1]], batch_size))
        out = []
        for _, _, outq in self.workers:
            out += outq.get()
        return out

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def load_eval_model(encoder, vocab_size, W, head, ckpt, collate):
    """EVAL_GPUS > 1 (opt in, capped at the visible GPUs): a ParallelModel; else one model."""
    from ..model.model import load_model
    n = min(torch.cuda.device_count(), int(os.environ.get("EVAL_GPUS", "1")))
    if n > 1:
        return ParallelModel(encoder, vocab_size, W, head, ckpt, collate, n)
    return load_model(encoder, vocab_size, W, head, ckpt)


@torch.no_grad()
def predict(model, encs, collate, device, batch_size=32):
    """Run the model; return per-row arrays of logits over supervised decisions.

    For an RLCD head the row also keeps the raw utilities `u` ([1+M, C], none
    first) and `known` ([M]), and `z` / `choice` come from with_temperature(T=1).
    """
    if isinstance(model, ParallelModel):
        return model.run("ner", encs, batch_size)
    model.eval()
    rlcd = getattr(model, "head_type", "bce") == "rlcd"
    out = []
    for k in range(0, len(encs), batch_size):
        chunk = encs[k:k + batch_size]
        inputs, tgt = collate(chunk)
        inputs = {n: v for n, v in inputs.items() if n not in ("rel_mask", "pair_h", "pair_t")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(**{n: v.to(device) for n, v in inputs.items()}).float().cpu()
        for b, e in enumerate(chunk):
            M, C = len(e["labels"]), len(e["cands"])
            o = {"y": tgt["y"][b, :M, :C].numpy(), "valid": tgt["valid"][b, :M, :C].numpy()}
            if rlcd:
                o["u"] = z[b, :1 + M, :C].numpy().copy()
                if NULL_SHIFT and o["u"].size:
                    o["u"][0] += NULL_SHIFT
                o["known"] = o["valid"].any(-1) if C else np.zeros(M, bool)
                o["unknown_row"] = bool((np.array(e["supervised"]) == False).any())  # noqa: E712
            else:
                o["z"] = z[b, :M, :C].numpy()
            out.append(o)
    model.train()
    return with_temperature(out, 1.0) if rlcd else out


@torch.no_grad()
def predict_re(model, encs, collate, device, batch_size=16):
    """Relation choice probabilities per row: p [1+R, P] over [no relation, queried
    relations] for each ordered candidate pair (softmax over known relation queries)."""
    if isinstance(model, ParallelModel):
        return re_temperature(model.run("re", encs, batch_size), RE_T[0])
    model.eval()
    out = []
    for k in range(0, len(encs), batch_size):
        chunk = encs[k:k + batch_size]
        inputs, tgt = collate(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, zr = model(**{n: v.to(device) for n, v in inputs.items()})
        zr = zr.float().cpu()
        for b, e in enumerate(chunk):
            R, P = len(e.get("rel_labels", [])), len(e.get("pairs", []))
            known = np.array(e.get("rel_supervised", []), bool)
            o = {"p": np.zeros((1 + R, P))}
            if R and P:
                u = zr[b, :1 + R, :P].numpy().copy()
                u[0] += NULL_SHIFT
                u = np.where(np.concatenate([[True], known])[:, None], u, -np.inf)
                u = u - u.max(0, keepdims=True)
                p = np.exp(u)
                o["p"] = p / p.sum(0, keepdims=True)
            out.append(o)
    model.train()
    return re_temperature(out, RE_T[0])


def decode_relations(e, o, threshold):
    """{(head span, tail span, label)} where 1 - p(no relation) >= threshold."""
    p = o["p"]
    if p.shape[1] == 0 or p.shape[0] == 1:
        return set()
    best = p[1:].argmax(0)
    keep = (1.0 - p[0]) >= threshold
    return {(e["pair_char"][k][0], e["pair_char"][k][1], e["rel_labels"][best[k]])
            for k in np.nonzero(keep)[0]}


def re_counts(encs, outs, threshold):
    tp = fp = fn = 0
    for e, o in zip(encs, outs):
        pred, gold = decode_relations(e, o, threshold), e.get("gold_triples", set())
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def best_re_threshold(encs, outs):
    """(best F1, threshold) over 0.05 ... 0.95."""
    best = (-1.0, 0.5)
    for t in np.linspace(0.05, 0.95, 19):
        v = f1(*re_counts(encs, outs, t))["f1"]
        if v > best[0]:
            best = (v, float(t))
    return best


def rlcd_probs(o, T):
    """Softmax over [none, known labels] of u / T -> [1+M, C]."""
    u = o["u"] / T
    ok = np.concatenate([[True], o["known"]])[:, None]
    u = np.where(ok, u, -np.inf)
    u = u - u.max(0, keepdims=True)
    p = np.exp(u)
    return p / p.sum(0, keepdims=True)


def with_temperature(outs, T, decode="label"):
    """RLCD rows -> binary-decision view: z = logit(p_label), choice = the span's
    most likely label. Decoding keeps a label only where choice is set and the
    decision score `dz` clears the threshold:
      label   dz = logit(p_label)       (the chosen label's own probability)
      entity  dz = logit(1 - p_none)    (probability that the span is an entity
              at all; the label is then the most likely one, so mass split
              between near-synonymous labels no longer suppresses the span)"""
    res = []
    for o in outs:
        p = rlcd_probs(o, T)
        pl = np.clip(p[1:], 1e-12, 1 - 1e-12)
        choice = np.zeros_like(pl, dtype=bool)
        if pl.size:
            best = pl.argmax(0)
            choice[best, np.arange(pl.shape[1])] = True
        z = np.log(pl) - np.log1p(-pl)
        if decode == "entity":
            pe = np.clip(1 - p[0], 1e-12, 1 - 1e-12)
            dz = np.broadcast_to(np.log(pe) - np.log1p(-pe), z.shape).copy()
        else:
            dz = z
        res.append({**o, "z": z, "dz": dz, "choice": choice, "p_none": p[0], "T": T})
    return res


def rlcd_brier(outs, T):
    """Mean multiway Brier over spans (the RLCD loss's targets and masking)."""
    tot, n = 0.0, 0
    for o in outs:
        if not o["u"].size:
            continue
        p = rlcd_probs(o, T)
        gold = (o["y"] > 0) & o["valid"]
        has = gold.any(0)
        ok = has | (not o["unknown_row"])
        t = np.concatenate([~has[None], gold], 0).astype(float)
        t = t / np.maximum(t.sum(0, keepdims=True), 1)
        b = 0.5 * ((p - t) ** 2).sum(0)
        tot += b[ok].sum()
        n += ok.sum()
    return tot / max(n, 1)


calon_probs, calon_brier = rlcd_probs, rlcd_brier          # earlier names


def decode(e, z, threshold_logit, choice=None):
    preds = set()
    hit = z >= threshold_logit
    if choice is not None:
        hit &= choice
    ms, cs = np.nonzero(hit)
    for m, c in zip(ms, cs):
        s, t, _, _ = e["cands"][c]
        preds.add((s, t, e["labels"][m]))
    return preds


def f1_counts(encs, outs, threshold_logit):
    tp = fp = fn = 0
    for e, o in zip(encs, outs):
        # only supervised queries are scored; FiNERWeb has no unknown queries
        keep = {lab for lab, sup in zip(e["labels"], e["supervised"]) if sup}
        pred = {p for p in decode(e, o.get("dz", o["z"]), threshold_logit, o.get("choice")) if p[2] in keep}
        gold = {g for g in e["gold_spans"] if g[2] in keep}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def ece(prob, y, bins=15):
    if len(prob) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(prob[sel].mean() - y[sel].mean())
    return float(err)


def reliability(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_p": float(prob[idx == b].mean()) if (idx == b).any() else None,
             "freq": float(y[idx == b].mean()) if (idx == b).any() else None}
            for b in range(bins)]


def metrics(encs, outs, temperature=1.0, threshold=0.5):
    """All metrics at probability threshold `threshold` on sigmoid(z / T)."""
    thr_logit = temperature * float(np.log(threshold / (1 - threshold)))
    res = f1(*f1_counts(encs, outs, thr_logit))
    zs = np.concatenate([o["z"][o["valid"]] for o in outs]) / temperature
    ys = np.concatenate([o["y"][o["valid"]] for o in outs])
    row_nll = [F.binary_cross_entropy_with_logits(
        torch.from_numpy(o["z"][o["valid"]] / temperature),
        torch.from_numpy(o["y"][o["valid"]])).item() for o in outs if o["valid"].any()]
    prob = 1 / (1 + np.exp(-zs))
    pred = prob >= threshold
    calon = bool(outs) and "choice" in outs[0]
    if calon:                                  # only the span's chosen label is a prediction
        dz = np.concatenate([o["dz"][o["valid"]] for o in outs])
        pred = (dz >= thr_logit) & np.concatenate([o["choice"][o["valid"]] for o in outs])
    gold = sum(len(e["gold_spans"]) for e in encs)
    res.update({
        "nll_row_mean": float(np.mean(row_nll)),
        "nll_decision": float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(zs), torch.from_numpy(ys)).item()),
        "brier": float(np.mean((prob - ys) ** 2)),
        "ece_all": ece(prob, ys),
        "ece_predicted": ece(prob[pred], ys[pred]),
        "n_decisions": int(len(ys)), "n_predicted": int(pred.sum()),
        "positive_rate": float(ys.mean()),
        "candidate_recall": 1 - sum(e["unrepresented"] for e in encs) / max(gold, 1),
        "temperature": temperature, "threshold": threshold,
    })
    if calon:
        res["brier_multiway"] = float(rlcd_brier(outs, outs[0]["T"]))
        res["temperature"] = outs[0]["T"]
    return res
