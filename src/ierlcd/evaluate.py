"""Extraction F1, candidate recall and calibration metrics (SPEC 6)."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict(model, encs, collate, device, batch_size=32):
    """Run the model; return per-row arrays of logits over supervised decisions.

    For an RLCD head the row also keeps the raw utilities `u` ([1+M, C], none
    first) and `known` ([M]), and `z` / `choice` come from with_temperature(T=1).
    """
    model.eval()
    rlcd = getattr(model, "head_type", "bce") == "rlcd"
    out = []
    for k in range(0, len(encs), batch_size):
        chunk = encs[k:k + batch_size]
        inputs, tgt = collate(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(**{n: v.to(device) for n, v in inputs.items()}).float().cpu()
        for b, e in enumerate(chunk):
            M, C = len(e["labels"]), len(e["cands"])
            o = {"y": tgt["y"][b, :M, :C].numpy(), "valid": tgt["valid"][b, :M, :C].numpy()}
            if rlcd:
                o["u"] = z[b, :1 + M, :C].numpy()
                o["known"] = o["valid"].any(-1) if C else np.zeros(M, bool)
                o["unknown_row"] = bool((np.array(e["supervised"]) == False).any())  # noqa: E712
            else:
                o["z"] = z[b, :M, :C].numpy()
            out.append(o)
    model.train()
    return with_temperature(out, 1.0) if rlcd else out


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
    rlcd = bool(outs) and "choice" in outs[0]
    if rlcd:                                  # only the span's chosen label is a prediction
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
    if rlcd:
        res["brier_multiway"] = float(rlcd_brier(outs, outs[0]["T"]))
        res["temperature"] = outs[0]["T"]
    return res
