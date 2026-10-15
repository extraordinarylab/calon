"""Training losses, plus a `bce_scaled` control.

Modes:
  bce         clean BCE only
  bce_scaled  (1 + lam) * clean BCE: matches the extra gradient mass that
              direct/pg add at small sigma, where J ~= -BCE - sigma^2/2 p(1-p)
  direct      clean + lam * (-mean noisy log score), reparameterized
  pg          clean + lam * Gaussian policy-gradient surrogate

`generator` draws the K Gaussian samples from its own stream so that the
global RNG (dropout) stays aligned across arms.
"""
import math

import torch
import torch.nn.functional as F


def rlcd_log_probs(u, label_ok):
    """RLCD choice distribution per span over [none, label_1..label_M].

    u: [B, 1+M, C] utilities (none first); label_ok: [B, M] bool, the labels that
    are queried with known status. Unqueried / unknown / padded labels get -inf.
    """
    ok = torch.cat([label_ok.new_ones(label_ok.shape[0], 1), label_ok], 1)[:, :, None]
    return torch.log_softmax(u.float().masked_fill(~ok, float("-inf")), 1)


def rlcd_loss(u, y, valid, query_mask, cand_mask, brier_weight=1.0, stats=None, none_weight=1.0):
    """Plackett-Luce single-choice loss plus multiway Brier score, per span.

    Each candidate span chooses one of [none, queried labels]. Target: its gold
    label(s) among known queries, else none. With several gold labels the
    choice loss is -log of their summed probability and the Brier target is
    uniform over them. In a row that has an unknown query, a span without a
    gold label is skipped: "none" is not known there. Reduction:
    mean over a row's valid spans, then over rows.
    """
    y = y.float()
    known = valid.any(-1)                                   # [B,M] queried, known status
    unknown_row = (query_mask & ~known).any(-1)             # [B]
    gold = (y > 0) & valid                                  # [B,M,C]
    has_gold = gold.any(1)                                  # [B,C]
    span_ok = cand_mask & (has_gold | ~unknown_row[:, None])
    logp = rlcd_log_probs(u, known)                          # [B,1+M,C]
    target = torch.cat([~has_gold[:, None, :], gold], 1)
    ce = -logp.masked_fill(~target, float("-inf")).logsumexp(1)
    p = logp.exp()
    t = target.float() / target.float().sum(1, keepdim=True).clamp_min(1)
    brier = 0.5 * (p - t).square().sum(1)
    # "none" spans are ~99% of candidates; none_weight < 1 down-weights them. The row
    # mean is a weighted mean, so none_weight = 1 is exactly the unweighted loss.
    w = torch.where(has_gold, torch.ones_like(ce), torch.full_like(ce, none_weight))
    w = torch.where(span_ok, w, torch.zeros_like(w))
    per_span = torch.where(span_ok, (ce + brier_weight * brier) * w, torch.zeros_like(ce))
    cnt = span_ok.sum(-1)
    active = cnt > 0
    if not bool(active.any()):
        return u.float().sum() * 0
    per_row = per_span.sum(-1) / w.sum(-1).clamp_min(1e-8)
    if stats is not None:
        def row_mean(x):
            return (torch.where(span_ok, x, torch.zeros_like(x)).sum(-1) / cnt.clamp_min(1))[active].mean().item()
        stats["rlcd_ce"] = row_mean(ce)
        stats["rlcd_brier"] = row_mean(brier)
    return per_row[active].mean()


def rlcd_pg(u, y, valid, query_mask, cand_mask, K=8, sigma=1.0, fp_cost=1.0, generator=None, stats=None,
            none_weight=1.0):
    """Policy-gradient term of RLCD: a Gaussian random-utility policy over each span's choice.

    Each candidate span samples K utility vectors u + sigma * eps over [none, known labels] and
    takes the argmax as its decision. The reward is that of the decision itself, as the evaluation
    scores it: +1 for a gold label, -fp_cost for a label that is not gold (a false positive or a
    wrong type), 0 for none. Unlike the likelihood terms, this rewards the decision, not the
    probability of the gold choice. Leave-one-out baseline over the K samples; the score function
    of the Gaussian, (u_sample - u) / sigma^2, carries the gradient. Spans are weighted and
    skipped as in rlcd_loss (a span without a gold label in a row with an unknown query has no
    known reward).
    """
    u, y = u.float(), y.float()
    known = valid.any(-1)                                   # [B,M]
    unknown_row = (query_mask & ~known).any(-1)
    gold = (y > 0) & valid                                  # [B,M,C]
    has_gold = gold.any(1)                                  # [B,C]
    span_ok = cand_mask & (has_gold | ~unknown_row[:, None])
    ok = torch.cat([known.new_ones(known.shape[0], 1), known], 1)[:, :, None].expand_as(u)
    eps = torch.randn((K,) + tuple(u.shape), device=u.device, dtype=u.dtype, generator=generator)
    sample = (u.detach()[None] + sigma * eps).masked_fill(~ok[None], float("-inf"))
    a = sample.argmax(2)                                    # [K,B,C], 0 = none
    target = torch.cat([~has_gold[:, None, :], gold], 1)    # [B,1+M,C]
    hit = target[None].expand(K, *target.shape).gather(2, a[:, :, None, :]).squeeze(2)
    pos = a > 0
    one = torch.ones_like(a, dtype=u.dtype)
    reward = torch.where(pos & hit, one, torch.where(pos, -fp_cost * one, 0 * one))
    baseline = (reward.sum(0, keepdim=True) - reward) / (K - 1)
    adv = reward - baseline
    # log N(sample; u, sigma^2) up to a constant, over the span's valid choices
    drift = (u.detach()[None] + sigma * eps - u[None]) / sigma
    log_policy = torch.where(ok[None], -0.5 * drift.square(), torch.zeros_like(drift)).sum(2)   # [K,B,C]
    surrogate = -(adv * log_policy).mean(0)                 # [B,C]
    w = torch.where(has_gold, torch.ones_like(surrogate), torch.full_like(surrogate, none_weight))
    w = torch.where(span_ok, w, torch.zeros_like(w))
    active = span_ok.any(-1)
    if not bool(active.any()):
        return u.sum() * 0
    per_row = (surrogate * w).sum(-1) / w.sum(-1).clamp_min(1e-8)
    if stats is not None:
        r = (reward * w[None]).sum(-1) / w.sum(-1).clamp_min(1e-8)
        stats["rl_reward"] = r[:, active].mean().item()
        stats["rl_pos_rate"] = pos.float()[:, span_ok].mean().item()
    return per_row[active].mean()


calon_log_probs, calon_loss = rlcd_log_probs, rlcd_loss    # earlier names


def spherical_term(u, y):
    """Spherical score: (y p + (1-y)(1-p)) / sqrt(p^2 + (1-p)^2) - 1, p = sigmoid(u)."""
    p = torch.sigmoid(u)
    return (y * p + (1 - y) * (1 - p)) / torch.sqrt(p.square() + (1 - p).square()) - 1


def loss_fn(z, y, valid, mode="pg", K=8, sigma=0.2, lam=0.1, generator=None, stats=None,
            alpha=0.0):
    z, y = z.float(), y.detach().float()
    mask = valid.float()
    counts = mask.sum((-2, -1))                 # [B]
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("Skip all-invalid batch before optimizer.step")

    def mean_valid(v):                         # v: [K,B,M,C]
        per_row = (v * mask[None]).sum((-2, -1)) / counts.clamp_min(1)[None]
        return per_row[:, active].mean()

    clean = mean_valid(F.binary_cross_entropy_with_logits(
        z, y, reduction="none")[None])
    if stats is not None:
        stats["clean_bce"] = clean.item()
    if mode == "bce":
        return clean
    if mode == "bce_scaled":
        return (1 + lam) * clean
    if mode not in {"pg", "direct"}:
        raise ValueError(mode)
    if K < 1 or sigma < 0 or (mode == "pg" and (K < 2 or sigma <= 0)):
        raise ValueError("PG needs K>=2 and sigma>0")

    noise = torch.randn((K,) + tuple(z.shape), device=z.device, dtype=z.dtype,
                        generator=generator)
    u = z[None] + sigma * noise
    if mode == "pg":
        u = u.detach()                         # fixed sampled action

    # log score, computed stably; expand y to the K-rollout shape
    reward = -F.binary_cross_entropy_with_logits(
        u, y[None].expand_as(u), reduction="none")
    if alpha:                                  # spherical-score ablation, same for pg/direct
        reward = reward + alpha * spherical_term(u, y[None].expand_as(u))
    if stats is not None:
        stats["mean_reward"] = mean_valid(reward.detach()).item()

    if mode == "direct":
        auxiliary = -mean_valid(reward)         # reward -> z -> model
    else:
        reward = reward.detach()
        baseline = (reward.sum(0, keepdim=True) - reward) / (K - 1)
        advantage = reward - baseline
        log_policy = (-0.5 * ((u - z[None]) / sigma).square()
                      - math.log(sigma) - 0.5 * math.log(2 * math.pi))
        auxiliary = -mean_valid(advantage * log_policy)
    if stats is not None:
        stats["auxiliary"] = auxiliary.item()

    return clean + lam * auxiliary
