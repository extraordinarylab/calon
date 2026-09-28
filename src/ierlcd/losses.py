"""SPEC 5 loss, plus a `bce_scaled` control.

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


def spherical_term(u, y):
    """SPEC appendix A: (y p + (1-y)(1-p)) / sqrt(p^2 + (1-p)^2) - 1, p = sigmoid(u)."""
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
