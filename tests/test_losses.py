"""SPEC 6 correctness tests 2, 4, 5 for the loss."""
import math

import pytest
import torch
import torch.nn.functional as F

from ierlcd.losses import loss_fn


def batch(B=3, M=4, C=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, M, C, generator=g)
    y = (torch.rand(B, M, C, generator=g) < 0.2).float()
    valid = torch.rand(B, M, C, generator=g) < 0.8
    return z, y, valid


@pytest.mark.parametrize("mode", ["bce", "bce_scaled", "direct", "pg"])
def test_padding_does_not_change_loss(mode):
    z, y, valid = batch()
    zp = F.pad(z, (0, 5, 0, 2), value=3.0)
    yp = F.pad(y, (0, 5, 0, 2), value=1.0)
    vp = F.pad(valid, (0, 5, 0, 2), value=False)
    # same noise on the valid region: draw full-size noise, compare with direct
    # recomputation only for deterministic modes; for stochastic modes compare
    # the estimates under a large K.
    if mode in {"bce", "bce_scaled"}:
        assert torch.allclose(loss_fn(z, y, valid, mode), loss_fn(zp, yp, vp, mode))
    else:
        a = loss_fn(z, y, valid, mode, K=4000, generator=torch.Generator().manual_seed(1))
        b = loss_fn(zp, yp, vp, mode, K=4000, generator=torch.Generator().manual_seed(2))
        assert abs(a.item() - b.item()) < 0.02


@pytest.mark.parametrize("mode", ["bce", "direct", "pg"])
def test_unknown_gets_zero_gradient(mode):
    z, y, valid = batch()
    z.requires_grad_(True)
    loss_fn(z, y, valid, mode, generator=torch.Generator().manual_seed(0)).backward()
    assert torch.all(z.grad[~valid] == 0)
    assert torch.any(z.grad[valid] != 0)


def test_all_invalid_raises():
    z, y, valid = batch()
    with pytest.raises(ValueError):
        loss_fn(z, y, torch.zeros_like(valid), "bce")


def test_pg_rejects_sigma_zero_and_k1():
    z, y, valid = batch()
    with pytest.raises(ValueError):
        loss_fn(z, y, valid, "pg", sigma=0.0)
    with pytest.raises(ValueError):
        loss_fn(z, y, valid, "pg", K=1)
    loss_fn(z, y, valid, "direct", sigma=0.0)  # direct with sigma=0 is allowed


def test_row_reduction_is_mean_of_row_means():
    z, y, valid = batch()
    per = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    rows = [(per[b][valid[b]]).mean() for b in range(z.shape[0])]
    assert torch.allclose(loss_fn(z, y, valid, "bce"), torch.stack(rows).mean())


def test_baseline_is_leave_one_out_along_k():
    """Advantages from one decision must not depend on other decisions' samples."""
    z = torch.zeros(1, 1, 2, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    loss_fn(z, y, valid, "pg", K=2, sigma=1.0, lam=1.0,
            generator=torch.Generator().manual_seed(3)).backward()
    g_pair = z.grad.clone()
    for c in range(2):
        z1 = torch.zeros(1, 1, 1, requires_grad=True)
        gen = torch.Generator().manual_seed(3)
        noise = torch.randn((2, 1, 1, 2), generator=gen)[..., c]
        u = noise  # z=0, sigma=1
        r = -F.binary_cross_entropy_with_logits(u, y[..., c:c + 1].expand_as(u), reduction="none")
        adv = r - r.flip(0)
        lp = -0.5 * (u - z1[None]).square()
        pg = -(adv * lp).mean()
        clean = F.binary_cross_entropy_with_logits(z1, y[..., c:c + 1])
        ((clean + pg) / 2).backward()
        assert torch.allclose(z1.grad.flatten(), g_pair[..., c].flatten(), atol=1e-6)


def expected_reward_grad(z0, y, sigma, n=200001):
    """d/dz E[log score(z + sigma*eps)] by finite difference with common eps."""
    eps = torch.randn(n, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    def J(z):
        u = z + sigma * eps
        return -F.binary_cross_entropy_with_logits(u, torch.full_like(u, y)).item()
    h = 1e-4
    return (J(z0 + h) - J(z0 - h)) / (2 * h)


@pytest.mark.parametrize("z0,y", [(-1.0, 1.0), (0.7, 0.0), (2.5, 1.0)])
def test_pg_and_direct_match_finite_difference(z0, y):
    sigma, K, n = 0.5, 8, 20000
    fd = expected_reward_grad(z0, y, sigma)
    grads = {}
    for mode in ("direct", "pg"):
        z = torch.full((n, 1, 1), z0, dtype=torch.float64, requires_grad=True)
        yt = torch.full_like(z, y)
        valid = torch.ones_like(z, dtype=torch.bool)
        # lam=1 and subtract the clean part to isolate the auxiliary gradient
        loss = loss_fn(z, yt, valid, mode, K=K, sigma=sigma, lam=1.0,
                       generator=torch.Generator().manual_seed(5))
        clean = F.binary_cross_entropy_with_logits(z, yt, reduction="none").mean()
        g, = torch.autograd.grad(loss - clean, z)
        per = -g.flatten() * n                 # reward-maximizing gradient per decision
        grads[mode] = (per.mean().item(), per.std().item() / math.sqrt(n))
    for mode, (m, se) in grads.items():
        assert abs(m - fd) < 5 * se + 1e-3, (mode, m, fd, se)
    # PG is the higher-variance estimator of the same quantity
    assert grads["pg"][1] > grads["direct"][1]


@pytest.mark.parametrize("z0,y", [(-1.0, 1.0), (0.7, 0.0)])
def test_spherical_pg_and_direct_match_finite_difference(z0, y):
    from ierlcd.losses import spherical_term
    sigma, K, n, alpha = 0.5, 8, 20000, 0.1
    eps = torch.randn(200001, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    def J(zz):
        u = zz + sigma * eps
        yy = torch.full_like(u, y)
        r = -F.binary_cross_entropy_with_logits(u, yy, reduction="none") + alpha * spherical_term(u, yy)
        return r.mean().item()
    fd = (J(z0 + 1e-4) - J(z0 - 1e-4)) / 2e-4
    for mode in ("direct", "pg"):
        z = torch.full((n, 1, 1), z0, dtype=torch.float64, requires_grad=True)
        yt = torch.full_like(z, y)
        valid = torch.ones_like(z, dtype=torch.bool)
        loss = loss_fn(z, yt, valid, mode, K=K, sigma=sigma, lam=1.0, alpha=alpha,
                       generator=torch.Generator().manual_seed(5))
        clean = F.binary_cross_entropy_with_logits(z, yt, reduction="none").mean()
        g, = torch.autograd.grad(loss - clean, z)
        per = -g.flatten() * n
        m, se = per.mean().item(), per.std().item() / math.sqrt(n)
        assert abs(m - fd) < 5 * se + 1e-3, (mode, m, fd, se)


def test_spherical_term_is_zero_at_certainty_and_proper():
    from ierlcd.losses import spherical_term
    big = torch.tensor([30.0, -30.0], dtype=torch.float64)
    assert torch.allclose(spherical_term(big, torch.tensor([1.0, 0.0], dtype=torch.float64)),
                          torch.zeros(2, dtype=torch.float64), atol=1e-9)
    # proper: for true probability q, expected score is maximized at p = q
    q = 0.3
    zs = torch.linspace(-4, 4, 801, dtype=torch.float64)
    exp = q * spherical_term(zs, torch.ones_like(zs)) + (1 - q) * spherical_term(zs, torch.zeros_like(zs))
    p_best = torch.sigmoid(zs[exp.argmax()]).item()
    assert abs(p_best - q) < 0.01


def test_bernoulli_surrogate_has_zero_expected_gradient():
    """Negative control: reward independent of a ~ Bernoulli(p), times log P(a)."""
    n = 400000
    z = torch.full((n,), 0.3, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(z)
    gen = torch.Generator().manual_seed(7)
    a = (torch.rand(n, generator=gen, dtype=torch.float64) < p.detach()).double()
    reward = torch.log(p.detach())             # y = 1 log score, does not depend on a
    logp_a = a * torch.log(p) + (1 - a) * torch.log(1 - p)
    g, = torch.autograd.grad((reward * logp_a).mean(), z)
    per = g * n
    assert abs(per.mean().item()) < 5 * per.std().item() / math.sqrt(n)


def test_detach_matters():
    """Without detaching u, (u - z) cancels and the PG surrogate gives zero z-gradient."""
    z = torch.randn(4, 3, 5, requires_grad=True)
    noise = torch.randn((8,) + tuple(z.shape))
    u = z[None] + 0.2 * noise                   # NOT detached (the bug)
    lp = -0.5 * ((u - z[None]) / 0.2).square()
    g, = torch.autograd.grad((torch.randn_like(u) * lp).sum(), z)
    assert torch.allclose(g, torch.zeros_like(g))
