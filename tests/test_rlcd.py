"""RLCD head: per-span choice over [none, labels], Plackett-Luce + Brier."""
import numpy as np
import torch
import torch.nn.functional as F

from ierlcd.evaluate import decode, rlcd_brier, with_temperature
from ierlcd.losses import rlcd_log_probs, rlcd_loss


def setup():
    # 1 row, 2 labels, 3 candidates; cand 0 gold for label 0, cand 1 gold for both, cand 2 none
    B, M, C = 1, 2, 3
    y = torch.zeros(B, M, C)
    y[0, 0, 0] = 1
    y[0, 0, 1] = y[0, 1, 1] = 1
    valid = torch.ones(B, M, C, dtype=torch.bool)
    qmask = torch.ones(B, M, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)
    return y, valid, qmask, cmask


def manual(u, y, brier_weight=1.0):
    total = []
    for c in range(u.shape[2]):
        p = F.softmax(u[0, :, c], 0)
        gold = [1 + m for m in range(y.shape[1]) if y[0, m, c] > 0] or [0]
        ce = -torch.log(p[gold].sum())
        t = torch.zeros_like(p)
        t[gold] = 1 / len(gold)
        total.append(ce + brier_weight * 0.5 * ((p - t) ** 2).sum())
    return torch.stack(total).mean()


def test_loss_matches_manual():
    y, valid, qmask, cmask = setup()
    u = torch.randn(1, 3, 3, generator=torch.Generator().manual_seed(0))
    got = rlcd_loss(u, y, valid, qmask, cmask)
    assert torch.allclose(got, manual(u, y), atol=1e-6)


def test_padding_labels_and_candidates_are_ignored():
    y, valid, qmask, cmask = setup()
    u = torch.randn(1, 3, 3, generator=torch.Generator().manual_seed(1))
    base = rlcd_loss(u, y, valid, qmask, cmask)
    up = F.pad(u, (0, 2, 0, 1), value=5.0)          # one padded label, two padded cands
    yp = F.pad(y, (0, 2, 0, 1))
    vp = F.pad(valid, (0, 2, 0, 1), value=False)
    qp = F.pad(qmask, (0, 1), value=False)
    cp = F.pad(cmask, (0, 2), value=False)
    assert torch.allclose(rlcd_loss(up, yp, vp, qp, cp), base, atol=1e-6)


def test_unknown_query_skips_spans_without_gold():
    y, valid, qmask, cmask = setup()
    y = F.pad(y, (0, 0, 0, 1))                      # third label, status unknown
    valid = torch.cat([valid, torch.zeros(1, 1, 3, dtype=torch.bool)], 1)
    qmask = F.pad(qmask, (0, 1), value=True)
    u = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(2), requires_grad=True)
    rlcd_loss(u, y, valid, qmask, cmask).backward()
    assert torch.all(u.grad[0, :, 2] == 0)           # cand 2 has no gold: skipped
    assert torch.all(u.grad[0, 3] == 0)              # unknown label never scored
    assert torch.any(u.grad[0, :, 0] != 0)


def test_log_probs_mask_unqueried_labels():
    u = torch.zeros(1, 3, 2)
    lp = rlcd_log_probs(u, torch.tensor([[True, False]]))
    assert torch.isneginf(lp[0, 2]).all()
    assert torch.allclose(lp[0, :2].exp().sum(0), torch.ones(2))


def test_decode_keeps_one_label_per_span():
    o = {"u": np.array([[0.0, 0.0], [2.0, -1.0], [1.9, 3.0]]), "known": np.array([True, True]),
         "y": np.zeros((2, 2)), "valid": np.ones((2, 2), bool), "unknown_row": False}
    (v,) = with_temperature([o], 1.0)
    e = {"cands": [(0, 4, 0, 0), (5, 9, 1, 1)], "labels": ["a", "b"]}
    preds = decode(e, v["z"], np.log(0.1 / 0.9), v["choice"])
    assert preds == {(0, 4, "a"), (5, 9, "b")}       # both labels pass 0.1 on span 0; only argmax kept
    assert 0 <= rlcd_brier([o], 1.0) <= 1
