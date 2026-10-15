"""losses.rlcd_pg: the policy gradient moves utilities towards the rewarded decision."""
import torch

from calon.model.losses import rlcd_pg


def grad_for(gold_label, K=4096, fp_cost=1.0):
    # one row, two queried labels, one candidate span; utilities tied at 0
    u = torch.zeros(1, 3, 1, requires_grad=True)
    y = torch.zeros(1, 2, 1)
    if gold_label is not None:
        y[0, gold_label, 0] = 1
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    cand_mask = torch.ones(1, 1, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    stats = {}
    rlcd_pg(u, y, valid, query_mask, cand_mask, K=K, sigma=1.0, fp_cost=fp_cost, generator=g, stats=stats).backward()
    return u.grad[0, :, 0], stats


def test_gold_label_is_pushed_up():
    g, stats = grad_for(0)
    # gradient descent: the gold label's utility rises, the wrong label's falls (none's reward, 0,
    # is the mean of the tied choices, so its gradient is about zero here)
    assert g[1] < 0 and g[2] > 0 and abs(float(g[0])) < abs(float(g[1]))
    assert 0 < stats["rl_pos_rate"] <= 1


def test_no_gold_pushes_labels_down():
    g, _ = grad_for(None)
    assert g[1] > 0 and g[2] > 0 and g[0] < 0


def test_unknown_row_skips_spans_without_gold():
    u = torch.zeros(1, 3, 1, requires_grad=True)
    y = torch.zeros(1, 2, 1)
    valid = torch.tensor([[[True], [False]]])            # the second query has unknown status
    loss = rlcd_pg(u, y, valid, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), K=4)
    loss.backward()
    assert float(loss) == 0 and torch.all(u.grad == 0)
