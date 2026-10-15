"""UniMax sampling shares (Chung et al., ICLR 2023) for a mix directory.

Budget B = updates x batch rows. Sources are visited from the smallest; each gets an equal share of
what is left, but at most MAX_EPOCHS passes over its train rows; what a capped source does not use
goes to the larger ones. Shares = allocation / B.
Writes <mix>/mixspec_unimax.txt and <mix>/unimax.json (rows, allocation, passes, task shares).
"""
import argparse
import json
from pathlib import Path


def unimax(rows, budget, max_epochs):
    alloc, left = {}, float(budget)
    order = sorted(rows, key=rows.get)
    for k, s in enumerate(order):
        fair = left / (len(order) - k)
        alloc[s] = min(fair, max_epochs * rows[s])
        left -= alloc[s]
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--updates", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    a = ap.parse_args()
    mix = Path(a.mix)
    summ = json.loads((mix / "summary.json").read_text())
    rows = {s: n for s, n in summ["rows"].items() if n}
    budget = a.updates * a.batch
    alloc = unimax(rows, budget, a.max_epochs)
    total = sum(alloc.values())
    share = {s: v / total for s, v in alloc.items()}
    spec = ",".join(f"{s}:{share[s]:.6f}" for s in sorted(share))
    (mix / "mixspec_unimax.txt").write_text(spec + "\n")
    from .mix_seeded import task_of
    task_share = {}
    for s, p in share.items():
        with open(mix / s / "train.jsonl") as f:
            t = task_of(s, json.loads(f.readline()))
        task_share[t] = task_share.get(t, 0.0) + p
    passes = {s: alloc[s] / rows[s] for s in rows}
    out = {"budget": budget, "max_epochs": a.max_epochs, "allocated": total, "share": share, "passes": passes,
           "task_share": task_share, "capped": sorted(s for s in rows if passes[s] >= a.max_epochs - 1e-9)}
    print("task share", {k: round(v, 3) for k, v in task_share.items()})
    (mix / "unimax.json").write_text(json.dumps(out, indent=1))
    print(f"budget {budget}, allocated {total:.0f}, sources {len(rows)}, capped at {a.max_epochs} passes: "
          f"{len(out['capped'])}")
    for s in sorted(share, key=share.get, reverse=True)[:8]:
        print(f"  {s:28s} rows {rows[s]:8d} share {share[s]:.4f} passes {passes[s]:.2f}")
    for s in sorted(share, key=share.get)[:5]:
        print(f"  {s:28s} rows {rows[s]:8d} share {share[s]:.5f} passes {passes[s]:.2f}")


if __name__ == "__main__":
    main()
