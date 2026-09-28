"""Warm-up and fork training (SPEC 6)."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .collator import Collator, add_control_tokens, encode_row, load_rows
from .evaluate import metrics, predict
from .losses import loss_fn
from .model import SpanLabelModel


def param_groups(model, enc_lr, head_lr, wd):
    groups = {(e, d): [] for e in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        is_enc = name.startswith("encoder.")
        no_decay = p.ndim < 2 or "norm" in name.lower() or name == "bias"
        groups[(is_enc, not no_decay)].append(p)
    return [{"params": ps, "lr": enc_lr if e else head_lr, "weight_decay": wd if d else 0.0}
            for (e, d), ps in groups.items() if ps]


def encode_split(path, tok, W, limit=None):
    rows = load_rows(path)[:limit]
    return [encode_row(r, tok, W) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--init", help="checkpoint to fork from")
    ap.add_argument("--mode", default="bce", choices=["bce", "bce_scaled", "direct", "pg"])
    ap.add_argument("--updates", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--lam_ramp", type=int, default=200)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_limit", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    t0 = time.time()
    train = encode_split(Path(args.data) / "train.jsonl", tok, args.W)
    dev = encode_split(Path(args.data) / "dev_select.jsonl", tok, args.W, args.eval_limit)
    print(f"encoded train={len(train)} dev={len(dev)} in {time.time() - t0:.1f}s", flush=True)
    collate = Collator(tok, lab_id, txt_id)

    model = SpanLabelModel(args.encoder, len(tok), W=args.W).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.AdamW(param_groups(model, args.enc_lr, args.head_lr, args.wd))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * args.updates), args.updates)
    data_rng = random.Random(args.seed)                    # identical row stream across arms
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    forked = args.init is not None

    log = open(out / "log.jsonl", "a")
    best = (-1.0, math.inf)
    t_start = time.time()
    for step in range(1, args.updates + 1):
        lam = args.lam * min(1.0, (step - 1) / args.lam_ramp) if forked and args.lam_ramp else args.lam
        rows = [train[data_rng.randrange(len(train))] for _ in range(args.batch)]
        opt.zero_grad(set_to_none=True)
        stats_acc, n_active_total = {}, 0
        micro = [rows[k:k + args.micro] for k in range(0, len(rows), args.micro)]
        prepared = [collate(mb) for mb in micro]
        actives = [int((t["valid"].sum((-2, -1)) > 0).sum()) for _, t in prepared]
        n_active_total = sum(actives)
        for (inputs, tgt), n_act in zip(prepared, actives):
            if n_act == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(**{k: v.to(device) for k, v in inputs.items()})
            stats = {}
            loss = loss_fn(z, tgt["y"].to(device), tgt["valid"].to(device), mode=args.mode,
                           K=args.K, sigma=args.sigma, lam=lam, generator=noise_gen, stats=stats)
            (loss * n_act / n_active_total).backward()
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v * n_act / n_active_total
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "lam": lam, "grad_norm": gnorm, "lr_enc": sched.get_last_lr()[0],
                   "sec_per_update": (time.time() - t_start) / step, **stats_acc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.updates:
            res = metrics(dev, predict(model, dev, collate, device))
            res["step"] = step
            print("EVAL", json.dumps(res), flush=True)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            key = (res["f1"], -res["nll_row_mean"])
            if key > (best[0], -best[1]):
                best = (res["f1"], res["nll_row_mean"])
                torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "final.pt")
    print(f"done {args.updates} updates in {time.time() - t_start:.0f}s; best dev f1={best[0]:.4f}")


if __name__ == "__main__":
    main()
