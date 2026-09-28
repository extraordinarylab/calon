"""Warm-up and fork training on a source mixture (SPEC 6).

--data is a prepare_mix directory (<source>/<split>.jsonl). --mix gives the
per-step source-family probabilities. Checkpoint selection: macro over sources
of dev_select F1, NLL (macro) breaks ties.
"""
import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .collator import Collator, add_control_tokens, encode_row, load_rows
from .evaluate import metrics, predict
from .losses import loss_fn
from .mixdata import StepDataset, load_eval, load_interned, parse_mix, step_schedule
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


def evaluate_sources(model, dev, collate, device):
    res = {src: metrics(encs, predict(model, encs, collate, device, batch_size=16))
           for src, encs in dev.items()}
    res["macro_f1"] = float(np.mean([res[s]["f1"] for s in dev]))
    res["macro_nll"] = float(np.mean([res[s]["nll_row_mean"] for s in dev]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default="finerweb:0.25,fewnerd:0.30,pile:0.30,massive:0.15")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--init", help="checkpoint to fork from")
    ap.add_argument("--mode", default="bce", choices=["bce", "bce_scaled", "direct", "pg"])
    ap.add_argument("--updates", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
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
    ap.add_argument("--eval_limit", type=int, default=1000, help="dev_select rows per source")
    ap.add_argument("--workers", type=int, default=12)
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
    collate = Collator(tok, lab_id, txt_id)
    mix = parse_mix(args.mix)
    t0 = time.time()
    rows = {src: load_interned(Path(args.data) / src / "train.jsonl") for src in mix}
    dev = load_eval(args.data, "dev_select", list(mix), tok, args.W, args.eval_limit)
    print(f"loaded train={ {s: len(r) for s, r in rows.items()} } "
          f"dev={ {s: len(d) for s, d in dev.items()} } in {time.time() - t0:.1f}s", flush=True)
    steps = step_schedule({s: len(r) for s, r in rows.items()}, mix, args.updates, args.batch,
                          args.seed)
    loader = torch.utils.data.DataLoader(
        StepDataset(rows, steps, tok, collate, args.W, args.micro), batch_size=None,
        shuffle=False, num_workers=args.workers, prefetch_factor=4, persistent_workers=False)

    model = SpanLabelModel(args.encoder, len(tok), W=args.W).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.AdamW(param_groups(model, args.enc_lr, args.head_lr, args.wd))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * args.updates), args.updates)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    forked = args.init is not None

    log = open(out / "log.jsonl", "a")
    best = (-1.0, math.inf)
    t_start = time.time()
    query_stats = collections.Counter()
    for step, (prepared, qstats) in enumerate(loader, start=1):
        lam = args.lam * min(1.0, (step - 1) / args.lam_ramp) if forked and args.lam_ramp else args.lam
        query_stats.update(qstats)
        opt.zero_grad(set_to_none=True)
        actives = [int((t["valid"].sum((-2, -1)) > 0).sum()) for _, t in prepared]
        n_active_total = sum(actives)
        stats_acc = {}
        for (inputs, tgt), n_act in zip(prepared, actives):
            if n_act == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(**{k: v.to(device, non_blocking=True) for k, v in inputs.items()})
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
            if step % 500 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.updates:
            res = evaluate_sources(model, dev, collate, device)
            res["step"] = step
            res["query_stats_cumulative"] = dict(query_stats)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            print("EVAL", step, json.dumps({s: round(res[s]["f1"], 4) for s in dev}),
                  "macro_f1", round(res["macro_f1"], 4), "macro_nll", round(res["macro_nll"], 5),
                  flush=True)
            key = (res["macro_f1"], -res["macro_nll"])
            if key > (best[0], -best[1]):
                best = (res["macro_f1"], res["macro_nll"])
                torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "final.pt")
    (out / "query_stats.json").write_text(json.dumps(dict(query_stats), indent=1))
    print(f"done {args.updates} updates in {time.time() - t_start:.0f}s; best dev macro f1={best[0]:.4f}")


if __name__ == "__main__":
    main()
