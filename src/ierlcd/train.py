"""Warm-up and fork training on a source mixture (SPEC 6).

--data is a prepare_mix directory (<source>/<split>.jsonl). --mix gives the
per-step source-family probabilities. Checkpoint selection: macro over sources
of dev_select F1, NLL (macro) breaks ties.
"""
import argparse
import collections
import contextlib
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .collator import Collator, add_control_tokens, encode_row, load_rows
from .evaluate import f1, f1_counts, metrics, predict, with_temperature
from .losses import loss_fn
from .losses import rlcd_loss
from .mixdata import (StepDataset, fewnerd_transform, filter_long, load_eval, load_interned,
                      parse_mix, step_schedule)
from .model import SpanLabelModel


def param_groups(model, enc_lr, head_lr, wd):
    groups = {(e, d): [] for e in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        is_enc = name.startswith("encoder.")
        no_decay = p.ndim < 2 or "norm" in name.lower() or name == "bias"
        groups[(is_enc, not no_decay)].append(p)
    return [{"params": ps, "lr": enc_lr if e else head_lr, "weight_decay": wd if d else 0.0}
            for (e, d), ps in groups.items() if ps]


def encode_split(path, tok, W, limit=None, transform=None):
    rows = filter_long(load_rows(path)[:limit], tok, str(path))
    if transform is not None:
        rows = transform(rows)
    return [encode_row(r, tok, W) for r in rows]


def best_threshold_f1(encs, outs):
    """F1 at the best of 19 probability thresholds (0.05 ... 0.95)."""
    return max(f1(*f1_counts(encs, outs, float(np.log(t / (1 - t)))))["f1"]
               for t in np.linspace(0.05, 0.95, 19))


def evaluate_sources(model, dev, collate, device, head="bce"):
    """Checkpoint selection, identical for both heads: each source's dev F1 at
    its best threshold, with the decision rule the head is used with (RLCD:
    entity-level decoding). Round 1-3 selected RLCD with label decoding at 0.5."""
    res = {}
    for src, encs in dev.items():
        outs = predict(model, encs, collate, device, batch_size=16)
        if head == "rlcd":
            outs = with_temperature(outs, 1.0, "entity")
        res[src] = metrics(encs, outs)
        res[src]["f1_at_0.5"] = res[src]["f1"]
        res[src]["f1"] = best_threshold_f1(encs, outs)
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
    ap.add_argument("--alpha", type=float, default=0.0, help="spherical-score weight (appendix A)")
    ap.add_argument("--head", default="bce", choices=["bce", "rlcd"],
                    help="rlcd: per-span choice over [none, labels], Plackett-Luce + Brier")
    ap.add_argument("--brier_weight", type=float, default=1.0)
    ap.add_argument("--none_weight", type=float, default=1.0,
                    help="RLCD: loss weight of spans whose answer is 'not an entity'")
    ap.add_argument("--fewnerd_fine_only", action="store_true", help="same as --fewnerd_mode fine")
    ap.add_argument("--fewnerd_mode", default="both", choices=["both", "fine", "precedence"])
    ap.add_argument("--desc_dropout", type=float, default=0.0,
                    help="probability of showing a described label by name only (training)")
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_limit", type=int, default=1000, help="dev_select rows per source")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_pool", default="mean", choices=["mean", "marker", "both"])
    ap.add_argument("--head_hidden", type=int, default=512)
    ap.add_argument("--head_out", type=int, default=256)
    ap.add_argument("--head_layers", type=int, default=0,
                    help="extra transformer layers over the encoder output (Laya uses 2)")
    ap.add_argument("--lowercase_aug", type=float, default=0.0,
                    help="share of training rows shown fully lowercased (MIT-style text)")
    ap.add_argument("--anon_labels", type=float, default=0.0,
                    help="share of described labels renamed 'entity k' (GLiNER2-style)")
    args = ap.parse_args()
    # DDP under torchrun: each rank takes batch/world rows of every global step; the
    # loss keeps SPEC's reduction (mean over the step's active rows across all ranks).
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    assert args.batch % world == 0, (args.batch, world)
    main_rank = rank == 0
    out = Path(args.out)
    if main_rank:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2))
        (out / "model_config.json").write_text(json.dumps(
            {"hidden": args.head_hidden, "out": args.head_out, "label_pool": args.label_pool,
             "head_layers": args.head_layers}))

    random.seed(args.seed)
    torch.manual_seed(args.seed + rank)
    device = f"cuda:{local}"
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    mix = parse_mix(args.mix)
    t0 = time.time()
    rows = {src: filter_long(load_interned(Path(args.data) / src / "train.jsonl"), tok,
                             f"{src}/train") for src in mix}
    if args.fewnerd_fine_only:
        args.fewnerd_mode = "fine"
    tf = fewnerd_transform(args.data, args.fewnerd_mode)
    if tf and "fewnerd" in rows:
        rows["fewnerd"] = tf(rows["fewnerd"])
    dev = load_eval(args.data, "dev_select", list(mix), tok, args.W, args.eval_limit,
                    args.fewnerd_mode) if main_rank else {}
    if main_rank:
        print(f"world={world} loaded train={ {s: len(r) for s, r in rows.items()} } "
              f"dev={ {s: len(d) for s, d in dev.items()} } in {time.time() - t0:.1f}s", flush=True)
    steps = step_schedule({s: len(r) for s, r in rows.items()}, mix, args.updates, args.batch,
                          args.seed)
    loader = torch.utils.data.DataLoader(
        StepDataset(rows, steps, tok, collate, args.W, args.micro, rank, world, args.desc_dropout,
                    args.lowercase_aug, args.anon_labels),
        batch_size=None,
        shuffle=False, num_workers=args.workers, prefetch_factor=4, persistent_workers=False)

    base = SpanLabelModel(args.encoder, len(tok), W=args.W, head=args.head, hidden=args.head_hidden,
                          out=args.head_out, label_pool=args.label_pool,
                          head_layers=args.head_layers).to(device)
    if args.init:
        base.load_state_dict(torch.load(args.init, map_location=device))
    model = DDP(base, device_ids=[local]) if world > 1 else base
    opt = torch.optim.AdamW(param_groups(base, args.enc_lr, args.head_lr, args.wd))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * args.updates), args.updates)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1 + 1000 * rank)
    forked = args.init is not None

    log = open(out / "log.jsonl", "a") if main_rank else None
    best = (-1.0, math.inf)
    t_start = time.time()
    query_stats = collections.Counter()
    for step, (prepared, qstats) in enumerate(loader, start=1):
        lam = args.lam * min(1.0, (step - 1) / args.lam_ramp) if forked and args.lam_ramp else args.lam
        query_stats.update(qstats)
        opt.zero_grad(set_to_none=True)
        actives = [int((t["valid"].sum((-2, -1)) > 0).sum()) for _, t in prepared]
        n_total = torch.tensor(float(sum(actives)), device=device)
        if world > 1:
            dist.all_reduce(n_total)
        n_total = n_total.item()
        stats_acc = {}
        for k_mb, ((inputs, tgt), n_act) in enumerate(zip(prepared, actives)):
            last = k_mb == len(prepared) - 1
            sync = contextlib.nullcontext() if (last or world == 1) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(**{k: v.to(device, non_blocking=True) for k, v in inputs.items()})
                if n_act == 0:                      # every rank must reach backward under DDP
                    (z.float().sum() * 0).backward()
                    continue
                stats = {}
                if args.head == "rlcd":
                    loss = rlcd_loss(z, tgt["y"].to(device), tgt["valid"].to(device),
                                     tgt["query_mask"].to(device), tgt["cand_mask"].to(device),
                                     brier_weight=args.brier_weight, stats=stats,
                                     none_weight=args.none_weight)
                else:
                    loss = loss_fn(z, tgt["y"].to(device), tgt["valid"].to(device), mode=args.mode,
                                   K=args.K, sigma=args.sigma, lam=lam, generator=noise_gen,
                                   stats=stats, alpha=args.alpha)
                # DDP averages gradients over ranks, hence the factor `world`
                (loss * n_act * world / n_total).backward()
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v * n_act / n_total
        if world > 1:                           # logged stats: sum of every rank's share
            keys = ("clean_bce", "mean_reward", "auxiliary", "rlcd_ce", "rlcd_brier")
            t = torch.tensor([stats_acc.get(k, 0.0) for k in keys], device=device)
            dist.all_reduce(t)
            stats_acc = {k: v for k, v in zip(keys, t.tolist()) if k in stats_acc}
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        if main_rank and (step % 50 == 0 or step == 1):
            rec = {"step": step, "lam": lam, "grad_norm": gnorm, "lr_enc": sched.get_last_lr()[0],
                   "sec_per_update": (time.time() - t_start) / step, **stats_acc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 500 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.updates:
            if main_rank:
                res = evaluate_sources(base, dev, collate, device, args.head)
                res["step"] = step
                res["query_stats_cumulative_rank0"] = dict(query_stats)
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print("EVAL", step, json.dumps({s: round(res[s]["f1"], 4) for s in dev}),
                      "macro_f1", round(res["macro_f1"], 4), "macro_nll",
                      round(res["macro_nll"], 5), flush=True)
                key = (res["macro_f1"], -res["macro_nll"])
                if key > (best[0], -best[1]):
                    best = (res["macro_f1"], res["macro_nll"])
                    torch.save(base.state_dict(), out / "best.pt")
            if world > 1:
                dist.barrier()
    if main_rank:
        torch.save(base.state_dict(), out / "final.pt")
        (out / "query_stats_rank0.json").write_text(json.dumps(dict(query_stats), indent=1))
        print(f"done {args.updates} updates in {time.time() - t_start:.0f}s; "
              f"best dev macro f1={best[0]:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
