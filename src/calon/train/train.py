"""Warm-up and fork training on a source mixture.

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

from ..data.collator import Collator, add_control_tokens, load_rows
from ..data.mixdata import (
    PackedRows,
    StepDataset,
    fewnerd_transform,
    filter_long,
    load_eval,
    load_interned,
    parse_mix,
    step_schedule,
)
from ..eval.evaluate import best_re_threshold, f1, f1_counts, metrics, predict, predict_re, with_temperature
from ..model.losses import loss_fn, rlcd_loss, rlcd_pg
from ..model.model import SpanLabelModel, canonical_head


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
    from ..data.mixdata import encode_kept
    return encode_kept(rows, tok, W, str(path))


def best_threshold_f1(encs, outs):
    """F1 at the best of 19 probability thresholds (0.05 ... 0.95)."""
    return max(f1(*f1_counts(encs, outs, float(np.log(t / (1 - t)))))["f1"]
               for t in np.linspace(0.05, 0.95, 19))


def eval_part(model, dev, collate, device, head="bce"):
    """Per-source dev results of the sources in `dev` (one rank's share): ({key: metrics}, keys)."""
    res, keys = {}, []
    for src, encs in dev.items():
        outs = predict(model, encs, collate, device, batch_size=16)
        if head == "rlcd":
            outs = with_temperature(outs, 1.0, "entity")
        res[src] = metrics(encs, outs)
        res[src]["f1_at_0.5"] = res[src]["f1"]
        res[src]["f1"] = best_threshold_f1(encs, outs)
        keys.append(src)
        if getattr(model, "relations", False) and any(e.get("rel_ids") for e in encs):
            v, t = best_re_threshold(encs, predict_re(model, encs, collate, device))
            res[f"{src}_re"] = {"f1": v, "threshold": t}
            keys.append(f"{src}_re")
    return res, keys


def evaluate_sources(parts, order):
    """Checkpoint selection, identical for both heads: each source's dev F1 at
    its best threshold, with the decision rule the head is used with (RLCD:
    entity-level decoding). Round 1-3 selected RLCD with label decoding at 0.5.
    Sources with relations add "<source>_re": relation F1 with gold entities at its
    best threshold, averaged into macro_f1 like one more source.
    parts: eval_part results of every rank (the dev sources are split over ranks); order: the
    sources in mix order, so keys and macros do not depend on the split."""
    res, got = {}, set()
    for r, k in parts:
        res.update(r)
        got.update(k)
    keys = [k for s in order for k in (s, f"{s}_re") if k in got]
    res["macro_f1"] = float(np.mean([res[s]["f1"] for s in keys]))
    res["macro_nll"] = float(np.mean([res[s]["nll_row_mean"] for s in order if s in got]))
    res["_keys"] = keys
    return res


def relation_params(model):
    return [p for n, p in model.named_parameters()
            if n.split(".")[0] in ("between", "pair_mlp", "rel_mlp", "rel_bias", "none_rel", "pair_proj", "rel_proj",
                                   "rel_kind", "none_rel_tok", "rel_mix", "Ur", "Vr")]


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
    # RLCD policy-gradient term (losses.rlcd_pg); off at rl_lam 0. --mode / --K / --sigma / --lam above
    # belong to the BCE head only.
    ap.add_argument("--rl_lam", type=float, default=0.0, help="weight of the RLCD policy-gradient term")
    ap.add_argument("--rl_K", type=int, default=8)
    ap.add_argument("--rl_sigma", type=float, default=1.0)
    ap.add_argument("--rl_fp_cost", type=float, default=1.0, help="reward of a predicted label that is not gold")
    ap.add_argument("--rl_re", type=int, default=1, help="1: the term on the relation head too")
    ap.add_argument("--head", default="bce", type=canonical_head, choices=["bce", "rlcd"],
                    help="rlcd (alias calon): per-span choice over [none, labels], Plackett-Luce + Brier")
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
    ap.add_argument("--head_arch", default="mlp", choices=["mlp", "attn"],
                    help="attn: self-attention over [none, labels, candidates] and a low-rank bilinear score")
    ap.add_argument("--attn_dim", type=int, default=256)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--attn_layers", type=int, default=2)
    ap.add_argument("--bilinear_rank", type=int, default=64)
    ap.add_argument("--head_layers", type=int, default=0,
                    help="extra transformer layers over the encoder output (Laya uses 2)")
    ap.add_argument("--lowercase_aug", type=float, default=0.0,
                    help="share of training rows shown fully lowercased (MIT-style text)")
    ap.add_argument("--anon_labels", type=float, default=0.0,
                    help="share of described labels renamed 'entity k' (GLiNER2-style)")
    ap.add_argument("--relations", action="store_true",
                    help="joint NER + RE: relation head over ordered entity pairs (rows with rel_queries)")
    ap.add_argument("--re_weight", type=float, default=1.0, help="weight of the relation loss")
    ap.add_argument("--re_none_weight", type=float, default=0.1,
                    help="RLCD: loss weight of pairs whose answer is 'no relation'")
    ap.add_argument("--resume", type=int, default=1,
                    help="1: continue from <out>/ckpt.pt when it exists (any number of nodes / GPUs)")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="steps between resume checkpoints (0 = at every evaluation)")
    ap.add_argument("--stop_after", type=int, default=0,
                    help="testing resume: leave after this step (after its checkpoint), as a killed job would")
    args = ap.parse_args()
    # DDP under torchrun: each rank takes batch/world rows of every global step; the
    # loss keeps the single-process reduction (mean over the step's active rows across all ranks).
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
             "head_layers": args.head_layers, "relations": args.relations, "head_arch": args.head_arch,
             "attn_dim": args.attn_dim, "attn_heads": args.attn_heads, "attn_layers": args.attn_layers,
             "bilinear_rank": args.bilinear_rank}))

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
    rows = {s: PackedRows(r) for s, r in rows.items()}   # shared with loader workers, no copies
    # dev sources split over the ranks (round robin in mix order); every rank evaluates its share
    dev_order = [s for s in mix if (Path(args.data) / s / "dev_select.jsonl").exists()
                 and (Path(args.data) / s / "dev_select.jsonl").stat().st_size > 0]
    dev = load_eval(args.data, "dev_select", dev_order[rank::world], tok, args.W, args.eval_limit,
                    args.fewnerd_mode)
    if main_rank:
        print(f"world={world} loaded train={ {s: len(r) for s, r in rows.items()} } "
              f"dev sources={len(dev_order)} (rank 0: { {s: len(d) for s, d in dev.items()} }) "
              f"in {time.time() - t0:.1f}s", flush=True)
    steps = step_schedule({s: len(r) for s, r in rows.items()}, mix, args.updates, args.batch,
                          args.seed)
    # resume: the step schedule is fixed by the seed, so the remaining steps are steps[done:] on any
    # world size; every rank reads the one checkpoint rank 0 wrote (model, optimizer, scheduler)
    ckpt_path = out / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) \
        if args.resume and ckpt_path.exists() else None
    done = ckpt["step"] if ckpt else 0
    loader = torch.utils.data.DataLoader(
        StepDataset(rows, steps[done:], tok, collate, args.W, args.micro, rank, world, args.desc_dropout,
                    args.lowercase_aug, args.anon_labels),
        batch_size=None,
        shuffle=False, num_workers=args.workers, prefetch_factor=4, persistent_workers=False)

    base = SpanLabelModel(args.encoder, len(tok), W=args.W, head=args.head, hidden=args.head_hidden,
                          out=args.head_out, label_pool=args.label_pool,
                          head_layers=args.head_layers, relations=args.relations, head_arch=args.head_arch,
                          attn_dim=args.attn_dim, attn_heads=args.attn_heads, attn_layers=args.attn_layers,
                          bilinear_rank=args.bilinear_rank).to(device)
    # the relation head is always an RLCD choice (no relation vs the queried relations); with --head bce only
    # the entity decisions are independent sigmoids (ablation, 2026-10-06)
    rel_params = relation_params(base) if args.relations else []
    if args.init and not ckpt:
        base.load_state_dict(torch.load(args.init, map_location=device))
    model = DDP(base, device_ids=[local]) if world > 1 else base
    opt = torch.optim.AdamW(param_groups(base, args.enc_lr, args.head_lr, args.wd))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * args.updates), args.updates)
    # noise for the sampled losses: a fresh stream per rank and start step (the rank count may change)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1 + 1000 * rank + 7919 * done)
    forked = args.init is not None

    best = (-1.0, math.inf)
    query_stats = collections.Counter()
    if ckpt:
        base.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        best, query_stats = tuple(ckpt["best"]), collections.Counter(ckpt["query_stats"])
        if main_rank:
            # records written after the checkpoint are redone
            for name in ("log.jsonl", "eval.jsonl"):
                p = out / name
                if p.exists():
                    keep = [x for x in p.read_text().splitlines() if x and json.loads(x)["step"] <= done]
                    p.write_text("".join(x + "\n" for x in keep))
            print(f"resumed from step {done} (best dev macro f1 {best[0]:.4f}), world={world}", flush=True)
        del ckpt
    ckpt_every = args.ckpt_every or args.eval_every

    def save_ckpt(step):
        """Atomic: written to a temporary name, then renamed over the previous checkpoint."""
        if main_rank:
            tmp = out / "ckpt.pt.tmp"
            torch.save({"model": base.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": list(best), "query_stats": dict(query_stats),
                        "world_size": world}, tmp)
            os.replace(tmp, ckpt_path)
        if world > 1:
            dist.barrier()

    log = open(out / "log.jsonl", "a") if main_rank else None
    t_start = time.time()
    for step, (prepared, qstats) in enumerate(loader, start=done + 1):
        lam = args.lam * min(1.0, (step - 1) / args.lam_ramp) if forked and args.lam_ramp else args.lam
        rl_lam = args.rl_lam * min(1.0, (step - 1) / args.lam_ramp) if args.lam_ramp else args.rl_lam
        query_stats.update(qstats)
        opt.zero_grad(set_to_none=True)
        actives = [int((t["valid"].sum((-2, -1)) > 0).sum()) for _, t in prepared]
        # relation rows are normalised on their own: mean over the step's active RE rows
        re_actives = [int((t["rel"]["valid"].sum((-2, -1)) > 0).sum()) if "rel" in t else 0
                      for _, t in prepared]
        totals = torch.tensor([float(sum(actives)), float(sum(re_actives))], device=device)
        if world > 1:
            dist.all_reduce(totals)
        n_total, r_total = totals.tolist()
        stats_acc = {}
        for k_mb, ((inputs, tgt), n_act, r_act) in enumerate(zip(prepared, actives, re_actives)):
            last = k_mb == len(prepared) - 1
            sync = contextlib.nullcontext() if (last or world == 1) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_mb = model(**{k: v.to(device, non_blocking=True) for k, v in inputs.items()})
                z, zr = out_mb if isinstance(out_mb, tuple) else (out_mb, None)
                # every rank must reach backward with every parameter used under DDP
                total = z.float().sum() * 0 + sum(p.sum() for p in rel_params) * 0
                if zr is not None:
                    total = total + zr.float().sum() * 0
                stats = {}
                if n_act:
                    if args.head == "rlcd":
                        loss = rlcd_loss(z, tgt["y"].to(device), tgt["valid"].to(device),
                                         tgt["query_mask"].to(device), tgt["cand_mask"].to(device),
                                         brier_weight=args.brier_weight, stats=stats,
                                         none_weight=args.none_weight)
                        if args.rl_lam:
                            loss = loss + rl_lam * rlcd_pg(
                                z, tgt["y"].to(device), tgt["valid"].to(device), tgt["query_mask"].to(device),
                                tgt["cand_mask"].to(device), K=args.rl_K, sigma=args.rl_sigma,
                                fp_cost=args.rl_fp_cost, generator=noise_gen, stats=stats,
                                none_weight=args.none_weight)
                    else:
                        loss = loss_fn(z, tgt["y"].to(device), tgt["valid"].to(device), mode=args.mode,
                                       K=args.K, sigma=args.sigma, lam=lam, generator=noise_gen,
                                       stats=stats, alpha=args.alpha)
                    # DDP averages gradients over ranks, hence the factor `world`
                    total = total + loss * n_act * world / n_total
                if r_act and zr is not None:
                    rt, rstats = tgt["rel"], {}
                    re_loss = rlcd_loss(zr, rt["y"].to(device), rt["valid"].to(device),
                                        rt["query_mask"].to(device), rt["cand_mask"].to(device),
                                        brier_weight=args.brier_weight, stats=rstats,
                                        none_weight=args.re_none_weight)
                    if args.rl_lam and args.rl_re:
                        re_loss = re_loss + rl_lam * rlcd_pg(
                            zr, rt["y"].to(device), rt["valid"].to(device), rt["query_mask"].to(device),
                            rt["cand_mask"].to(device), K=args.rl_K, sigma=args.rl_sigma,
                            fp_cost=args.rl_fp_cost, generator=noise_gen, none_weight=args.re_none_weight)
                    total = total + args.re_weight * re_loss * r_act * world / r_total
                    stats_acc["re_ce"] = stats_acc.get("re_ce", 0.0) + rstats.get("rlcd_ce", 0.0) * r_act / r_total
                total.backward()
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v * n_act / n_total
        if world > 1:                           # logged stats: sum of every rank's share
            keys = ("clean_bce", "mean_reward", "auxiliary", "rlcd_ce", "rlcd_brier", "re_ce", "rl_reward",
                    "rl_pos_rate")
            t = torch.tensor([stats_acc.get(k, 0.0) for k in keys], device=device)
            dist.all_reduce(t)
            stats_acc = {k: v for k, v in zip(keys, t.tolist()) if k in stats_acc}
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        if main_rank and (step % 50 == 0 or step == 1):
            rec = {"step": step, "lam": lam, "grad_norm": gnorm, "lr_enc": sched.get_last_lr()[0],
                   "sec_per_update": (time.time() - t_start) / (step - done), **stats_acc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 500 == 0 or step == 1:
                print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.updates:
            t_eval = time.time()
            part = eval_part(base, dev, collate, device, args.head)
            parts = [None] * world
            if world > 1:
                dist.all_gather_object(parts, part)
            else:
                parts = [part]
            if main_rank:
                res = evaluate_sources(parts, dev_order)
                res["step"] = step
                res["eval_seconds"] = time.time() - t_eval
                res["query_stats_cumulative_rank0"] = dict(query_stats)
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print("EVAL", step, json.dumps({s: round(res[s]["f1"], 4) for s in res["_keys"]}),
                      "macro_f1", round(res["macro_f1"], 4), "macro_nll",
                      round(res["macro_nll"], 5), flush=True)
                key = (res["macro_f1"], -res["macro_nll"])
                if key > (best[0], -best[1]):
                    best = (res["macro_f1"], res["macro_nll"])
                    torch.save(base.state_dict(), out / "best.pt")
            if world > 1:
                dist.barrier()
        if step % ckpt_every == 0 or step == args.updates:
            save_ckpt(step)
        if args.stop_after and step >= args.stop_after:
            if main_rank:
                print(f"stopped after step {step} (--stop_after)", flush=True)
            if world > 1:
                dist.destroy_process_group()
            return
    if main_rank:
        torch.save(base.state_dict(), out / "final.pt")
        (out / "query_stats_rank0.json").write_text(json.dumps(dict(query_stats), indent=1))
        print(f"done {args.updates} updates in {time.time() - t_start:.0f}s this run (from step {done}); "
              f"best dev macro f1={best[0]:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
