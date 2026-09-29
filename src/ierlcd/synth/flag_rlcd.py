"""Round-5 refinement, step 2: our RLCD model reads the synthetic train rows and
flags spans for the judge.

  extra  spans with p(entity) >= --hi that the current (Qwen) labels lack, labelled
         with the model's most likely type; greedy, no overlap with each other or
         with a current span
  doubt  current spans with p(entity) <= --lo (logged for the summary; the judge
         already sees every span the two LLMs disagree on)
One shard of the rows per call; output synth2/rlcd_flags/shard-*.jsonl.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from ..data.collator import Collator, add_control_tokens, encode_row
from ..eval.evaluate import predict, rlcd_probs
from ..model.model import load_model
from .llm_annotate import read_jsonl
from .refine import MIX_DIR, S2, SOURCES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--W", type=int, default=32)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    lab_id, txt_id = add_control_tokens(tok)
    collate = Collator(tok, lab_id, txt_id)
    model = load_model(args.encoder, len(tok), args.W, "rlcd", args.ckpt)
    out_path = S2 / "rlcd_flags" / f"shard-{args.shard:05d}-of-{args.num_shards:05d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for src in SOURCES for r in read_jsonl(MIX_DIR / src / "train.jsonl")]
    rows = rows[args.shard::args.num_shards]
    n_extra = n_doubt = 0
    with open(out_path, "w") as f:
        for k in range(0, len(rows), 256):
            chunk, encs = [], []
            for r in rows[k:k + 256]:
                try:
                    encs.append(encode_row(r, tok, args.W))
                    chunk.append(r)
                except NotImplementedError:
                    continue
            with torch.no_grad():
                outs = predict(model, encs, collate, "cuda")
            for r, e, o in zip(chunk, encs, outs):
                p = rlcd_probs(o, 1.0)                              # [1+M, C]
                if p.shape[1] == 0:
                    continue
                p_ent = 1.0 - p[0]
                best = p[1:].argmax(0)
                gold = {(g["start"], g["end"]) for g in r["entities"]}
                gold_lab = {(g["start"], g["end"], g["label"]) for g in r["entities"]}
                taken = [(s, t) for s, t in gold]
                extra = []
                for c in np.argsort(-p_ent):
                    if p_ent[c] < args.hi:
                        break
                    s, t = int(e["cands"][c][0]), int(e["cands"][c][1])
                    if any(s < b and a < t for a, b in taken):
                        continue
                    extra.append([s, t, e["labels"][best[c]], round(float(p_ent[c]), 3)])
                    taken.append((s, t))
                index = {(int(c[0]), int(c[1])): i for i, c in enumerate(e["cands"])}
                doubt = [[s, t, lab, round(float(p_ent[index[(s, t)]]), 3)] for s, t, lab in gold_lab
                         if (s, t) in index and p_ent[index[(s, t)]] <= args.lo]
                n_extra += len(extra)
                n_doubt += len(doubt)
                if extra or doubt:
                    f.write(json.dumps({"id": r["id"], "extra": extra, "doubt": doubt}) + "\n")
    print(f"shard {args.shard}: rows {len(rows)} extra {n_extra} doubt {n_doubt}", flush=True)


if __name__ == "__main__":
    main()
