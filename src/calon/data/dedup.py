"""Near-duplicate removal inside each synthetic source (round 6, from the distribution analysis).

Texts of a source's train split are embedded (all-mpnet-base-v2, normalised, as in the analysis);
a text is dropped when an earlier text of the same source (file order, which is shuffled) has
cosine >= THRESHOLD. Comparing with every earlier text, kept or not, can drop slightly more than
a greedy pass over kept texts would. All rows of a dropped text go (EE sources have a trigger row
and an argument row per text). dev / test files are linked unchanged.
Output: <out>/<source>/train.jsonl plus links; summary.json with kept / dropped per source.
"""
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

THRESHOLD = 0.9


def max_earlier_cos(emb, block=8192):
    import torch
    x = torch.tensor(emb, dtype=torch.float16, device="cuda")
    best = torch.full((len(x),), -1.0, device="cuda")
    for i in range(0, len(x), block):
        q = x[i:i + block]
        for j in range(0, i + block, block):
            s = (q @ x[j:j + block].T).float()
            if j + block > i:            # the diagonal block: only earlier texts count
                r = torch.arange(i, i + len(q), device="cuda")[:, None]
                c = torch.arange(j, j + s.shape[1], device="cuda")[None, :]
                s = s.masked_fill(c >= r, -1.0)
            best[i:i + len(q)] = torch.maximum(best[i:i + len(q)], s.max(1).values)
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="mix directory whose sources are read")
    ap.add_argument("--sources", required=True, help="comma-separated synthetic sources to deduplicate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cuda")
    pool = model.start_multi_process_pool() if os.environ.get("DEDUP_GPUS", "1") != "1" else None
    summary = {}
    for src in args.sources.split(","):
        d = Path(args.mix) / src
        rows = [json.loads(line) for line in open(d / "train.jsonl")]
        texts = list(dict.fromkeys(r["text"] for r in rows))
        if pool:
            emb = model.encode_multi_process(texts, pool, batch_size=256, normalize_embeddings=True)
        else:
            emb = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        m = max_earlier_cos(np.asarray(emb))
        drop = {t for t, s in zip(texts, m) if s >= args.threshold}
        kept = [r for r in rows if r["text"] not in drop]
        o = Path(args.out) / src
        o.mkdir(parents=True, exist_ok=True)
        with open(o / "train.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for p in d.iterdir():
            if p.name != "train.jsonl" and not (o / p.name).exists():
                os.symlink(p.resolve(), o / p.name)
        hist = collections.Counter(np.digitize(m, [0.7, 0.8, 0.9, 0.95]).tolist())
        summary[src] = {"texts": len(texts), "dropped_texts": len(drop),
                        "dropped_share": len(drop) / max(1, len(texts)),
                        "rows": len(rows), "rows_kept": len(kept),
                        "nn_cos_bins": {b: hist.get(k, 0) for k, b in enumerate(["<0.7", "0.7-0.8", "0.8-0.9",
                                                                                  "0.9-0.95", ">=0.95"])}}
        print(src, json.dumps(summary[src]), flush=True)
    if pool:
        model.stop_multi_process_pool(pool)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
