"""Source-mixture sampling and lazy per-step encoding (SPEC 2.5).

Each update draws a source family by probability, then a train row uniformly,
then its query subset (prepare_mix.sample_queries). The whole step schedule
comes from one seeded RNG in the main process, so every arm sees the same rows
and queries; DataLoader workers only tokenize and collate.
"""
import collections
import json
import random
from pathlib import Path

import torch

from .alignment import MAX_TEXT_TOKENS
from .collator import encode_row
from .prepare_mix import sample_queries


def load_interned(path):
    """Rows with identical query dicts shared (Few-NERD has 74 per row)."""
    pool, rows = {}, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["queries"] = [pool.setdefault(tuple(sorted(q.items())), q) for q in r["queries"]]
            rows.append(r)
    return rows


def parse_mix(spec):
    out = {}
    for part in spec.split(","):
        name, p = part.split(":")
        out[name] = float(p)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def step_schedule(sizes, mix, updates, batch, seed):
    rng = random.Random(seed)
    names, probs = list(mix), [mix[n] for n in mix]
    steps = []
    for _ in range(updates):
        srcs = rng.choices(names, probs, k=batch)
        steps.append([(s, rng.randrange(sizes[s]), rng.getrandbits(48)) for s in srcs])
    return steps


class StepDataset(torch.utils.data.Dataset):
    """Step k yields this rank's contiguous slice of the global batch."""

    def __init__(self, rows, steps, tokenizer, collate, W, micro, rank=0, world=1):
        self.rows, self.steps, self.tok = rows, steps, tokenizer
        self.collate, self.W, self.micro = collate, W, micro
        self.rank, self.world = rank, world

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, k):
        encs, stats = [], collections.Counter()
        per = len(self.steps[k]) // self.world
        for src, idx, qseed in self.steps[k][self.rank * per:(self.rank + 1) * per]:
            row = dict(self.rows[src][idx])
            row["queries"] = sample_queries(row["queries"], random.Random(qseed))
            for q in row["queries"]:
                stats[f"{src}/{q['status']}"] += 1
                stats[f"desc/{'with' if q['description'] else 'name_only'}"] += 1
            stats[f"rows/{src}"] += 1
            encs.append(encode_row(row, self.tok, self.W))
        # group similar lengths into microbatches to limit padding
        encs.sort(key=lambda e: len(e["cands"]) * max(1, len(e["labels"])))
        micro = [self.collate(encs[i:i + self.micro]) for i in range(0, len(encs), self.micro)]
        return micro, dict(stats)


def filter_long(rows, tok, what):
    """Drop rows whose text exceeds one 512-token window under this tokenizer
    (windowing is not implemented) and say how many; datasets are pre-filtered
    with ModernBERT's tokenizer, so this only acts for other encoders."""
    lengths = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
    kept = [r for r, ids in zip(rows, lengths) if len(ids) <= MAX_TEXT_TOKENS]
    if len(kept) < len(rows):
        print(f"filter_long {what}: dropped {len(rows) - len(kept)} of {len(rows)} rows "
              f"over {MAX_TEXT_TOKENS} tokens", flush=True)
    return kept


def load_eval(data, split, sources, tok, W, limit):
    out = {}
    for src in sources:
        path = Path(data) / src / f"{split}.jsonl"
        rows = filter_long([json.loads(line) for line in open(path)][:limit], tok, f"{src}/{split}")
        out[src] = [encode_row(r, tok, W) for r in rows]
    return out
