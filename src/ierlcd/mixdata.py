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

    def __init__(self, rows, steps, tokenizer, collate, W, micro, rank=0, world=1, desc_dropout=0.0):
        self.rows, self.steps, self.tok = rows, steps, tokenizer
        self.collate, self.W, self.micro = collate, W, micro
        self.rank, self.world = rank, world
        self.desc_dropout = desc_dropout

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, k):
        encs, stats = [], collections.Counter()
        per = len(self.steps[k]) // self.world
        for src, idx, qseed in self.steps[k][self.rank * per:(self.rank + 1) * per]:
            row = dict(self.rows[src][idx])
            rng = random.Random(qseed)
            row["queries"] = sample_queries(row["queries"], rng)
            if self.desc_dropout:
                # regularizer: each described label is sometimes shown by name only
                # (copies: query dicts are shared between rows)
                row["queries"] = [dict(q, description="") if q["description"] and
                                  rng.random() < self.desc_dropout else q for q in row["queries"]]
            for q in row["queries"]:
                stats[f"{src}/{q['status']}"] += 1
                stats[f"desc/{'with' if q['description'] else 'name_only'}"] += 1
            stats[f"rows/{src}"] += 1
            encs.append(encode_row(row, self.tok, self.W))
        # group similar lengths into microbatches to limit padding
        encs.sort(key=lambda e: len(e["cands"]) * max(1, len(e["labels"])))
        micro = [self.collate(encs[i:i + self.micro]) for i in range(0, len(encs), self.micro)]
        return micro, dict(stats)


def fewnerd_coarse_names(data):
    inv = json.loads((Path(data) / "inventory.json").read_text())
    return {x["name"] for x in inv["fewnerd"] if not x.get("parent")}


def fine_only(rows, coarse):
    """Few-NERD with its 8 coarse types removed (queries and mentions): each span
    then has exactly one gold type, as a single-choice RLCD head needs."""
    out = []
    for r in rows:
        r = dict(r)
        r["queries"] = [q for q in r["queries"] if q["label"] not in coarse]
        r["entities"] = [e for e in r["entities"] if e["label"] not in coarse]
        out.append(r)
    return out


PRECEDENCE_NOTE = " Use this type only when no more specific type in the list applies."


def fewnerd_precedence(rows, data):
    """Few-NERD with coarse and fine types asked together, most specific wins:
    a mention of a specific fine type (politician) keeps only that type; a
    mention of an "other X" fine type is labelled with the coarse type X instead,
    and the "other X" queries are dropped. Coarse descriptions get a precedence note."""
    inv = json.loads((Path(data) / "inventory.json").read_text())["fewnerd"]
    coarse = {x["name"] for x in inv if not x.get("parent")}
    other_to_coarse = {x["name"]: x["parent"] for x in inv
                       if x.get("parent") and x["name"] == f"other {x['parent']}"}
    out = []
    for r in rows:
        spans = collections.defaultdict(set)
        for e in r["entities"]:
            spans[(e["start"], e["end"])].add(e["label"])
        ents = []
        for (s, t), labs in spans.items():
            fine = [l for l in labs if l not in coarse]
            for l in fine:
                ents.append({"start": s, "end": t, "label": other_to_coarse.get(l, l)})
            if not fine:                               # coarse-only mention (should not occur)
                ents += [{"start": s, "end": t, "label": l} for l in labs]
        present = {e["label"] for e in ents}
        queries = []
        for q in r["queries"]:
            if q["label"] in other_to_coarse:
                continue
            q = dict(q)
            if q["label"] in coarse:
                q["description"] = (q["description"] or "") + PRECEDENCE_NOTE
            if q["status"] in ("positive", "scope_assumed_absent"):
                q["status"] = "positive" if q["label"] in present else "scope_assumed_absent"
            queries.append(q)
        out.append({**r, "queries": queries, "entities": ents})
    return out


def fewnerd_transform(data, mode):
    """mode: both (as stored) | fine (fine types only) | precedence."""
    if mode == "fine":
        c = fewnerd_coarse_names(data)
        return lambda rows: fine_only(rows, c)
    if mode == "precedence":
        return lambda rows: fewnerd_precedence(rows, data)
    return None


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


def load_eval(data, split, sources, tok, W, limit, fewnerd_mode="both"):
    out = {}
    tf = fewnerd_transform(data, fewnerd_mode)
    for src in sources:
        path = Path(data) / src / f"{split}.jsonl"
        rows = filter_long([json.loads(line) for line in open(path)][:limit], tok, f"{src}/{split}")
        if tf and src == "fewnerd":
            rows = tf(rows)
        out[src] = [encode_row(r, tok, W) for r in rows]
    return out
