"""Pre-tokenized rows and batch collation into z-shaped [B, M, C] targets."""
import json
import os

import torch

from .alignment import MAX_TEXT_TOKENS, build_candidates_np, tokenize_text

LABEL_TOKEN, TEXT_TOKEN, REL_TOKEN = "<LABEL>", "<TEXT>", "<REL>"
# 1024 text tokens + schema + specials; evaluation sets with long schemas (Re-TACRED asks 38
# relation types with descriptions) raise both through the environment
SCHEMA_BUDGET = int(os.environ.get("CALON_SCHEMA_BUDGET", 1500))
TOTAL_BUDGET = int(os.environ.get("CALON_TOTAL_BUDGET", 3072))
STATUS_SUPERVISED = {"positive", "explicit_absent", "scope_assumed_absent"}
# ablation (2026-10-06): CALON_UNKNOWN_AS_NEGATIVE=1 supervises "unknown" types as absent (no label statuses)
if os.environ.get("CALON_UNKNOWN_AS_NEGATIVE") == "1":
    STATUS_SUPERVISED = STATUS_SUPERVISED | {"unknown"}
MAX_PAIR_ENTITIES = 24                     # ordered pairs among at most this many entity spans


def add_control_tokens(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": [LABEL_TOKEN, TEXT_TOKEN, REL_TOKEN]})
    return tokenizer.convert_tokens_to_ids(LABEL_TOKEN), tokenizer.convert_tokens_to_ids(TEXT_TOKEN)


def rel_token_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(REL_TOKEN)


def encode_relations(row, tokenizer, index, pair_spans=None):
    """Relation part of a row: relation label ids, the entity spans that form ordered
    pairs (the gold entity spans, or `pair_spans` at end-to-end evaluation), and gold
    relation labels per pair. Spans that are not candidates are left out."""
    rel_ids = [tokenizer(label_string(q), add_special_tokens=False)["input_ids"] for q in row["rel_queries"]]
    rels = row.get("relations", [])
    rel_spans = {tuple(r["head"]) for r in rels} | {tuple(r["tail"]) for r in rels}
    if pair_spans is None:
        spans = sorted({(e["start"], e["end"]) for e in row["entities"]}, key=lambda s: (s not in rel_spans, s))
    else:
        spans = list(dict.fromkeys(pair_spans))
    spans = [s for s in spans if s in index][:MAX_PAIR_ENTITIES]
    ent = {s: k for k, s in enumerate(spans)}
    pairs = [(a, b) for a in range(len(spans)) for b in range(len(spans)) if a != b]
    pidx = {p: k for k, p in enumerate(pairs)}
    ridx = {q["label"]: m for m, q in enumerate(row["rel_queries"])}
    gold = [[] for _ in row["rel_queries"]]
    missed = 0
    for r in row.get("relations", []):
        h, t = tuple(r["head"]), tuple(r["tail"])
        if r["label"] in ridx and h in ent and t in ent and h != t:
            gold[ridx[r["label"]]].append(pidx[(ent[h], ent[t])])
        elif r["label"] in ridx:
            missed += 1
    return {"rel_ids": rel_ids, "rel_labels": [q["label"] for q in row["rel_queries"]],
            "rel_supervised": [q["status"] in STATUS_SUPERVISED for q in row["rel_queries"]],
            "pair_spans": spans, "pairs": [(index[spans[a]], index[spans[b]]) for a, b in pairs],
            "pair_char": [(spans[a], spans[b]) for a, b in pairs], "rel_gold": gold, "rel_missed": missed}


def label_string(q):
    return f"{q['label']} : {q['description']}" if q["description"] else q["label"]


def encode_row(row, tokenizer, W, pair_spans=None):
    text_ids, offsets = tokenize_text(tokenizer, row["text"])
    if len(text_ids) > MAX_TEXT_TOKENS:
        raise NotImplementedError(
            f"{row['id']}: {len(text_ids)} text tokens; windowing not implemented yet")
    cands = build_candidates_np(row["text"], offsets, W)
    index = {(s, e): c for c, (s, e, _, _) in enumerate(cands)}
    label_ids = [tokenizer(label_string(q), add_special_tokens=False)["input_ids"]
                 for q in row["queries"]]
    rel = encode_relations(row, tokenizer, index, pair_spans) if row.get("rel_queries") else None
    schema_len = sum(1 + len(x) for x in label_ids) + (sum(1 + len(x) for x in rel["rel_ids"]) if rel else 0)
    if schema_len > SCHEMA_BUDGET or schema_len + len(text_ids) + 3 > TOTAL_BUDGET:
        raise NotImplementedError(f"{row['id']}: schema blocks not implemented yet")
    gold = [[] for _ in row["queries"]]
    qidx = {q["label"]: m for m, q in enumerate(row["queries"])}
    unrepresented = 0
    entities = [e for e in row["entities"] if e["label"] in qidx]   # queries may be a subset
    for ent in entities:
        c = index.get((ent["start"], ent["end"]))
        if c is None:
            unrepresented += 1
        else:
            gold[qidx[ent["label"]]].append(c)
    out = {
        "id": row["id"], "text": row["text"], "text_ids": text_ids, "label_ids": label_ids,
        "cands": cands, "labels": [q["label"] for q in row["queries"]],
        "supervised": [q["status"] in STATUS_SUPERVISED for q in row["queries"]],
        "gold": gold, "gold_spans": {(e["start"], e["end"], e["label"]) for e in entities},
        "unrepresented": unrepresented,
    }
    if rel:
        out.update(rel)
        out["gold_triples"] = {(tuple(r["head"]), tuple(r["tail"]), r["label"]) for r in row.get("relations", [])
                               if r["label"] in set(rel["rel_labels"])}
    return out


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class Collator:
    def __init__(self, tokenizer, label_token_id, text_token_id):
        # decoder tokenizers (Qwen) have no CLS/SEP: the sequence then starts at <LABEL>
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.lab, self.txt = label_token_id, text_token_id
        self.rel = rel_token_id(tokenizer)

    def __call__(self, encs):
        B = len(encs)
        seqs, label_spans, rel_spans, text_start = [], [], [], []
        for e in encs:
            ids, spans, rspans = ([self.cls] if self.cls is not None else []), [], []
            for lid in e["label_ids"]:
                ids.append(self.lab)
                spans.append((len(ids), len(ids) + len(lid)))
                ids.extend(lid)
            for rid in e.get("rel_ids", []):
                ids.append(self.rel)
                rspans.append((len(ids), len(ids) + len(rid)))
                ids.extend(rid)
            rel_spans.append(rspans)
            ids.append(self.txt)
            text_start.append(len(ids))
            ids.extend(e["text_ids"])
            if self.sep is not None:
                ids.append(self.sep)
            seqs.append(ids)
            label_spans.append(spans)
        T = max(map(len, seqs))
        M = max(len(e["label_ids"]) for e in encs)
        C = max(len(e["cands"]) for e in encs)
        input_ids = torch.full((B, T), self.pad, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        label_mask = torch.zeros((B, M, T))
        cand_i = torch.zeros((B, C), dtype=torch.long)
        cand_j = torch.zeros((B, C), dtype=torch.long)
        cand_mask = torch.zeros((B, C), dtype=torch.bool)
        query_mask = torch.zeros((B, M), dtype=torch.bool)
        supervised = torch.zeros((B, M), dtype=torch.bool)
        y = torch.zeros((B, M, C))
        for b, e in enumerate(encs):
            input_ids[b, :len(seqs[b])] = torch.tensor(seqs[b])
            attn[b, :len(seqs[b])] = 1
            for m, (s, t) in enumerate(label_spans[b]):
                label_mask[b, m, s:t] = 1
                query_mask[b, m] = True
                supervised[b, m] = e["supervised"][m]
                y[b, m, e["gold"][m]] = 1
            n = len(e["cands"])
            cand_i[b, :n] = torch.tensor([c[2] for c in e["cands"]]) + text_start[b]
            cand_j[b, :n] = torch.tensor([c[3] for c in e["cands"]]) + text_start[b]
            cand_mask[b, :n] = True
        valid = supervised[:, :, None] & cand_mask[:, None, :]
        model_inputs = {"input_ids": input_ids, "attention_mask": attn, "label_mask": label_mask,
                        "cand_i": cand_i, "cand_j": cand_j}
        targets = {"y": y, "valid": valid, "query_mask": query_mask, "cand_mask": cand_mask}
        R = max(len(e.get("rel_ids", [])) for e in encs)
        if R:
            P = max(1, max(len(e.get("pairs", [])) for e in encs))
            rel_mask = torch.zeros((B, R, T))
            pair_h = torch.zeros((B, P), dtype=torch.long)
            pair_t = torch.zeros((B, P), dtype=torch.long)
            pair_mask = torch.zeros((B, P), dtype=torch.bool)
            rq_mask = torch.zeros((B, R), dtype=torch.bool)
            r_sup = torch.zeros((B, R), dtype=torch.bool)
            ry = torch.zeros((B, R, P))
            for b, e in enumerate(encs):
                for m, (s, t) in enumerate(rel_spans[b]):
                    rel_mask[b, m, s:t] = 1
                    rq_mask[b, m] = True
                    r_sup[b, m] = e["rel_supervised"][m]
                    ry[b, m, e["rel_gold"][m]] = 1
                n = len(e.get("pairs", []))
                if n:
                    pair_h[b, :n] = torch.tensor([p[0] for p in e["pairs"]])
                    pair_t[b, :n] = torch.tensor([p[1] for p in e["pairs"]])
                    pair_mask[b, :n] = True
            model_inputs.update({"rel_mask": rel_mask, "pair_h": pair_h, "pair_t": pair_t})
            targets["rel"] = {"y": ry, "valid": r_sup[:, :, None] & pair_mask[:, None, :],
                              "query_mask": rq_mask, "cand_mask": pair_mask}
        return model_inputs, targets
