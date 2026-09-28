"""Pre-tokenized rows and batch collation into z-shaped [B, M, C] targets."""
import json

import torch

from .alignment import MAX_TEXT_TOKENS, build_candidates, tokenize_text

LABEL_TOKEN, TEXT_TOKEN = "<LABEL>", "<TEXT>"
SCHEMA_BUDGET, TOTAL_BUDGET = 1500, 2048
STATUS_SUPERVISED = {"positive", "explicit_absent", "scope_assumed_absent"}


def add_control_tokens(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": [LABEL_TOKEN, TEXT_TOKEN]})
    return tokenizer.convert_tokens_to_ids(LABEL_TOKEN), tokenizer.convert_tokens_to_ids(TEXT_TOKEN)


def label_string(q):
    return f"{q['label']} : {q['description']}" if q["description"] else q["label"]


def encode_row(row, tokenizer, W):
    text_ids, offsets = tokenize_text(tokenizer, row["text"])
    if len(text_ids) > MAX_TEXT_TOKENS:
        raise NotImplementedError(
            f"{row['id']}: {len(text_ids)} text tokens; windowing (SPEC 3) not implemented yet")
    cands = build_candidates(row["text"], offsets, W)
    index = {(s, e): c for c, (s, e, _, _) in enumerate(cands)}
    label_ids = [tokenizer(label_string(q), add_special_tokens=False)["input_ids"]
                 for q in row["queries"]]
    schema_len = sum(1 + len(x) for x in label_ids)
    if schema_len > SCHEMA_BUDGET or schema_len + len(text_ids) + 3 > TOTAL_BUDGET:
        raise NotImplementedError(f"{row['id']}: schema blocks (SPEC 3) not implemented yet")
    gold = [[] for _ in row["queries"]]
    qidx = {q["label"]: m for m, q in enumerate(row["queries"])}
    unrepresented = 0
    for ent in row["entities"]:
        c = index.get((ent["start"], ent["end"]))
        if c is None:
            unrepresented += 1
        else:
            gold[qidx[ent["label"]]].append(c)
    return {
        "id": row["id"], "text": row["text"], "text_ids": text_ids, "label_ids": label_ids,
        "cands": cands, "labels": [q["label"] for q in row["queries"]],
        "supervised": [q["status"] in STATUS_SUPERVISED for q in row["queries"]],
        "gold": gold, "gold_spans": {(e["start"], e["end"], e["label"]) for e in row["entities"]},
        "unrepresented": unrepresented,
    }


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class Collator:
    def __init__(self, tokenizer, label_token_id, text_token_id):
        self.cls, self.sep, self.pad = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                                        tokenizer.pad_token_id)
        self.lab, self.txt = label_token_id, text_token_id

    def __call__(self, encs):
        B = len(encs)
        seqs, label_spans, text_start = [], [], []
        for e in encs:
            ids, spans = [self.cls], []
            for lid in e["label_ids"]:
                ids.append(self.lab)
                spans.append((len(ids), len(ids) + len(lid)))
                ids.extend(lid)
            ids.append(self.txt)
            text_start.append(len(ids))
            ids.extend(e["text_ids"])
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
        return model_inputs, {"y": y, "valid": valid, "query_mask": query_mask,
                              "cand_mask": cand_mask}
