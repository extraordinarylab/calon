"""Zero-shot InstructUIE (ZWK/InstructUIE, Flan-T5 11B) on the OOD test sets, scored
like our models and the GLiNER baselines (strict typed character-span F1).

Prompt as in the InstructUIE repo (configs/instruction_config.json, first NER
instruction; src/uie_dataset.py):
  <instruction>Option: l1, l2, ... \nText: <text> \nAnswer:
and the model answers "type1: word1; type2: word2" or "None". Each answered word
is mapped to every occurrence in the text (the model gives no offsets). Label names
only: InstructUIE takes no label descriptions.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .gliner2_eval import f1, find_all, read_jsonl
from .llm_ner_eval import write_shard

INSTRUCTION = ("Please list all entity words in the text that fit the category."
               "Output format is \"type1: word1; type2: word2\".")


def prompt(text, labels):
    return INSTRUCTION + "Option: " + ", ".join(labels) + " \nText: " + text + " \nAnswer:"


def parse(answer, text, labels):
    """'type: word; type: word' -> {(start, end, label)}; unknown types are dropped."""
    by_lower = {l.lower(): l for l in labels}
    out = set()
    if answer.strip().lower() == "none":
        return out
    for part in answer.split(";"):
        if ":" not in part:
            continue
        typ, word = part.split(":", 1)
        lab, word = by_lower.get(typ.strip().lower()), word.strip()
        # the model often copies a trailing comma / period / quote ("Rural England,")
        # a final period is kept in abbreviations such as "U.S."
        word = word.rstrip(",;:'\" ") or word
        if word.endswith(".") and word.count(".") == 1:
            word = word[:-1].rstrip() or word
        if lab and word:
            out |= {(a, b, lab) for a, b in find_all(text, word)}
    return out


def merge(out_dir, sets, pii_sets, out):
    """Per-set files -> one result file in the GLiNER2 baseline layout (PII kept apart)."""
    load = lambda n: json.loads((Path(out_dir) / f"{n}.json").read_text())
    res = {"model": "ZWK/InstructUIE", "name_only": True, "sets": {n: load(n) for n in sets},
           "pii": {n: load(n) for n in pii_sets}}
    s = res["sets"]
    cn = [v["f1"] for k, v in s.items() if k.startswith("crossner")]
    res["macro_f1_crossner"] = sum(cn) / len(cn)
    res["macro_f1_all"] = sum(v["f1"] for v in s.values()) / len(s)
    res["mean_miss_rate"] = sum(v["entity_miss_rate"] for v in s.values()) / len(s)
    res["mean_absent_fpr"] = sum(v["absent_query_fpr"] for v in s.values()) / len(s)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


def main():
    ap = argparse.ArgumentParser()
    if __import__("sys").argv[1:2] == ["merge"]:
        ap.add_argument("cmd")
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--sets", nargs="+", required=True)
        ap.add_argument("--pii_sets", nargs="*", default=[])
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        return merge(a.out_dir, a.sets, a.pii_sets, a.out)
    ap.add_argument("--ood", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_source", type=int, default=512)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0, help="rows shard::nshard; preds to <set>.json.shard<k>")
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        rows = read_jsonl(Path(args.ood) / f"{name}.jsonl")
        idx = list(range(args.shard, len(rows), args.nshard))
        rows = [rows[i] for i in idx]
        tp = fp = fn = absent_q = absent_fp = present_q = present_miss = truncated = 0
        per, examples, preds = {}, [], []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            labels = [[q["label"] for q in r["queries"]] for r in chunk]
            enc = tok([prompt(r["text"], l) for r, l in zip(chunk, labels)], return_tensors="pt",
                      padding=True, truncation=True, max_length=args.max_source).to("cuda")
            truncated += int((enc["attention_mask"].sum(-1) >= args.max_source).sum())
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, num_beams=1)
            answers = tok.batch_decode(gen, skip_special_tokens=True)
            for r, labs, ans in zip(chunk, labels, answers):
                pred = parse(ans, r["text"], labs)
                preds.append(pred)
                gold = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                for k, part in ((0, pred & gold), (1, pred - gold), (2, gold - pred)):
                    for s in part:
                        per.setdefault(s[2], [0, 0, 0])[k] += 1
                for q in r["queries"]:
                    predicted = any(p[2] == q["label"] for p in pred)
                    if not any(g[2] == q["label"] for g in gold):
                        absent_q += 1
                        absent_fp += predicted
                    else:
                        present_q += 1
                        present_miss += not predicted
                if len(examples) < 5:
                    examples.append({"text": r["text"][:200], "answer": ans[:300]})
        if args.nshard > 1:
            write_shard(out_dir / f"{name}.json.shard{args.shard}", idx, preds)
            continue
        m = f1(tp, fp, fn)
        m["absent_query_fpr"] = absent_fp / absent_q if absent_q else float("nan")
        m["present_query_fnr"] = present_miss / present_q if present_q else float("nan")
        m["entity_miss_rate"] = fn / (tp + fn) if tp + fn else float("nan")
        m["truncated_prompts"] = truncated
        m["per_label"] = {lab: f1(*c) for lab, c in per.items()}
        m["examples"] = examples
        (out_dir / f"{name}.json").write_text(json.dumps(m, indent=1))
        print(name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
