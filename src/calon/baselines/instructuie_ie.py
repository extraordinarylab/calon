"""Zero-shot InstructUIE (ZWK/InstructUIE, Flan-T5 11B) for relations and events, scored like
re_eval / ee_eval. Names only (InstructUIE takes no descriptions); greedy decoding.

Prompts and answer formats as in the InstructUIE repo (configs/instruction_config.json, first
instruction of each task; src/uie_dataset.py):
  RE   <instruction>Option: r1, r2 \\nText: <text> \\nAnswer:     -> "rel: head, tail; rel: head, tail"
  EET  <instruction> Option: t1, t2 \\nText: <text> \\nAnswer:    -> "type: trigger; type: trigger"
  EEA  <instruction>Event type: <type> \\n  Option: role1, role2 \\nText: <text> \\nAnswer:
                                                                  -> "argument: role; argument: role"
Strings are mapped to the nearest occurrences (the model gives no offsets); "None" is no answer.
  re   end to end, and the gold-entity proxy of the GLiNER baselines (re_baselines.to_gold_entities)
  ee   end to end: EET, then EEA for each predicted type; with gold triggers: EEA for each gold
       event type (EEA takes the type, not the trigger, so an argument goes to the nearest gold
       trigger of that type)
Answer budget 256 new tokens (the repo uses 50, which cuts long relation and event lists).
Training-data overlap: InstructUIE's supervised mixture has CASIE (EE), SciERC (RE), MIT
movie / restaurant and ACE05 / GENIA; CrossRE, SciER, MLEE, M2E2 and the other test sets are not in it.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..eval.ee_eval import score as ee_score
from .ee_baselines import span_near
from .re_baselines import f1, fold, to_gold_entities, triples

RE_INSTR = ("Given a phrase that describes the relationship between two words, extract the words and the lexical "
            "relationship between them. The output format should be \"relation1: word1, word2; relation2: word3, "
            "word4\". \n")
EET_INSTR = ("Please tell me event type and its trigger word from given type options. "
             "Output format is \"type: trigger\". ")
EEA_INSTR = ("Given event type and trigger, please tell me the arguments of all the listed option. Output format is "
             "\"name1: role1; name2: role2\". \n")


def re_prompt(text, rels):
    return RE_INSTR + "Option: " + ", ".join(rels) + " \nText: " + text + " \nAnswer:"


def eet_prompt(text, types):
    return EET_INSTR + " Option: " + ", ".join(types) + " \nText: " + text + " \nAnswer:"


def eea_prompt(text, etype, roles):
    return (EEA_INSTR + "Event type: " + etype + " \n " + " Option: " + ", ".join(roles)
            + " \nText: " + text + " \nAnswer:")


def pairs(answer, sep_last=False):
    """'a: b; c: d' -> [(a, b)]; sep_last splits at the last colon (EEA: the argument text may hold one)."""
    if answer.strip().lower() in ("", "none"):
        return []
    out = []
    for part in answer.split(";"):
        if ":" not in part:
            continue
        a, b = part.rsplit(":", 1) if sep_last else part.split(":", 1)
        a, b = a.strip(), b.strip().rstrip(".")
        if a and b and b.lower() != "none":
            out.append((a, b))
    return out


def parse_re(answer, labels):
    by_lower = {lab.lower(): lab for lab in labels}
    items = []
    for rel, rest in pairs(answer):
        lab = by_lower.get(rel.lower())
        if lab and "," in rest:
            h, t = rest.split(",", 1)
            items.append({"head": h.strip(), "relation": lab, "tail": t.strip()})
    return items


class Generator:
    def __init__(self, model, batch, max_source, max_new):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
        self.batch, self.max_source, self.max_new, self.truncated = batch, max_source, max_new, 0

    def __call__(self, prompts):
        out = []
        for i in range(0, len(prompts), self.batch):
            enc = self.tok(prompts[i:i + self.batch], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_source).to("cuda")
            self.truncated += int((enc["attention_mask"].sum(-1) >= self.max_source).sum())
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, num_beams=1)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out


def run_re(gen, rows):
    answers = gen([re_prompt(r["text"], [q["label"] for q in r["rel_queries"]]) for r in rows])
    c = {"end2end": [0, 0, 0], "given_filter": [0, 0, 0]}
    for r, a in zip(rows, answers):
        labels = {q["label"] for q in r["rel_queries"]}
        sym = set(r.get("symmetric_labels", []))
        pred = triples(parse_re(a, labels), r["text"], labels)
        gold = fold({(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}, sym)
        for mode, p in (("end2end", fold(pred, sym)), ("given_filter", fold(to_gold_entities(pred, r), sym))):
            c[mode][0] += len(p & gold)
            c[mode][1] += len(p - gold)
            c[mode][2] += len(gold - p)
    res = {mode: f1(*v) for mode, v in c.items()}
    res["examples"] = [{"text": r["text"][:200], "answer": a[:300]} for r, a in zip(rows[:3], answers)]
    return res


def args_for(gen, rows, events, meta):
    """events: per row [(ts, te, type)]; EEA once per (row, type), each argument attached to the
    nearest event of that type. Returns per row the argument tuples of ee_eval."""
    jobs = [(k, t) for k in range(len(rows)) for t in dict.fromkeys(e[2] for e in events[k])]
    answers = gen([eea_prompt(rows[k]["text"], t, meta["event_roles"].get(t, [])) for k, t in jobs])
    out = [set() for _ in rows]
    for (k, t), a in zip(jobs, answers):
        r, roles = rows[k], {x.lower(): x for x in meta["event_roles"].get(t, [])}
        evs = [e for e in events[k] if e[2] == t]
        for text, role in pairs(a, sep_last=True):
            role = roles.get(role.lower())
            occ = span_near(r["text"], text)
            if not role or occ is None:
                continue
            ev = min(evs, key=lambda e: abs(e[0] - occ[0]))
            a_span = span_near(r["text"], text, ev[0])
            if a_span and a_span != ev[:2]:
                out[k].add((ev[0], ev[1], t, a_span[0], a_span[1], role))
    return out


def run_ee(gen, rows, meta):
    types = list(meta["event_types"])
    by_lower = {t.lower(): t for t in types}
    res = {}
    # gold triggers given
    gold = [[(e["trigger"][0], e["trigger"][1], e["type"]) for e in r["events"]] for r in rows]
    pa = args_for(gen, rows, gold, meta)
    res["given"] = ee_score(rows, [(set(g), a) for g, a in zip(gold, pa)], types)
    # end to end
    answers = gen([eet_prompt(r["text"], types) for r in rows])
    pt = []
    for r, a in zip(rows, answers):
        trig = set()
        for t, word in pairs(a):
            t = by_lower.get(t.lower())
            s = span_near(r["text"], word) if t else None
            if s:
                trig.add((s[0], s[1], t))
        pt.append(trig)
    pa = args_for(gen, rows, [sorted(t) for t in pt], meta)
    res["end2end"] = ee_score(rows, list(zip(pt, pa)), types)
    res["examples"] = [{"text": r["text"][:200], "eet": a[:300]} for r, a in zip(rows[:3], answers)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["re", "ee"])
    ap.add_argument("--ood", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", help="default: every set of the manifest")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_source", type=int, default=512)
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    gen = Generator(args.model, args.batch, args.max_source, args.max_new)
    res = {"model": "ZWK/InstructUIE", "name_only": True, "sets": {}}
    for name in args.sets or list(manifest["sets"]):
        rows = [json.loads(line) for line in open(Path(args.ood) / f"{name}.jsonl")]
        if args.task == "re":
            res["sets"][name] = run_re(gen, rows)
            print(name, {m: round(res["sets"][name][m]["f1"], 4) for m in ("end2end", "given_filter")}, flush=True)
        else:
            res["sets"][name] = run_ee(gen, rows, manifest["sets"][name])
            v = res["sets"][name]
            print(name, "given AC", round(v["given"]["AC"]["f1"], 4), "e2e TC", round(v["end2end"]["TC"]["f1"], 4),
                  "AC", round(v["end2end"]["AC"]["f1"], 4), flush=True)
        res["truncated_prompts"] = gen.truncated
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
