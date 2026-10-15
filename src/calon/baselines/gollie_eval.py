"""Zero-shot GoLLIE-7B (HiTZ/GoLLIE-7B, Code Llama fine-tuned on guidelines written as Python classes) for
entities, relations and events, scored like the other baselines.

Prompt as in the GoLLIE repo (templates/prompt.txt; notebooks/Create Custom Task.ipynb): the guideline
classes, then `text = <repr>`, then "# The annotation instances that take place in the text above are listed
here" and `result =`; the model writes a Python list of class instances. One class per type, its name the
type in CamelCase, its docstring the type description (no docstring with --mode name_only); no examples.
  NER  class <Type>(Entity): span: str                     -> <Type>(span="...")
  RE   class <Relation>(Relation): arg1: str; arg2: str     -> <Relation>(arg1="head", arg2="tail")
  EE   class <EventType>(Event): mention: str; <role>: List[str] (role descriptions as comments)
                                                            -> <EventType>(mention="trigger", <role>=[...])
The answer is read with ast (never eval); classes that are not in the prompt are dropped. GoLLIE gives no
offsets: an entity counts at every occurrence of its text (as InstructUIE); relation arguments take the
nearest pair of occurrences; a trigger its first occurrence and arguments the occurrence nearest to it.
Settings with gold entities / gold triggers do not exist in GoLLIE, so they use the proxies of the other
generative baselines: re_baselines.to_gold_entities ("given_filter") and ee_baselines.to_gold_triggers.
Greedy decoding with vLLM (bfloat16), answers up to --max_new tokens.

  run    one evaluation set: --task ner|re|ee --data <dir> --set <name> --mode desc|name_only --out <json>
         (layout {"model", "name_only", "sets": {<name>: scores}}; --limit / --dump for smoke tests)
"""
import argparse
import ast
import json
import keyword
import re
from pathlib import Path

from .ee_baselines import span_near, to_gold_triggers
from .llm_ner_eval import score as ner_score
from .re_baselines import f1, find_all, fold, to_gold_entities

HEAD = "# The following lines describe the task definition"
TEXT = "# This is the text to analyze"
RESULT = "# The annotation instances that take place in the text above are listed here"
MENTION_DOC = '"""The text span that most clearly expresses the event."""'


def class_name(label, taken):
    words = re.findall(r"[A-Za-z0-9]+", label)
    name = "".join(w[:1].upper() + w[1:] for w in words) or "Type"
    if name[0].isdigit():
        name = "T" + name
    base, k = name, 2
    while name in taken or keyword.iskeyword(name) or name in ("Entity", "Relation", "Event", "List"):
        name, k = f"{base}{k}", k + 1
    taken.add(name)
    return name


def field_name(role, taken):
    name = "_".join(w.lower() for w in re.findall(r"[A-Za-z0-9]+", role)) or "role"
    if name[0].isdigit():
        name = "r_" + name
    base, k = name, 2
    while name in taken or keyword.iskeyword(name) or name == "mention":
        name, k = f"{base}_{k}", k + 1
    taken.add(name)
    return name


def doc(desc):
    desc = " ".join(str(desc or "").split()).replace('"""', "'''")
    return f'    """{desc}"""\n\n' if desc else ""


def comment(desc):
    desc = " ".join(str(desc or "").split())
    return f"  # {desc}" if desc else ""


def entity_classes(queries, name_only):
    taken, names, defs = set(), {}, []
    for q in queries:
        n = class_name(q["label"], taken)
        names[n] = q["label"]
        defs.append(f"@dataclass\nclass {n}(Entity):\n{doc(None if name_only else q.get('description'))}"
                    f"    span: str\n")
    return names, defs


def relation_classes(queries, name_only):
    taken, names, defs = set(), {}, []
    for q in queries:
        n = class_name(q["label"], taken)
        names[n] = q["label"]
        defs.append(f"@dataclass\nclass {n}(Relation):\n{doc(None if name_only else q.get('description'))}"
                    f"    arg1: str\n    arg2: str\n")
    return names, defs


def event_classes(meta, name_only):
    taken, names, roles, defs = set(), {}, {}, []
    for t, desc in meta["event_types"].items():
        n = class_name(t, taken)
        names[n] = t
        ftaken, fields = set(), []
        roles[n] = {}
        for r in meta["event_roles"].get(t, []):
            f = field_name(r, ftaken)
            roles[n][f] = r
            fields.append(f"    {f}: List[str]{'' if name_only else comment(meta.get('roles', {}).get(r))}\n")
        defs.append(f"@dataclass\nclass {n}(Event):\n{doc(None if name_only else desc)}"
                    f"    mention: str\n    {MENTION_DOC}\n" + "".join(fields))
    return names, roles, defs


def prompt(defs, text):
    return f"{HEAD}\n" + "\n".join(defs) + f"\n\n{TEXT}\ntext = {text!r}\n\n{RESULT}\nresult ="


def calls(answer):
    """[(class name, {field: str | [str]})] from the generated list; a cut-off answer keeps its complete
    instances."""
    s = answer.strip()
    if not s.startswith("["):
        s = "[" + s
    for cut in range(6):
        try:
            tree = ast.parse(s if s.rstrip().endswith("]") else s + "\n]", mode="eval")
            break
        except SyntaxError:
            k = s.rfind("),")
            if k < 0:
                return []
            s = s[:k + 1] + "\n]"
    else:
        return []
    out = []
    for node in getattr(tree.body, "elts", []):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        kw = {}
        for k in node.keywords:
            if k.arg is None:
                continue
            try:
                v = ast.literal_eval(k.value)
            except Exception:
                continue
            if isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                kw[k.arg] = v
        out.append((node.func.id, kw))
    return out


def ner_preds(answer, text, names):
    out = set()
    for cls, kw in calls(answer):
        lab, span = names.get(cls), kw.get("span")
        if lab and isinstance(span, str) and span.strip():
            out |= {(a, b, lab) for a, b in find_all(text, span.strip())}
    return out


def re_preds(answer, text, names):
    out = set()
    for cls, kw in calls(answer):
        lab, h, t = names.get(cls), kw.get("arg1"), kw.get("arg2")
        if not (lab and isinstance(h, str) and isinstance(t, str)):
            continue
        pairs = [(a, b) for a in find_all(text, h.strip()) for b in find_all(text, t.strip()) if a != b]
        if pairs:
            a, b = min(pairs, key=lambda p: abs(p[0][0] - p[1][0]))
            out.add((a, b, lab))
    return out


def ee_preds(answer, text, names, roles):
    pt, pa = set(), set()
    for cls, kw in calls(answer):
        t, m = names.get(cls), kw.get("mention")
        if not t or not isinstance(m, str):
            continue
        trig = span_near(text, m)
        if trig is None:
            continue
        pt.add((trig[0], trig[1], t))
        for f, role in roles[cls].items():
            v = kw.get(f, [])
            for a in [v] if isinstance(v, str) else v:
                sp = span_near(text, a, trig[0]) if a and a.strip() else None
                if sp and sp != trig:
                    pa.add((trig[0], trig[1], t, sp[0], sp[1], role))
    return pt, pa


class Generator:
    def __init__(self, model, max_len, max_new):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len, enable_prefix_caching=True,
                       gpu_memory_utilization=0.85)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new, stop=["\n]"])
        self.max_len, self.max_new, self.truncated = max_len, max_new, 0

    def __call__(self, prompts):
        from vllm.inputs import TokensPrompt
        ids = []
        for p in prompts:
            x = self.tok(p, add_special_tokens=True)["input_ids"]
            if x and x[-1] == self.tok.eos_token_id:        # GoLLIE's tokenizer appends </s>; the notebook drops it
                x = x[:-1]
            if len(x) > self.max_len - self.max_new:
                self.truncated += 1
                x = x[:1] + x[-(self.max_len - self.max_new - 1):]   # keep BOS and the end (text, result =)
            ids.append(TokensPrompt(prompt_token_ids=x))
        outs = self.llm.generate(ids, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def run(a):
    gen = Generator(a.model, a.max_len, a.max_new)
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    rows = [json.loads(line) for line in open(Path(a.data) / f"{a.set}.jsonl")]
    if a.limit:
        rows = rows[:a.limit]
    name_only = a.mode == "name_only"
    prompts, ctx = [], []
    if a.task == "ee":
        meta = manifest["sets"][a.set]
        names, roles, defs = event_classes(meta, name_only)
        prompts = [prompt(defs, r["text"]) for r in rows]
    else:
        for r in rows:
            names, defs = (entity_classes(r["queries"], name_only) if a.task == "ner"
                           else relation_classes(r["rel_queries"], name_only))
            prompts.append(prompt(defs, r["text"]))
            ctx.append(names)
    answers = gen(prompts)
    if a.task == "ner":
        preds = [ner_preds(ans, r["text"], n) for ans, r, n in zip(answers, rows, ctx)]
        res = ner_score(rows, preds)
    elif a.task == "re":
        c = {"end2end": [0, 0, 0], "given_filter": [0, 0, 0]}
        preds = []
        for ans, r, n in zip(answers, rows, ctx):
            sym = set(r.get("symmetric_labels", []))
            pred = re_preds(ans, r["text"], n)
            preds.append(pred)
            gold = fold({(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}, sym)
            for mode, p in (("end2end", fold(pred, sym)), ("given_filter", fold(to_gold_entities(pred, r), sym))):
                c[mode][0] += len(p & gold)
                c[mode][1] += len(p - gold)
                c[mode][2] += len(gold - p)
        res = {mode: f1(*v) for mode, v in c.items()}
        if "end2end" not in manifest["sets"][a.set].get("settings", ["end2end"]):
            del res["end2end"]                    # only some pairs annotated: end to end is undefined
    else:
        from ..eval.ee_eval import score as ee_score
        types = list(meta["event_types"])
        preds = [ee_preds(ans, r["text"], names, roles) for ans, r in zip(answers, rows)]
        res = {"end2end": ee_score(rows, preds, types),
               "given": ee_score(rows, [to_gold_triggers(pt, pa, r) for (pt, pa), r in zip(preds, rows)], types)}
    res["n_rows"], res["truncated_prompts"] = len(rows), gen.truncated
    res["empty_answers"] = sum(1 for x in answers if x.strip() in ("[]", "[", ""))
    res["unparsed_answers"] = sum(1 for x in answers if x.strip() not in ("[]", "[", "") and not calls(x))
    out = {"model": f"gollie-7b-{a.mode}", "name_only": name_only, "sets": {a.set: res}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=list))
    if a.dump:
        lines = []
        for p, ans, r, pr in list(zip(prompts, answers, rows, preds))[:a.dump]:
            lines += ["=" * 80, p[-1500:], "--- answer:", ans[:1500], "--- parsed:", str(pr)[:800]]
            if a.task == "ner":
                lines.append("--- gold: "
                             + str(sorted((e["start"], e["end"], e["label"]) for e in r["entities"]))[:800])
            elif a.task == "re":
                lines.append("--- gold: " + str(sorted((tuple(x["head"]), tuple(x["tail"]), x["label"])
                                                       for x in r["relations"]))[:800])
            else:
                lines.append("--- gold: " + str([(e["trigger"], e["type"], [(x["span"], x["role"]) for x in e["args"]])
                                                 for e in r["events"]])[:800])
        Path(a.dump_file).write_text("\n".join(lines) + "\n")
    brief = {k: (round(v["f1"], 4) if isinstance(v, dict) and "f1" in v else
                 {m: round(v[m]["f1"], 4) for m in ("TC", "AC")} if isinstance(v, dict) and "AC" in v else v)
             for k, v in res.items() if k not in ("per_label",)}
    print(a.task, a.set, a.mode, json.dumps(brief), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=["ner", "re", "ee"])
    r.add_argument("--data", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--mode", required=True, choices=["desc", "name_only"])
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max_len", type=int, default=8192)
    r.add_argument("--max_new", type=int, default=512)
    r.add_argument("--dump", type=int, default=0, help="write the first N prompts / answers / parses")
    r.add_argument("--dump_file", default="")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
