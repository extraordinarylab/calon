"""Zero-shot decoder-LLM NER baselines trained only on Pile-NER data, scored like
our models (strict typed character-span F1, absent-query FPR, miss rates).

  gner    dyyyyyyyy/GNER-LLaMA-7B: all labels in one prompt, the model tags every
          word "word(B-type)"; words are the text's whitespace tokens, so predicted
          tags map back to exact character offsets (sequence alignment on words).
  uniner  Universal-NER/UniNER-7B-type: one question per type, answers a JSON list
          of strings, mapped to every occurrence in the text.
  slimer  expertai/SLIMER-LLaMA3: one type per prompt with DEFINITION (our label
          description) and GUIDELINES (none available: left empty); JSON list.
Prompts follow the model cards. Runs with vLLM.
"""
import argparse
import ast
import difflib
import json
import re
from pathlib import Path

from .gliner2_eval import f1, find_all, read_jsonl

GNER_INSTRUCTION = (
    "Please analyze the sentence provided, identifying the type of entity for each word on a "
    "token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the "
    "BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n"
    "2. I- (Inside) is used for words within a named entity but are not the first word.\n"
    "3. O (Outside) denotes words that are not part of a named entity.\n")
UNINER = ("A virtual assistant answers questions from a user based on the provided text.\n"
          "USER: Text: {text}\nASSISTANT: I've read this text.\nUSER: What describes {label} in the text?\n"
          "ASSISTANT:")
SLIMER_SYSTEM = "You are an expert in Named Entity Recognition designed to output JSON only."
SLIMER_USER = (
    "You are given a text chunk (delimited by triple quotes) and an instruction.\n"
    "Read the text and answer to the instruction in the end.\n\"\"\"\n{text}\n\"\"\"\n"
    "Instruction: Extract the Named Entities of type {label} from the text chunk you have read. "
    "You are given a DEFINITION and some GUIDELINES.\nDEFINITION: {definition}\nGUIDELINES: \n"
    "Return a JSON list of instances of this Named Entity type (for example [\"text_span_1\", "
    "\"text_span_2\"]. Return an empty list [] if no instances are present. Return only the JSON "
    "list, no further motivations or introduction to the answer.")
TAG = re.compile(r"(\S+)\(((?:B-|I-)[^()]*|O)\)")


def words_of(text):
    """CoNLL-style words (punctuation split off, as in GNER's training data), so a
    tagged word never carries a trailing comma into the entity span."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def gner_prompt(text, labels):
    words = " ".join(w for w, _, _ in words_of(text))
    body = f"{GNER_INSTRUCTION}\nUse the specific entity tags: {', '.join(labels)} and O.\nSentence: {words}"
    return f"[INST] {body} [/INST]"


def gner_parse(answer, text, labels):
    words = words_of(text)
    pred = TAG.findall(answer)
    tags = ["O"] * len(words)
    sm = difflib.SequenceMatcher(a=[w for w, _, _ in words], b=[w for w, _ in pred], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            tags[a + k] = pred[b + k][1].strip()
    by_lower = {l.lower(): l for l in labels}
    out, cur = set(), None
    for (w, s, e), t in zip(words + [("", 0, 0)], tags + ["O"]):
        lab = by_lower.get(t[2:].strip().lower()) if t[:2] in ("B-", "I-") else None
        if cur and (t.startswith("I-") and lab == cur[2]):
            cur[1] = e
            continue
        if cur:
            out.add(tuple(cur))
        cur = [s, e, lab] if lab else None
    return out


def json_list(answer):
    answer = answer.strip()
    m = re.search(r"\[.*\]", answer, re.S)
    if not m:
        return []
    for load in (json.loads, ast.literal_eval):
        try:
            v = load(m.group())
            return [x if isinstance(x, str) else str(x[0]) for x in v if x]
        except Exception:
            pass
    return re.findall(r'"([^"]+)"', m.group())


def score(rows, preds):
    tp = fp = fn = absent_q = absent_fp = present_q = present_miss = 0
    per = {}
    for r, pred in zip(rows, preds):
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
    m = f1(tp, fp, fn)
    m["absent_query_fpr"] = absent_fp / absent_q if absent_q else float("nan")
    m["present_query_fnr"] = present_miss / present_q if present_q else float("nan")
    m["entity_miss_rate"] = fn / (tp + fn) if tp + fn else float("nan")
    m["per_label"] = {lab: f1(*c) for lab, c in per.items()}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gner", "uniner", "slimer"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ood", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_model_len", type=int, default=0, help="0: the model's own context")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    kw = {"max_model_len": args.max_model_len} if args.max_model_len else {}
    llm = LLM(model=args.model, gpu_memory_utilization=0.9, **kw)
    tok = llm.get_tokenizer()
    try:
        args.max_model_len = llm.llm_engine.model_config.max_model_len
    except AttributeError:
        args.max_model_len = args.max_model_len or 2048
    print("context", args.max_model_len, flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        rows = read_jsonl(Path(args.ood) / f"{name}.jsonl")
        labels = [[q["label"] for q in r["queries"]] for r in rows]
        skipped, examples = 0, []
        if args.backend == "gner":
            prompts, params = [], []
            for r, labs in zip(rows, labels):
                p = gner_prompt(r["text"], labs)
                n_in = len(tok(p)["input_ids"])
                budget = args.max_model_len - n_in
                need = 12 * len(words_of(r["text"])) + 64
                if budget < 32:
                    skipped += 1
                    prompts.append(None)
                    params.append(None)
                    continue
                prompts.append(p)
                params.append(SamplingParams(temperature=0.0, max_tokens=min(need, budget)))
            live = [k for k, p in enumerate(prompts) if p is not None]
            outs = llm.generate([prompts[k] for k in live], [params[k] for k in live])
            answers = [""] * len(rows)
            for k, o in zip(live, outs):
                answers[k] = o.outputs[0].text
            preds = [gner_parse(a, r["text"], labs) for a, r, labs in zip(answers, rows, labels)]
            examples = [{"text": r["text"][:200], "answer": a[:300]} for r, a in zip(rows[:3], answers)]
        else:
            items, prompts = [], []
            for i, r in enumerate(rows):
                for q in r["queries"]:
                    if args.backend == "uniner":
                        p = UNINER.format(text=r["text"], label=q["label"])
                    else:
                        msgs = [{"role": "system", "content": SLIMER_SYSTEM},
                                {"role": "user", "content": SLIMER_USER.format(
                                    text=r["text"], label=q["label"], definition=q["description"] or q["label"])}]
                        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if len(tok(p)["input_ids"]) > args.max_model_len - 256:
                        skipped += 1
                        continue
                    items.append((i, q["label"]))
                    prompts.append(p)
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
            preds = [set() for _ in rows]
            for (i, lab), o in zip(items, outs):
                ans = o.outputs[0].text
                for s in json_list(ans):
                    s = s.strip()
                    if s:
                        preds[i] |= {(a, b, lab) for a, b in find_all(rows[i]["text"], s)}
                if len(examples) < 5:
                    examples.append({"text": rows[i]["text"][:150], "label": lab, "answer": ans[:200]})
        m = score(rows, preds)
        m["skipped_too_long"] = skipped
        m["examples"] = examples
        (out_dir / f"{name}.json").write_text(json.dumps(m, indent=1))
        print(name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, "skipped", skipped,
              flush=True)


if __name__ == "__main__":
    main()
