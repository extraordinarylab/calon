"""One type per gold span in the *_new evaluation sets (data/ood_new).

Their gold is the 2-of-3 annotator consensus per (span, type), so a span can keep two types ("China":
country and location). The per-annotator votes are gone, so gpt-oss-120b picks, for each such span,
the one type of the two (or more) that fits the span in its sentence, given the types' descriptions.
Originals are kept as <set>.jsonl.multi; the manifest's entity counts are updated.
"""
import collections
import json
import shutil

from calon.data.roots import MODELS, ROOT

D = ROOT / "data/ood_new"
PROMPT = """Text:
{text}

In this text, the span "{span}" was annotated with more than one of these entity types:
{types}

Which single type fits "{span}" best in this text? Answer with JSON only: {{"type": "<one of the types above>"}}"""


def main():
    from .newtest import chat, final_text, last_json, llm
    man = json.loads((D / "manifest.json").read_text())
    jobs, data = [], {}
    for name in man["sets"]:
        src = D / f"{name}.jsonl.multi"
        if not src.exists():
            shutil.copy(D / f"{name}.jsonl", src)
        rows = [json.loads(line) for line in open(src)]
        data[name] = rows
        for i, r in enumerate(rows):
            desc = {q["label"]: q["description"] for q in r["queries"]}
            by = collections.defaultdict(list)
            for e in r["entities"]:
                by[(e["start"], e["end"])].append(e["label"])
            for (s, t), labs in by.items():
                if len(labs) > 1:
                    types = "\n".join(f'- "{x}": {desc.get(x, "")}' for x in sorted(labs))
                    jobs.append((name, i, s, t, sorted(labs),
                                 PROMPT.format(text=r["text"], span=r["text"][s:t], types=types)))
    print("multi-type spans", len(jobs), collections.Counter(j[0] for j in jobs), flush=True)
    engine = llm(str(MODELS / "openai/gpt-oss-120b"), max_len=8192)
    answers = chat(engine, [j[5] for j in jobs], 2048)
    pick, stats = {}, collections.Counter()
    for (name, i, s, t, labs, _), a in zip(jobs, answers):
        js = last_json(final_text(a))
        choice = js.get("type") if isinstance(js, dict) else None
        if choice not in labs:
            stats["unparsed_kept_first"] += 1
            choice = labs[0]
        pick[(name, i, s, t)] = choice
        stats[f"{labs[0]}|{labs[1]} -> {choice}"] += 1
    for name, rows in data.items():
        for i, r in enumerate(rows):
            ents = [e for e in r["entities"]
                    if pick.get((name, i, e["start"], e["end"]), e["label"]) == e["label"]]
            r["entities"] = ents
            present = {e["label"] for e in ents}
            for q in r["queries"]:
                if q["status"] == "positive" and q["label"] not in present:
                    q["status"], q["supervision_origin"] = "scope_assumed_absent", "benchmark_closed_world"
        with open(D / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["sets"][name]["entities"] = sum(len(r["entities"]) for r in rows)
    man["rule"] += "; a span with two or more consensus types keeps the one gpt-oss-120b picks in context"
    (D / "manifest.json").write_text(json.dumps(man, indent=2))
    for k, v in stats.most_common(40):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
