"""Fresh test sets for the contamination check (data/ood_new), built without Qwen.

CrossNER-style domains (ai, literature, music, politics, science): lead sections of
English Wikipedia articles created after gliformer's release (2026-09-11), so no
evaluated model can have trained on the text. A non-Qwen model assigns each article
a domain; snippets of 1-3 sentences are taken from the lead.
MIT-style domains (movie, restaurant): no fresh real source exists, so a non-Qwen
model writes lowercase spoken requests in the style of MIT *train* examples.
Every snippet is annotated by three non-Qwen models with the benchmark's own label
set and descriptions; a span is kept when at least two of them give it with the
same type. Rows where any annotator's answer cannot be parsed are dropped.

Stages: fetch-wiki (network, CPU) -> classify (LLM) -> snippets -> gen-mit (LLM)
-> annotate --model M (LLM, once per annotator) -> assemble.
"""
import argparse
import collections
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .llm_annotate import A2_PROMPT, align, clean_type, read_jsonl, write_jsonl

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
W = ROOT / "newtest"
OUT = ROOT / "data/ood_new"
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "ie-rlcd-research/0.1 (NER evaluation; contact via project maintainer)"}
START, END = "2026-09-12T00:00:00Z", "2026-09-29T00:00:00Z"
CN_DOMAINS = ["ai", "literature", "music", "politics", "science"]
MIT = {"movie": "mit_movie", "restaurant": "mit_restaurant"}
MIT_TRAIN = {
    "movie": Path.home() / ".cache/huggingface/hub/datasets--mneb--mit-movie/snapshots/"
             "7713e53a7e6b4b181ebdb984803d024a825b83c6/data/mit_movie_complex_train.jsonl",
    "restaurant": Path.home() / ".cache/huggingface/hub/datasets--mneb--mit-restaurant/snapshots/"
                  "9dc75d3666eea550722ae4cc3bb9dfc9d3b7ceed/data/mit_restaurant_train.jsonl"}

CLASSIFY = """Article title: {title}
Opening text: {text}

Which one of these areas is this article mainly about?
- "ai": artificial intelligence, machine learning, computing research
- "literature": books, writers, poetry, literary works and awards
- "music": musicians, bands, albums, songs, music genres and events
- "politics": politicians, parties, elections, governments, political events
- "science": natural sciences, scientists, research in physics, chemistry, biology, astronomy and similar
- "none": anything else
Return JSON only: {{"domain": "..."}}"""

GEN_MIT = """Here are examples of short spoken requests that people say to a {what} search assistant:
{examples}

Write {n} new, different requests of the same kind, about different {things}. Write them the same way: all lowercase, no punctuation, casual spoken English, varied length and wording. Do not copy or paraphrase the examples.
Return JSON only: a list of strings."""


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as exc:                  # transient network / rate limit
            time.sleep(5 * (attempt + 1))
            err = exc
    raise err


def fetch_wiki(args):
    titles, cont = [], {}
    while True:
        d = api({"action": "query", "list": "logevents", "letype": "create", "lenamespace": 0,
                 "lestart": START, "leend": END, "ledir": "newer", "lelimit": 500, **cont})
        titles += [e["title"] for e in d["query"]["logevents"] if "title" in e]
        if "continue" not in d:
            break
        cont = d["continue"]
        time.sleep(0.5)
    titles = sorted(set(titles))
    pages = []
    for k in range(0, len(titles), 20):
        d = api({"action": "query", "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1,
                 "exlimit": 20, "redirects": 1, "titles": "|".join(titles[k:k + 20])})
        for p in d["query"].get("pages", []):
            text = (p.get("extract") or "").strip()
            if p.get("missing") or "disambiguation" in (p.get("pageprops") or {}) or len(text.split()) < 30:
                continue
            pages.append({"id": f"wiki:{p['pageid']}", "title": p["title"], "text": text})
        time.sleep(0.3)
    W.mkdir(parents=True, exist_ok=True)
    write_jsonl(W / "wiki_pages.jsonl", pages)
    print(json.dumps({"created_titles": len(titles), "pages_with_lead": len(pages), "window": [START, END]}))


def llm(model, max_len=8192):
    from vllm import LLM
    kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=0.9, max_num_seqs=128,
              trust_remote_code=True)
    try:
        return LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as exc:
        print("retrying without limit_mm_per_prompt:", repr(exc)[:200], flush=True)
        return LLM(**kw)


def chat(engine, prompts, max_tokens, temperature=0.0, seed=0):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = engine.chat(msgs, SamplingParams(temperature=temperature, top_p=0.95 if temperature else 1.0,
                                            max_tokens=max_tokens, seed=seed),
                       chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
    return [final_text(o.outputs[0].text) for o in outs]


def final_text(t):
    """gpt-oss prints its analysis channel before the answer; keep the final part."""
    for marker in ("assistantfinal", "<|channel|>final<|message|>"):
        if marker in t:
            t = t.split(marker)[-1]
    return t


def last_json(t):
    """The longest parseable JSON object or list in the text (the latest one on a tie),
    so an inner list of the answer object is never returned instead of the object."""
    dec = json.JSONDecoder()
    best, best_len = None, -1
    for m in re.finditer(r"[\[{]", t):
        try:
            v, end = dec.raw_decode(t[m.start():])
        except json.JSONDecodeError:
            continue
        if end >= best_len:
            best, best_len = v, end
    return best


def classify(args):
    pages = read_jsonl(W / "wiki_pages.jsonl")
    engine = llm(args.model)
    outs = chat(engine, [CLASSIFY.format(title=p["title"], text=" ".join(p["text"].split()[:150]))
                         for p in pages], 64)
    for p, o in zip(pages, outs):
        js = last_json(o)
        p["domain"] = str(js.get("domain", "none")).strip().lower() if isinstance(js, dict) else "none"
    write_jsonl(W / "wiki_classified.jsonl", pages)
    print(json.dumps(dict(collections.Counter(p["domain"] for p in pages))))


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s]


def snippets(args):
    rng = random.Random(args.seed)
    pages = [p for p in read_jsonl(W / "wiki_classified.jsonl") if p["domain"] in CN_DOMAINS]
    rows = []
    for p in pages:
        sents, cur, got = sentences(p["text"]), [], 0
        for s in sents:
            cur.append(s)
            n = len(" ".join(cur).split())
            if n >= 25 or len(cur) == 3:
                if 8 <= n <= 90:
                    rows.append({"id": f"{p['id']}:{got}", "domain": p["domain"], "title": p["title"],
                                 "text": " ".join(cur)})
                    got += 1
                cur = []
            if got == 2:
                break
    by = collections.defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    keep = []
    for d, rs in by.items():
        rng.shuffle(rs)
        keep += rs[:args.per_domain]
    write_jsonl(W / "snippets_cn.jsonl", keep)
    print(json.dumps({d: min(len(rs), args.per_domain) for d, rs in by.items()}))


def gen_mit(args):
    engine = llm(args.model)
    rows = []
    for dom, (what, things) in {"movie": ("movie", "movies, actors and plots"),
                                "restaurant": ("restaurant", "restaurants, dishes and places")}.items():
        pool = [json.loads(l)["input"] for l in open(MIT_TRAIN[dom])]
        prompts, rng = [], random.Random(args.seed)
        for k in range(args.mit_requests):
            ex = rng.sample(pool, 5)
            prompts.append(GEN_MIT.format(what=what, things=things, n=10,
                                          examples="\n".join(f"- {e}" for e in ex)))
        outs = chat(engine, prompts, 1024, temperature=0.9, seed=args.seed)
        seen = set(pool)
        for k, o in enumerate(outs):
            js = last_json(o)
            for j, t in enumerate(js if isinstance(js, list) else []):
                t = re.sub(r"[^\w\s']", " ", str(t).lower())
                t = re.sub(r"\s+", " ", t).strip()
                if 3 <= len(t.split()) <= 40 and t not in seen:
                    seen.add(t)
                    rows.append({"id": f"gen:{dom}:{k}:{j}", "domain": dom, "text": t})
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    by = collections.defaultdict(list)
    for r in rows:
        if len(by[r["domain"]]) < args.per_domain:
            by[r["domain"]].append(r)
    write_jsonl(W / "snippets_mit.jsonl", [r for rs in by.values() for r in rs])
    print(json.dumps({d: len(rs) for d, rs in by.items()}))


def schemas():
    """Each benchmark's own label set and descriptions, from our test files."""
    out = {}
    for d in CN_DOMAINS:
        r = json.loads(open(ROOT / f"data/ood/crossner_{d}.jsonl").readline())
        out[d] = [(q["label"], q["description"]) for q in r["queries"]]
    for d, name in MIT.items():
        r = json.loads(open(ROOT / f"data/ood/{name}.jsonl").readline())
        out[d] = [(q["label"], q["description"]) for q in r["queries"]]
    return out


def all_snippets():
    return read_jsonl(W / "snippets_cn.jsonl") + read_jsonl(W / "snippets_mit.jsonl")


def annotate(args):
    sch = schemas()
    rows = all_snippets()
    engine = llm(args.model)
    prompts = []
    for r in rows:
        types = "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in sch[r["domain"]])
        prompts.append(A2_PROMPT.format(text=r["text"], types=types))
    outs = chat(engine, prompts, 1024)
    name = Path(args.model).name
    write_jsonl(W / f"annot-{name}.jsonl", [{"id": r["id"], "output": o} for r, o in zip(rows, outs)])
    print(name, "annotated", len(rows))


def assemble(args):
    from .prepare_mix import find_all
    sch = schemas()
    rows = all_snippets()
    files = sorted(W.glob("annot-*.jsonl"))
    votes_by = {f.stem[6:]: {o["id"]: o["output"] for o in read_jsonl(f)} for f in files}
    stats = collections.Counter()
    by_set = collections.defaultdict(list)
    for r in rows:
        labels = [t for t, _ in sch[r["domain"]]]
        votes, ok = collections.Counter(), True
        for model, outs in votes_by.items():
            js = last_json(outs.get(r["id"], ""))
            if not isinstance(js, dict):
                ok = False
                break
            ans = {clean_type(k): v for k, v in js.items() if isinstance(v, list)}
            spans = set()
            for t in labels:
                v = [str(x).strip() for x in ans.get(clean_type(t), []) if str(x).strip()]
                occ = align(r["text"], v, find_all) if v else []
                if occ is None:
                    stats[f"{model}_unaligned"] += 1
                    continue
                spans |= {(s, e, t) for s, e in occ}
            votes.update(spans)
        if not ok:
            stats["rows_dropped_parse"] += 1
            continue
        ents = sorted((s, e, t) for (s, e, t), n in votes.items() if n >= 2)
        stats["spans_3_votes"] += sum(n == 3 for n in votes.values())
        stats["spans_2_votes"] += sum(n == 2 for n in votes.values())
        stats["spans_1_vote_dropped"] += sum(n == 1 for n in votes.values())
        present = {t for _, _, t in ents}
        name = MIT.get(r["domain"], f"crossner_{r['domain']}") + "_new"
        queries = [{"label": t, "description": d,
                    "status": "positive" if t in present else "scope_assumed_absent",
                    "description_origin": "source",
                    "supervision_origin": "llm_consensus" if t in present else "benchmark_closed_world"}
                   for t, d in sch[r["domain"]]]
        by_set[name].append({"id": f"{name}:{r['id']}", "source": name, "text": r["text"], "queries": queries,
                             "entities": [{"start": s, "end": e, "label": t} for s, e, t in ents]})
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"scope_policy": "benchmark_closed_world",
                "splits_used": "new text: Wikipedia articles created " + f"{START}..{END}" +
                               "; generated MIT-style requests",
                "annotators": list(votes_by), "rule": "span kept when >= 2 annotators agree on span and type",
                "threshold_policy": "zero-shot: training sources' dev_calib T and threshold", "sets": {}}
    for name, rs in sorted(by_set.items()):
        write_jsonl(OUT / f"{name}.jsonl", rs)
        labels = [q["label"] for q in rs[0]["queries"]]
        manifest["sets"][name] = {"rows": len(rs), "entities": sum(len(r["entities"]) for r in rs),
                                  "labels": labels, "seen_labels": [], "unseen_labels": labels}
    manifest["stats"] = dict(stats)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {"rows": v["rows"], "entities": v["entities"]} for k, v in manifest["sets"].items()}))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-wiki", "classify", "snippets", "gen-mit", "annotate", "assemble"])
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--mit-requests", type=int, default=150)
    args = ap.parse_args()
    {"fetch-wiki": fetch_wiki, "classify": classify, "snippets": snippets, "gen-mit": gen_mit,
     "annotate": annotate, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
