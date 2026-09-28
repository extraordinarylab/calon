"""Qwen3.8 data generation, sharded by data.

Two products:
  A  open-type annotation of non-Wikipedia web passages (FineWeb), two passes:
     a1: the LLM lists mentions with free-form types and a definition per type;
     a2: each passage is asked about 8 types it did not list (from the global
         a1 type pool, half sharing a head word with a listed type); [] answers
         become explicit_absent (supervision_origin=llm_absent), found spans
         become positives. Zero-shot: no CrossNER/MIT data is involved.
  C  target-seeded generation: 5 CrossNER/MIT *train* examples + that domain's
     label set and descriptions -> 5 new annotated examples. Types not used in
     an example are scope_assumed_absent (llm_generated_closed_world). Outputs
     are checked against the CrossNER/MIT *test* sets and against the seeds.
     Results trained on C must be reported apart from zero-shot results.

Subcommands: prepare-a, prepare-c, run, prepare-a2, assemble. `run` processes
shard k of n of a request file in chunks and resumes from what is on disk.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
from pathlib import Path

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
SYNTH = ROOT / "synth"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
CROSSNER = HUB / "datasets--mneb--cross-ner/snapshots/64e9a4a56dbe9457c7ad8fe68a710b81ed63599f/data"
MIT = {"mit_movie": HUB / "datasets--mneb--mit-movie/snapshots/7713e53a7e6b4b181ebdb984803d024a825b83c6/data/mit_movie_complex_train.jsonl",
       "mit_restaurant": HUB / "datasets--mneb--mit-restaurant/snapshots/9dc75d3666eea550722ae4cc3bb9dfc9d3b7ceed/data/mit_restaurant_train.jsonl"}
CROSSNER_DOMAINS = ["ai", "literature", "music", "politics", "science"]
PROMPT_VERSION = "v1"

A1_PROMPT = """Text:
{text}

List every entity mention in the text: named entities (people, organizations, places, creative works, products, events, laws, ...) and other specific typed spans a reader might want to extract (dates, quantities, technical terms, diseases, ...).
For each mention give the exact text as it appears and an entity type. Use specific but reusable type names in lowercase English (for example "politician", "programming language", "river"), not generic ones like "entity" or "thing", and not names that only make sense for this text.
Then give a one-sentence definition for every type you used, general enough to apply to other texts; do not mention this text.
Return JSON only: {{"entities": [{{"text": "...", "type": "..."}}], "types": {{"type name": "definition"}}}}"""

A2_PROMPT = """Text:
{text}

For each entity type below, list every span in the text that belongs to it (exact text as it appears), or [] if there is none.
Types:
{types}

Return JSON only, one key per type name above: {{"type name": ["span", ...]}}"""

C_PROMPT = """You are creating training data for entity extraction in this domain: {domain}.
Entity types and definitions:
{types}

Annotated examples from this domain:
{examples}

Write {n} new examples in the same style, register and length as the examples above, about different content. Do not copy or paraphrase the examples.
Annotate every mention of every listed type exactly as it appears in your text; a type that does not occur in an example gets no entries.
Return JSON only: {{"examples": [{{"text": "...", "entities": [{{"text": "...", "type": "..."}}]}}]}}"""
C_DOMAIN_NAMES = {"ai": "artificial intelligence (encyclopedic text)", "literature": "literature (encyclopedic text)",
                  "music": "music (encyclopedic text)", "politics": "politics (encyclopedic text)",
                  "science": "natural science (encyclopedic text)",
                  "mit_movie": "spoken queries to a movie search assistant",
                  "mit_restaurant": "spoken queries to a restaurant search assistant"}


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- A: passages
def chunk_passages(text, max_words=180, min_words=40):
    """Split a document on sentence ends into passages of at most max_words."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], []
    for s in sents:
        w = s.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return [p for p in out if min_words <= len(p.split()) <= max_words]


def prepare_a(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from .prepare_data import SPLITS  # noqa: F401  (split names used in assemble)
    from .prepare_mix import contamination_filter

    api = HfApi()
    repo = "HuggingFaceFW/fineweb"
    info = api.dataset_info(repo)
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=info.sha)
                   if f.startswith("sample/10BT/") and f.endswith(".parquet"))
    path = hf_hub_download(repo, files[0], repo_type="dataset", revision=info.sha,
                           cache_dir=str(ROOT / "raw/hf"))
    rng = random.Random(args.seed)
    docs = pq.read_table(path, columns=["id", "text", "url"]).to_pylist()
    docs = [d for d in docs if "wikipedia.org" not in (d["url"] or "")]
    rng.shuffle(docs)
    passages = []
    for d in docs:
        for k, p in enumerate(chunk_passages(d["text"])):
            passages.append({"id": f"fw:{d['id']}:{k}", "text": p, "url": d["url"]})
        if len(passages) >= args.n * 1.2:
            break
    stats = collections.Counter(docs_seen=len(docs), passages=len(passages))
    eval_texts = []
    for split in ("dev_select", "dev_calib", "test"):
        for src in ("finerweb", "fewnerd", "pile", "massive"):
            eval_texts += [r["text"] for r in read_jsonl(ROOT / f"data/mix/{src}/{split}.jsonl")]
    for f in (ROOT / "data/ood").glob("*.jsonl"):
        eval_texts += [r["text"] for r in read_jsonl(f)]
    passages = contamination_filter(passages, eval_texts, stats, "a")[:args.n]
    write_jsonl(SYNTH / "a1_requests.jsonl", passages)
    meta = {"repo": repo, "revision": info.sha, "file": files[0], "n": len(passages),
            "stats": dict(stats), "prompt_version": PROMPT_VERSION}
    (SYNTH / "a1_requests.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


# ---------------------------------------------------------------- C: seeded requests
def seed_pool():
    pools = {}
    for dom in CROSSNER_DOMAINS:
        pools[dom] = read_jsonl(CROSSNER / dom / "train.jsonl")
    for dom, path in MIT.items():
        pools[dom] = read_jsonl(path)
    return pools


def schema_items(schema):
    items = schema.get("entities", []) if isinstance(schema, dict) else schema
    return [(it["label"], it.get("description") or "") for it in items]


def format_example(rec):
    ents = [{"text": m["text"], "type": lab}
            for lab, ms in (rec["output"].get("entities") or {}).items() for m in ms]
    return f"text: {rec['input']}\nentities: {json.dumps(ents, ensure_ascii=False)}"


def prepare_c(args):
    pools = seed_pool()
    reqs = []
    for dom, pool in pools.items():
        labels = schema_items(pool[0]["schema"])
        for k in range(args.per_domain):
            rng = random.Random(seed_of(f"c:{dom}:{k}"))
            ex = rng.sample(pool, min(args.k, len(pool)))
            reqs.append({"id": f"c:{dom}:{k}", "domain": dom, "labels": labels,
                         "seed_ids": [pool.index(e) for e in ex],
                         "examples": [format_example(e) for e in ex]})
    write_jsonl(SYNTH / "c_requests.jsonl", reqs)
    print({d: len(p) for d, p in pools.items()}, "requests", len(reqs))


# ---------------------------------------------------------------- prompts
def build_prompt(task, r, n_new):
    if task == "a1":
        return A1_PROMPT.format(text=r["text"])
    if task == "a2":
        types = "\n".join(f'- "{t}": {d}' for t, d in r["query_types"])
        return A2_PROMPT.format(text=r["text"], types=types)
    types = "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in r["labels"])
    examples = "\n\n".join(f"Example {i + 1}:\n{e}" for i, e in enumerate(r["examples"]))
    return C_PROMPT.format(domain=C_DOMAIN_NAMES[r["domain"]], types=types, examples=examples,
                           n=n_new)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- run (one shard)
def run(args):
    from vllm import LLM, SamplingParams

    reqs = read_jsonl(SYNTH / f"{args.task}_requests.jsonl")
    mine = [r for i, r in enumerate(reqs) if i % args.num_shards == args.shard][:args.limit]
    out_path = SYNTH / args.task / f"shard-{args.shard:05d}-of-{args.num_shards:05d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path)}
    todo = [r for r in mine if r["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} items, {len(done)} done, "
          f"{len(todo)} to go", flush=True)
    if not todo:
        return
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, enforce_eager=args.eager,
              max_num_seqs=args.max_num_seqs,     # hybrid model: 620 Mamba cache blocks here
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    import time
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        t0 = time.time()
        msgs = [[{"role": "user", "content": build_prompt(args.task, r, args.n_new)}] for r in chunk]
        if args.task == "c":
            params = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens,
                                     seed=seed_of(r["id"])) for r in chunk]
        else:
            params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for r, o in zip(chunk, outs):
                g = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "output": g.text,
                                    "finish_reason": g.finish_reason,
                                    "n_out_tokens": len(g.token_ids)}, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"chunk {c // args.chunk}: {len(chunk)} items {dt:.0f}s "
              f"{len(chunk) / dt:.2f} items/s {ntok / dt:.0f} out tok/s", flush=True)


def load_outputs(task):
    out = {}
    for p in sorted((SYNTH / task).glob("shard-*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------- A pass 2 requests
def clean_type(t):
    return re.sub(r"\s+", " ", str(t).strip().lower().replace("_", " "))


def parse_a1(text):
    js = parse_json(text)
    if not isinstance(js, dict) or not isinstance(js.get("entities"), list):
        return None
    types = {clean_type(k): str(v).strip() for k, v in (js.get("types") or {}).items()
             if isinstance(v, str)}
    ents = [(str(e["text"]).strip(), clean_type(e["type"])) for e in js["entities"]
            if isinstance(e, dict) and e.get("text") and e.get("type")]
    return ents, types


def prepare_a2(args):
    reqs = {r["id"]: r for r in read_jsonl(SYNTH / "a1_requests.jsonl")}
    outs = load_outputs("a1")
    type_count, type_desc = collections.Counter(), {}
    parsed = {}
    for pid in sorted(outs):
        p = parse_a1(outs[pid]["output"])
        if p is None:
            continue
        parsed[pid] = p
        ents, types = p
        for t in {t for _, t in ents}:
            type_count[t] += 1
            if t not in type_desc and types.get(t):
                type_desc[t] = types[t]            # first definition in id order
    pool = [t for t, c in type_count.most_common() if c >= args.min_type_count and t in type_desc]
    by_head = collections.defaultdict(list)
    for t in pool:
        by_head[t.split()[-1]].append(t)
    frequent = pool[:2000]
    a2 = []
    for pid, (ents, _) in parsed.items():
        present = {t for _, t in ents}
        rng = random.Random(seed_of(f"a2:{pid}"))
        near = sorted({t for p in present for t in by_head.get(p.split()[-1], []) if t not in present})
        rng.shuffle(near)
        picked = near[:4]
        rest = [t for t in frequent if t not in present and t not in picked]
        picked += rng.sample(rest, min(8 - len(picked), len(rest)))
        a2.append({"id": pid, "text": reqs[pid]["text"],
                   "query_types": [(t, type_desc[t]) for t in picked]})
    write_jsonl(SYNTH / "a2_requests.jsonl", a2)
    (SYNTH / "a_type_pool.json").write_text(json.dumps(
        {"min_count": args.min_type_count, "types": {t: {"count": type_count[t],
                                                         "definition": type_desc[t]} for t in pool}},
        indent=1))
    print(f"a1 outputs={len(outs)} parsed={len(parsed)} types_seen={len(type_count)} "
          f"pool={len(pool)} a2_requests={len(a2)}")


# ---------------------------------------------------------------- assemble
def align(text, spans, find_all):
    """All occurrences of every span string; None if any string is not in the text.
    Occurrences nested inside a longer occurrence of the same type are dropped
    ("Still Modernism" inside "Still Modernism: Photography, Literature, Film")."""
    out = set()
    for s in spans:
        occ = find_all(text, s)
        if not occ:
            return None
        out.update(occ)
    return sorted(o for o in out
                  if not any(p != o and p[0] <= o[0] and o[1] <= p[1] for p in out))


def assemble(args):
    from .prepare_data import SPLITS, norm_text, sha, split_groups, word_ngrams
    from .prepare_mix import contamination_filter, find_all

    stats = collections.Counter()
    rows_by_src = {}
    if (SYNTH / "a1").exists():
        reqs = {r["id"]: r for r in read_jsonl(SYNTH / "a1_requests.jsonl")}
        a1, a2 = load_outputs("a1"), load_outputs("a2")
        a2_reqs = {r["id"]: r for r in read_jsonl(SYNTH / "a2_requests.jsonl")} \
            if (SYNTH / "a2_requests.jsonl").exists() else {}
        rows = []
        for pid, o in a1.items():
            p = parse_a1(o["output"])
            if p is None:
                stats["a1_parse_fail"] += 1
                continue
            text = reqs[pid]["text"]
            ents, types = p
            by_type = collections.defaultdict(list)
            for s, t in ents:
                by_type[t].append(s)
            queries, entities = [], []
            for t, spans in by_type.items():
                occ = align(text, spans, find_all)
                q = {"label": t, "description": types.get(t, ""),
                     "description_origin": "llm_authored" if types.get(t) else "name_only"}
                if occ is None:
                    q.update(status="unknown", supervision_origin="llm_unaligned")
                    stats["a1_unaligned_types"] += 1
                else:
                    q.update(status="positive", supervision_origin="llm_annotated")
                    entities += [{"start": s, "end": e, "label": t} for s, e in occ]
                queries.append(q)
            if pid in a2 and pid in a2_reqs:
                js = parse_json(a2[pid]["output"])
                if not isinstance(js, dict):
                    stats["a2_parse_fail"] += 1
                else:
                    asked = dict(a2_reqs[pid]["query_types"])
                    answers = {clean_type(k): v for k, v in js.items()}
                    for t, d in asked.items():
                        v = answers.get(t)
                        if not isinstance(v, list):
                            stats["a2_missing_answer"] += 1
                            continue
                        v = [str(x).strip() for x in v if str(x).strip()]
                        q = {"label": t, "description": d, "description_origin": "llm_authored"}
                        if not v:
                            q.update(status="explicit_absent", supervision_origin="llm_absent")
                            stats["a2_absent"] += 1
                        else:
                            occ = align(text, v, find_all)
                            if occ is None:
                                q.update(status="unknown", supervision_origin="llm_unaligned")
                            else:
                                q.update(status="positive", supervision_origin="llm_a2_found")
                                entities += [{"start": s, "end": e, "label": t} for s, e in occ]
                                stats["a2_found"] += 1
                        queries.append(q)
            entities = [dict(x) for x in {tuple(sorted(e.items())) for e in entities}]
            if any(q["status"] != "unknown" for q in queries):
                rows.append({"id": f"synth_a:{pid}", "source": "synth_a", "text": text,
                             "queries": queries, "entities": sorted(entities, key=lambda e: (e["start"], e["label"]))})
        rows_by_src["synth_a"] = rows

    if (SYNTH / "c").exists():
        reqs = {r["id"]: r for r in read_jsonl(SYNTH / "c_requests.jsonl")}
        pools = seed_pool()
        seed_grams = {d: [word_ngrams(r["input"]) for r in pool] for d, pool in pools.items()}
        rows = []
        for rid, o in load_outputs("c").items():
            req = reqs[rid]
            js = parse_json(o["output"])
            if not isinstance(js, dict) or not isinstance(js.get("examples"), list):
                stats["c_parse_fail"] += 1
                continue
            labels = dict(req["labels"])
            for k, ex in enumerate(js["examples"]):
                if not isinstance(ex, dict) or not isinstance(ex.get("text"), str):
                    stats["c_bad_example"] += 1
                    continue
                text = ex["text"].strip()
                g = word_ngrams(text)
                if g and any(len(g & sg) / len(g) >= 0.8 for sg in seed_grams[req["domain"]] if sg):
                    stats["c_copied_seed"] += 1
                    continue
                by_type, bad = collections.defaultdict(list), False
                for e in ex.get("entities") or []:
                    if not isinstance(e, dict) or e.get("type") not in labels or not e.get("text"):
                        stats["c_bad_entity"] += 1
                        bad = True
                        continue
                    by_type[e["type"]].append(str(e["text"]).strip())
                entities, queries = [], []
                for lab, desc in labels.items():
                    q = {"label": lab, "description": desc, "description_origin": "source"}
                    if lab in by_type:
                        occ = align(text, by_type[lab], find_all)
                        if occ is None:
                            q.update(status="unknown", supervision_origin="llm_unaligned")
                            stats["c_unaligned_types"] += 1
                        else:
                            q.update(status="positive", supervision_origin="llm_generated")
                            entities += [{"start": s, "end": e, "label": lab} for s, e in occ]
                    else:
                        # an invalid entity may have been meant for this type: keep it unknown
                        q.update(status="unknown" if bad else "scope_assumed_absent",
                                 supervision_origin="llm_generated_closed_world")
                    queries.append(q)
                entities = [dict(x) for x in {tuple(sorted(e.items())) for e in entities}]
                rows.append({"id": f"synth_c:{rid}:{k}", "source": "synth_c", "domain": req["domain"],
                             "text": text, "queries": queries,
                             "entities": sorted(entities, key=lambda e: (e["start"], e["label"]))})
        rows_by_src["synth_c"] = rows

    ood_test = []
    for f in (ROOT / "data/ood").glob("*.jsonl"):
        ood_test += [r["text"] for r in read_jsonl(f)]
    from .prepare_mix import fixed_eval_queries
    summary = {}
    for src, rows in rows_by_src.items():
        # keep every row's full schema well under the 1,500-token schema budget
        before = len(rows)
        rows = [r for r in rows if sum(len(q["label"]) + len(q["description"]) + 3
                                       for q in r["queries"]) <= 4000]
        stats[f"{src}_drop_long_schema"] = before - len(rows)
        rows = contamination_filter(rows, ood_test, stats, src)
        by_hash = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_hash[sha(norm_text(r["text"]))].append(i)
        assign = split_groups(list(by_hash.values()), len(rows), args.seed)
        out = ROOT / "data/mix" / src
        info = {}
        for s, _ in SPLITS:
            part = [rows[i] for i in range(len(rows)) if assign[i] == s]
            part = part if s == "train" else fixed_eval_queries(part)
            write_jsonl(out / f"{s}.jsonl", part)
            info[s] = {"rows": len(part), "entities": sum(len(r["entities"]) for r in part),
                       "status": dict(collections.Counter(q["status"] for r in part for q in r["queries"]))}
        summary[src] = info
    summary["stats"] = dict(stats)
    (SYNTH / "assemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-a"); p.add_argument("--n", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-c"); p.add_argument("--per-domain", type=int, default=1000)
    p.add_argument("--k", type=int, default=5)
    p = sub.add_parser("prepare-a2"); p.add_argument("--min-type-count", type=int, default=3)
    p = sub.add_parser("assemble"); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("run")
    p.add_argument("--task", choices=["a1", "a2", "c"], required=True)
    p.add_argument("--model", default="/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--n-new", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=256)
    args = ap.parse_args()
    {"prepare-a": prepare_a, "prepare-c": prepare_c, "prepare-a2": prepare_a2,
     "assemble": assemble, "run": run}[args.cmd.replace("_", "-")](args)


if __name__ == "__main__":
    main()
