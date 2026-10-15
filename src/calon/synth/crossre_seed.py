"""CrossRE-seeded relation data (synth_cre), sharded by data.

Each request shows 5 CrossRE *train* sentences of one domain with their entities and
relations, the domain's entity types (CrossNER descriptions) and the 17 CrossRE
relation types (CrossRE paper definitions), and asks for 5 new annotated sentences.
Like the CrossNER / MIT seeded NER data (synth_c), this uses the target training
split only; CrossRE validation and test are never shown.

Rows: entity queries and relation queries are positive when used, otherwise
scope_assumed_absent (the generator was told to annotate everything), or unknown
when the example had an invalid or unaligned item of that kind. Rows that copy a
seed (word 5-gram containment >= 0.8) or come near any evaluation text are dropped.

Subcommands: prepare, run, assemble.
"""
import argparse
import collections
import json
import random
import time

from calon.data.roots import MODELS, ROOT

from .llm_annotate import align, parse_json, read_jsonl, seed_of, write_jsonl

S = ROOT / "synth_cre"
OUT = ROOT / "data/re/synth_cre"
DOMAIN_NAMES = {"ai": "artificial intelligence", "literature": "literature", "music": "music",
                "news": "news", "politics": "politics", "science": "natural science"}

PROMPT = """You are creating training data for entity and relation extraction in this domain: {domain} (encyclopedic or news text).

Entity types:
{etypes}

Relation types (directed, from head to tail):
{rtypes}

Annotated examples from this domain:
{examples}

Write 5 new examples in the same style, register and length as the examples above, each about a different, specific sub-topic that the examples do not cover. Do not copy or paraphrase the examples.
Annotate every mention of every listed entity type exactly as it appears in your text, and every relation of a listed type that the text states between two of your entity mentions, following the conventions of the examples (the same pair may carry more than one relation type).
Return JSON only: {{"examples": [{{"text": "...", "entities": [{{"text": "...", "type": "..."}}], "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}]}}"""


def domain_schema(d):
    # data modules need pyarrow, which the vLLM env lacks: requests carry their schema
    from ..data.re_prep import CROSSRE, CROSSRE_DEFS
    rec =json.loads(open(CROSSRE / d / "train.jsonl").readline())
    ents = [(s["label"], s.get("description") or "") for s in rec["schema"] if s["task"] == "entities"]
    return ents, sorted(CROSSRE_DEFS.items())


def format_example(rec):
    ents = [{"text": m["text"], "type": lab} for lab, ms in rec["output"].get("entities", {}).items() for m in ms]
    rels = [{"head": m["head"]["text"], "relation": lab, "tail": m["tail"]["text"]}
            for lab, ms in rec["output"].get("relations", {}).items() for m in ms]
    return (f"text: {rec['input']}\nentities: {json.dumps(ents, ensure_ascii=False)}\n"
            f"relations: {json.dumps(rels, ensure_ascii=False)}")


def prepare(args):
    from ..data.re_prep import CR_DOMAINS, CROSSRE
    reqs = []
    for d in CR_DOMAINS:
        pool = read_jsonl(CROSSRE / d / "train.jsonl")
        ents, rels = domain_schema(d)
        for k in range(args.per_domain):
            rng = random.Random(seed_of(f"cre:{d}:{k}"))
            ex = rng.sample(pool, 5)
            reqs.append({"id": f"cre:{d}:{k}", "domain": d, "etypes": ents, "rtypes": rels,
                         "examples": [format_example(e) for e in ex]})
    rng = random.Random(0)
    rng.shuffle(reqs)                      # shards get every domain
    write_jsonl(S / "requests.jsonl", reqs)
    print("requests", len(reqs), {d: len(read_jsonl(CROSSRE / d / "train.jsonl")) for d in CR_DOMAINS})


def build_prompt(r):
    etypes = "\n".join(f'- "{t}": {x}' if x else f'- "{t}"' for t, x in r["etypes"])
    rtypes = "\n".join(f'- "{t}": {x}' for t, x in r["rtypes"])
    examples = "\n\n".join(f"Example {i + 1}:\n{e}" for i, e in enumerate(r["examples"]))
    return PROMPT.format(domain=DOMAIN_NAMES[r["domain"]], etypes=etypes, rtypes=rtypes, examples=examples)


def run(args, base=S, build=None):
    """Shard `args.shard` of base/requests.jsonl -> base/out/; resumes from disk.
    Also used by re_gen (base and prompt builder passed in)."""
    from vllm import LLM, SamplingParams
    build = build or build_prompt
    reqs = read_jsonl(base / "requests.jsonl")
    mine = [r for i, r in enumerate(reqs) if i % args.num_shards == args.shard][:args.limit]
    out_path = base / "out" / f"shard-{args.shard:05d}-of-{args.num_shards:05d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    todo = [r for r in mine if r["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} items, {len(done)} done, {len(todo)} to go",
          flush=True)
    if not todo:
        return
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9,
              max_num_seqs=256, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        t0 = time.time()
        params = [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens, seed=seed_of(r["id"]))
                  for r in chunk]
        outs = llm.chat([[{"role": "user", "content": build(r)}] for r in chunk], params,
                        chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for r, o in zip(chunk, outs):
                g = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "output": g.text, "finish_reason": g.finish_reason},
                                   ensure_ascii=False) + "\n")
        dt = time.time() - t0
        print(f"chunk {c // args.chunk}: {len(chunk)} items {dt:.0f}s {len(chunk) / dt:.2f} items/s", flush=True)


def example_row(rid, d, ex, schema, stats):
    """One generated example -> training row, or None."""
    from ..data.prepare_mix import find_all
    from ..data.re_prep import nearest
    ent_labels, rel_labels = schema
    text = ex["text"].strip()
    by, mention_type, bad_ent = collections.defaultdict(list), {}, False
    for e in ex.get("entities") or []:
        if not isinstance(e, dict) or e.get("type") not in ent_labels or not e.get("text"):
            bad_ent = True
            continue
        s = str(e["text"]).strip()
        by[e["type"]].append(s)
        mention_type.setdefault(s, e["type"])
    queries, ents = [], set()
    for lab, desc in ent_labels.items():
        if lab in by:
            occ = align(text, by[lab], find_all)
            if occ is None:
                stats["entity_type_unaligned"] += 1
                queries.append({"label": lab, "description": desc, "status": "unknown"})
                for s in by[lab]:
                    mention_type.pop(s, None)
            else:
                queries.append({"label": lab, "description": desc, "status": "positive"})
                ents |= {(a, b, lab) for a, b in occ}
        else:
            queries.append({"label": lab, "description": desc,
                            "status": "unknown" if bad_ent else "scope_assumed_absent"})
    if not ents:
        stats["no_entities"] += 1
        return None
    spans = {(a, b) for a, b, _ in ents}
    rels, bad_rel = set(), False
    for r in ex.get("relations") or []:
        if not isinstance(r, dict):
            bad_rel = True
            continue
        h, t, lab = str(r.get("head", "")).strip(), str(r.get("tail", "")).strip(), r.get("relation")
        if lab not in rel_labels or h not in mention_type or t not in mention_type:
            stats["relation_invalid"] += 1
            bad_rel = True
            continue
        pair = nearest(text, h, t)
        if pair is None or pair[0] not in spans or pair[1] not in spans:
            stats["relation_unaligned"] += 1
            bad_rel = True
            continue
        rels.add((pair[0][0], pair[0][1], pair[1][0], pair[1][1], lab))
    used = {x[4] for x in rels}
    rel_queries = [{"label": lab, "description": desc,
                    "status": "positive" if lab in used else ("unknown" if bad_rel else "scope_assumed_absent")}
                   for lab, desc in rel_labels.items()]
    stats["relations"] += len(rels)
    stats["rows_with_bad_relation"] += bad_rel
    for q in queries:
        q.update(description_origin="source", supervision_origin="llm_generated_crossre_seeded")
    return {"id": f"synth_cre:{rid}", "source": "synth_cre", "domain": d, "text": text, "queries": queries,
            "entities": [{"start": a, "end": b, "label": lab} for a, b, lab in sorted(ents)],
            "rel_queries": rel_queries,
            "relations": [{"head": [a, b], "tail": [c, e], "label": lab} for a, b, c, e, lab in sorted(rels)]}


def assemble(args):
    from ..data.prepare_data import word_ngrams
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import CR_DOMAINS, CROSSRE, eval_texts
    stats = collections.Counter()
    reqs = {r["id"]: r for r in read_jsonl(S / "requests.jsonl")}
    schemas = {d: (dict(domain_schema(d)[0]), dict(domain_schema(d)[1])) for d in CR_DOMAINS}
    seed_grams = {d: [word_ngrams(r["input"]) for r in read_jsonl(CROSSRE / d / "train.jsonl")] for d in CR_DOMAINS}
    rows_by_req = collections.defaultdict(list)
    for p in sorted((S / "out").glob("shard-*.jsonl")):
        for o in read_jsonl(p):
            req, js = reqs[o["id"]], parse_json(o["output"])
            stats["requests"] += 1
            if not isinstance(js, dict) or not isinstance(js.get("examples"), list):
                stats["parse_fail"] += 1
                continue
            for k, ex in enumerate(js["examples"]):
                if not isinstance(ex, dict) or not isinstance(ex.get("text"), str) or not ex["text"].strip():
                    continue
                stats["examples"] += 1
                g = word_ngrams(ex["text"])
                if g and any(sg and len(g & sg) / len(g) >= 0.8 for sg in seed_grams[req["domain"]]):
                    stats["copied_seed"] += 1
                    continue
                row = example_row(f"{o['id']}:{k}", req["domain"], ex, schemas[req["domain"]], stats)
                if row is not None:
                    rows_by_req[o["id"]].append(row)
    ids = sorted(rows_by_req)
    random.Random(args.seed).shuffle(ids)
    n = args.n_dev
    splits = {"dev_select": ids[:n], "dev_calib": ids[n:2 * n], "test": ids[2 * n:3 * n], "train": ids[3 * n:]}
    texts = eval_texts()
    summary = {"stats": {}, "splits": {}}
    for sp, keys in splits.items():
        rows = contamination_filter([r for k in keys for r in rows_by_req[k]], texts, stats, sp)
        write_jsonl(OUT / f"{sp}.jsonl", rows)
        summary["splits"][sp] = {"rows": len(rows), "relations": sum(len(r["relations"]) for r in rows),
                                 "by_domain": dict(collections.Counter(r["domain"] for r in rows))}
    summary["stats"] = dict(stats)
    (OUT.parent / "synth_cre_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if args.show:
        r = read_jsonl(OUT / "train.jsonl")[:args.show]
        for x in r:
            t = x["text"]
            print(t[:300])
            print("  E:", [(t[e["start"]:e["end"]], e["label"]) for e in x["entities"]][:12])
            print("  R:", [(t[y["head"][0]:y["head"][1]], y["label"], t[y["tail"][0]:y["tail"][1]])
                           for y in x["relations"]][:10])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--per-domain", type=int, default=4000)
    p = sub.add_parser("run")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--model", default=str(MODELS / "Qwen/Qwen3.8-27B"))
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=3072)
    p = sub.add_parser("assemble")
    p.add_argument("--n-dev", type=int, default=200, help="requests per held-out split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
