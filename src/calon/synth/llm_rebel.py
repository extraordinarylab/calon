"""Qwen3.8 annotation of mneb/rebel (entities + relations), sharded by data.

For later multi-task (NER + RE) training; not used for NER training yet
(REBEL text is Wikipedia, like CrossNER). Each sentence keeps its REBEL
relations; Qwen adds entity types with definitions (heads, tails and other
mentions), missing explicit relations, definitions for every relation name,
and 2-4 relation types the text does not state (negatives).
Subcommands: prepare (download, count, write NSHARDS request files), run.
"""
import argparse
import json
import time

from calon.data.roots import MODELS, ROOT

from .llm_annotate import seed_of

S3 = ROOT / "synth_rebel"
REPO = "mneb/rebel"
PROMPT = """Text:
{text}

Known relations in this text (head -> relation -> tail):
{rels}

1. List every entity mention in the text, including the heads and tails above, with a specific but reusable entity type in lowercase English, and give a one-sentence definition for every type you use.
2. List further relations between entity mentions that the text states explicitly and that are missing above, using the same style of relation names. Give a one-sentence definition for every relation name you use, including the known ones.
3. List 2 to 4 relation types that would be plausible for these entities but are not stated in the text, each with a one-sentence definition.
Return JSON only: {{"entities": [{{"text": "...", "type": "..."}}], "entity_types": {{"type": "definition"}}, "relations": [{{"head": "...", "relation": "...", "tail": "..."}}], "relation_types": {{"relation": "definition"}}, "absent_relations": {{"relation": "definition"}}}}"""


def relations_of(rec):
    rels = []
    for name, items in ((rec.get("output") or {}).get("relations") or {}).items():
        for it in items or []:
            h, t = it.get("head") or {}, it.get("tail") or {}
            rels.append({"relation": name, "head": h.get("text"), "head_start": h.get("start"),
                         "head_end": h.get("end"), "tail": t.get("text"), "tail_start": t.get("start"),
                         "tail_end": t.get("end")})
    return rels


def prepare(args):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(REPO)
    S3.mkdir(parents=True, exist_ok=True)
    outs = [open(S3 / f"req-{k:05d}.jsonl", "w") for k in range(args.num_shards)]
    counts, n = {}, 0
    for split in ("train", "validation", "test"):
        path = hf_hub_download(REPO, f"data/{split}.jsonl", repo_type="dataset", revision=info.sha,
                               cache_dir=str(ROOT / "raw/hf"))
        c = 0
        with open(path) as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                r = {"id": f"rebel:{split}:{i}", "split": split, "text": rec["input"], "relations": relations_of(rec)}
                outs[seed_of(r["id"]) % args.num_shards].write(json.dumps(r, ensure_ascii=False) + "\n")
                c += 1
        counts[split] = c
        n += c
    for f in outs:
        f.close()
    meta = {"repo": REPO, "revision": info.sha, "rows": counts, "total": n, "num_shards": args.num_shards}
    (S3 / "meta.json").write_text(json.dumps(meta, indent=1))
    first = json.loads(open(S3 / "req-00000.jsonl").readline())
    print(json.dumps(meta))
    print("example:", json.dumps(first, ensure_ascii=False)[:500])


def build_prompt(r):
    rels = "\n".join(f"- {x['head']} -> {x['relation']} -> {x['tail']}" for x in r["relations"]) or "- (none)"
    return PROMPT.format(text=r["text"], rels=rels)


def run(args):
    from vllm import LLM, SamplingParams
    src = S3 / f"req-{args.shard:05d}.jsonl"
    reqs = [json.loads(l) for l in open(src)]
    out_path = S3 / "out" / f"out-{args.shard:05d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in reqs if r["id"] not in done][:args.limit]
    print(f"shard {args.shard}: {len(reqs)} items, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9,
              max_num_seqs=256, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        t0 = time.time()
        outs = llm.chat([[{"role": "user", "content": build_prompt(r)}] for r in chunk], params,
                        chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for r, o in zip(chunk, outs):
                g = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "output": g.text, "finish_reason": g.finish_reason},
                                   ensure_ascii=False) + "\n")
        dt = time.time() - t0
        print(f"chunk {c // args.chunk}: {len(chunk)} items {dt:.0f}s {len(chunk) / dt:.2f} items/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--num-shards", type=int, default=256)
    p = sub.add_parser("run")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--model", default=str(MODELS / "Qwen/Qwen3.8-27B"))
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()
    {"prepare": prepare, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
