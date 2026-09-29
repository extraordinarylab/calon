"""Smoke test of candidate non-Qwen annotators under vLLM: load, run the A2
annotation prompt on N synth_g train rows, report speed, JSON parse rate and span
agreement with the existing (Qwen) annotations. One model per call."""
import argparse
import json
import time
from pathlib import Path

from ierlcd.synth.llm_annotate import A2_PROMPT, clean_type

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    with open(ROOT / "data/mix3/synth_g/train.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break
    prompts = []
    for r in rows:
        qs = [q for q in r["queries"] if q["status"] in ("positive", "explicit_absent")]
        r["_types"] = [q["label"] for q in qs]
        types = "\n".join(f'- "{q["label"]}": {q["description"]}' if q["description"] else f'- "{q["label"]}"'
                          for q in qs)
        prompts.append([{"role": "user", "content": A2_PROMPT.format(text=r["text"], types=types)}])
    from ierlcd.eval.newtest import chat, last_json, llm
    t0 = time.time()
    engine = llm(args.model)
    load = time.time() - t0
    t0 = time.time()
    outs = chat(engine, [p[0]["content"] for p in prompts], 1024)
    gen = time.time() - t0
    parsed = agree = qwen_n = new_n = 0
    samples = []
    for r, text in zip(rows, outs):
        js = last_json(text)
        if isinstance(js, dict):
            parsed += 1
            ans = {clean_type(k): {str(x).strip() for x in v} for k, v in js.items() if isinstance(v, list)}
            for t in r["_types"]:
                q = {r["text"][e["start"]:e["end"]] for e in r["entities"] if e["label"] == t}
                g = ans.get(clean_type(t), set())
                agree += len(q & g)
                qwen_n += len(q)
                new_n += len(g)
        if len(samples) < 3:
            samples.append(text[:400])
    res = {"model": Path(args.model).name, "load_s": round(load), "gen_s": round(gen),
           "items_per_s": round(len(rows) / gen, 2), "parse_rate": parsed / len(rows),
           "span_precision_vs_qwen": agree / max(new_n, 1), "span_recall_vs_qwen": agree / max(qwen_n, 1),
           "samples": samples}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}))


if __name__ == "__main__":
    main()
