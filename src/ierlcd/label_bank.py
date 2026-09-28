"""Authored label descriptions for sources that have none (SPEC 2.3, 2.4).

Reads the label inventory written by `prepare_mix --stage inventory` and asks
Qwen3.8-27B (vLLM, greedy, thinking off) for a one-sentence definition of each
label. The prompt sees only the label name, its parent category and a fixed
description of the source dataset: never rows, answers or any test data.
Output entries are marked description_origin=authored with the model and
prompt version. Runs in a vLLM env (e.g. lco-embed), not ie-rlcd.
"""
import argparse
import json
import re
from pathlib import Path

PROMPT_VERSION = "v1"
SOURCE_CONTEXT = {
    "fewnerd": ("a fine-grained named entity recognition dataset over English Wikipedia text, "
                "with 8 coarse entity categories and 66 fine-grained subcategories"),
    "massive": ("slot annotations for short spoken commands to a voice assistant (alarms, "
                "music, weather, calendar, shopping, transport, smart home and similar)"),
}
SYSTEM = ("You write concise label definitions for an entity extraction schema. "
          "Answer with JSON only.")
TEMPLATE = """Dataset context: {context}.
Label name: "{name}"{parent}

Write one sentence, at most 30 words, defining which text spans this label covers.
You may mention generic kinds of things (for example "airports, train stations"), but do not
name specific real-world entities and do not describe any particular sentence.
Return {{"description": "..."}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    inventory = json.loads(Path(args.inventory).read_text())
    items = [(src, lab) for src, labs in inventory.items() for lab in labs]
    prompts = []
    for src, lab in items:
        parent = f'\nParent category: "{lab["parent"]}"' if lab.get("parent") else ""
        prompts.append([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": TEMPLATE.format(
                            context=SOURCE_CONTEXT[src], name=lab["name"], parent=parent)}])
    # enforce_eager: CUDA-graph capture fails on this node's driver ("driver version is
    # insufficient"), while eager forward passes run; ~130 prompts do not need graphs.
    kw = dict(model=args.model, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True)
    try:                                   # Qwen3.5-family checkpoints carry a vision tower
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as exc:
        print("retrying without limit_mm_per_prompt:", repr(exc)[:200])
        llm = LLM(**kw)
    params = SamplingParams(temperature=0.0, max_tokens=200)
    outs = llm.chat(prompts, params, chat_template_kwargs={"enable_thinking": False})
    bank, failed = {}, []
    for (src, lab), out in zip(items, outs):
        text = out.outputs[0].text
        m = re.search(r"\{.*\}", text, re.S)
        try:
            desc = json.loads(m.group(0))["description"].strip()
        except Exception:
            failed.append({"source": src, "raw": lab["raw"], "output": text[:300]})
            desc = ""
        bank.setdefault(src, {})[lab["raw"]] = {
            "name": lab["name"], "description": desc,
            "description_origin": "authored" if desc else "name_only",
            "model": Path(args.model).name, "prompt_version": PROMPT_VERSION}
    Path(args.out).write_text(json.dumps({"bank": bank, "failed": failed}, indent=2))
    print(f"labels={len(items)} failed={len(failed)}")
    for src in bank:
        for raw, v in list(bank[src].items())[:3]:
            print(f"{src} | {v['name']} | {v['description']}")


if __name__ == "__main__":
    main()
