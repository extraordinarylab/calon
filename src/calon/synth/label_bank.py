"""Authored label descriptions for sources that have none.

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

PROMPT_VERSION = "v2"   # v2: coarse labels see their subcategory names (ontology only)
SOURCE_CONTEXT = {
    "fewnerd": ("a fine-grained named entity recognition dataset over English Wikipedia text, "
                "with 8 coarse entity categories and 66 fine-grained subcategories"),
    "massive": ("slot annotations for short spoken commands to a voice assistant (alarms, "
                "music, weather, calendar, shopping, transport, smart home and similar)"),
    "pii": ("personally identifiable information in messages, forms and documents (names, contact "
            "details, identifiers, addresses, dates)"),
    "spy": ("personal identifiers in English legal advice requests written by members of the public "
            "(names, contact details, addresses, web addresses, usernames, identification numbers)"),
    "dynamicner": ("a fine-grained named entity recognition dataset over English sentences from news and "
                   "encyclopedic text, with a three-level category hierarchy"),
    "buster": ("entities in US SEC filings about mergers and acquisitions (the companies involved and "
               "their advisors, and company revenues)"),
    "scier": "scientific entities in sentences from machine learning and AI research papers",
    "scier_rel": ("relations between methods, tasks and datasets in sentences from machine learning and "
                  "AI research papers"),
    "casie": ("cybersecurity events in English news articles (data breaches, phishing, ransom attacks, "
              "discovery and patching of vulnerabilities)"),
    "casie_role": "argument roles of cybersecurity events in English news articles",
    "wikievents": ("events in English news articles linked from Wikipedia (conflicts, attacks, arrests, "
                   "deaths, disasters, movements and similar), with a three-level event type hierarchy"),
    "wikievents_role": "argument roles of events in English news articles linked from Wikipedia",
    "mlee": ("biological events at the molecular, cellular, tissue and organ levels in English biomedical "
             "research abstracts"),
    "mlee_role": "argument roles of biological events in English biomedical research abstracts",
    "m2e2": ("events in English news articles (conflict, movement, contact, justice, life and transaction "
             "events such as attacks, arrests, meetings and deaths)"),
    "m2e2_role": "argument roles of events in English news articles",
    "gentle": ("all entity mentions, named, nominal and pronominal, in English texts of unusual genres "
               "(dictionary entries, legal texts, poetry, medical notes, syllabuses, threats, esports and "
               "proof texts)"),
    "biored": "biomedical entities in English PubMed titles and abstracts",
    "biored_rel": ("undirected relations between two biomedical concepts (genes, diseases, chemicals, "
                   "variants) stated in English PubMed titles and abstracts"),
    "measeval": "measurements in English scientific articles: quantities and what they measure",
    "measeval_rel": ("links inside a measurement in English scientific articles: a measured entity or "
                     "property to its quantity, an entity to its measured property, a qualifier to what it "
                     "qualifies"),
    "dwie": "entities in English news articles",
    "dwie_rel": "relations between two entities in English news articles",
    "retacred": "the subject and object entities of a relation in English news and web sentences",
    "retacred_rel": ("relations from a subject (a person or an organization) to an object in English news "
                     "and web sentences, as in TAC knowledge base population slot filling"),
    "aste": "aspect terms and opinion terms in English laptop and restaurant reviews",
    "aste_rel": ("the sentiment that an opinion term expresses towards an aspect term in English laptop "
                 "and restaurant reviews (head: aspect term, tail: opinion term)"),
    "gap": "pronouns and person names in English Wikipedia text",
    "gap_rel": "a pronoun (head) referring to the person a name (tail) denotes, in English Wikipedia text",
    "litbank": "mentions of people, places, organizations and other entities in English fiction",
    "litbank_rel": ("coreference in English fiction: a mention (head) and the nearest earlier mention "
                    "(tail) of the same entity"),
    "maven": "event mentions and time expressions in English Wikipedia articles",
    "maven_rel": ("relations between events and times in English Wikipedia articles: temporal order, "
                  "causation, subevents, and coreference of a later mention (head) with the first mention "
                  "(tail) of the same event"),
    "speed": ("epidemic events (spread, infection, prevention, control, symptoms, cures, deaths) in English "
              "social media posts about mpox and zika"),
    "speed_role": "argument roles of epidemic events in English social media posts",
    "sentivent": "company-specific economic events in English business news",
    "sentivent_role": "argument roles of company-specific economic events in English business news",
    "mailex": "requests and deliveries of actions, data and meetings in English business emails",
    "mailex_role": "argument roles of request and delivery events in English business emails",
    "maven_arg": "general-domain events in English Wikipedia articles about historical and recent events",
    "maven_arg_role": "argument roles of general-domain events in English Wikipedia articles",
    "geneva": "events of many kinds in English sentences from news and fiction, with FrameNet-style event types",
    "geneva_role": "argument roles of FrameNet-style events in English sentences",
    "rams": "events in English news articles (conflict, life, movement, transaction, justice and others)",
    "rams_role": "argument roles of events in English news articles",
    "phee": "pharmacovigilance events (adverse drug effects, therapeutic effects) in English medical case reports",
    "phee_role": "argument roles of pharmacovigilance events in English medical case reports",
}
EVENT_TEMPLATE = """Dataset context: {context}.
Event type name: "{name}"{parent}

Write one sentence, at most 30 words, defining which words or phrases in a text (event triggers) express an event of this type.
Do not name specific real-world entities and do not describe any particular sentence.
Return {{"description": "..."}}"""
ROLE_TEMPLATE = """Dataset context: {context}.
Event argument role name: "{name}" (the part an entity, value or phrase plays in an event)

Write one sentence, at most 30 words, defining which text spans fill this role in an event.
Do not name specific real-world entities and do not describe any particular sentence.
Return {{"description": "..."}}"""
REL_TEMPLATE = """Dataset context: {context}.
Relation type name: "{name}" (a directed relation from a head span to a tail span)

Write one sentence, at most 30 words, defining when a head span and a tail span stand in this relation.
Do not name specific real-world entities and do not describe any particular sentence.
Return {{"description": "..."}}"""
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
    children = {}
    for src, lab in items:
        if lab.get("parent"):
            children.setdefault((src, lab["parent"]), []).append(lab["name"])
    prompts = []
    for src, lab in items:
        parent = f'\nParent category: "{lab["parent"]}"' if lab.get("parent") else ""
        subs = children.get((src, lab["name"]))
        if subs:
            parent += "\nSubcategories: " + ", ".join(f'"{s}"' for s in subs)
        tpl = {"relation": REL_TEMPLATE, "event": EVENT_TEMPLATE, "role": ROLE_TEMPLATE}.get(lab.get("kind"), TEMPLATE)
        prompts.append([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": tpl.format(
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
