"""Round-4 targeted synthetic data, from the error analysis of the joint model (no target
examples are shown to the generator; zero-shot).

  gfine  fine-grained NER (DynamicNER gap): each text with its own two-level type hierarchy,
         3-6 fine types per coarse category plus a residual "other <category>" type; every
         mention labelled with its most specific fine type. Every fine type of the hierarchy
         is asked (siblings that do not occur are explicit negatives).
         -> data/r4/synth_fine (refined with refine.py, NER)
  gsci   scientific relations (SciER gap): paragraphs of research papers on many ML and
         AI4Science topics, entity types method / task / dataset and nine relation types by
         name (used-for, part-of, subclass-of, synonym-of, evaluated-with, compare-with,
         benchmark-for, subtask-of, trained-with); the generator writes the definitions.
         Same output format as re_gen gre. -> data/re/synth_sci (refined with re_gen.py)
  gnest  nested events (MLEE gap): like ee_gen gee, but at least half of the events take
         another event of the text as an argument (cause, inhibit, increase, regulate, lead to,
         prevent, ...); 40% of the requests in biomedical domains.
         -> data/ee/synth_nest (refined with ee_gen.py)
Requests in synth2/<task>_requests.jsonl, run by llm_gen2 run (gen_node.sh).
Subcommands: prepare, assemble-fine, assemble-sci, assemble-nest.
"""
import argparse
import collections
import json
import random

from calon.data.roots import ROOT

from .llm_annotate import align, clean_type, write_jsonl
from .paths import DATA_OUT, S2  # noqa: E402

LENGTHS = [60, 120, 200, 300]
SCI_RELATIONS = ["used-for", "part-of", "subclass-of", "synonym-of", "evaluated-with", "compare-with",
                 "benchmark-for", "subtask-of", "trained-with"]
SCI_ENTITIES = ["method", "task", "dataset"]
SCI_GENRES = ["abstract", "introduction paragraph", "related-work paragraph", "method-section paragraph",
              "experimental-setup paragraph", "results paragraph", "ablation-study paragraph",
              "conclusion paragraph", "survey paragraph", "dataset-description paragraph"]
SCI_TOPICS = ["image classification", "object detection", "semantic segmentation", "video understanding",
              "3D vision", "image generation with diffusion models", "machine translation", "question answering",
              "summarisation", "information extraction", "large language model alignment", "code generation",
              "speech recognition", "text-to-speech", "reinforcement learning for games", "robot learning",
              "graph neural networks", "recommender systems", "time-series forecasting", "anomaly detection",
              "federated learning", "differential privacy", "adversarial robustness", "interpretability",
              "fairness in machine learning", "AutoML and neural architecture search", "optimisation for deep learning",
              "self-supervised representation learning", "multimodal learning", "retrieval-augmented generation",
              "knowledge graphs", "causal inference", "tabular data learning", "continual learning",
              "few-shot learning", "medical image analysis", "clinical text mining", "protein structure prediction",
              "drug discovery", "molecular property prediction", "materials discovery", "weather forecasting",
              "climate modelling", "genomics", "single-cell analysis", "neuroscience data analysis", "astronomy",
              "particle physics", "chemical reaction prediction", "molecular dynamics", "epidemiological modelling",
              "remote sensing", "autonomous driving", "quantum machine learning", "theorem proving",
              "scientific document understanding", "education technology", "legal text analysis",
              "financial forecasting", "agriculture monitoring"]
BIO_DOMAINS = ["molecular biology", "cell biology", "cancer biology", "immunology", "neurobiology", "pharmacology",
               "genetics and gene regulation", "vascular biology and angiogenesis", "developmental biology",
               "biochemistry and signalling pathways", "microbiology", "endocrinology"]

GFINE_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English, that mentions many specific named things (people, organisations, places, creative works, products, events, technologies, organisms and the like).

Then design a fine-grained entity type hierarchy for this text: 3 to 5 coarse categories, each split into 3 to 6 fine types that must be told apart (for example "politician", "athlete" and "musician" under person), plus, for every coarse category, one residual fine type named "other <coarse category>" for members that fit none of its other fine types. Give every fine type a one-sentence definition, at most 30 words, of which text spans it covers, general enough to apply to other texts.
Annotate every mention in the text that belongs to one of the coarse categories with its most specific fine type (the residual type when no other fine type fits), copied exactly as it appears in the text.

Return JSON only: {{"text": "...", "hierarchy": {{"coarse category": {{"fine type": "definition"}}}}, "entities": [{{"text": "...", "type": "fine type"}}]}}"""

GSCI_PROMPT = """Write one realistic {genre} of a research paper on {topic}, about {length} words long, in English, with specific method, model, task and dataset names (real or plausible).

Use exactly these entity types: "method" (methods, models, algorithms, architectures, techniques), "task" (problems or applications addressed), "dataset" (named collections of data or benchmarks). Use exactly these directed relation types (head -> tail): {relations}. Give a one-sentence definition for every entity type and every relation type, general enough to apply to other papers.
Annotate every mention of the three entity types exactly as it appears in the text, and every relation of the listed types that the text states between two of those mentions (head and tail copied exactly as they appear). Relation types that do not occur get no entries.

Return JSON only: {{"text": "...", "entity_types": {{"type": "definition"}}, "entities": [{{"text": "...", "type": "..."}}], "relation_types": {{"relation": "definition"}}, "relations": [{{"head": "...", "relation": "...", "tail": "..."}}], "absent_relations": {{}}}}"""

GNEST_EXTRA = """
Important: at least half of the events must be about another event of the same text: events of causing, triggering, enabling, preventing, inhibiting, increasing, decreasing, regulating, following or responding to something, whose argument (for example a "theme" or "cause" role) is another event. Annotate such an argument with the trigger of that other event as its text (the other event is annotated as well)."""


def prompt_for(task, r):
    if task == "gfine":
        return GFINE_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"])
    if task == "gsci":
        from .diversity import hint
        return GSCI_PROMPT.format(genre=r["genre"], topic=r["domain"], length=r["length"],
                                  relations=", ".join(f'"{x}"' for x in SCI_RELATIONS)) \
            + hint(r["id"], names_only=True)
    from .ee_gen import GEE_PROMPT
    return GEE_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"]) + GNEST_EXTRA


def prepare(args):
    from .llm_gen2 import DOMAINS3, GENRES3
    rng = random.Random(args.seed + 13)
    cells = [(g, d) for g in GENRES3 for d in DOMAINS3]
    reqs = {"gfine": [], "gsci": [], "gnest": []}
    for k in range(args.n_fine):
        g, d = cells[k % len(cells)]
        reqs["gfine"].append({"id": f"gfine:{k}", "genre": g, "domain": d, "length": rng.choice(LENGTHS)})
    sci = [(g, t) for g in SCI_GENRES for t in SCI_TOPICS]
    for k in range(args.n_sci):
        g, t = sci[k % len(sci)]
        reqs["gsci"].append({"id": f"gsci:{k}", "genre": g, "domain": t, "length": rng.choice(LENGTHS)})
    for k in range(args.n_nest):
        g, d = cells[k % len(cells)]
        if rng.random() < 0.4:
            d = rng.choice(BIO_DOMAINS)
            g = rng.choice(["research paper abstract", "results paragraph of a research paper",
                            "review article paragraph", "clinical case report", "science news article"])
        reqs["gnest"].append({"id": f"gnest:{k}", "genre": g, "domain": d, "length": rng.choice(LENGTHS)})
    for task, rs in reqs.items():
        rng.shuffle(rs)
        write_jsonl(S2 / f"{task}_requests.jsonl", rs)
    print({t: len(r) for t, r in reqs.items()})


def split_write(rows, out, n_dev, seed, stats, key):
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    random.Random(seed).shuffle(rows)
    rows = contamination_filter(rows, eval_texts(), stats, key)
    parts = {"dev_select": rows[:n_dev], "dev_calib": rows[n_dev:2 * n_dev], "test": rows[2 * n_dev:3 * n_dev],
             "train": rows[3 * n_dev:]}
    for sp, rs in parts.items():
        write_jsonl(out / f"{sp}.jsonl", rs)
        stats[f"{key}_{sp}"] = len(rs)
    return parts


def assemble_fine(args):
    from ..data.prepare_mix import find_all
    from .ee_gen import load_outputs, parse
    stats, rows = collections.Counter(), []
    for rid, o in load_outputs("gfine").items():
        js = parse(o["output"])
        if not isinstance(js, dict) or not isinstance(js.get("text"), str) or not isinstance(js.get("hierarchy"), dict):
            stats["parse_fail"] += 1
            continue
        text, fine = js["text"].strip(), {}
        for coarse, subs in js["hierarchy"].items():
            if isinstance(subs, dict):
                for t, d in subs.items():
                    if clean_type(t):
                        fine[clean_type(t)] = str(d)
        by = collections.defaultdict(list)
        for e in js.get("entities") or []:
            t = clean_type(e.get("type", "")) if isinstance(e, dict) else ""
            if t in fine and e.get("text"):
                by[t].append(str(e["text"]).strip())
            else:
                stats["entity_type_not_in_hierarchy"] += 1
        queries, ents = [], []
        for t, d in fine.items():
            q = {"label": t, "description": d, "description_origin": "llm_authored",
                 "supervision_origin": "llm_generated"}
            if t in by:
                occ = align(text, by[t], find_all)
                if occ is None:
                    stats["unaligned_types"] += 1
                    queries.append(dict(q, status="unknown"))
                else:
                    queries.append(dict(q, status="positive"))
                    ents += [{"start": s, "end": e, "label": t} for s, e in occ]
            else:
                queries.append(dict(q, status="explicit_absent"))
        if ents:
            ents = sorted({(e["start"], e["end"], e["label"]) for e in ents})
            rows.append({"id": f"synth_fine:{rid}", "source": "synth_fine", "text": text, "queries": queries,
                         "entities": [{"start": s, "end": e, "label": t} for s, e, t in ents]})
            stats["fine_types"] += len(fine)
            stats["entities"] += len(ents)
    split_write(rows, ROOT / "data/r4/synth_fine", args.n_dev, args.seed, stats, "synth_fine")
    print(json.dumps(dict(stats)))


def assemble_sci(args):
    from .ee_gen import load_outputs, parse
    from .re_gen import gre_row
    stats, rows = collections.Counter(), []
    for rid, o in load_outputs("gsci").items():
        js = parse(o["output"])
        if not isinstance(js, dict):
            stats["parse_fail"] += 1
            continue
        # the schema is fixed: keep the generator's definitions of the fixed names only
        et = {clean_type(k): str(v) for k, v in (js.get("entity_types") or {}).items()}
        rt = {clean_type(k): str(v) for k, v in (js.get("relation_types") or {}).items()}
        js["entity_types"] = {t: et.get(t, "") for t in SCI_ENTITIES}
        js["relation_types"] = {t: rt.get(t, "") for t in SCI_RELATIONS}
        js["absent_relations"] = {}
        row = gre_row(rid, js, stats)
        if row is not None:
            rows.append(dict(row, id=f"synth_sci:{rid}", source="synth_sci"))
    split_write(rows, DATA_OUT / "re/synth_sci", args.n_dev, args.seed, stats, "synth_sci")
    print(json.dumps(dict(stats)))


def assemble_nest(args):
    from .ee_gen import EE_DIR, doc_of, load_outputs, parse
    stats, docs = collections.Counter(), []
    for rid, o in load_outputs("gnest").items():
        js = parse(o["output"])
        if not isinstance(js, dict) or not isinstance(js.get("text"), str):
            stats["parse_fail"] += 1
            continue
        schema = {}
        for t, v in (js.get("event_types") or {}).items():
            if clean_type(t) and isinstance(v, dict):
                schema[clean_type(t)] = [str(v.get("definition", "")),
                                         {clean_type(r): str(d) for r, d in (v.get("roles") or {}).items() if clean_type(r)}]
        absent = {clean_type(k): str(v) for k, v in (js.get("absent_event_types") or {}).items()
                  if clean_type(k) and clean_type(k) not in schema}
        d = doc_of(js["text"].strip(), schema, js.get("events"), absent, stats, "gnest", False)
        if d["triggers"]:
            trig = {(s, e) for s, e, _ in d["triggers"]}
            stats["event_args"] += sum((a[3], a[4]) in trig for a in d["args"])
            stats["args"] += len(d["args"])
            docs.append(dict(d, id=f"synth_nest:{rid}"))
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    random.Random(args.seed).shuffle(docs)
    docs = contamination_filter(docs, eval_texts(), stats, "synth_nest")
    n = args.n_dev
    for sp, part in (("dev_select", docs[:n]), ("dev_calib", docs[n:2 * n]), ("test", docs[2 * n:3 * n]),
                     ("train", docs[3 * n:])):
        write_jsonl(EE_DIR / "synth_nest" / f"docs_{sp}.jsonl", part)
        stats[f"synth_nest_{sp}_docs"] = len(part)
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--n-fine", type=int, default=60000)
    p.add_argument("--n-sci", type=int, default=50000)
    p.add_argument("--n-nest", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    for c in ("assemble-fine", "assemble-sci", "assemble-nest"):
        p = sub.add_parser(c)
        p.add_argument("--n-dev", type=int, default=1500)
        p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"prepare": prepare, "assemble-fine": assemble_fine, "assemble-sci": assemble_sci,
     "assemble-nest": assemble_nest}[args.cmd](args)


if __name__ == "__main__":
    main()
