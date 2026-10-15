"""Relation data over many domains (synth_gre), and refinement of the relation sources.

gre     Like the NER genre x domain data (g3): 40 genres x 120 domains; Qwen3.8 writes
        one text, its own entity and relation schema (definitions, confusable pairs),
        the mentions and the relations, plus relation types that are not stated.
        No target-dataset examples or labels (zero-shot).
Refinement of synth_gre and synth_cre train rows, as for the NER sources:
  rra   a second, non-Qwen model re-annotates entities and relations with the same
        entity and relation types;
  rjd   every disputed entity span or relation (found by only one annotator) goes to
        a third model, which answers yes / no;
  apply keep what both found plus what the judge accepts; a relation needs both of its
        spans kept. Types left without items become explicit_absent, types that gain
        items become positive. Backups: <source>/train.jsonl.pre_refine.
Requests live in synth2/<task>_requests.jsonl and run through llm_gen2 run (gen_node.sh).
Subcommands: prepare-gre, assemble-gre, prepare-rra, prepare-rjd, apply.
"""
import argparse
import collections
import json
import os
import random

from .llm_annotate import align, clean_type, read_jsonl, write_jsonl
from .paths import DATA_OUT, S2  # noqa: E402

RE_DIR = DATA_OUT / "re"
REFINE_SOURCES = os.environ.get("RE_REFINE", "synth_cre,synth_gre").split(",")
ASKED = ("positive", "explicit_absent", "scope_assumed_absent")
LENGTHS = [60, 120, 200, 300]

GRE_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English. Make it specific and natural, with concrete names, places, organisations, products, numbers and dates where the genre calls for them, and with several entities that are related to each other.

Then design an extraction schema for this text:
- 6 to 12 entity types a user working in {domain} would want to extract, each with a one-sentence definition that would apply to other texts too;
- 5 to 10 directed relation types between those entities (from a head entity to a tail entity), each with a one-sentence definition. Include at least two pairs of closely related relation types that must be told apart (for example "founded" and "employed by", or a relation and its reverse direction).
Annotate every mention of every entity type exactly as it appears in the text, and every relation of a schema type that the text states between two of those mentions (head and tail copied exactly as they appear). Schema types that do not occur get no entries.
Finally list 3 to 5 further relation types that would be plausible between these entities but are not stated in the text, each with a one-sentence definition.

Return JSON only: {{"text": "...", "entity_types": {{"type": "definition"}}, "entities": [{{"text": "...", "type": "..."}}], "relation_types": {{"relation": "definition"}}, "relations": [{{"head": "...", "relation": "...", "tail": "..."}}], "absent_relations": {{"relation": "definition"}}}}"""

RRA_PROMPT = """Text:
{text}

Entity types:
{etypes}

Relation types (directed, from head to tail):
{rtypes}

List every mention in the text of the entity types above (exact text as it appears), and every relation of the relation types above that the text states between two of those mentions (head and tail copied exactly as they appear). Use only the listed types.
Return JSON only: {{"entities": [{{"text": "...", "type": "..."}}], "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}"""

RJD_PROMPT = """Text:
{text}

Entity types:
{etypes}

Relation types (directed, from head to tail):
{rtypes}

Candidate annotations:
{cands}

For each candidate answer "yes" if it is correct for this text, otherwise "no". An entity candidate is correct if the span is a complete mention of that type (right boundaries, right type). A relation candidate is correct if the text states that relation from the head to the tail.
Return JSON only: {{"E1": "yes" or "no", ..., "R1": "yes" or "no", ...}}"""


def type_lines(types):
    return "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in types)


def asked(qs):
    return [(q["label"], q.get("description") or "") for q in qs if q["status"] in ASKED]


def prompt_for(task, r):
    if task == "gre":
        return GRE_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"])
    if task == "rra":
        return RRA_PROMPT.format(text=r["text"], etypes=type_lines(r["etypes"]), rtypes=type_lines(r["rtypes"]))
    lines = [f'E{k + 1}. "{c[3]}" as {c[2]}' for k, c in enumerate(r["ecands"])]
    lines += [f'R{k + 1}. "{c[5]}" --{c[4]}--> "{c[6]}"' for k, c in enumerate(r["rcands"])]
    return RJD_PROMPT.format(text=r["text"], etypes=type_lines(r["etypes"]), rtypes=type_lines(r["rtypes"]),
                             cands="\n".join(lines))


def parse(text):
    from ..eval.newtest import final_text, last_json
    js = last_json(final_text(text))
    return js if isinstance(js, dict) else None


def load_outputs(task):
    out = {}
    for p in sorted((S2 / task).glob("shard-*.jsonl")):
        for line in open(p):
            o = json.loads(line)
            out[o["id"]] = o
    return out


def annotate(text, ents_in, rels_in, etypes, rtypes, stats, key):
    """LLM entity / relation strings -> ({(s, e, type)}, {(hs, he, ts, te, rel)}, bad entity types, bad rels)."""
    from ..data.prepare_mix import find_all
    from ..data.re_prep import nearest
    by, bad_e = collections.defaultdict(list), set()
    for e in ents_in or []:
        if isinstance(e, dict) and e.get("text") and clean_type(e.get("type", "")) in etypes:
            by[clean_type(e["type"])].append(str(e["text"]).strip())
    ents, mention = set(), {}
    for t, strs in by.items():
        occ = align(text, strs, find_all)
        if occ is None:
            stats[f"{key}_entity_type_unaligned"] += 1
            bad_e.add(t)
            continue
        ents |= {(a, b, t) for a, b in occ}
        for s in strs:
            mention.setdefault(s, t)
    spans = {(a, b) for a, b, _ in ents}
    rels, bad_r = set(), set()
    for r in rels_in or []:
        if not isinstance(r, dict):
            continue
        h, t, lab = str(r.get("head", "")).strip(), str(r.get("tail", "")).strip(), clean_type(r.get("relation", ""))
        if lab not in rtypes:
            stats[f"{key}_relation_type_unknown"] += 1
            continue
        pair = nearest(text, h, t) if h in mention and t in mention else None
        if pair is None or pair[0] not in spans or pair[1] not in spans:
            stats[f"{key}_relation_unaligned"] += 1
            bad_r.add(lab)
            continue
        rels.add((pair[0][0], pair[0][1], pair[1][0], pair[1][1], lab))
    return ents, rels, bad_e, bad_r


# ---------------------------------------------------------------- gre
def prepare_gre(args):
    from .llm_gen2 import DOMAINS3, GENRES3
    rng = random.Random(args.seed + 7)
    cells = [(g, d) for g in GENRES3 for d in DOMAINS3]
    reqs = [{"id": f"gre:{k}", "genre": cells[k % len(cells)][0], "domain": cells[k % len(cells)][1],
             "length": rng.choice(LENGTHS)} for k in range(args.n)]
    rng.shuffle(reqs)
    write_jsonl(S2 / "gre_requests.jsonl", reqs)
    print(f"gre requests {len(reqs)} over {len(cells)} genre x domain cells")


def gre_row(rid, js, stats):
    if not isinstance(js.get("text"), str) or not js["text"].strip():
        stats["gre_no_text"] += 1
        return None
    text = js["text"].strip()
    etypes = {clean_type(k): str(v) for k, v in (js.get("entity_types") or {}).items() if clean_type(k)}
    rtypes = {clean_type(k): str(v) for k, v in (js.get("relation_types") or {}).items() if clean_type(k)}
    absent = {clean_type(k): str(v) for k, v in (js.get("absent_relations") or {}).items() if clean_type(k)}
    ents, rels, bad_e, bad_r = annotate(text, js.get("entities"), js.get("relations"), etypes, rtypes, stats, "gre")
    if not ents:
        stats["gre_no_entities"] += 1
        return None
    has_e, has_r = {t for _, _, t in ents}, {x[4] for x in rels}
    queries = [{"label": t, "description": d,
                "status": "positive" if t in has_e else ("unknown" if t in bad_e else "explicit_absent"),
                "description_origin": "llm_authored", "supervision_origin": "llm_generated"}
               for t, d in etypes.items()]
    rel_queries = [{"label": t, "description": d,
                    "status": "positive" if t in has_r else ("unknown" if t in bad_r else "explicit_absent")}
                   for t, d in rtypes.items()]
    rel_queries += [{"label": t, "description": d, "status": "explicit_absent"}
                    for t, d in absent.items() if t not in rtypes]
    stats["gre_relations"] += len(rels)
    stats["gre_rows_with_relations"] += bool(rels)
    return {"id": f"synth_gre:{rid}", "source": "synth_gre", "text": text, "queries": queries,
            "entities": [{"start": a, "end": b, "label": t} for a, b, t in sorted(ents)],
            "rel_queries": rel_queries,
            "relations": [{"head": [a, b], "tail": [c, d], "label": l} for a, b, c, d, l in sorted(rels)]}


def assemble_gre(args):
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    stats = collections.Counter()
    rows = []
    for rid, o in load_outputs("gre").items():
        js = parse(o["output"])
        if js is None:
            stats["gre_parse_fail"] += 1
            continue
        row = gre_row(rid, js, stats)
        if row is not None:
            rows.append(row)
    random.Random(args.seed).shuffle(rows)
    rows = contamination_filter(rows, eval_texts(), stats, "gre")
    n = args.n_dev
    out = RE_DIR / "synth_gre"
    splits = {"dev_select": rows[:n], "dev_calib": rows[n:2 * n], "test": rows[2 * n:3 * n], "train": rows[3 * n:]}
    for sp, rs in splits.items():
        write_jsonl(out / f"{sp}.jsonl", rs)
        stats[f"{sp}_rows"] = len(rs)
    (RE_DIR / "synth_gre_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))
    for x in splits["train"][:args.show]:
        t = x["text"]
        print(t[:300])
        print("  E:", [(t[e["start"]:e["end"]], e["label"]) for e in x["entities"]][:12])
        print("  R:", [(t[y["head"][0]:y["head"][1]], y["label"], t[y["tail"][0]:y["tail"][1]])
                       for y in x["relations"]][:10])


# ---------------------------------------------------------------- refinement
def prepare_rra(args):
    reqs = []
    for src in REFINE_SOURCES:
        for r in read_jsonl(RE_DIR / src / "train.jsonl"):
            et, rt = asked(r["queries"]), asked(r["rel_queries"])
            if et:
                reqs.append({"id": r["id"], "source": src, "text": r["text"], "etypes": et, "rtypes": rt})
    write_jsonl(S2 / "rra_requests.jsonl", reqs)
    print("rra requests", len(reqs), dict(collections.Counter(r["source"] for r in reqs)))


def disputes(stats):
    ra = load_outputs("rra")
    for src in REFINE_SOURCES:
        for r in read_jsonl(RE_DIR / src / "train.jsonl"):
            o = ra.get(r["id"])
            js = parse(o["output"]) if o else None
            if js is None:
                stats[f"{src}_no_second_opinion"] += 1
                yield src, r, None
                continue
            et, rt = dict(asked(r["queries"])), dict(asked(r["rel_queries"]))
            ge, gr, _, _ = annotate(r["text"], js.get("entities"), js.get("relations"), et, rt, stats, src)
            qe = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
            qr = {(x["head"][0], x["head"][1], x["tail"][0], x["tail"][1], x["label"]) for x in r["relations"]}
            d = {"agreed_e": qe & ge, "agreed_r": qr & gr, "disp_e": sorted(qe ^ ge), "disp_r": sorted(qr ^ gr)}
            stats[f"{src}_e_agreed"] += len(d["agreed_e"])
            stats[f"{src}_e_qwen_only"] += len(qe - ge)
            stats[f"{src}_e_second_only"] += len(ge - qe)
            stats[f"{src}_r_agreed"] += len(d["agreed_r"])
            stats[f"{src}_r_qwen_only"] += len(qr - gr)
            stats[f"{src}_r_second_only"] += len(gr - qr)
            yield src, r, d


def prepare_rjd(args):
    stats, reqs = collections.Counter(), []
    for src, r, d in disputes(stats):
        if d is None or not (d["disp_e"] or d["disp_r"]):
            continue
        t = r["text"]
        reqs.append({"id": r["id"], "source": src, "text": t,
                     "etypes": asked(r["queries"]), "rtypes": asked(r["rel_queries"]),
                     "ecands": [[s, e, lab, t[s:e]] for s, e, lab in d["disp_e"]],
                     "rcands": [[a, b, c, e, lab, t[a:b], t[c:e]] for a, b, c, e, lab in d["disp_r"]]})
    write_jsonl(S2 / "rjd_requests.jsonl", reqs)
    stats["rjd_requests"] = len(reqs)
    (S2 / "rjd_prepare_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def apply(args):
    jreq = {r["id"]: r for r in read_jsonl(S2 / "rjd_requests.jsonl")}
    jout = load_outputs("rjd")
    stats = collections.Counter()
    by_src = collections.defaultdict(list)
    for src, r, d in disputes(stats):
        if d is None:
            by_src[src].append(r)
            continue
        keep_e, keep_r = set(d["agreed_e"]), set(d["agreed_r"])
        if d["disp_e"] or d["disp_r"]:
            req, o = jreq.get(r["id"]), jout.get(r["id"])
            js = parse(o["output"]) if o else None
            if req is None or js is None:
                stats[f"{src}_judge_missing"] += 1
                keep_e |= {(e["start"], e["end"], e["label"]) for e in r["entities"]}
                keep_r |= {(x["head"][0], x["head"][1], x["tail"][0], x["tail"][1], x["label"]) for x in r["relations"]}
            else:
                for k, c in enumerate(req["ecands"]):
                    yes = str(js.get(f"E{k + 1}", "")).strip().lower().startswith("y")
                    stats[f"{src}_e_judge_{'yes' if yes else 'no'}"] += 1
                    if yes:
                        keep_e.add(tuple(c[:3]))
                for k, c in enumerate(req["rcands"]):
                    yes = str(js.get(f"R{k + 1}", "")).strip().lower().startswith("y")
                    stats[f"{src}_r_judge_{'yes' if yes else 'no'}"] += 1
                    if yes:
                        keep_r.add(tuple(c[:5]))
        spans = {(s, e) for s, e, _ in keep_e}
        n_r = len(keep_r)
        keep_r = {x for x in keep_r if (x[0], x[1]) in spans and (x[2], x[3]) in spans}
        stats[f"{src}_r_dropped_span_removed"] += n_r - len(keep_r)
        before_e = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
        before_r = {(x["head"][0], x["head"][1], x["tail"][0], x["tail"][1], x["label"]) for x in r["relations"]}
        stats[f"{src}_e_removed"] += len(before_e - keep_e)
        stats[f"{src}_e_added"] += len(keep_e - before_e)
        stats[f"{src}_r_removed"] += len(before_r - keep_r)
        stats[f"{src}_r_added"] += len(keep_r - before_r)
        for qs, has, tag in ((r["queries"], {t for _, _, t in keep_e}, "e"),
                             (r["rel_queries"], {x[4] for x in keep_r}, "r")):
            for q in qs:
                if q["status"] not in ASKED:
                    continue
                if q["label"] in has and q["status"] != "positive":
                    q["status"] = "positive"
                    stats[f"{src}_{tag}_absent_to_positive"] += 1
                elif q["label"] not in has and q["status"] == "positive":
                    q["status"] = "explicit_absent"
                    stats[f"{src}_{tag}_positive_to_absent"] += 1
        r["entities"] = [{"start": s, "end": e, "label": t} for s, e, t in sorted(keep_e)]
        r["relations"] = [{"head": [a, b], "tail": [c, e], "label": l} for a, b, c, e, l in sorted(keep_r)]
        if keep_e:
            by_src[src].append(r)
        else:
            stats[f"{src}_rows_dropped_no_entity"] += 1
    for src, rows in by_src.items():
        path = RE_DIR / src / "train.jsonl"
        path.rename(path.with_suffix(".jsonl.pre_refine"))
        write_jsonl(path, rows)
        stats[f"{src}_rows_after"] = len(rows)
        stats[f"{src}_relations_after"] = sum(len(x["relations"]) for x in rows)
    (RE_DIR / "refine_re_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-gre")
    p.add_argument("--n", type=int, default=150000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("assemble-gre")
    p.add_argument("--n-dev", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", type=int, default=0)
    sub.add_parser("prepare-rra")
    sub.add_parser("prepare-rjd")
    sub.add_parser("apply")
    args = ap.parse_args()
    {"prepare-gre": prepare_gre, "assemble-gre": assemble_gre, "prepare-rra": prepare_rra,
     "prepare-rjd": prepare_rjd, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
