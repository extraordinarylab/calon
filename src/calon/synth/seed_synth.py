"""Target-seeded synthesis: new annotated texts in the style of every evaluation set, checked by
two models (relaxed zero-shot: no evaluation labels are used; evaluation texts are shown as
unlabeled style references only).

Pools come from calon.data.seed_pools ($CALON_ROOT/seed_pools/<set>.json).

  prepare   one generation request per K examples: the set's types (with their descriptions),
            up to 3 labeled train / validation examples, unlabeled evaluation texts (2 when there are
            labeled examples, otherwise 4), K new texts asked (K by the set's text length)
  run sg    Qwen3.8-27B writes and annotates the texts (llm_gen2 run, task sg)
  verify    parse; drop texts that copy a seed, a style reference or any evaluation text (word
            5-gram containment >= 0.8); one re-annotation request per kept text, with the same
            types and labeled examples (task sv)
  run sv    gpt-oss-120b re-annotates independently (greedy)
  assemble  agreement per type and text:
              NER    a type is positive when both found the same mentions, explicit_absent when
                     both found none, unknown (not supervised, its spans removed) otherwise
              RE     the same per relation type, over (head span, tail span); a relation type is
                     unknown when one of its pairs has an endpoint without an agreed entity type
              EE     an event type is kept when both found the same triggers and the same
                     arguments of those triggers; other event types are left out of the schema
            texts without a positive type are dropped. Sources: synth_sner, synth_sre, synth_see
            (domain = the evaluation set), under $IE_DATA_OUT/seed/<source>/{train,dev_select,
            dev_calib,test}.jsonl; summary in $IE_SYNTH2/seed_summary.json.
Requests and outputs: $IE_SYNTH2 (default $CALON_ROOT/synth_seed).
"""
import argparse
import collections
import json
import math
import random

from calon.data.roots import ROOT

from .llm_annotate import align, clean_type, read_jsonl, seed_of, write_jsonl
from .paths import DATA_OUT, S2

POOLS = ROOT / "seed_pools"
OUT = DATA_OUT / "seed"
SOURCE = {"NER": "synth_sner", "RE": "synth_sre", "EE": "synth_see"}
MAX_TYPES = 24              # entity types per request (DynamicNER asks 133): sampled, seed types first
MAX_EXAMPLE_CHARS = 1500    # labeled examples longer than this are not shown
MAX_REF_CHARS = 1200        # style references are cut here
TASK_TEXT = {"NER": "named entity recognition", "RE": "entity and relation extraction",
             "EE": "event extraction"}

GEN = """You are creating training data for {task} on this kind of text: {genre}.

{schema}
{labeled}{refs}
Write {n} new {texts} of this kind. Match the style, register, length, formatting, casing and punctuation habits of the texts above, but write about different, specific content: new names, new facts, new sub-topics. Do not copy or paraphrase any text above.{anchor}
{annotate}
Return JSON only: {{"examples": [{fmt}]}}"""

# round 2 for sets the first round left far from their texts: each new text follows one unannotated
# text's kind of subject and setting (a different case of the same kind), not only its style
ANCHOR = (" Base each new text on one of the unannotated texts in turn: the same kind of subject, situation and "
          "setting (for example another person of the same kind of career, another dispute of the same kind, "
          "another message in the same kind of exchange), with different specific names, facts and wording.")

# round 3: every evaluation text in turn is the anchor (Text 1) of one request
ANCHOR_ONE = (" Base every new text on Text 1: the same kind of subject, situation and setting (a different case "
               "of the same kind), with different specific names, facts and wording; the other texts only show the style.")

VERIFY = """Text:
{text}

{schema}
{labeled}
{annotate}
Return JSON only: {fmt}"""

ANNOTATE = {
    "NER": "Annotate every mention of every listed entity type, copied exactly as it appears in the text, "
           "following the conventions of the annotated examples. A type that does not occur gets no mentions.",
    "RE": "Annotate every mention of every listed entity type, copied exactly as it appears in the text, and "
          "every relation of a listed type that the text states between two of those mentions (head and tail "
          "copied exactly), following the conventions of the annotated examples.",
    "EE": "Annotate every event of the listed types: its type, its trigger (copied exactly) and its arguments "
          "(role and span, copied exactly; only roles of that event type), following the conventions of the "
          "annotated examples.",
}
FMT = {
    "NER": '{"text": "...", "entities": [{"text": "...", "type": "..."}]}',
    "RE": '{"text": "...", "entities": [{"text": "...", "type": "..."}], '
          '"relations": [{"head": "...", "relation": "...", "tail": "..."}]}',
    "EE": '{"text": "...", "events": [{"type": "...", "trigger": "...", '
          '"arguments": [{"role": "...", "text": "..."}]}]}',
}
FMT_V = {
    "NER": '{"entities": [{"text": "...", "type": "..."}]}',
    "RE": '{"entities": [{"text": "...", "type": "..."}], "relations": [{"head": "...", "relation": "...", "tail": "..."}]}',
    "EE": '{"events": [{"type": "...", "trigger": "...", "arguments": [{"role": "...", "text": "..."}]}]}',
}


# ---------------------------------------------------------------- prompts
def type_lines(items):
    return "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in items)


def schema_block(task, r):
    if task == "EE":
        from .ee_gen import schema_lines
        return "Event types and their argument roles:\n" + schema_lines(r["schema"])
    s = "Entity types:\n" + type_lines(r["labels"])
    if task == "RE":
        s += "\n\nRelation types (directed, from head to tail):\n" + type_lines(r["rel_labels"])
    return s


def show_example(task, x):
    t = x["text"]
    if task == "EE":
        from .ee_gen import format_example
        return format_example(t, x["events"])
    ents = [{"text": t[s:e], "type": lab} for s, e, lab in x["entities"]]
    out = f"text: {t}\nentities: {json.dumps(ents, ensure_ascii=False)}"
    if task == "RE":
        rels = [{"head": t[a:b], "relation": lab, "tail": t[c:d]} for a, b, c, d, lab in x.get("relations", [])]
        out += f"\nrelations: {json.dumps(rels, ensure_ascii=False)}"
    return out


def labeled_block(task, r):
    if not r["examples"]:
        return ""
    return "\nAnnotated examples from this dataset:\n\n" + "\n\n".join(
        f"Example {i + 1}:\n{show_example(task, x)}" for i, x in enumerate(r["examples"])) + "\n"


def prompt_for(task, r):
    t = r["task"]
    if task == "sg":
        refs = ""
        if r["refs"]:
            refs = ("\nUnannotated texts from the same source (only for style, register, length and the kind of "
                    "content; their annotations are not given):\n\n" +
                    "\n\n".join(f"Text {i + 1}:\n{x}" for i, x in enumerate(r["refs"])) + "\n")
        from .diversity import hint
        anchor = {"one": ANCHOR_ONE, True: ANCHOR}.get(r.get("anchor"), "")
        return GEN.format(task=TASK_TEXT[t], genre=r["genre"], schema=schema_block(t, r),
                          labeled=labeled_block(t, r), refs=refs, n=r["n"], anchor=anchor,
                          texts="text" if r["n"] == 1 else "texts", annotate=ANNOTATE[t], fmt=FMT[t]) \
            + hint(r["id"], names_only=True)
    return VERIFY.format(text=r["text"], schema=schema_block(t, r), labeled=labeled_block(t, r),
                         annotate=ANNOTATE[t].replace(" in the text", " in the text above"), fmt=FMT_V[t])


# ---------------------------------------------------------------- requests
def load_pool(name):
    return json.loads((POOLS / f"{name}.json").read_text())


def per_request(pool):
    texts = pool["texts"]
    med = sorted(len(x) for x in texts)[len(texts) // 2] if texts else 300
    return 4 if med < 400 else (2 if med < 1200 else 1)


def request_types(pool, examples, rng):
    """All types, or for very large type sets the examples' types plus a random sample."""
    labels = pool["labels"]
    if len(labels) <= MAX_TYPES:
        return labels
    want = {lab for x in examples for _, _, lab in x["entities"]}
    rest = [x for x in labels if x[0] not in want]
    rng.shuffle(rest)
    keep = [x for x in labels if x[0] in want][:MAX_TYPES]
    return sorted(keep + rest[:MAX_TYPES - len(keep)])


def prepare(args):
    names = sorted(p.stem for p in POOLS.glob("*.json") if p.stem != "summary")
    if args.sets:
        names = [n for n in names if n in args.sets.split(",")]
    reqs, plan = [], {}
    boost = {k: float(v) for k, v in (x.split("=") for x in args.boost.split(",") if x)}
    for name in names:
        pool = load_pool(name)
        k = per_request(pool)
        n_req = math.ceil(args.per_set * boost.get(name, 1.0) / k)
        texts = pool["texts"]
        med = sorted(len(x) for x in texts)[len(texts) // 2] if texts else 0
        cap = max(MAX_EXAMPLE_CHARS, int(1.2 * med))      # long-document sets keep whole examples
        lab = [x for x in pool["labeled"] if len(x["text"]) <= cap]
        if args.per_text:
            # every evaluation text (not only the pool's sample) anchors one request
            ev = sorted({json.loads(line)["text"] for line in open(ROOT / "data" / pool["group"] / f"{name}.jsonl")})
            for i, anchor_text in enumerate(ev):
                rng = random.Random(seed_of(f"{args.prefix}:{name}:{i}"))
                refs = [anchor_text[:max(MAX_REF_CHARS, cap)]] + \
                    [t[:max(MAX_REF_CHARS, cap)] for t in rng.sample(texts, min(2, len(texts)))]
                r = {"id": f"{args.prefix}:{name}:{i}", "set": name, "task": pool["task"], "genre": pool["genre"],
                     "n": round(args.per_text * boost.get(name, 1.0)), "examples": rng.sample(lab, min(2, len(lab))),
                     "refs": refs, "anchor": "one"}
                if pool["task"] == "EE":
                    r["schema"] = pool["schema"]
                else:
                    r["labels"] = request_types(pool, r["examples"], rng)
                    r["rel_labels"] = pool.get("rel_labels", [])
                reqs.append(r)
            plan[name] = {"task": pool["task"], "requests": len(ev), "per_request": round(args.per_text * boost.get(name, 1.0)),
                          "labeled_pool": len(lab), "texts": len(ev)}
            continue
        n_ref = 4 if args.anchor or not lab else 2
        n_lab = 2 if args.anchor else 3
        for i in range(n_req):
            rng = random.Random(seed_of(f"{args.prefix}:{name}:{i}"))
            ex = rng.sample(lab, min(n_lab, len(lab)))
            refs = [t[:max(MAX_REF_CHARS, cap)] for t in rng.sample(texts, min(n_ref, len(texts)))]
            r = {"id": f"{args.prefix}:{name}:{i}", "set": name, "task": pool["task"], "genre": pool["genre"],
                 "n": min(k, len(refs)) if args.anchor else k, "examples": ex, "refs": refs, "anchor": args.anchor}
            if pool["task"] == "EE":
                r["schema"] = pool["schema"]
            else:
                r["labels"] = request_types(pool, ex, rng)
                r["rel_labels"] = pool.get("rel_labels", [])
                if len(pool["labels"]) > MAX_TYPES:      # examples keep only the asked types
                    asked = {t for t, _ in r["labels"]}
                    r["examples"] = [dict(x, entities=[e for e in x["entities"] if e[2] in asked]) for x in ex]
            reqs.append(r)
        plan[name] = {"task": pool["task"], "requests": n_req, "per_request": k, "labeled_pool": len(lab),
                      "texts": len(pool["texts"])}
    random.Random(0).shuffle(reqs)
    write_jsonl(S2 / "sg_requests.jsonl", reqs)
    (S2 / "sg_plan.json").write_text(json.dumps(plan, indent=1))
    print("sg requests", len(reqs))
    for k, v in plan.items():
        print(f"  {k:26s} {json.dumps(v)}")


# ---------------------------------------------------------------- parsing
def parse(text):
    from ..eval.newtest import final_text, last_json
    return last_json(final_text(text))


def ner_spans(text, items, labels, stats, key):
    """[{text, type}] -> ({label: {(s, e)}}, bad labels). labels: {clean_type(name): name}; answers
    are matched loosely (case, underscores) and keyed by the set's own name. A mention that cannot
    be found marks its type bad."""
    from ..data.prepare_mix import find_all
    by, bad = collections.defaultdict(list), set()
    for e in items or []:
        if not isinstance(e, dict):
            continue
        lab, s = labels.get(clean_type(e.get("type", ""))), str(e.get("text", "")).strip()
        if lab is None or not s:
            stats[f"{key}_type_unknown"] += 1
            continue
        by[lab].append(s)
    out = {}
    for lab, ss in by.items():
        occ = align(text, ss, find_all)
        if occ is None:
            stats[f"{key}_unaligned"] += 1
            bad.add(lab)
            continue
        out[lab] = set(occ)
    return out, bad


def re_pairs(text, items, rel_labels, mentions, stats, key):
    """[{head, relation, tail}] -> ({label: {(hs, he, ts, te)}}, bad labels)."""
    from ..data.re_prep import nearest
    out, bad = collections.defaultdict(set), set()
    for r in items or []:
        if not isinstance(r, dict):
            continue
        lab = rel_labels.get(clean_type(r.get("relation", "")))
        h, t = str(r.get("head", "")).strip(), str(r.get("tail", "")).strip()
        if lab is None:
            stats[f"{key}_relation_unknown"] += 1
            continue
        p = nearest(text, h, t) if h and t else None
        if p is None or p[0] not in mentions or p[1] not in mentions:
            stats[f"{key}_relation_unaligned"] += 1
            bad.add(lab)
            continue
        out[lab].add((*p[0], *p[1]))
    return dict(out), bad


def annotation(task, text, js, r, stats, key):
    """One annotator's answer for one text -> comparable sets, or None if unusable."""
    if not isinstance(js, dict):
        return None
    if task == "EE":
        from .ee_gen import align_events
        tmap = {clean_type(t): t for t in r["schema"]}
        rmap = {clean_type(x): x for _, (_, roles) in r["schema"].items() for x in roles}
        schema = {clean_type(t): [d, {clean_type(x): y for x, y in roles.items()}] for t, (d, roles) in r["schema"].items()}
        trig, args = align_events(text, js.get("events"), schema, stats, key)
        return {"trig": {(s, e, tmap[t]) for s, e, t in trig},
                "args": {(a, b, tmap[t], c, d, rmap[role]) for a, b, t, c, d, role in args}}
    labels = {clean_type(t): t for t, _ in r["labels"]}
    ents, bad = ner_spans(text, js.get("entities"), labels, stats, key)
    out = {"ents": ents, "bad": bad}
    if task == "RE":
        mentions = {sp for v in ents.values() for sp in v}
        rels, rbad = re_pairs(text, js.get("relations"), {clean_type(t): t for t, _ in r["rel_labels"]},
                              mentions, stats, key)
        out.update(rels=rels, rbad=rbad)
    return out


# ---------------------------------------------------------------- verify
def verify(args):
    from ..data.prepare_data import word_ngrams
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    from .llm_gen2 import load_outputs
    stats = collections.Counter()
    reqs = {r["id"]: r for r in read_jsonl(S2 / "sg_requests.jsonl")}
    cands = []
    for rid, o in load_outputs("sg").items():
        r, js = reqs.get(rid), parse(o["output"])
        if r is None:
            continue
        stats["requests"] += 1
        if not isinstance(js, dict) or not isinstance(js.get("examples"), list):
            stats["parse_fail"] += 1
            continue
        seeds = [word_ngrams(x["text"]) for x in r["examples"]] + [word_ngrams(x) for x in r["refs"]]
        for j, ex in enumerate(js["examples"][:r["n"]]):
            if not isinstance(ex, dict) or not isinstance(ex.get("text"), str) or not ex["text"].strip():
                stats["empty_example"] += 1
                continue
            text = ex["text"].strip()
            g = word_ngrams(text)
            if g and any(sg and len(g & sg) / len(g) >= 0.8 for sg in seeds):
                stats["copied_seed_or_reference"] += 1
                continue
            a = annotation(r["task"], text, ex, r, stats, "gen")
            if a is None:
                continue
            cands.append({"id": f"{rid}:{j}", "set": r["set"], "task": r["task"], "text": text,
                          "gen": ex})
    stats["candidates"] = len(cands)
    cands = contamination_filter(cands, eval_texts(), stats, "eval")
    seen, uniq = set(), []
    for c in cands:                               # the same text written twice: keep the first
        if c["text"] not in seen:
            seen.add(c["text"])
            uniq.append(c)
    stats["duplicate_text"] = len(cands) - len(uniq)
    vreqs = []
    for c in uniq:
        r = reqs[c["id"].rsplit(":", 1)[0]]
        v = {k: r[k] for k in ("task", "examples", "labels", "rel_labels", "schema") if k in r}
        vreqs.append(dict(v, id=c["id"], text=c["text"]))
    write_jsonl(S2 / "sg_candidates.jsonl", uniq)
    write_jsonl(S2 / "sv_requests.jsonl", vreqs)
    stats["sv_requests"] = len(vreqs)
    (S2 / "verify_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))
    print(json.dumps(dict(collections.Counter(c["set"] for c in uniq))))


# ---------------------------------------------------------------- earlier seeded sources
OLD = {"synth_c": "NER", "synth_cre": "RE", "synth_cee": "EE"}


def old_set(src, row):
    if src == "synth_c":
        d = row["domain"]
        return d if d.startswith("mit_") else f"crossner_{d}"
    if src == "synth_cre":
        return f"crossre_{row['domain']}"
    return row["set"]


def prepare_old(args):
    """The train rows of synth_c / synth_cre (and synth_cee's documents) as candidates: their
    annotation is the generator's side; gpt-oss re-annotates them with the same types and three
    labeled examples of the target set (task sv). Asked types: those with a known status."""
    mix = ROOT / "data" / args.mix
    pools, cands, vreqs = {}, [], []
    stats = collections.Counter()
    for src, task in OLD.items():
        path = mix / src / ("docs_train.jsonl" if task == "EE" else "train.jsonl")
        for row in read_jsonl(path):
            name = old_set(src, row)
            if name not in pools:
                pools[name] = load_pool(name)
            pool, text = pools[name], row["text"]
            rng = random.Random(seed_of(f"old:{row['id']}"))
            lab = [x for x in pool["labeled"] if len(x["text"]) <= MAX_EXAMPLE_CHARS]
            req = {"task": task, "examples": rng.sample(lab, min(3, len(lab)))}
            if task == "EE":
                req["schema"] = row["schema"]
                gen = {"events": [{"type": t, "trigger": text[s:e],
                                   "arguments": [{"role": a[5], "text": text[a[3]:a[4]]} for a in row["args"]
                                                 if (a[0], a[1], a[2]) == (s, e, t)]}
                                  for s, e, t in row["triggers"]]}
            else:
                asked = [q for q in row["queries"] if q["status"] != "unknown"]
                req["labels"] = [(q["label"], q.get("description") or "") for q in asked]
                req["rel_labels"] = [(q["label"], q.get("description") or "") for q in row.get("rel_queries", [])
                                     if q["status"] != "unknown"]
                gen = {"entities": [{"text": text[e["start"]:e["end"]], "type": e["label"]} for e in row["entities"]],
                       "relations": [{"head": text[x["head"][0]:x["head"][1]], "relation": x["label"],
                                      "tail": text[x["tail"][0]:x["tail"][1]]} for x in row.get("relations", [])]}
                if not req["labels"]:
                    stats[f"{src}_no_known_type"] += 1
                    continue
            cands.append({"id": row["id"], "set": name, "task": task, "text": text, "gen": gen, "req": req,
                          "source": src})
            vreqs.append(dict(req, id=row["id"], text=text))
            stats[f"{src}_rows"] += 1
    write_jsonl(S2 / "sg_candidates.jsonl", cands)
    write_jsonl(S2 / "sv_requests.jsonl", vreqs)
    print(json.dumps(dict(stats)))


# ---------------------------------------------------------------- assemble
def agree_ner(text, q, g, labels, descs, stats, key):
    """-> (entities, queries) from the two annotators; per type positive / explicit_absent / unknown."""
    ents, queries, status = [], [], {}
    for lab in labels:
        a, b = q["ents"].get(lab, set()), g["ents"].get(lab, set())
        if lab in q["bad"] or lab in g["bad"] or a != b:
            st = "unknown"
            stats[f"{key}_type_disagree"] += 1
            if a and b and len(a) == len(b) and all(any(x[0] < y[1] and y[0] < x[1] for y in b) for x in a):
                stats[f"{key}_type_boundary_only"] += 1
        elif a:
            st = "positive"
            ents += [{"start": s, "end": e, "label": lab} for s, e in sorted(a)]
            stats[f"{key}_type_agree_positive"] += 1
        else:
            st = "explicit_absent"
            stats[f"{key}_type_agree_absent"] += 1
        status[lab] = st
        queries.append({"label": lab, "description": descs.get(lab, ""), "status": st,
                        "description_origin": "source",
                        "supervision_origin": {"positive": "two_annotators_agree", "explicit_absent":
                                               "two_annotators_agree_absent", "unknown": "annotators_disagree"}[st]})
    return ents, queries, status


def merge_overlaps(q_args, g_args):
    """Earlier CASIE-seeded documents often tag one role of one trigger with several overlapping spans
    ("14,000" and "14,000 residents"); the evaluation sets tag one. Each group of overlapping spans of
    the same (trigger, type, role) becomes the one span the verifier also chose from it; a group from
    which the verifier chose none, or more than one, stays as it is (and then fails the agreement)."""
    by = collections.defaultdict(list)
    for a in q_args:
        by[(a[0], a[1], a[2], a[5])].append((a[3], a[4]))
    g_by = collections.defaultdict(set)
    for a in g_args:
        g_by[(a[0], a[1], a[2], a[5])].add((a[3], a[4]))
    out = set()
    for key, spans in by.items():
        groups = []
        for sp in sorted(spans):
            if groups and sp[0] < max(e for _, e in groups[-1]):
                groups[-1].append(sp)
            else:
                groups.append([sp])
        for grp in groups:
            pick = [sp for sp in g_by.get(key, ()) if sp in grp]
            for s, e in (pick if len(grp) > 1 and len(pick) == 1 else grp):
                out.add((key[0], key[1], key[2], s, e, key[3]))
    return out


def assemble(args):
    from .ee_gen import rows_of
    from .llm_gen2 import load_outputs
    stats = collections.Counter()
    reqs = {r["id"]: r for r in read_jsonl(S2 / "sg_requests.jsonl")} if (S2 / "sg_requests.jsonl").exists() else {}
    vout = load_outputs("sv")
    by_src = collections.defaultdict(list)
    per_set = collections.defaultdict(collections.Counter)
    out_dir = DATA_OUT / args.out_name
    for c in read_jsonl(S2 / "sg_candidates.jsonl"):
        r = c.get("req") or reqs[c["id"].rsplit(":", 1)[0]]
        task, text, ps = c["task"], c["text"], per_set[c["set"]]
        src_name = c.get("source") or SOURCE[task]
        row_id = c["id"] if c.get("source") else f"{SOURCE[task]}:{c['id']}"
        ps["candidates"] += 1
        o = vout.get(c["id"])
        if o is None:
            stats["no_verifier_output"] += 1
            ps["no_verifier_output"] += 1
            continue
        q = annotation(task, text, c["gen"], r, stats, "gen")
        g = annotation(task, text, parse(o["output"]), r, stats, "ver")
        if g is None:
            stats["verifier_parse_fail"] += 1
            ps["verifier_parse_fail"] += 1
            continue
        key = f"{task}"
        if task == "EE":
            schema = r["schema"]
            if args.merge_overlaps:
                merged = merge_overlaps(q["args"], g["args"])
                stats["EE_args_merged_away"] += len(q["args"]) - len(merged)
                q["args"] = merged
            keep = {}
            for t in schema:
                tq = {x for x in q["trig"] if x[2] == t}
                tg = {x for x in g["trig"] if x[2] == t}
                aq = {x for x in q["args"] if x[2] == t}
                ag = {x for x in g["args"] if x[2] == t}
                if tq == tg and aq == ag:
                    keep[t] = schema[t]
                    ps["type_agree"] += 1
                else:
                    ps["type_disagree"] += 1
                    stats["EE_type_disagree"] += 1
                    stats["EE_type_trigger_agree_args_differ"] += tq == tg
            trig = sorted(x for x in q["trig"] if x[2] in keep)
            if not trig:
                stats["EE_no_agreed_event"] += 1
                ps["dropped_no_positive"] += 1
                continue
            d = {"id": row_id, "text": text, "schema": keep, "absent": {},
                 "closed_world": True, "triggers": trig,
                 "args": sorted(x for x in q["args"] if x[2] in keep)}
            for row in rows_of(d):
                row["domain"] = c["set"]
                by_src[src_name].append((c["id"], row))
            ps["kept"] += 1
            ps["triggers"] += len(trig)
            continue
        labels = [t for t, _ in r["labels"]]
        descs = dict(r["labels"])
        ents, queries, status = agree_ner(text, q, g, labels, descs, stats, key)
        row = {"id": row_id, "source": src_name, "domain": c["set"], "text": text,
               "queries": queries, "entities": ents}
        if task == "RE":
            kept = {(e["start"], e["end"]) for e in ents}
            rels, rq = [], []
            for lab, desc in r["rel_labels"]:
                a, b = q["rels"].get(lab, set()), g["rels"].get(lab, set())
                ends_ok = all((x[0], x[1]) in kept and (x[2], x[3]) in kept for x in a)
                if lab in q["rbad"] or lab in g["rbad"] or a != b or not ends_ok:
                    st = "unknown"
                    stats["RE_relation_type_disagree"] += 1
                    stats["RE_relation_type_endpoint_unknown"] += a == b and not ends_ok
                elif a:
                    st = "positive"
                    rels += [{"head": [x[0], x[1]], "tail": [x[2], x[3]], "label": lab} for x in sorted(a)]
                else:
                    st = "explicit_absent"
                rq.append({"label": lab, "description": desc, "status": st})
            row.update(rel_queries=rq, relations=rels)
            ps["relations"] += len(rels)
        ps["type_agree"] += sum(s != "unknown" for s in status.values())
        ps["type_disagree"] += sum(s == "unknown" for s in status.values())
        if not any(s == "positive" for s in status.values()):
            ps["dropped_no_positive"] += 1
            continue
        ps["kept"] += 1
        ps["entities"] += len(ents)
        by_src[src_name].append((c["id"], row))
    summary = {"stats": dict(stats), "sets": {k: dict(v) for k, v in sorted(per_set.items())}, "splits": {}}
    for src, items in by_src.items():
        texts = sorted({cid for cid, _ in items})
        random.Random(args.seed).shuffle(texts)
        n = args.n_dev
        split_of = {cid: sp for sp, part in (("dev_select", texts[:n]), ("dev_calib", texts[n:2 * n]),
                                             ("test", texts[2 * n:3 * n]), ("train", texts[3 * n:]))
                    for cid in part}
        for sp in ("train", "dev_select", "dev_calib", "test"):
            rows = [row for cid, row in items if split_of[cid] == sp]
            write_jsonl(out_dir / src / f"{sp}.jsonl", rows)
            summary["splits"][f"{src}/{sp}"] = len(rows)
    (S2 / "seed_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["splits"]))
    print(json.dumps(summary["stats"])[:3000])
    print(f"{'set':26s} {'cand':>6s} {'kept':>6s} {'agree':>6s} {'disagr':>6s}")
    for k, v in summary["sets"].items():
        print(f"{k:26s} {v.get('candidates', 0):6d} {v.get('kept', 0):6d} {v.get('type_agree', 0):6d} "
              f"{v.get('type_disagree', 0):6d}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--per-set", type=int, default=3000, help="generated texts asked per evaluation set")
    p.add_argument("--sets", default="", help="comma-separated subset (default: every pool)")
    p.add_argument("--boost", default="", help="set=factor,...: more requests for sets whose texts the two "
                                               "annotators often disagree on (smoke-test yield)")
    p.add_argument("--anchor", action="store_true", help="4 unannotated texts, each new text follows one of them")
    p.add_argument("--prefix", default="sg", help="request id prefix (a new round gets its own)")
    p.add_argument("--per-text", type=int, default=0,
                   help="one request per evaluation text, anchored on it, asking this many new texts")
    sub.add_parser("verify")
    p = sub.add_parser("prepare-old", help="re-check synth_c / synth_cre / synth_cee train rows (task sv)")
    p.add_argument("--mix", default="mix14")
    p = sub.add_parser("assemble")
    p.add_argument("--n-dev", type=int, default=300, help="texts per held-out split, per source")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", default="seed", help="output under $IE_DATA_OUT/<name>/<source>/")
    p.add_argument("--merge-overlaps", action="store_true",
                   help="EE: overlapping spans of one role and trigger -> the one the verifier chose")
    args = ap.parse_args()
    {"prepare": prepare, "verify": verify, "prepare-old": prepare_old, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
