"""Event data for training (synth_gee, synth_cee) and its refinement.

gee     Multi-domain events, like the relation data (re_gen gre): 40 genres x 120 domains.
        Qwen3.8 writes one text with several events, its own event schema (3-6 event types
        with definitions, 2-6 roles each with definitions, confusable types), every event
        (type, trigger, arguments) and 2-4 event types that do not occur. Zero-shot.
cee     Seeded by the CASIE / MLEE / M2E2 *train* splits (5 examples, the dataset's event
        types and roles): 5 new annotated sentences per request. CASIE seeds carry all the
        events of the original annotation (mneb keeps one per sentence).
Refinement (both sources), as for NER and relations:
  era   Nemotron re-annotates every text with the same event types and roles;
  ejd   triggers and arguments found by only one annotator go to gpt-oss (yes / no);
  keep what both found plus what the judge accepts; an argument needs its trigger kept.
Rows (two per text) for the joint model:
  trigger row   event types as span labels, triggers as their spans;
  argument row  event types and roles as span labels (triggers and arguments), roles as
                relation types over (trigger -> argument) pairs.
  Types / roles without items: explicit_absent (gee) or scope_assumed_absent (cee).
Requests: synth2/<task>_requests.jsonl, run by llm_gen2 run (gen_node.sh).
Docs (before rows): data/ee/<source>/docs_{train,dev_select,dev_calib,test}.jsonl.
Subcommands: prepare-gee, prepare-cee, assemble, prepare-era, prepare-ejd, apply, rows.
"""
import argparse
import collections
import json
import os
import random

from calon.data.roots import ROOT

from .llm_annotate import clean_type, read_jsonl, seed_of, write_jsonl
from .paths import DATA_OUT, S2  # noqa: E402

EE_DIR = DATA_OUT / "ee"
SOURCES = os.environ.get("EE_REFINE", "synth_gee,synth_cee").split(",")   # refined / rows sources
LENGTHS = [60, 120, 200, 300]
# M2E2 has a test split only (its training data would be ACE, LDC-licensed): no seeds
SEED_SETS = {"casie": "cybersecurity news", "mlee": "biomedical research abstracts"}
MAX_ARG_LABELS = 24                # event types + roles asked in one argument row
SEED_MAX_CHARS = 600              # seed examples cut to about sentence length (MLEE: abstracts)

GEE_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English. Make it specific and natural, with concrete names, places, numbers and dates where the genre calls for them, and let it report several events (things that happen or are done), with their participants, times, places and other details.

Then design an event schema for this text: 3 to 6 event types a user working in {domain} would want to extract, each with a one-sentence definition that would apply to other texts too, and for each event type 2 to 6 argument roles (the participants and attributes of such an event), each with a one-sentence definition. Include at least one pair of closely related event types that must be told apart.
Annotate every event of every schema type in the text: its type, its trigger (the single word, occasionally a short phrase, that most directly expresses the event, copied exactly) and its arguments (role and the span filling it, copied exactly as it appears; only roles of that event type). Event types of the schema that do not occur get no events.
Finally list 2 to 4 further event types that are relevant to {domain} but do not occur anywhere in this text, each with a one-sentence definition.

Return JSON only: {{"text": "...", "event_types": {{"type": {{"definition": "...", "roles": {{"role": "definition"}}}}}}, "events": [{{"type": "...", "trigger": "...", "arguments": [{{"role": "...", "text": "..."}}]}}], "absent_event_types": {{"type": "definition"}}}}"""

CEE_PROMPT = """You are creating training data for event extraction from {domain}.
Event types and their argument roles:
{schema}

Annotated examples:
{examples}

Write 5 new examples in the same style, register and length as the examples above, each about a different, specific topic that the examples do not cover. Do not copy or paraphrase the examples.
Annotate every event of the listed types in your text: its type, its trigger (copied exactly) and its arguments (role and span, copied exactly; only roles of that event type), following the conventions of the examples.
Return JSON only: {{"examples": [{{"text": "...", "events": [{{"type": "...", "trigger": "...", "arguments": [{{"role": "...", "text": "..."}}]}}]}}]}}"""

CCEE_PROMPT = """Text (a real news passage):
{text}

Design an event schema for this text: 3 to 6 event types that a user reading such news would want to extract, each with a one-sentence definition that would apply to other texts too, and for each event type 2 to 6 argument roles (the participants and attributes of such an event), each with a one-sentence definition. Include at least one pair of closely related event types that must be told apart.
Annotate every event of every schema type in the text: its type, its trigger (the single word, occasionally a short phrase, that most directly expresses the event, copied exactly) and its arguments (role and the span filling it, copied exactly as it appears; only roles of that event type). Event types of the schema that do not occur get no events.
Finally list 2 to 4 further event types that are relevant to this kind of news but do not occur anywhere in this text, each with a one-sentence definition.

Return JSON only: {{"event_types": {{"type": {{"definition": "...", "roles": {{"role": "definition"}}}}}}, "events": [{{"type": "...", "trigger": "...", "arguments": [{{"role": "...", "text": "..."}}]}}], "absent_event_types": {{"type": "definition"}}}}"""
# round 7, from the ee3 error analysis: CASIE / SENTiVENT / MAILEX gold triggers are mostly 2-4+ word
# phrases while ours were single words; arguments that are clauses (an action requested, a means,
# a source) were almost never found
PHR_GENRES = ["business email", "email thread reply", "meeting request email", "earnings news report",
              "company press release", "analyst note", "health advice post on social media", "public health bulletin",
              "cybersecurity incident report", "IT support ticket", "forum discussion post", "project status update",
              "customer complaint letter", "government announcement"]
PHR_EXTRA = """
Triggers and arguments in this text: when an event is expressed by several words, the trigger is the whole expression (for example "let me know", "filed for bankruptcy", "raw cost", "took over", "carried out an attack", "rolled out a patch"); otherwise it is the single word. Arguments are complete spans: full noun phrases, and full clauses or verb phrases where the role is an action, a description, a means, a purpose, a reason or a source of information (for example "to send the revised contract by Friday", "by washing hands regularly", "according to the state health department")."""
CC_EXCLUDE_DOMAINS =("voanews",)    # M2E2 (an evaluation set) is Voice of America news

ERA_PROMPT = """Text:
{text}

Event types and their argument roles:
{schema}

List every event of the types above that the text reports: its type, its trigger (the single word, occasionally a short phrase, that most directly expresses the event, copied exactly) and its arguments (role and span, copied exactly as it appears; only roles of that event type).
Return JSON only: {{"events": [{{"type": "...", "trigger": "...", "arguments": [{{"role": "...", "text": "..."}}]}}]}}"""

EJD_PROMPT = """Text:
{text}

Event types and their argument roles:
{schema}

Candidate annotations:
{cands}

For each candidate answer "yes" if it is correct for this text, otherwise "no". A trigger candidate is correct if the span is the word or phrase that expresses an event of that type (right boundaries, right type). An argument candidate is correct if the span fills that role in the event expressed by the given trigger.
Return JSON only: {{"T1": "yes" or "no", ..., "A1": "yes" or "no", ...}}"""


def schema_lines(schema):
    """schema: {type: [definition, {role: definition}]}"""
    out = []
    for t, (d, roles) in schema.items():
        out.append(f'- "{t}"' + (f": {d}" if d else ""))
        for r, rd in roles.items():
            out.append(f'    - role "{r}"' + (f": {rd}" if rd else ""))
    return "\n".join(out)


def format_example(text, events):
    evs = [{"type": e["type"], "trigger": text[e["trigger"][0]:e["trigger"][1]],
            "arguments": [{"role": a["role"], "text": text[a["span"][0]:a["span"][1]]} for a in e["args"]]}
           for e in events]
    return f"text: {text}\nevents: {json.dumps(evs, ensure_ascii=False)}"


def prompt_for(task, r):
    if task == "gee":
        return GEE_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"])
    if task == "cee":
        ex = "\n\n".join(f"Example {i + 1}:\n{e}" for i, e in enumerate(r["examples"]))
        from .diversity import hint
        return CEE_PROMPT.format(domain=r["domain"], schema=schema_lines(r["schema"]), examples=ex) \
            + hint(r["id"], names_only=True)
    if task == "ccee":
        return CCEE_PROMPT.format(text=r["text"])
    if task == "gphr":
        from .diversity import hint
        base = GEE_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"]).replace(
            "its trigger (the single word, occasionally a short phrase, that most directly expresses the event, copied exactly)",
            "its trigger (the word or the whole multi-word expression that expresses the event, copied exactly; see below)")
        return base + PHR_EXTRA + "\nWrite the text so that at least half of the events are expressed by multi-word " \
            "expressions." + hint(r["id"])
    if task == "era":
        return ERA_PROMPT.format(text=r["text"], schema=schema_lines(r["schema"]))
    lines = [f'T{k + 1}. "{c[3]}" as {c[2]}' for k, c in enumerate(r["tcands"])]
    lines += [f'A{k + 1}. "{c[6]}" as {c[5]} of the {c[2]} event with trigger "{c[7]}"' for k, c in enumerate(r["acands"])]
    return EJD_PROMPT.format(text=r["text"], schema=schema_lines(r["schema"]), cands="\n".join(lines))


def parse(text):
    from ..eval.newtest import final_text, last_json
    js = last_json(final_text(text))
    return js if isinstance(js, (dict, list)) else None


def load_outputs(task):
    out = {}
    for p in sorted((S2 / task).glob("shard-*.jsonl")):
        for line in open(p):
            o = json.loads(line)
            out[o["id"]] = o
    return out


def align_events(text, items, schema, stats, key):
    """LLM events -> ({(ts, te, type)}, {(ts, te, type, as, ae, role)}); triggers take the
    first unused occurrence, arguments the occurrence nearest their trigger."""
    from ..data.prepare_mix import find_all
    trig, args, used = set(), set(), set()
    for ev in items or []:
        if not isinstance(ev, dict):
            continue
        t = clean_type(ev.get("type", ""))
        if t not in schema:
            stats[f"{key}_event_type_unknown"] += 1
            continue
        trig_text = str(ev.get("trigger", "")).strip()
        occ = [o for o in find_all(text, trig_text) if o not in used] if trig_text else []
        if not occ:
            stats[f"{key}_trigger_unaligned"] += 1
            continue
        ts = occ[0]
        used.add(ts)
        trig.add((ts[0], ts[1], t))
        roles = schema[t][1]
        for a in ev.get("arguments") or []:
            if not isinstance(a, dict):
                continue
            role = clean_type(a.get("role", ""))
            s = str(a.get("text", "")).strip()
            if role not in roles or not s:
                stats[f"{key}_role_unknown"] += 1
                continue
            ao = [o for o in find_all(text, s) if o != ts]
            if not ao:
                stats[f"{key}_argument_unaligned"] += 1
                continue
            sp = min(ao, key=lambda o: abs(o[0] - ts[0]))
            args.add((ts[0], ts[1], t, sp[0], sp[1], role))
    return trig, args


# ---------------------------------------------------------------- requests
def prepare_gee(args):
    from .llm_gen2 import DOMAINS3, GENRES3
    rng = random.Random(args.seed + 11)
    cells = [(g, d) for g in GENRES3 for d in DOMAINS3]
    reqs = [{"id": f"gee:{k}", "genre": cells[k % len(cells)][0], "domain": cells[k % len(cells)][1],
             "length": rng.choice(LENGTHS)} for k in range(args.n)]
    rng.shuffle(reqs)
    write_jsonl(S2 / "gee_requests.jsonl", reqs)
    print(f"gee requests {len(reqs)} over {len(cells)} genre x domain cells")


def prepare_gphr(args):
    from .llm_gen2 import DOMAINS3
    rng = random.Random(args.seed + 29)
    reqs = [{"id": f"gphr:{k}", "genre": PHR_GENRES[k % len(PHR_GENRES)], "domain": rng.choice(DOMAINS3),
             "length": rng.choice(LENGTHS)} for k in range(args.n)]
    rng.shuffle(reqs)
    write_jsonl(S2 / "gphr_requests.jsonl", reqs)
    print(f"gphr requests {len(reqs)}")


def prepare_ccee(args):
    """One passage (whole paragraphs, 60 to MAX words) from each of args.n random CC-News
    articles (vblagoje/cc_news, 2017-2019 English news); Voice of America articles left out."""
    import glob

    import pandas as pd
    from huggingface_hub import snapshot_download
    path = snapshot_download("vblagoje/cc_news", repo_type="dataset", cache_dir=str(ROOT / "raw/hf"), max_workers=2)
    df = pd.concat([pd.read_parquet(f, columns=["title", "text", "domain", "date", "url"])
                    for f in sorted(glob.glob(f"{path}/plain_text/*.parquet"))], ignore_index=True)
    stats = collections.Counter(articles=len(df))
    df = df[~df["domain"].fillna("").str.contains("|".join(CC_EXCLUDE_DOMAINS))]
    stats["after_domain_exclusion"] = len(df)
    order = list(range(len(df)))
    random.Random(args.seed).shuffle(order)
    reqs = []
    for k in order:
        rec = df.iloc[k]
        paras = [p.strip() for p in str(rec["text"]).split("\n") if len(p.split()) >= 8]
        cur = []
        for p in paras:
            if sum(len(x.split()) for x in cur) + len(p.split()) > args.max_words:
                break
            cur.append(p)
        words = sum(len(x.split()) for x in cur)
        if words < 60:
            stats["too_short"] += 1
            continue
        reqs.append({"id": f"ccee:{k}", "text": "\n".join(cur), "url": rec["url"], "domain": rec["domain"],
                     "date": str(rec["date"])})
        if len(reqs) == args.n:
            break
    write_jsonl(S2 / "ccee_requests.jsonl", reqs)
    stats["requests"] = len(reqs)
    stats["news_sites"] = len({r["domain"] for r in reqs})
    print(json.dumps(dict(stats)))


def pieces(r):
    """A seed text cut into <= SEED_MAX_CHARS pieces at whitespace outside every trigger and
    argument; events kept whole (trigger and all arguments inside the piece)."""
    from ..data.prepare_spy import chunks
    marks = [{"start": e["trigger"][0], "end": e["trigger"][1]} for e in r["events"]]
    marks += [{"start": a["span"][0], "end": a["span"][1]} for e in r["events"] for a in e["args"]]
    out = []
    for a, b in chunks(r["text"], marks, SEED_MAX_CHARS):
        evs = [e for e in r["events"] if a <= e["trigger"][0] and e["trigger"][1] <= b
               and all(a <= x["span"][0] and x["span"][1] <= b for x in e["args"])]
        out.append({"text": r["text"][a:b], "events": [
            {"type": e["type"], "trigger": [e["trigger"][0] - a, e["trigger"][1] - a],
             "args": [{"role": x["role"], "span": [x["span"][0] - a, x["span"][1] - a]} for x in e["args"]]}
            for e in evs]})
    return out


def prepare_cee(args):
    from ..data.prepare_ee import casie_index, load_split
    bank = json.loads((ROOT / "data/ood_ee/label_bank.json").read_text())["bank"]
    index = casie_index()
    reqs, meta = [], {}
    for name, domain in SEED_SETS.items():
        rows, schema, info = load_split(name, "train", index)
        pool = [p for r in rows for p in pieces(r) if p["events"]]
        edesc = {v["name"]: v["description"] for v in bank.get(name, {}).values()}
        rdesc = {v["name"]: v["description"] for v in bank.get(f"{name}_role", {}).values()}
        sch = {t: [edesc.get(t, ""), {r: rdesc.get(r, "") for r in roles}] for t, roles in sorted(schema.items())}
        meta[name] = dict(info, seed_pool=len(pool), event_types=len(sch))
        for k in range(args.per_set):
            rng = random.Random(seed_of(f"cee:{name}:{k}"))
            ex = rng.sample(pool, 5)
            reqs.append({"id": f"cee:{name}:{k}", "set": name, "domain": domain, "schema": sch,
                         "examples": [format_example(e["text"], e["events"]) for e in ex]})
    random.Random(0).shuffle(reqs)
    write_jsonl(S2 / "cee_requests.jsonl", reqs)
    (EE_DIR).mkdir(parents=True, exist_ok=True)
    (EE_DIR / "cee_seed_meta.json").write_text(json.dumps(meta, indent=1))
    print("cee requests", len(reqs), json.dumps({k: {x: v[x] for x in ("seed_pool", "event_types")} for k, v in meta.items()}))


# ---------------------------------------------------------------- docs
def doc_of(text, schema, items, absent, stats, key, closed):
    trig, args = align_events(text, items, schema, stats, key)
    return {"text": text, "schema": schema, "absent": absent, "closed_world": closed,
            "triggers": sorted(trig), "args": sorted(args)}


def assemble(args):
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    stats = collections.Counter()
    docs = {"synth_gee": [], "synth_cee": [], "ccnews_ee": [], "synth_phr": []}

    def open_schema(js):
        schema = {}
        for t, v in (js.get("event_types") or {}).items():
            if clean_type(t) and isinstance(v, dict):
                schema[clean_type(t)] = [str(v.get("definition", "")),
                                         {clean_type(r): str(d) for r, d in (v.get("roles") or {}).items() if clean_type(r)}]
        absent = {clean_type(k): str(v) for k, v in (js.get("absent_event_types") or {}).items()
                  if clean_type(k) and clean_type(k) not in schema}
        return schema, absent
    for task, src in (("gee", "synth_gee"), ("gphr", "synth_phr")):
        for rid, o in load_outputs(task).items():
            js = parse(o["output"])
            if not isinstance(js, dict) or not isinstance(js.get("text"), str):
                stats[f"{task}_parse_fail"] += 1
                continue
            schema, absent = open_schema(js)
            d = doc_of(js["text"].strip(), schema, js.get("events"), absent, stats, task, False)
            if d["triggers"]:
                docs[src].append(dict(d, id=f"{src}:{rid}"))
                stats[f"{task}_multiword_triggers"] += sum(len(d["text"][s:e].split()) > 1 for s, e, _ in d["triggers"])
                stats[f"{task}_triggers"] += len(d["triggers"])
            else:
                stats[f"{task}_no_events"] += 1
    creqs = {r["id"]: r for r in read_jsonl(S2 / "ccee_requests.jsonl")} if (S2 / "ccee_requests.jsonl").exists() else {}
    for rid, o in load_outputs("ccee").items():           # real news text, annotated (not generated)
        js = parse(o["output"])
        if not isinstance(js, dict) or rid not in creqs:
            stats["ccee_parse_fail"] += 1
            continue
        schema, absent = open_schema(js)
        d = doc_of(creqs[rid]["text"], schema, js.get("events"), absent, stats, "ccee", False)
        if d["triggers"]:
            docs["ccnews_ee"].append(dict(d, id=f"ccnews_ee:{rid}"))
        else:
            stats["ccee_no_events"] += 1
    reqs = {r["id"]: r for r in read_jsonl(S2 / "cee_requests.jsonl")} if (S2 / "cee_requests.jsonl").exists() else {}
    for rid, o in load_outputs("cee").items():
        req, js = reqs[rid], parse(o["output"])
        if not isinstance(js, dict) or not isinstance(js.get("examples"), list):
            stats["cee_parse_fail"] += 1
            continue
        schema = {t: [d, dict(roles)] for t, (d, roles) in req["schema"].items()}
        for k, ex in enumerate(js["examples"]):
            if not isinstance(ex, dict) or not isinstance(ex.get("text"), str) or not ex["text"].strip():
                continue
            d = doc_of(ex["text"].strip(), schema, ex.get("events"), {}, stats, "cee", True)
            if d["triggers"]:
                docs["synth_cee"].append(dict(d, id=f"synth_cee:{rid}:{k}", set=req["set"]))
            else:
                stats["cee_no_events"] += 1
    # copies of a seed (word 5-gram containment >= 0.8), with the indexed filter
    seed_texts = sorted({e.split("\nevents:")[0][len("text: "):] for r in reqs.values() for e in r["examples"]})
    docs["synth_cee"] = contamination_filter(docs["synth_cee"], seed_texts, stats, "cee_seed")
    texts = eval_texts()
    for src, ds in docs.items():
        if not ds:
            continue
        random.Random(args.seed).shuffle(ds)
        ds = contamination_filter(ds, texts, stats, src)
        n = args.n_dev
        for sp, part in (("dev_select", ds[:n]), ("dev_calib", ds[n:2 * n]), ("test", ds[2 * n:3 * n]),
                         ("train", ds[3 * n:])):
            write_jsonl(EE_DIR / src / f"docs_{sp}.jsonl", part)
            stats[f"{src}_{sp}_docs"] = len(part)
        stats[f"{src}_triggers"] = sum(len(d["triggers"]) for d in ds)
        stats[f"{src}_args"] = sum(len(d["args"]) for d in ds)
    (EE_DIR / "assemble_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))
    for d in docs["synth_gee"][:args.show] + docs["synth_cee"][:args.show]:
        t = d["text"]
        print(t[:250])
        print("  T:", [(t[s:e], ty) for s, e, ty in d["triggers"]][:8])
        print("  A:", [(t[x[0]:x[1]], x[5], t[x[3]:x[4]]) for x in d["args"]][:10])


# ---------------------------------------------------------------- refinement
def prepare_era(args):
    reqs = []
    for src in SOURCES:
        for d in read_jsonl(EE_DIR / src / "docs_train.jsonl"):
            reqs.append({"id": d["id"], "source": src, "text": d["text"], "schema": d["schema"]})
    write_jsonl(S2 / "era_requests.jsonl", reqs)
    print("era requests", len(reqs), dict(collections.Counter(r["source"] for r in reqs)))


def disputes(stats):
    ra = load_outputs("era")
    for src in SOURCES:
        for d in read_jsonl(EE_DIR / src / "docs_train.jsonl"):
            o = ra.get(d["id"])
            js = parse(o["output"]) if o else None
            items = js.get("events") if isinstance(js, dict) else (js if isinstance(js, list) else None)
            if items is None:
                stats[f"{src}_no_second_opinion"] += 1
                yield src, d, None
                continue
            gt, ga = align_events(d["text"], items, d["schema"], stats, f"{src}_era")
            qt, qa = {tuple(x) for x in d["triggers"]}, {tuple(x) for x in d["args"]}
            res = {"agreed_t": qt & gt, "agreed_a": qa & ga, "disp_t": sorted(qt ^ gt), "disp_a": sorted(qa ^ ga)}
            for k, (q, g) in (("t", (qt, gt)), ("a", (qa, ga))):
                stats[f"{src}_{k}_agreed"] += len(q & g)
                stats[f"{src}_{k}_qwen_only"] += len(q - g)
                stats[f"{src}_{k}_second_only"] += len(g - q)
            yield src, d, res


def prepare_ejd(args):
    stats, reqs = collections.Counter(), []
    for src, d, res in disputes(stats):
        if res is None or not (res["disp_t"] or res["disp_a"]):
            continue
        t = d["text"]
        reqs.append({"id": d["id"], "source": src, "text": t, "schema": d["schema"],
                     "tcands": [[s, e, ty, t[s:e]] for s, e, ty in res["disp_t"]],
                     "acands": [[a, b, ty, s, e, role, t[s:e], t[a:b]] for a, b, ty, s, e, role in res["disp_a"]]})
    write_jsonl(S2 / "ejd_requests.jsonl", reqs)
    stats["ejd_requests"] = len(reqs)
    (EE_DIR / "ejd_prepare_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def apply(args):
    jreq = {r["id"]: r for r in read_jsonl(S2 / "ejd_requests.jsonl")}
    jout = load_outputs("ejd")
    stats, by_src = collections.Counter(), collections.defaultdict(list)
    for src, d, res in disputes(stats):
        if res is None:
            by_src[src].append(d)
            continue
        kt, ka = set(res["agreed_t"]), set(res["agreed_a"])
        if res["disp_t"] or res["disp_a"]:
            req, o = jreq.get(d["id"]), jout.get(d["id"])
            js = parse(o["output"]) if o else None
            if req is None or not isinstance(js, dict):
                stats[f"{src}_judge_missing"] += 1
                kt |= {tuple(x) for x in d["triggers"]}
                ka |= {tuple(x) for x in d["args"]}
            else:
                for k, c in enumerate(req["tcands"]):
                    yes = str(js.get(f"T{k + 1}", "")).strip().lower().startswith("y")
                    stats[f"{src}_t_judge_{'yes' if yes else 'no'}"] += 1
                    if yes:
                        kt.add(tuple(c[:3]))
                for k, c in enumerate(req["acands"]):
                    yes = str(js.get(f"A{k + 1}", "")).strip().lower().startswith("y")
                    stats[f"{src}_a_judge_{'yes' if yes else 'no'}"] += 1
                    if yes:
                        ka.add(tuple(c[:6]))
        n = len(ka)
        ka = {a for a in ka if a[:3] in kt}
        stats[f"{src}_a_dropped_trigger_removed"] += n - len(ka)
        before_t, before_a = {tuple(x) for x in d["triggers"]}, {tuple(x) for x in d["args"]}
        stats[f"{src}_t_removed"] += len(before_t - kt)
        stats[f"{src}_t_added"] += len(kt - before_t)
        stats[f"{src}_a_removed"] += len(before_a - ka)
        stats[f"{src}_a_added"] += len(ka - before_a)
        if kt:
            by_src[src].append(dict(d, triggers=sorted(kt), args=sorted(ka), refined=True))
        else:
            stats[f"{src}_docs_dropped_no_event"] += 1
    for src, ds in by_src.items():
        p = EE_DIR / src / "docs_train.jsonl"
        p.rename(p.with_suffix(".jsonl.pre_refine"))
        write_jsonl(p, ds)
        stats[f"{src}_docs_after"] = len(ds)
        stats[f"{src}_args_after"] = sum(len(d["args"]) for d in ds)
    (EE_DIR / "refine_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def merge_nested(d):
    """synth_phr: refinement keeps both annotators' triggers, so one event can carry a phrase trigger
    ("sent the invoice") and a word inside it ("sent"). For triggers of the same type nested in one
    another the outer span is kept and the inner one's arguments move to it."""
    trig = sorted({tuple(x) for x in d["triggers"]}, key=lambda t: (t[1] - t[0]), reverse=True)
    outer = {}
    for t in trig:
        host = next((o for o in outer if o[2] == t[2] and o[0] <= t[0] and t[1] <= o[1]), None)
        outer[t] = host if host is not None else t
    keep = sorted({o for t, o in outer.items() if o == t})
    args = set()
    for a in d["args"]:
        o = outer.get((a[0], a[1], a[2]), (a[0], a[1], a[2]))
        if (a[3], a[4]) != (o[0], o[1]):
            args.add((o[0], o[1], o[2], a[3], a[4], a[5]))
    return dict(d, triggers=keep, args=sorted(args)), len(trig) - len(keep)


def merge_nested_cmd(args):
    stats = collections.Counter()
    for src in args.sources.split(","):
        for sp in ("train", "dev_select", "dev_calib", "test"):
            p = EE_DIR / src / f"docs_{sp}.jsonl"
            out = []
            for d in read_jsonl(p):
                d2, n = merge_nested(d)
                stats[f"{src}_{sp}_nested_triggers_merged"] += n
                out.append(d2)
            write_jsonl(p, out)
    print(json.dumps(dict(stats)))


# ---------------------------------------------------------------- training rows
def rows_of(d):
    # closed_world: True = every event of the schema is annotated (absent types are negatives);
    # "partial" = only some events are (RAMS: one per passage, GENEVA: selected frames), so a type
    # without a gold event is unknown, not absent
    absent_status = {True: "scope_assumed_absent", "partial": "unknown"}.get(d["closed_world"], "explicit_absent")
    origin = "llm_generated_seeded" if d["closed_world"] else "llm_generated"
    has_t = {t for _, _, t in d["triggers"]}

    def q(label, desc, pos):
        return {"label": label, "description": desc, "status": "positive" if pos else absent_status,
                "description_origin": "llm_authored", "supervision_origin": origin}
    tq = [q(t, v[0], t in has_t) for t, v in d["schema"].items()]
    tq += [dict(q(t, v, False), status="explicit_absent") for t, v in d["absent"].items()]
    trig_ents = [{"start": s, "end": e, "label": t} for s, e, t in d["triggers"]]
    out = [{"id": f"{d['id']}:trig", "source": d["id"].split(":")[0], "text": d["text"], "queries": tq,
            "entities": trig_ents}]
    if d["args"]:
        # argument rows ask the present event types and their roles; texts with many event types
        # (MAVEN-Arg documents) are split into several rows of at most MAX_ARG_LABELS labels, each
        # with the events of its types only, so that the schema fits the encoder's budget
        groups, cur, size = [], [], 0
        for t in sorted(has_t):
            n = 1 + len(d["schema"][t][1])
            if cur and size + n > MAX_ARG_LABELS:
                groups.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += n
        if cur:
            groups.append(cur)
        for k, types in enumerate(groups):
            roles = {}
            for t in types:
                roles.update(d["schema"][t][1])
            args = sorted({tuple(x) for x in d["args"] if x[2] in types})
            if not args:
                continue
            has_g = {a[5] for a in args}
            rq = [q(r, desc, r in has_g) for r, desc in sorted(roles.items())]
            tg = [e for e in trig_ents if e["label"] in types]
            ents = tg + [{"start": a[3], "end": a[4], "label": a[5]} for a in args]
            ents = [e for j, e in enumerate(ents) if e not in ents[:j]]
            suffix = "" if len(groups) == 1 else f":{k}"
            out.append({"id": f"{d['id']}:args{suffix}", "source": d["id"].split(":")[0], "text": d["text"],
                        "queries": [x for x in tq if x["label"] in types] + rq, "entities": ents,
                        "rel_queries": [{"label": x["label"], "description": x["description"], "status": x["status"]}
                                        for x in rq],
                        "relations": [{"head": [a[0], a[1]], "tail": [a[3], a[4]], "label": a[5]} for a in args]})
    return out


def rows(args):
    stats = collections.Counter()
    for src in SOURCES:
        for sp in ("train", "dev_select", "dev_calib", "test"):
            rs = [x for d in read_jsonl(EE_DIR / src / f"docs_{sp}.jsonl") for x in rows_of(d)]
            write_jsonl(EE_DIR / src / f"{sp}.jsonl", rs)
            stats[f"{src}_{sp}_rows"] = len(rs)
    (EE_DIR / "rows_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-gee")
    p.add_argument("--n", type=int, default=150000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-cee")
    p.add_argument("--per-set", type=int, default=12000)
    p = sub.add_parser("merge-nested")
    p.add_argument("--sources", default="synth_phr")
    p = sub.add_parser("prepare-gphr")
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-ccee")
    p.add_argument("--n", type=int, default=150000)
    p.add_argument("--max-words", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("assemble")
    p.add_argument("--n-dev", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", type=int, default=0)
    sub.add_parser("prepare-era")
    sub.add_parser("prepare-ejd")
    sub.add_parser("apply")
    sub.add_parser("rows")
    args = ap.parse_args()
    {"prepare-gee": prepare_gee, "prepare-cee": prepare_cee, "prepare-ccee": prepare_ccee, "prepare-gphr": prepare_gphr, "merge-nested": merge_nested_cmd, "assemble": assemble, "prepare-era": prepare_era,
     "prepare-ejd": prepare_ejd, "apply": apply, "rows": rows}[args.cmd](args)


if __name__ == "__main__":
    main()
