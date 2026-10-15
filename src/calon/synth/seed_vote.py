"""Five annotators and two judges for the target-seeded data (replaces seed_synth's two-annotator
agreement, which marked every disagreement unknown: such a row teaches no "none" at all).

Annotators of each text: Qwen (the generator's own annotation), gpt-oss (task sv), Nemotron (sv_nem),
Phi-4 (sv_phi; Hunyuan-A13B hung in vLLM 0.26 with TP 4), Mistral (sv_mis); all but Qwen see the same types and labeled examples (seed_synth's
VERIFY prompt). An annotator whose answer cannot be parsed, or whose mentions of a type cannot be found
in the text, abstains (for that text / type). Items: NER and RE entity mentions (span, type), RE relations
(head span, tail span, type), EE triggers (span, event type) and arguments (trigger, type, span, role).
  >= a majority of the voters (3 of 5)   accepted
  no voter                               the type is absent
  in between                             two judges (gpt-oss: sj_oss, Qwen: sj_qwen) answer yes / no:
                                         both yes -> accepted, both no -> rejected, otherwise uncertain
Per text, a type (entity type, relation type, event type; a role of an event type) with an uncertain item
is removed from the text's queries; the others are positive or explicit_absent. So every row is closed
over what it asks and teaches "none" for the rest.

Stages, in a requests dir $IE_SYNTH2 that has sg_candidates.jsonl and the sv outputs (seed_synth):
  prepare-annot   sv_<model>_requests.jsonl (= sv_requests.jsonl) for the three new annotators
  prepare-judge   votes; the items in between go to sj_requests.jsonl (and sj_oss / sj_qwen links)
  assemble        rows into $IE_DATA_OUT/<out-name>/<source>/train.jsonl, summary in vote_summary.json
"""
import argparse
import collections
import json
import os
import random

from .llm_annotate import read_jsonl, write_jsonl
from .paths import DATA_OUT, S2
from .seed_synth import SOURCE, annotation, parse, schema_block

ANNOT = {"oss": "sv", "nem": "sv_nem", "phi": "sv_phi", "mis": "sv_mis"}      # + "qwen": the generator
JUDGES = ("sj_oss", "sj_qwen")

JUDGE = """Text:
{text}

{schema}

Candidate annotations:
{cands}

For each candidate answer "yes" if it is correct for this text, otherwise "no". An entity candidate is correct if the span is a complete mention of that type (right boundaries, right type). A relation candidate is correct if the text states that relation from the head to the tail. A trigger candidate is correct if the span is the word or phrase expressing an event of that type. An argument candidate is correct if the span fills that role in the event with the given trigger.
Return JSON only: {{"1": "yes" or "no", "2": ...}}"""


def judge_prompt(r):
    lines = [f"{k + 1}. {c}" for k, c in enumerate(r["shown"])]
    return JUDGE.format(text=r["text"], schema=schema_block(r["task"], r), cands="\n".join(lines))


def load(task):
    from .llm_gen2 import load_outputs
    return load_outputs(task) if (S2 / task).exists() else {}


def request_of(c, reqs):
    return c.get("req") or reqs[c["id"].rsplit(":", 1)[0]]


def items_of(task, ann, r):
    """One annotation -> {kind: {key: set(items)}} and the keys it abstains on. Keys: entity / relation /
    event type; items are tuples with the key last."""
    out, abstain = collections.defaultdict(lambda: collections.defaultdict(set)), set()
    if task == "EE":
        for s, e, t in ann["trig"]:
            out["trig"][t].add((s, e, t))
        for ts, te, t, s, e, role in ann["args"]:
            out["arg"][(t, role)].add((ts, te, t, s, e, role))
        return out, abstain
    for lab, spans in ann["ents"].items():
        out["ent"][lab] = {(s, e, lab) for s, e in spans}
    abstain |= {("ent", lab) for lab in ann["bad"]}
    if task == "RE":
        for lab, pairs in ann["rels"].items():
            out["rel"][lab] = {(*p, lab) for p in pairs}
        abstain |= {("rel", lab) for lab in ann["rbad"]}
    return out, abstain


def keys_of(task, r):
    if task == "EE":
        return {"trig": list(r["schema"]),
                "arg": [(t, role) for t, (_, roles) in r["schema"].items() for role in roles]}
    k = {"ent": [t for t, _ in r["labels"]]}
    if task == "RE":
        k["rel"] = [t for t, _ in r["rel_labels"]]
    return k


def votes_of(c, r, outs, stats):
    """{kind: {key: (n_voters, Counter(item -> votes))}} for one candidate text."""
    task, text = c["task"], c["text"]
    anns = [annotation(task, text, c["gen"], r, stats, "qwen")]
    for name, t in ANNOT.items():
        o = outs[t].get(c["id"])
        anns.append(annotation(task, text, parse(o["output"]), r, stats, name) if o else None)
        stats[f"{name}_{'ok' if anns[-1] is not None else 'missing'}"] += 1
    keys = keys_of(task, r)
    res = {kind: {k: [0, collections.Counter()] for k in ks} for kind, ks in keys.items()}
    for ann in anns:
        if ann is None:
            continue
        got, abstain = items_of(task, ann, r)
        for kind, ks in keys.items():
            for k in ks:
                if (kind, k) in abstain:
                    continue
                res[kind][k][0] += 1
                res[kind][k][1].update(got[kind].get(k, ()))
    return res


def show(task, text, item, kind, trig_text=None):
    if kind == "ent":
        s, e, lab = item
        return f'entity "{text[s:e]}" as {lab}'
    if kind == "rel":
        hs, he, ts, te, lab = item
        return f'relation {lab}: head "{text[hs:he]}" -> tail "{text[ts:te]}"'
    if kind == "trig":
        s, e, t = item
        return f'trigger "{text[s:e]}" of a {t} event'
    ts, te, t, s, e, role = item
    return f'argument "{text[s:e]}" as {role} of the {t} event with trigger "{text[ts:te]}"'


def need(n):
    return n // 2 + 1


def prepare_annot(args):
    for t in ANNOT.values():
        if t != "sv":
            dst = S2 / f"{t}_requests.jsonl"
            if not dst.exists():
                os.symlink(S2 / "sv_requests.jsonl", dst)
    print("annotation requests linked:", [t for t in ANNOT.values() if t != "sv"])


def prepare_judge(args):
    stats = collections.Counter()
    reqs = {r["id"]: r for r in read_jsonl(S2 / "sg_requests.jsonl")} if (S2 / "sg_requests.jsonl").exists() else {}
    outs = {t: load(t) for t in ANNOT.values()}
    jreqs = []
    for c in read_jsonl(S2 / "sg_candidates.jsonl"):
        r = request_of(c, reqs)
        v = votes_of(c, r, outs, stats)
        shown, items = [], []
        for kind, per in v.items():
            for k, (n, cnt) in per.items():
                for it, m in cnt.items():
                    if n and 0 < m < need(n):
                        if kind == "arg" and v["trig"][it[2]][1][(it[0], it[1], it[2])] == 0:
                            continue           # an argument of a trigger nobody proposed
                        items.append([kind, list(it)])
                        shown.append(show(c["task"], c["text"], it, kind))
                        stats[f"judge_{kind}"] += 1
        stats["texts"] += 1
        if items:
            jreqs.append(dict({k: r[k] for k in ("task", "labels", "rel_labels", "schema") if k in r},
                              id=c["id"], text=c["text"], items=items, shown=shown))
    write_jsonl(S2 / "sj_requests.jsonl", jreqs)
    for t in JUDGES:
        dst = S2 / f"{t}_requests.jsonl"
        if not dst.exists():
            os.symlink(S2 / "sj_requests.jsonl", dst)
    stats["judge_requests"] = len(jreqs)
    (S2 / "vote_prepare_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def judged(jouts, rid, n):
    """[True / False / None] per item from both judges: accepted, rejected, uncertain."""
    from ..eval.newtest import final_text, last_json
    ans = []
    for t in JUDGES:
        o = jouts[t].get(rid)
        js = last_json(final_text(o["output"])) if o else None
        ans.append([str(js.get(str(k + 1), "")).strip().lower()[:1] if isinstance(js, dict) else "" for k in range(n)])
    out = []
    for a, b in zip(*ans):
        out.append(True if a == b == "y" else False if a == b == "n" else None)
    return out


def assemble(args):
    from .ee_gen import rows_of
    stats = collections.Counter()
    reqs = {r["id"]: r for r in read_jsonl(S2 / "sg_requests.jsonl")} if (S2 / "sg_requests.jsonl").exists() else {}
    outs = {t: load(t) for t in ANNOT.values()}
    jreq = {r["id"]: r for r in read_jsonl(S2 / "sj_requests.jsonl")} if (S2 / "sj_requests.jsonl").exists() else {}
    jouts = {t: load(t) for t in JUDGES}
    by_src = collections.defaultdict(list)
    per_set = collections.defaultdict(collections.Counter)
    for c in read_jsonl(S2 / "sg_candidates.jsonl"):
        r, task, text = request_of(c, reqs), c["task"], c["text"]
        ps = per_set[c["set"]]
        ps["texts"] += 1
        v = votes_of(c, r, outs, stats)
        decided = {}
        if c["id"] in jreq:
            jr = jreq[c["id"]]
            for (kind, it), d in zip(jr["items"], judged(jouts, c["id"], len(jr["items"]))):
                decided[(kind, tuple(it))] = d
        accepted, removed = collections.defaultdict(set), set()
        for kind, per in v.items():
            for k, (n, cnt) in per.items():
                if n == 0:
                    removed.add((kind, k))                 # nobody could answer for this type
                    continue
                for it, m in cnt.items():
                    if m >= need(n):
                        accepted[(kind, k)].add(it)
                        stats[f"{kind}_majority"] += 1
                    elif m > 0:
                        d = decided.get((kind, it))
                        if d is True:
                            accepted[(kind, k)].add(it)
                        elif d is None and not (kind == "arg" and v["trig"][it[2]][1][(it[0], it[1], it[2])] == 0):
                            removed.add((kind, k))
                        stats[f"{kind}_judged_{d}"] += 1
        src = c.get("source") or SOURCE[task]
        rid = c["id"] if c.get("source") else f"{SOURCE[task]}:{c['id']}"
        if task == "EE":
            schema = {}
            for t, (d, roles) in r["schema"].items():
                if ("trig", t) in removed:
                    stats["EE_type_removed"] += 1
                    continue
                schema[t] = [d, {ro: rd for ro, rd in roles.items() if ("arg", (t, ro)) not in removed}]
            trig = sorted(x for t in schema for x in accepted[("trig", t)])
            tset = {(s, e, t) for s, e, t in trig}
            args_ = sorted(x for (kind, k), xs in accepted.items() if kind == "arg" and k[0] in schema
                           and k[1] in schema[k[0]][1] for x in xs if (x[0], x[1], x[2]) in tset)
            if not trig:
                ps["dropped_no_positive"] += 1
                continue
            d = {"id": rid, "text": text, "schema": schema, "absent": {}, "closed_world": True,
                 "triggers": trig, "args": args_}
            for row in rows_of(d):
                row["domain"] = c["set"]
                by_src[src].append(row)
            ps["kept"] += 1
            continue
        queries, ents = [], []
        for lab, desc in r["labels"]:
            if ("ent", lab) in removed:
                stats["ent_type_removed"] += 1
                continue
            sp = accepted[("ent", lab)]
            ents += [{"start": s, "end": e, "label": lab} for s, e, _ in sorted(sp)]
            queries.append({"label": lab, "description": desc, "status": "positive" if sp else "explicit_absent",
                            "description_origin": "source", "supervision_origin": "vote"})
        row = {"id": rid, "source": src, "domain": c["set"], "text": text, "queries": queries, "entities": ents}
        if task == "RE":
            spans = {(e["start"], e["end"]) for e in ents}
            rq, rels = [], []
            for lab, desc in r["rel_labels"]:
                xs = accepted[("rel", lab)]
                if ("rel", lab) in removed or any((x[0], x[1]) not in spans or (x[2], x[3]) not in spans for x in xs):
                    stats["rel_type_removed"] += 1
                    continue
                rels += [{"head": [x[0], x[1]], "tail": [x[2], x[3]], "label": lab} for x in sorted(xs)]
                rq.append({"label": lab, "description": desc, "status": "positive" if xs else "explicit_absent"})
            row.update(rel_queries=rq, relations=rels)
        if not any(q["status"] == "positive" for q in queries):
            ps["dropped_no_positive"] += 1
            continue
        ps["kept"] += 1
        ps["entities"] += len(ents)
        by_src[src].append(row)
    out_dir = DATA_OUT / args.out_name
    summary = {"stats": dict(stats), "sets": {k: dict(v) for k, v in sorted(per_set.items())}, "rows": {}}
    for src, rows in by_src.items():
        random.Random(0).shuffle(rows)
        write_jsonl(out_dir / src / "train.jsonl", rows)
        for sp in ("dev_select", "dev_calib", "test"):
            write_jsonl(out_dir / src / f"{sp}.jsonl", [])
        summary["rows"][src] = len(rows)
    (S2 / "vote_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["rows"]))
    print(json.dumps(summary["stats"])[:2500])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-annot")
    sub.add_parser("prepare-judge")
    p = sub.add_parser("assemble")
    p.add_argument("--out-name", required=True)
    args = ap.parse_args()
    {"prepare-annot": prepare_annot, "prepare-judge": prepare_judge, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
