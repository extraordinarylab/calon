"""Human-annotated event data for training (round 6; never evaluated on):
  maven_arg  MAVEN-Arg train split only (its valid split is MAVEN-ERE valid, an evaluation set;
             its test labels are hidden); Wikipedia, 162 event types, document-level arguments.
             Every trigger mention is an event; an entity argument takes the entity mention
             nearest the trigger.
  geneva     GENEVA all data (FrameNet sentences, 115 event types)
  rams       RAMS train + dev + test (news, 139 types; five-sentence windows)
  phee       PHEE train + dev + test (pharmacovigilance case reports; nested arguments become
             their own roles, e.g. "drug" inside "treatment")
Long texts are cut at whitespace outside every trigger and argument into pieces of at most
MAX_CHARS characters; an argument outside its trigger's piece is dropped. Each piece asks its gold
event types plus sampled absent types of the same dataset up to N_TYPES (closed world: the
datasets annotate every event of their schema, so absent types are scope_assumed_absent).
Descriptions: label_bank.py (name and dataset context only).
Docs as ee_gen: <DATA_OUT>/ee/<source>/docs_{train,dev_select,dev_calib,test}.jsonl, then
`ee_gen rows` (EE_REFINE lists the sources). Stages: build -> label_bank -> finish.
"""
import argparse
import collections
import json
import random
import re

from calon.data.roots import ROOT

from ..synth.paths import DATA_OUT
from .prepare_spy import chunks

RAW = ROOT / "raw/ee_train"
WORK = DATA_OUT / "ee_train_build"
MAX_CHARS, N_TYPES, DEV_FRAC = 2000, 10, 0.04
# annotated only in part: RAMS has one event per passage, GENEVA selected FrameNet frames; their
# absent types and roles are unknown rather than negatives
PARTIAL = {"rams", "geneva"}


def words(code):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(code))
    return re.sub(r"\s+", " ", re.sub(r"[_\-.]+", " ", s)).strip().lower()


def rams_words(code):
    """RAMS names run words together ("deathcausedbyviolentevents"): split each part."""
    import wordninja
    parts = [p for p in str(code).split(".") if p.lower() not in ("n/a", "na")]   # "conflict.attack.n/a"
    return " ".join(" ".join(wordninja.split(p)) for p in parts).lower()


def join_tokens(tokens):
    pos, offs = 0, []
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    return " ".join(tokens), offs


def pieces_of(name, key, text, events, stats):
    """events: [{type, trigger (s, e), args [(role, (s, e))]}] -> docs per piece."""
    marks = [{"start": e["trigger"][0], "end": e["trigger"][1]} for e in events]
    marks += [{"start": s[0], "end": s[1]} for e in events for _, s in e["args"]]
    out = []
    for a, b in (chunks(text, marks, MAX_CHARS) if len(text) > MAX_CHARS else [(0, len(text))]):
        trig, args = set(), set()
        for e in events:
            ts, te = e["trigger"]
            if not (a <= ts and te <= b):
                continue
            trig.add((ts - a, te - a, e["type"]))
            for role, (s, t) in e["args"]:
                if a <= s and t <= b and (s, t) != (ts, te):
                    args.add((ts - a, te - a, e["type"], s - a, t - a, role))
                else:
                    stats[f"{name}_args_dropped"] += 1
        if trig:
            out.append({"id": f"{name}:{key}:{a}", "text": text[a:b], "triggers": sorted(trig), "args": sorted(args)})
    return out


def maven_arg(stats):
    docs = []
    for line in open(RAW / "maven_arg/train.jsonl"):
        d = json.loads(line)
        text = d["document"]
        ents = {e["id"]: [tuple(m["offset"]) for m in e["mention"]] for e in d.get("entities", [])}
        evs = []
        for ev in d["events"]:
            for m in ev["mention"]:
                ts = tuple(m["offset"])
                if text[ts[0]:ts[1]] != m["trigger_word"]:
                    stats["maven_trigger_mismatch"] += 1
                    continue
                args = []
                for role, fillers in (ev.get("argument") or {}).items():
                    for f in fillers:
                        if "entity_id" in f:
                            ms = ents.get(f["entity_id"], [])
                            if ms:
                                args.append((words(role), min(ms, key=lambda s: abs(s[0] - ts[0]))))
                        elif "offset" in f:
                            args.append((words(role), tuple(f["offset"])))
                evs.append({"type": words(ev["type"]), "trigger": ts, "args": args})
        docs += pieces_of("maven_arg", d["id"], text, evs, stats)
    return docs


def geneva(stats):
    docs = []
    for line in open(RAW / "GENEVA/data/all_data.json"):
        d = json.loads(line)
        text, offs = join_tokens(d["tokens"])
        span = {e["id"]: (offs[e["start"]][0], offs[e["end"] - 1][1]) for e in d["entity_mentions"]}
        evs = []
        for ev in d["event_mentions"]:
            t = ev["trigger"]
            evs.append({"type": words(ev["event_type"]), "trigger": (offs[t["start"]][0], offs[t["end"] - 1][1]),
                        "args": [(words(a["role"]), span[a["entity_id"]]) for a in ev["arguments"]
                                 if a["entity_id"] in span]})
        docs += pieces_of("geneva", d["wnd_id"], text, evs, stats)
    return docs


def rams(stats):
    docs = []
    for sp in ("train", "dev", "test"):
        for line in open(RAW / f"RAMS_1.0c/data/{sp}.jsonlines"):
            d = json.loads(line)
            text, offs = join_tokens([t for s in d["sentences"] for t in s])
            evs = []
            for s, e, types in d["evt_triggers"]:
                ts = (offs[s][0], offs[e][1])
                args = [(rams_words(re.sub(r"^evt\d+arg\d+", "", role)), (offs[a[0]][0], offs[a[1]][1]))
                        for (t0, t1), a, role in d["gold_evt_links"] if (t0, t1) == (s, e)]
                evs.append({"type": rams_words(types[0][0]), "trigger": ts, "args": args})
            docs += pieces_of("rams", d["doc_key"], text, evs, stats)
    return docs


def phee(stats):
    docs = []

    def spans(v, text):
        out = []
        for pieces, starts in zip(v.get("text") or [], v.get("start") or []):
            if pieces and starts:
                s, w = int(starts[0]), pieces[0]
                if text[s:s + len(w)] == w:
                    out.append((s, s + len(w)))
                else:
                    stats["phee_span_mismatch"] += 1
        return out
    for sp in ("train", "dev", "test"):
        raw = open(RAW / f"PHEE/data/json/{sp}.json").read()
        recs = (json.loads(raw) if raw.lstrip().startswith("[")
                else [json.loads(x) for x in raw.splitlines() if x.strip()])
        for d in recs:
            text, evs = d["context"], []
            for ann in d["annotations"]:
                for ev in ann["events"]:
                    tr = spans(ev.get("Trigger", {}), text)
                    if not tr:
                        continue
                    args = []
                    for role, v in ev.items():
                        if role in ("event_id", "event_type", "Trigger") or not isinstance(v, dict):
                            continue
                        args += [(words(role), s) for s in spans(v, text)]
                        for sub, sv in v.items():          # nested: Treatment -> Drug, Disorder, ...
                            if isinstance(sv, dict):
                                args += [(words(sub), s) for s in spans(sv, text)]
                    evs.append({"type": words(ev["event_type"]), "trigger": tr[0], "args": args})
            docs += pieces_of("phee", d["id"], text, evs, stats)
    return docs


def build(args):
    stats, inv, all_docs = collections.Counter(), {}, {}
    for name, fn in (("maven_arg", maven_arg), ("geneva", geneva), ("rams", rams), ("phee", phee)):
        docs = fn(stats)
        types = collections.defaultdict(set)
        for d in docs:
            for x in d["args"]:
                types[x[2]].add(x[5])
            for t in d["triggers"]:
                types.setdefault(t[2], set())
        all_docs[name] = docs
        inv[name] = [{"raw": t, "name": t, "parent": None, "kind": "event"} for t in sorted(types)]
        inv[f"{name}_role"] = [{"raw": r, "name": r, "parent": None, "kind": "role"}
                               for r in sorted({x for v in types.values() for x in v})]
        stats[f"{name}_docs"] = len(docs)
        stats[f"{name}_events"] = sum(len(d["triggers"]) for d in docs)
        stats[f"{name}_args"] = sum(len(d["args"]) for d in docs)
        stats[f"{name}_types"] = len(types)
        stats[f"{name}_roles"] = len(inv[f"{name}_role"])
        stats[f"{name}_max_chars"] = max(len(d["text"]) for d in docs)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "docs.json").write_text(json.dumps(all_docs))
    (WORK / "inventory.json").write_text(json.dumps(inv, indent=1))
    print(json.dumps(dict(sorted(stats.items()))))
    for k, v in inv.items():
        print(k, len(v), [x["name"] for x in v][:12])


def finish(args):
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    from ..synth.llm_annotate import write_jsonl
    bank = json.loads((WORK / "label_bank.json").read_text())["bank"]
    all_docs = json.loads((WORK / "docs.json").read_text())
    stats, texts = collections.Counter(), eval_texts()
    for name, docs in all_docs.items():
        eb, rb = bank[name], bank[f"{name}_role"]
        roles_of = collections.defaultdict(set)
        for d in docs:
            for x in d["args"]:
                roles_of[x[2]].add(x[5])
        all_types = sorted(eb)
        out = []
        for d in docs:
            rng = random.Random(d["id"])
            gold = sorted({t[2] for t in d["triggers"]})
            rest = [t for t in all_types if t not in gold]
            rng.shuffle(rest)
            ask = gold + rest[:max(0, N_TYPES - len(gold))]
            schema = {t: [eb[t]["description"], {r: rb[r]["description"] for r in sorted(roles_of[t])}] for t in ask}
            out.append({"id": d["id"], "text": d["text"], "schema": schema, "absent": {},
                        "closed_world": "partial" if name in PARTIAL else True,
                        "triggers": d["triggers"], "args": d["args"]})
        random.Random(42).shuffle(out)
        out = contamination_filter(out, texts, stats, name)
        n = max(200, int(len(out) * DEV_FRAC))
        for sp, part in (("dev_select", out[:n]), ("dev_calib", out[n:2 * n]), ("test", out[2 * n:3 * n]),
                         ("train", out[3 * n:])):
            write_jsonl(DATA_OUT / "ee" / name / f"docs_{sp}.jsonl", part)
            stats[f"{name}_{sp}_docs"] = len(part)
    print(json.dumps(dict(sorted(stats.items()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
