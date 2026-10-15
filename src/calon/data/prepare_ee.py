"""Event extraction evaluation sets (evaluation only): mneb/casie and mneb/wiki-events test
splits -> data/ood_ee/<set>.jsonl. mneb/casie keeps one event per sentence (1,493 events
where the original annotation has 2,105 in the same sentences), so CASIE sentences take
all their events from the original annotation (github.com/Ebiquity/CASIE, cloned
under raw/evalsets/ee/CASIE).

Row: {id, source, text, events: [{type, trigger: [s, e], args: [{role, span: [s, e]}]}],
      event_types: [label ...], roles: {event label: [role label ...]}}
Labels are the dataset codes made readable ("Conflict.Attack.DetonateExplode" ->
"conflict attack detonate explode", "Number-of-Victim" -> "number of victim");
descriptions for event types and roles come from label_bank.py (name, parent and a
fixed dataset context only), in label_bank.json next to the rows.
WikiEvents documents are cut into <= MAX_CHARS-character chunks at whitespace outside
every trigger and argument; an argument that falls in another chunk than its trigger
is dropped from the gold (counted in build_meta.json), for every model alike.
Every row asks all of its dataset's event types (CASIE 5, WikiEvents about 50).
Stages: build -> label_bank -> finish.
"""
import argparse
import collections
import json
import re
from pathlib import Path

from calon.data.roots import ROOT

from .prepare_spy import chunks

OUT = ROOT / "data/ood_ee"
# set -> (repo, test file or None = the repo's only *test*.jsonl)
REPOS = {"casie": ("mneb/casie", "data/test.jsonl"), "mlee": ("mneb/mlee", None), "m2e2": ("mneb/m2e2", None)}
MAX_CHARS = 2000


def readable(code):
    parts = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", p) for p in re.split(r"[._\-]+", code) if p]
    return " ".join(parts).lower()


def parent_of(code):
    parts = code.split(".")
    return readable(".".join(parts[:-1])) if len(parts) > 1 else None


def events_of(rec, stats, key):
    text, evs = rec["input"], []
    for etype, items in ((rec.get("output") or {}).get("json_structures") or {}).items():
        for it in items or []:
            tr = it.get("trigger") or {}
            s, e = tr.get("start"), tr.get("end")
            if s is None or e > len(text) or text[s:e] != tr.get("text", text[s:e]):
                stats[f"{key}_trigger_mismatch"] += 1
                continue
            args = []
            for a in it.get("arguments") or []:
                if (a.get("start") is None or a["end"] > len(text)
                        or text[a["start"]:a["end"]] != a.get("text", "")):
                    stats[f"{key}_argument_mismatch"] += 1
                    continue
                args.append({"role": a["role"], "span": [a["start"], a["end"]]})
            evs.append({"type": etype, "trigger": [s, e], "args": args})
    return evs


CASIE_RAW = ROOT / "raw/evalsets/ee/CASIE/data/annotation"


def casie_index():
    """Original CASIE articles (github.com/Ebiquity/CASIE): whitespace-free content, the
    content position of each kept character, and the event mentions (nugget = trigger)."""
    out = []
    for f in sorted(CASIE_RAW.glob("*.json")):
        try:
            js = json.load(open(f))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        evs = []

        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("nugget"), dict):
                    evs.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(js)
        c = js.get("content") or ""
        pos = [k for k, ch in enumerate(c) if not ch.isspace()]
        out.append(("".join(c[k] for k in pos), pos, evs))
    return out


def casie_full_events(text, index, codes, stats):
    """All original CASIE events whose trigger lies in this mneb sentence (mneb keeps one
    event per sentence), with offsets moved into the sentence; None if not matched."""
    mpos = [k for k, ch in enumerate(text) if not ch.isspace()]
    key = "".join(text[k] for k in mpos)
    hit = next(((z, pos, evs) for z, pos, evs in index if key in z), None)
    if hit is None:
        return None
    z, pos, evs = hit
    i = z.index(key)
    to_m = {pos[i + k]: mpos[k] for k in range(len(mpos))}
    raw_type = {readable(c): c for c in codes}
    raw_role = {readable(r): r for rs in codes.values() for r in rs}

    def span(s, e):
        a, b = to_m.get(int(s)), to_m.get(int(e) - 1)
        return None if a is None or b is None else [a, b + 1]
    out = []
    for ev in evs:
        tr = span(ev["nugget"]["startOffset"], ev["nugget"]["endOffset"])
        if tr is None:
            continue
        tag = f"{ev.get('type', '')} {ev.get('subtype', '')}".lower()
        name = next((v for k, v in (("breach", "data breach"), ("phish", "phishing"), ("ransom", "ransom"),
                                    ("discover", "discover vulnerability"), ("patch", "patch vulnerability"))
                     if k in tag), "")
        et = raw_type.get(name)
        if et is None:
            stats[f"casie_unmapped_tag:{tag}"] += 1
            continue
        args = []
        for a in ev.get("argument") or []:
            sp = span(a["startOffset"], a["endOffset"])
            role = raw_role.get(readable((a.get("role") or {}).get("type", "")))
            if sp is None or role is None:
                stats["casie_original_arg_outside_or_unmapped"] += 1
                continue
            args.append({"role": role, "span": sp})
        out.append({"type": et, "trigger": tr, "args": args})
    return out


def load_split(name, split, index=None):
    """One split of a set with readable labels: [{text, events}] (CASIE events restored from
    the original annotation) and its schema {event type: [roles]}. Used for seed examples."""
    from huggingface_hub import HfApi, hf_hub_download
    repo = REPOS[name][0]
    sha = HfApi().dataset_info(repo).sha
    f = next(x for x in HfApi().list_repo_files(repo, repo_type="dataset", revision=sha)
             if x.endswith(".jsonl") and split in Path(x).stem)
    path = hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=str(ROOT / "raw/hf"))
    stats, recs, codes = collections.Counter(), [], collections.defaultdict(set)
    for line in open(path):
        rec = json.loads(line)
        evs = events_of(rec, stats, name)
        for ev in evs:
            codes[ev["type"]] |= {a["role"] for a in ev["args"]}
        recs.append((rec["input"], evs))
    if name == "casie":
        index = index or casie_index()
        recs = [(t, casie_full_events(t, index, codes, stats) or evs) for t, evs in recs]
    rows = [{"text": t, "events": [{"type": readable(e["type"]), "trigger": e["trigger"],
                                    "args": [{"role": readable(a["role"]), "span": a["span"]} for a in e["args"]]}
                                   for e in evs]} for t, evs in recs]
    schema = {readable(c): sorted({readable(r) for r in rs}) for c, rs in codes.items()}
    return rows, schema, {"repo": repo, "revision": sha, "file": f, "stats": dict(stats)}


def build(args):
    from huggingface_hub import HfApi, hf_hub_download
    stats, rows, codes = collections.Counter(), [], collections.defaultdict(lambda: collections.defaultdict(set))
    meta = {}
    for name, (repo, f) in REPOS.items():
        sha = HfApi().dataset_info(repo).sha
        if f is None:
            f = next(x for x in HfApi().list_repo_files(repo, repo_type="dataset", revision=sha)
                     if x.endswith(".jsonl") and "test" in Path(x).stem)
        meta[name] = {"repo": repo, "revision": sha, "file": f}
        path = hf_hub_download(repo, f, repo_type="dataset", revision=sha, cache_dir=str(ROOT / "raw/hf"))
        for k, line in enumerate(open(path)):
            rec = json.loads(line)
            evs = events_of(rec, stats, name)
            for ev in evs:
                codes[name][ev["type"]] |= {a["role"] for a in ev["args"]}
            text = rec["input"]
            if len(text) <= MAX_CHARS:
                pieces = [(0, len(text))]
            else:
                marks = [{"start": ev["trigger"][0], "end": ev["trigger"][1]} for ev in evs]
                marks += [{"start": a["span"][0], "end": a["span"][1]} for ev in evs for a in ev["args"]]
                pieces = chunks(text, marks, MAX_CHARS)
            for a, b in pieces:
                cev = []
                for ev in evs:
                    if not (a <= ev["trigger"][0] and ev["trigger"][1] <= b):
                        continue
                    inside = [x for x in ev["args"] if a <= x["span"][0] and x["span"][1] <= b]
                    stats[f"{name}_args_cross_chunk_dropped"] += len(ev["args"]) - len(inside)
                    cev.append({"type": ev["type"], "trigger": [ev["trigger"][0] - a, ev["trigger"][1] - a],
                                "args": [{"role": x["role"], "span": [x["span"][0] - a, x["span"][1] - a]}
                                         for x in inside]})
                rows.append({"set": name, "id": f"{name}:test:{k}:{a}", "text": text[a:b], "events": cev})
            stats[f"{name}_docs"] += 1
            stats[f"{name}_max_chars"] = max(stats[f"{name}_max_chars"], len(text))
    merged = {}                    # the same text listed more than once (one event each): merge
    for r in rows:
        key = (r["set"], r["text"])
        if key in merged:
            stats[f"{r['set']}_duplicate_texts_merged"] += 1
            merged[key]["events"] += [e for e in r["events"] if e not in merged[key]["events"]]
        else:
            merged[key] = r
    rows = list(merged.values())
    index = casie_index()          # mneb keeps one event per CASIE sentence: restore all
    for r in rows:
        if r["set"] != "casie":
            continue
        full = casie_full_events(r["text"], index, codes["casie"], stats)
        if full is None:
            stats["casie_sentence_not_in_original"] += 1
            continue
        stats["casie_events_mneb"] += len(r["events"])
        r["events"] = [e for k, e in enumerate(full) if e not in full[:k]]
        stats["casie_events_restored"] += len(r["events"])
    for name in REPOS:
        rs = [r for r in rows if r["set"] == name]
        stats[f"{name}_rows"] = len(rs)
        stats[f"{name}_events"] = sum(len(r["events"]) for r in rs)
        stats[f"{name}_args"] = sum(len(e["args"]) for r in rs for e in r["events"])
        stats[f"{name}_event_types"] = len(codes[name])
        stats[f"{name}_roles"] = len({x for v in codes[name].values() for x in v})
        # one event per row everywhere would mean events were dropped (as in mneb/casie)
        stats[f"{name}_rows_with_2plus_events"] = sum(len(r["events"]) >= 2 for r in rs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_rows.json").write_text(json.dumps(rows))
    inv = {}
    for name, types in codes.items():
        inv[name] = [{"raw": c, "name": readable(c), "parent": parent_of(c), "kind": "event"} for c in sorted(types)]
        inv[f"{name}_role"] = [{"raw": r, "name": readable(r), "parent": None, "kind": "role"}
                               for r in sorted({x for v in types.values() for x in v})]
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1))
    schema = {n: {c: sorted(v) for c, v in t.items()} for n, t in codes.items()}
    (OUT / "build_meta.json").write_text(json.dumps({"sources": meta, "stats": dict(stats), "schema": schema},
                                                    indent=1))
    print(json.dumps(dict(stats)))
    for n, t in schema.items():
        print(n, json.dumps({c: v for c, v in list(t.items())[:8]})[:1500])


def finish(args):
    bank = json.loads((OUT / "label_bank.json").read_text())["bank"]
    meta = json.loads((OUT / "build_meta.json").read_text())
    rows = json.loads((OUT / "raw_rows.json").read_text())
    manifest = {"splits_used": "test", "scope_policy": "benchmark_closed_world",
                "descriptions": "authored by label_bank.py from the label name, parent and dataset context",
                "chunk_chars": MAX_CHARS, "sets": {}}
    for name in REPOS:
        eb, rb, schema = bank[name], bank[f"{name}_role"], meta["schema"][name]
        out = []
        for r in [x for x in rows if x["set"] == name]:
            out.append({"id": r["id"], "source": name, "text": r["text"],
                        "events": [{"type": eb[e["type"]]["name"], "trigger": e["trigger"],
                                    "args": [{"role": rb[a["role"]]["name"], "span": a["span"]} for a in e["args"]]}
                                   for e in r["events"]]})
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {
            "rows": len(out), "events": sum(len(r["events"]) for r in out),
            "args": sum(len(e["args"]) for r in out for e in r["events"]),
            "event_types": {v["name"]: v["description"] for v in eb.values()},
            "roles": {v["name"]: v["description"] for v in rb.values()},
            "event_roles": {eb[c]["name"]: [rb[x]["name"] for x in rs] for c, rs in schema.items()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("rows", "events", "args")} for k, v in manifest["sets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
