"""Relation completeness pass (round 8, from the case study against gpt-oss): relations the generator
did not list are negatives in training, which keeps the relation score low on true pairs (SciER,
GAP, LitBank, Re-TACRED: most misses had the right label but scored under the threshold).

  prepare  requests rfill: each train row of the sources, entities numbered, relation types with their
           definitions, the relations already annotated; the filler (Nemotron) lists further relations
  judge    requests rfjd: each proposed relation not yet annotated, yes / no by gpt-oss (round-5 judge
           prompt, coreference wording for refers to / coreference)
  apply    accepted relations added (both directions for undirected labels); a relation type that gains
           relations becomes positive. Output <out>/<source>/train.jsonl, other files linked.
"""
import argparse
import collections
import json
import os
from pathlib import Path

from .llm_annotate import read_jsonl, write_jsonl
from .paths import S2

RFILL_PROMPT = """Text (entity mentions numbered in brackets):
{text}

Relation types (directed, from head to tail):
{rtypes}

Relations already found:
{known}

List every further relation of the types above that the text states or clearly implies between two numbered mentions and that is not in the list already found. Check every pair of mentions; include relations between mentions in different sentences.
Return JSON only: a list of {{"head": "E<n>", "relation": "type", "tail": "E<n>"}}; return [] if there is none."""


def numbered(r):
    """Text with entities as [mention](E<k>); entity k = position in r["entities"] (overlaps skipped)."""
    order = sorted(range(len(r["entities"])), key=lambda k: (r["entities"][k]["start"], -r["entities"][k]["end"]))
    out, pos, shown = [], 0, set()
    for k in order:
        e = r["entities"][k]
        if e["start"] < pos:
            continue
        out.append(r["text"][pos:e["start"]] + f"[{r['text'][e['start']:e['end']]}](E{k + 1})")
        pos = e["end"]
        shown.add(k)
    return "".join(out) + r["text"][pos:], shown


def rtype_lines(r):
    return "\n".join(f'- "{q["label"]}"' + (f": {q['description']}" if q["description"] else "") for q in r["rel_queries"])


def ent_index(r):
    return {(e["start"], e["end"]): k for k, e in enumerate(r["entities"])}


def prompt_for(task, r):
    if task == "rfill":
        return RFILL_PROMPT.format(text=r["text"], rtypes=r["rtypes"], known=r["known"] or "(none)")
    from .round5 import JUDGE_PROMPT, LINK_EXTRA
    return JUDGE_PROMPT.format(text=r["text"], rtypes=r["rtypes"], cands=r["cands"], extra=LINK_EXTRA if r["links"] else "")


def prepare(args):
    reqs, stats = [], collections.Counter()
    for src in args.sources.split(","):
        for k, r in enumerate(read_jsonl(Path(args.mix) / src / "train.jsonl")):
            if not r.get("rel_queries") or len(r["entities"]) < 2:
                continue
            text, shown = numbered(r)
            idx = ent_index(r)
            known = "\n".join(f'E{idx[tuple(x["head"])] + 1} -{x["label"]}-> E{idx[tuple(x["tail"])] + 1}'
                              for x in r["relations"] if tuple(x["head"]) in idx and tuple(x["tail"]) in idx)
            reqs.append({"id": f"{src}#{k}", "text": text, "rtypes": rtype_lines(r), "known": known})
            stats[src] += 1
    write_jsonl(S2 / "rfill_requests.jsonl", reqs)
    print(json.dumps(dict(stats)), len(reqs))


def proposals(args):
    """{(source, row index): [(head k, tail k, label)]} new relations proposed by the filler."""
    from .ee_gen import load_outputs, parse
    out = load_outputs("rfill")
    rows = {src: read_jsonl(Path(args.mix) / src / "train.jsonl") for src in args.sources.split(",")}
    props = {}
    for rid, o in out.items():
        src, k = rid.split("#")
        r = rows[src][int(k)]
        labels = {q["label"] for q in r["rel_queries"]}
        have = {(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}
        js = parse(o["output"])
        new = []
        for it in js if isinstance(js, list) else []:
            try:
                h, t = int(str(it["head"]).lstrip("E")) - 1, int(str(it["tail"]).lstrip("E")) - 1
                eh, et = r["entities"][h], r["entities"][t]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            lab = it.get("relation")
            key = ((eh["start"], eh["end"]), (et["start"], et["end"]), lab)
            if lab in labels and h != t and key not in have and (h, t, lab) not in new:
                new.append((h, t, lab))
        if new:
            props[(src, int(k))] = new
    return rows, props


def judge(args):
    rows, props = proposals(args)
    reqs = []
    for (src, k), new in props.items():
        r = rows[src][k]
        text, _ = numbered(r)
        for j in range(0, len(new), 40):
            part = new[j:j + 40]
            lines = "\n".join(f'R{i + 1}: head E{h + 1} "{r["text"][r["entities"][h]["start"]:r["entities"][h]["end"]]}" -> '
                              f'"{lab}" -> tail E{t + 1} "{r["text"][r["entities"][t]["start"]:r["entities"][t]["end"]]}"'
                              for i, (h, t, lab) in enumerate(part))
            reqs.append({"id": f"{src}#{k}#{j}", "text": text, "rtypes": rtype_lines(r), "cands": lines,
                         "links": any(lab in ("refers to", "coreference", "person identity") for _, _, lab in part)})
    write_jsonl(S2 / "rfjd_requests.jsonl", reqs)
    print("proposed", sum(len(v) for v in props.values()), "in", len(props), "rows; judge requests", len(reqs))


def apply(args):
    from .ee_gen import load_outputs, parse
    rows, props = proposals(args)
    ans = load_outputs("rfjd")
    stats = collections.Counter()
    for (src, k), new in props.items():
        r = rows[src][k]
        sym = set(r.get("symmetric_labels", []))
        if src in ("synth_bio", "synth_bio2"):
            sym = {q["label"] for q in r["rel_queries"]}
        have = {(tuple(x["head"]), tuple(x["tail"]), x["label"]) for x in r["relations"]}
        for j in range(0, len(new), 40):
            o = ans.get(f"{src}#{k}#{j}")
            js = parse(o["output"]) if o else None
            for i, (h, t, lab) in enumerate(new[j:j + 40]):
                yes = isinstance(js, dict) and str(js.get(f"R{i + 1}", "")).strip().lower().startswith("y")
                stats[f"{src}_{'added' if yes else 'rejected'}"] += 1
                if not yes:
                    continue
                eh, et = r["entities"][h], r["entities"][t]
                pair = [((eh["start"], eh["end"]), (et["start"], et["end"]))]
                if lab in sym:
                    pair.append(pair[0][::-1])
                for a, b in pair:
                    if (a, b, lab) not in have:
                        r["relations"].append({"head": list(a), "tail": list(b), "label": lab})
                        have.add((a, b, lab))
                for q in r["rel_queries"]:
                    if q["label"] == lab and q["status"] != "positive":
                        q["status"] = "positive"
                        stats[f"{src}_query_status_to_positive"] += 1
    for src, rs in rows.items():
        o = Path(args.out) / src
        o.mkdir(parents=True, exist_ok=True)
        write_jsonl(o / "train.jsonl", rs)
        d = Path(args.mix) / src
        for p in d.iterdir():
            if p.name != "train.jsonl" and not (o / p.name).exists():
                os.symlink(p.resolve(), o / p.name)
        stats[f"{src}_relations_after"] = sum(len(r["relations"]) for r in rs)
    (Path(args.out) / "summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "judge", "apply"])
    ap.add_argument("--mix", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"prepare": prepare, "judge": judge, "apply": apply}[args.stage](args)


if __name__ == "__main__":
    main()
