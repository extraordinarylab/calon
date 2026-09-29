"""Round 5: refine the LLM-annotated training data with independent models.

1. prepare-ra   A second, non-Qwen model re-annotates every train row of the
                synthetic sources with the same types (task "ra", A2 prompt).
2. flag_rlcd.py Our RLCD model marks spans it is confident about that the Qwen
                labels lack (p(entity) >= 0.9), and Qwen spans it doubts.
3. prepare-jd   Every disputed span (found by only one of Qwen / the re-annotator,
                or flagged by RLCD) goes to a third model, which answers yes / no
                per span (task "jd").
4. apply        Keep spans both Qwen and the re-annotator found, plus disputed
                spans the judge accepts. A type left without spans becomes
                explicit_absent; one that gains spans becomes positive.
Backups: <source>/train.jsonl.pre_refine. Summary: synth2/refine_summary.json.
"""
import argparse
import collections
import json
import os
from pathlib import Path

from .llm_annotate import align, clean_type, parse_json, read_jsonl, write_jsonl
from .newtest import final_text, last_json

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
S2 = ROOT / "synth2"
MIX_DIR = Path(os.environ.get("IE_MIX", str(ROOT / "data/mix4")))
SOURCES = ["synth_a", "synth_r", "synth_g", "synth_g2", "synth_g3", "synth_c"]
ASKED = ("positive", "explicit_absent", "scope_assumed_absent")

JD_PROMPT = """Text:
{text}

Entity types:
{types}

Candidate annotations (span as it appears in the text, and its proposed type):
{cands}

For each candidate, answer "yes" if the span is a correct, complete mention of that type in this text (right boundaries, right type), otherwise "no".
Return JSON only: {{"1": "yes" or "no", "2": ...}}"""


def type_lines(types):
    return "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in types)


def jd_prompt(r):
    cands = "\n".join(f'{k + 1}. "{c[3]}" as {c[2]}' for k, c in enumerate(r["cands"]))
    return JD_PROMPT.format(text=r["text"], types=type_lines(r["types"]), cands=cands)


def asked_types(r):
    return [(q["label"], q["description"]) for q in r["queries"] if q["status"] in ASKED]


def prepare_ra(args):
    reqs = []
    for src in SOURCES:
        for r in read_jsonl(MIX_DIR / src / "train.jsonl"):
            types = asked_types(r)
            if types:
                reqs.append({"id": r["id"], "source": src, "text": r["text"], "query_types": types})
    write_jsonl(S2 / "ra_requests.jsonl", reqs)
    print("ra requests", len(reqs), dict(collections.Counter(r["source"] for r in reqs)))


def load_outputs(task):
    out = {}
    for p in sorted((S2 / task).glob("shard-*.jsonl")):
        for line in open(p):
            o = json.loads(line)
            out[o["id"]] = o
    return out


def second_opinion(r, output, stats, find_all):
    """Re-annotator output -> {(start, end, label)}; None if unparseable."""
    js = parse_json(output)
    if not isinstance(js, dict):
        return None
    ans = {clean_type(k): v for k, v in js.items() if isinstance(v, list)}
    spans = set()
    for t, _ in asked_types(r):
        v = [str(x).strip() for x in ans.get(clean_type(t), []) if str(x).strip()]
        if not v:
            continue
        occ = align(r["text"], v, find_all)
        if occ is None:
            stats["ra_unaligned_types"] += 1
            continue
        spans |= {(s, e, t) for s, e in occ}
    return spans


def flags_of():
    out = {}
    for p in sorted((S2 / "rlcd_flags").glob("shard-*.jsonl")):
        for line in open(p):
            o = json.loads(line)
            out[o["id"]] = o
    return out


def disputes(args, stats=None):
    """Per row: agreed spans and disputed candidates."""
    from .prepare_mix import find_all
    stats = stats if stats is not None else collections.Counter()
    ra = load_outputs("ra")
    fl = flags_of()
    for src in SOURCES:
        for r in read_jsonl(MIX_DIR / src / "train.jsonl"):
            o = ra.get(r["id"])
            if o is None:
                stats[f"{src}_no_ra"] += 1
                yield src, r, None, None
                continue
            g = second_opinion(r, o["output"], stats, find_all)
            if g is None:
                stats[f"{src}_ra_parse_fail"] += 1
                yield src, r, None, None
                continue
            q = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
            extra = {tuple(x[:3]) for x in fl.get(r["id"], {}).get("extra", [])}
            agreed = q & g
            disputed = (q ^ g) | (extra - agreed)
            stats[f"{src}_agreed"] += len(agreed)
            stats[f"{src}_qwen_only"] += len(q - g)
            stats[f"{src}_second_only"] += len(g - q)
            stats[f"{src}_rlcd_extra_new"] += len(extra - q - g)
            yield src, r, agreed, disputed


def prepare_jd(args):
    reqs, stats = [], collections.Counter()
    for src, r, agreed, disputed in disputes(args, stats):
        if not disputed:
            continue
        cands = sorted(disputed)
        reqs.append({"id": r["id"], "source": src, "text": r["text"], "types": asked_types(r),
                     "cands": [[s, e, t, r["text"][s:e]] for s, e, t in cands]})
    write_jsonl(S2 / "jd_requests.jsonl", reqs)
    stats["jd_requests"] = len(reqs)
    stats["jd_candidates"] = sum(len(r["cands"]) for r in reqs)
    (S2 / "jd_prepare_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def apply(args):
    jreq = {r["id"]: r for r in read_jsonl(S2 / "jd_requests.jsonl")}
    jout = load_outputs("jd")
    stats = collections.Counter()
    by_src = collections.defaultdict(list)
    for src, r, agreed, disputed in disputes(args, stats):
        if agreed is None:                      # no second opinion: keep the row unchanged
            by_src[src].append(r)
            continue
        keep = set(agreed)
        if disputed:
            req, o = jreq.get(r["id"]), jout.get(r["id"])
            js = last_json(final_text(o["output"])) if o else None
            if req is None or not isinstance(js, dict):
                stats[f"{src}_jd_missing"] += 1
                keep |= {(e["start"], e["end"], e["label"]) for e in r["entities"]}   # fall back to Qwen
            else:
                for k, (s, e, t, _) in enumerate(req["cands"]):
                    yes = str(js.get(str(k + 1), "")).strip().lower().startswith("y")
                    stats[f"{src}_jd_{'yes' if yes else 'no'}"] += 1
                    if yes:
                        keep.add((s, e, t))
        before = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
        stats[f"{src}_removed"] += len(before - keep)
        stats[f"{src}_added"] += len(keep - before)
        has = {t for _, _, t in keep}
        for q in r["queries"]:
            if q["status"] not in ASKED:
                continue
            if q["label"] in has and q["status"] != "positive":
                q["status"], q["supervision_origin"] = "positive", "refine_found"
                stats[f"{src}_absent_to_positive"] += 1
            elif q["label"] not in has and q["status"] == "positive":
                q["status"], q["supervision_origin"] = "explicit_absent", "refine_removed"
                stats[f"{src}_positive_to_absent"] += 1
        r["entities"] = sorted(({"start": s, "end": e, "label": t} for s, e, t in keep),
                               key=lambda x: (x["start"], x["label"]))
        if any(q["status"] == "positive" for q in r["queries"]):
            by_src[src].append(r)
        else:
            stats[f"{src}_rows_dropped_no_positive"] += 1
    for src, rows in by_src.items():
        path = MIX_DIR / src / "train.jsonl"
        path.rename(path.with_suffix(".jsonl.pre_refine"))
        write_jsonl(path, rows)
        stats[f"{src}_rows_after"] = len(rows)
    (S2 / "refine_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare-ra", "prepare-jd", "apply"])
    args = ap.parse_args()
    {"prepare-ra": prepare_ra, "prepare-jd": prepare_jd, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
