"""Event extraction baselines on data/ood_ee, scored like eval/ee_eval.py.

  gliner2   structured extraction (extract_json): one structure per event type with a
            "trigger" field and one list field per role (descriptions unless
            --name_only); end to end, plus a gold-trigger proxy (to_gold_triggers)
            since the trigger cannot be given; `proxy` adds it to an existing result.
  gliformer the same schemas through GLiFormer's structuring head (model.structure), field
            names only (its templates take no field descriptions); same proxy.
  llm       prompted LLM (vLLM, e.g. gpt-oss-120b): given (gold triggers listed with
            ids, the model names arguments) and end2end (the model lists events).
Strings are mapped to offsets: a span dict with start / end is used as is; otherwise
the trigger is the first occurrence and an argument the occurrence nearest the trigger.
"""
import argparse
import json
from pathlib import Path

from calon.data.roots import ROOT

from ..eval.ee_eval import gold_sets, score
from .re_baselines import find_all

GIVEN = """Text:
{text}

Events (id: trigger [event type]):
{events}

Argument roles per event type:
{roles}

For every event above, list the text spans that fill its argument roles. Copy each span exactly as it appears in the text; use only the roles of that event's type.
Return JSON only: a list of {{"event": id, "role": "role", "text": "span"}}; return [] if there is none."""

E2E = """Text:
{text}

Event types:
{types}

Argument roles per event type:
{roles}

List every event of the types above that the text mentions: its trigger (the single word, occasionally a short phrase, that most directly expresses the event, copied exactly), its type, and the spans filling its argument roles (copied exactly; only the roles of that type).
Return JSON only: a list of {{"type": "event type", "trigger": "text", "arguments": [{{"role": "role", "text": "span"}}]}}; return [] if there is none."""


def span_near(text, s, anchor=None):
    if isinstance(s, dict) and "start" in s and "end" in s:
        return int(s["start"]), int(s["end"])
    if isinstance(s, dict):
        s = s.get("text", "")
    occ = find_all(text, str(s).strip())
    if not occ:
        return None
    return occ[0] if anchor is None else min(occ, key=lambda o: abs(o[0] - anchor))


def type_lines(d, name_only):
    return "\n".join(f'- "{k}"' + ("" if name_only or not v else f": {v}") for k, v in d.items())


def role_lines(meta, types, name_only):
    return "\n".join(f'- "{t}": ' + ", ".join(
        f'"{x}"' + ("" if name_only or not meta["roles"].get(x) else f" ({meta['roles'][x]})")
        for x in meta["event_roles"].get(t, [])) for t in types)


def to_gold_triggers(pt, pa, r):
    """Closest stand-in for the gold-trigger setting for GLiNER2, which cannot be given a trigger
    (like re_baselines.to_gold_entities): each predicted event is moved to the gold event of its
    type with the same trigger span, else one whose trigger lies inside the predicted trigger, else
    the nearest with the same trigger text (case-insensitive); its arguments follow it, and the
    arguments of events that match no gold event are dropped. Triggers count as given (gold)."""
    gt = gold_sets(r)[0]
    text = r["text"]

    def snap(s, e, t):
        same = [g for g in gt if g[2] == t]
        if (s, e, t) in gt:
            return (s, e, t)
        inside = [g for g in same if s <= g[0] and g[1] <= e]
        if inside:
            return min(inside, key=lambda g: g[0])
        txt = text[s:e].casefold()
        same = [g for g in same if text[g[0]:g[1]].casefold() == txt]
        return min(same, key=lambda g: abs(g[0] - s)) if same else None
    moved = {x: snap(*x) for x in pt}
    pa = {moved[a[:3]][:3] + a[3:] for a in pa if moved.get(a[:3]) and a[3:5] != moved[a[:3]][:2]}
    return gt, pa


def gold_trigger_proxy(res, ood):
    """Adds "given" scores to a GLiNER2 result from its saved end-to-end predictions."""
    manifest = json.loads((Path(ood) / "manifest.json").read_text())
    for name, v in res["sets"].items():
        rows = [json.loads(l) for l in open(Path(ood) / f"{name}.jsonl")]
        preds = [to_gold_triggers({tuple(x) for x in t}, {tuple(x) for x in a}, r)
                 for (t, a), r in zip(v["preds"]["end2end"], rows)]
        v["given"] = score(rows, preds, list(manifest["sets"][name]["event_types"]))
        print(name, "given AC (gold-trigger proxy)", round(v["given"]["AC"]["f1"], 4), flush=True)
    return res


def decode_records(ans, r, meta):
    """{event type: [record with "trigger" and one field per role]} -> (triggers, arguments)."""
    pt, pa = set(), set()
    for t, insts in (ans or {}).items():
        if t not in meta["event_types"]:
            continue
        for inst in insts if isinstance(insts, list) else [insts]:
            if not isinstance(inst, dict):
                continue
            tr = inst.get("trigger")
            tr = tr[0] if isinstance(tr, list) and tr else tr
            ts = span_near(r["text"], tr) if tr else None
            if ts is None:
                continue
            pt.add((ts[0], ts[1], t))
            for role in meta["event_roles"].get(t, []):
                vals = inst.get(role) or []
                for v in vals if isinstance(vals, list) else [vals]:
                    a = span_near(r["text"], v, ts[0])
                    if a and a != ts:
                        pa.add((ts[0], ts[1], t, a[0], a[1], role))
    return pt, pa


def run_structured(args, manifest):
    """GLiNER2 (extract_json) or GLiFormer (structuring head): one record schema per event type."""
    if args.backend == "gliformer":
        from gliformer import GLiFormer
        model = GLiFormer.from_pretrained(args.model, load_tokenizer=True).to("cuda").eval()
    else:
        try:
            from gliner2 import AutoExtractor as Loader
        except ImportError:
            from gliner2 import GLiNER2 as Loader
        model = Loader.from_pretrained(args.model)
        try:
            model = model.to("cuda")
        except Exception:
            pass
    res = {"model": args.model, "name_only": args.name_only, "sets": {}, "samples": []}
    for name, meta in manifest["sets"].items():
        if args.sets and name not in args.sets:
            continue
        rows = [json.loads(l) for l in open(Path(args.ood) / f"{name}.jsonl")]
        schema = {}
        for t, d in meta["event_types"].items():
            if args.backend == "gliformer":     # field names only (the template takes no field descriptions)
                schema[t] = ["trigger"] + list(meta["event_roles"].get(t, []))
                continue
            fields = ["trigger::str" + ("" if args.name_only or not d else f"::{d}")]
            for x in meta["event_roles"].get(t, []):
                rd = meta["roles"].get(x, "")
                fields.append(f"{x}::list" + ("" if args.name_only or not rd else f"::{rd}"))
            schema[t] = fields
        if args.backend == "gliformer":
            answers = model.structure([r["text"] for r in rows], schema, threshold=args.threshold)
        else:
            answers = []
            for r in rows:
                try:
                    answers.append(model.extract_json(r["text"], schema, threshold=args.threshold, include_spans=True))
                except TypeError:
                    answers.append(model.extract_json(r["text"], schema, threshold=args.threshold))
        res["samples"] += [{"text": r["text"][:200], "answer": json.loads(json.dumps(a, default=str))}
                           for r, a in zip(rows[:2], answers)]
        preds = [decode_records(a, r, meta) for a, r in zip(answers, rows)]
        res["sets"][name] = {"end2end": score(rows, preds, list(meta["event_types"])),
                             "preds": {"end2end": [[sorted(t), sorted(a)] for t, a in preds]}}
        print(name, "TC", round(res["sets"][name]["end2end"]["TC"]["f1"], 4),
              "AC", round(res["sets"][name]["end2end"]["AC"]["f1"], 4), flush=True)
    return gold_trigger_proxy(res, args.ood)


def run_llm(args, manifest):
    from ..eval.newtest import chat, last_json, llm
    engine = llm(args.model, max_len=16384, tp=args.tp)   # long WikiEvents chunks, many events
    res = {"model": args.model, "name_only": args.name_only, "sets": {}, "samples": []}
    for name, meta in manifest["sets"].items():
        rows = [json.loads(l) for l in open(Path(args.ood) / f"{name}.jsonl")]
        res["sets"][name] = {}
        for mode in ("given", "end2end"):
            prompts = []
            for r in rows:
                if mode == "given":
                    evs = "\n".join(f'{k}: {r["text"][e["trigger"][0]:e["trigger"][1]]} [{e["type"]}]'
                                    for k, e in enumerate(r["events"]))
                    types = list(dict.fromkeys(e["type"] for e in r["events"]))
                    prompts.append(GIVEN.format(text=r["text"], events=evs or "(none)",
                                                roles=role_lines(meta, types, args.name_only)))
                else:
                    prompts.append(E2E.format(text=r["text"], types=type_lines(meta["event_types"], args.name_only),
                                              roles=role_lines(meta, list(meta["event_types"]), args.name_only)))
            outs = chat(engine, prompts, 6144)
            preds = []
            for r, o in zip(rows, outs):
                js = last_json(o)
                items = js if isinstance(js, list) else []
                pt, pa = set(), set()
                if mode == "given":
                    pt = gold_sets(r)[0]
                    for it in items:
                        try:
                            e = r["events"][int(it["event"])]
                        except (KeyError, ValueError, IndexError, TypeError):
                            continue
                        if it.get("role") not in meta["event_roles"].get(e["type"], []):
                            continue
                        a = span_near(r["text"], it.get("text", ""), e["trigger"][0])
                        if a and list(a) != e["trigger"]:
                            pa.add((e["trigger"][0], e["trigger"][1], e["type"], a[0], a[1], it["role"]))
                else:
                    for it in items:
                        if not isinstance(it, dict) or it.get("type") not in meta["event_types"]:
                            continue
                        ts = span_near(r["text"], it.get("trigger", ""))
                        if ts is None:
                            continue
                        pt.add((ts[0], ts[1], it["type"]))
                        for x in it.get("arguments") or []:
                            if not isinstance(x, dict) or x.get("role") not in meta["event_roles"].get(it["type"], []):
                                continue
                            a = span_near(r["text"], x.get("text", ""), ts[0])
                            if a and a != ts:
                                pa.add((ts[0], ts[1], it["type"], a[0], a[1], x["role"]))
                preds.append((pt, pa))
                if len(res["samples"]) < 4:
                    res["samples"].append({"mode": mode, "text": r["text"][:200], "answer": o[:500]})
            res["sets"][name][mode] = score(rows, preds, list(meta["event_types"]))
            res["sets"][name].setdefault("preds", {})[mode] = [[sorted(t), sorted(a)] for t, a in preds]
            print(name, mode, "AC", round(res["sets"][name][mode]["AC"]["f1"], 4), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gliner2", "gliformer", "llm", "proxy"],
                    help="proxy: add the gold-trigger proxy to an existing GLiNER2 result (--model is that file)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ood", default=str(ROOT / "data/ood_ee"))
    ap.add_argument("--sets", nargs="*", help="gliner2 / gliformer: only these sets (default: all)")
    ap.add_argument("--name_only", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    if args.backend == "proxy":
        res = gold_trigger_proxy(json.loads(Path(args.model).read_text()), args.ood)
    else:
        res = (run_llm if args.backend == "llm" else run_structured)(args, manifest)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
