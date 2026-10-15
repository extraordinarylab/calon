"""Statistics of a training mix directory (data/mixN): per source the task, mix share, rows per
split, entity / relation counts, label inventory size, and for generated sources the genres
and domains of their generation requests (synth2/*_requests.jsonl). Writes a JSON summary and
prints a short table."""
import argparse
import collections
import glob
import json
from pathlib import Path

from calon.data.roots import ROOT

S2 = ROOT / "synth2"


def requests_index():
    """request id -> (genre, domain or topic or item)."""
    idx = {}
    for f in glob.glob(str(S2 / "*_requests.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if "genre" in r or "item" in r:
                    idx[r["id"]] = (r.get("genre") or r.get("style"),
                                    r.get("domain") or r.get("topic") or r.get("item"))
    return idx


def req_of(row_id, idx):
    parts = row_id.split(":")
    for a in range(len(parts)):
        for b in range(len(parts), a, -1):
            k = ":".join(parts[a:b])
            if k in idx:
                return idx[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=str(ROOT / "data/mix10"))
    ap.add_argument("--mixspec", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    share = dict((s.split(":")[0], float(s.split(":")[1])) for s in args.mixspec.split(",") if s)
    idx = requests_index()
    res = {}
    for d in sorted(Path(args.mix).iterdir()):
        if not d.is_dir():
            continue
        src = d.name
        s = {"share": share.get(src), "task": "EE" if src in ("synth_gee", "synth_cee", "synth_nest") else None}
        labels, rlabels, genres, domains = collections.Counter(), collections.Counter(), collections.Counter(), \
            collections.Counter()
        kinds = collections.Counter()
        for sp in ("train", "dev_select", "dev_calib", "test"):
            p = d / f"{sp}.jsonl"
            if not p.exists():
                continue
            n = ents = rels = 0
            with open(p) as fh:
                for line in fh:
                    r = json.loads(line)
                    n += 1
                    ents += len(r.get("entities", []))
                    rels += len(r.get("relations", []))
                    if sp == "train":
                        labels.update(q["label"] for q in r.get("queries", []))
                        rlabels.update(q["label"] for q in r.get("rel_queries", []))
                        kinds["relation rows" if r.get("rel_queries") else "entity rows"] += 1
                        g = req_of(r.get("id", ""), idx)
                        if g:
                            genres[g[0]] += 1
                            domains[g[1]] += 1
            s[sp] = {"rows": n, "entities": ents, "relations": rels}
        s["row_kinds"] = dict(kinds)
        s["task"] = s["task"] or ("RE" if kinds["relation rows"] else "NER")
        s["entity_labels"], s["relation_labels"] = len(labels), len(rlabels)
        s["top_entity_labels"] = [x for x, _ in labels.most_common(12)]
        s["top_relation_labels"] = [x for x, _ in rlabels.most_common(12)]
        s["genres"], s["domains"] = len(genres), len(domains)
        s["top_genres"] = genres.most_common(6)
        s["top_domains"] = domains.most_common(6)
        res[src] = s
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for src, s in res.items():
        tr = s.get("train", {})
        print(f"{src:14s} {s['task']:3s} share {s['share']} train {tr.get('rows')} ents {tr.get('entities')} "
              f"rels {tr.get('relations')}"
              f" | labels {s['entity_labels']}/{s['relation_labels']} | genres {s['genres']} domains {s['domains']}"
              f" | {s['row_kinds']}")


if __name__ == "__main__":
    main()
