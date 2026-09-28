"""FiNERWeb -> unified rows (SPEC 2.2), group dedup and 80/10/5/5 split (SPEC 2.5).

Only the FiNERWeb adapter exists so far. Every FiNERWeb schema label has a
positive in its row, so all queries are `positive`; the source provides no
absent queries and none are invented here.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

FINERWEB_REPO = "mneb/finerweb-english"
FINERWEB_REV = "46a29532929a2cd439bf5c3df2281d7760d28731"
NORM_VERSION = "v1-nfkc-ws-lower"
SPLITS = [("train", 0.80), ("dev_select", 0.10), ("dev_calib", 0.05), ("test", 0.05)]


def norm_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_ngrams(text, n=5):
    words = norm_text(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def schema_labels(schema):
    """schema may be {"entities": [...]} or a list; items are dicts or strings."""
    items = schema.get("entities", []) if isinstance(schema, dict) else (schema or [])
    out = []
    for it in items:
        if isinstance(it, str):
            out.append((it, ""))
        else:
            out.append((it["label"], it.get("description") or ""))
    return out


def adapt_finerweb(path):
    audit = collections.Counter()
    rows = []
    with open(path) as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            text = rec["input"]
            ents = rec["output"].get("entities", {}) or {}
            queries, seen = [], {}
            for label, desc in schema_labels(rec["schema"]):
                if label in seen:
                    audit["schema_duplicate_label"] += 1
                    if seen[label] != desc:
                        audit["schema_duplicate_label_desc_differs"] += 1
                    continue
                seen[label] = desc
                queries.append({
                    "label": label, "description": desc, "status": "positive",
                    "description_origin": "source" if desc else "name_only",
                    "supervision_origin": "source"})
            entities = []
            for label, mentions in ents.items():
                if label not in seen:
                    audit["entity_label_not_in_schema"] += len(mentions)
                    continue
                for m in mentions:
                    audit["annotations"] += 1
                    if text[m["start"]:m["end"]] != m["text"]:
                        audit["offset_mismatch"] += 1
                        continue
                    entities.append({"start": m["start"], "end": m["end"], "label": label})
            pos_labels = {e["label"] for e in entities}
            for q in queries:
                if q["label"] not in pos_labels:
                    # SPEC: FiNERWeb label absence is not evidence of absence.
                    q["status"] = "unknown"
                    audit["schema_label_without_positive"] += 1
            audit["rows"] += 1
            audit["queries"] += len(queries)
            if not entities:
                audit["rows_without_entities"] += 1
            rows.append({"id": f"finerweb:train:{idx}", "source": "finerweb",
                         "text": text, "queries": queries, "entities": entities})
    return rows, audit


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def group_rows(rows, jaccard=0.8):
    dsu, stats = DSU(len(rows)), collections.Counter()
    for key, fn in (("exact", lambda t: sha(t)), ("normalized", lambda t: sha(norm_text(t)))):
        first = {}
        for i, r in enumerate(rows):
            h = fn(r["text"])
            if h in first:
                stats[f"{key}_dup_rows"] += 1
                dsu.union(first[h], i)
            else:
                first[h] = i
    grams = [word_ngrams(r["text"]) for r in rows]
    index = collections.defaultdict(list)
    for i, g in enumerate(grams):
        for x in g:
            index[x].append(i)
    stats["max_gram_df"] = max((len(v) for v in index.values()), default=0)
    for i, g in enumerate(grams):
        if not g:
            stats["rows_under_5_words"] += 1
            continue
        shared = collections.Counter(j for x in g for j in index[x] if j > i)
        for j, s in shared.items():
            if s / (len(g) + len(grams[j]) - s) >= jaccard:
                if dsu.find(i) != dsu.find(j):
                    stats["near_dup_unions"] += 1
                dsu.union(i, j)
    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[dsu.find(i)].append(i)
    stats["groups"] = len(groups)
    stats["rows_in_multi_row_groups"] = sum(len(v) for v in groups.values() if len(v) > 1)
    stats["largest_group"] = max(len(v) for v in groups.values())
    return list(groups.values()), stats


def split_groups(groups, n_rows, seed):
    order = sorted(groups, key=lambda g: min(g))
    random.Random(seed).shuffle(order)
    out, k, acc, bounds = {}, 0, 0, []
    for name, frac in SPLITS:
        acc += frac
        bounds.append((name, acc * n_rows))
    done = 0
    for g in order:
        while done >= bounds[k][1] and k < len(bounds) - 1:
            k += 1
        for i in g:
            out[i] = bounds[k][0]
        done += len(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finerweb", required=True, help="path to finerweb train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, audit = adapt_finerweb(args.finerweb)
    groups, gstats = group_rows(rows)
    assign = split_groups(groups, len(rows), args.seed)
    group_of = {i: min(g) for g in groups for i in g}
    by_split = collections.defaultdict(list)
    for i, r in enumerate(rows):
        r["group"] = f"finerweb:{group_of[i]}"
        by_split[assign[i]].append(r)
    for name, _ in SPLITS:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in by_split[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # leakage check: no group spans two splits
    seen = {}
    for name, rs in by_split.items():
        for r in rs:
            assert seen.setdefault(r["group"], name) == name, r["group"]
    status = collections.Counter(q["status"] for r in rows for q in r["queries"])
    desc = collections.Counter(q["description_origin"] for r in rows for q in r["queries"])
    manifest = {
        "source": {"repo": FINERWEB_REPO, "revision": FINERWEB_REV, "file": "data/train.jsonl"},
        "normalization": NORM_VERSION, "near_dup": "word-5gram jaccard>=0.8, connected groups",
        "language_id": "not applied (FiNERWeb-only preliminary run)",
        "seed": args.seed, "audit": dict(audit), "grouping": dict(gstats),
        "query_status": dict(status), "description_origin": dict(desc),
        "splits": {n: {"rows": len(by_split[n]),
                       "groups": len({r["group"] for r in by_split[n]}),
                       "entities": sum(len(r["entities"]) for r in by_split[n])}
                   for n, _ in SPLITS},
        "labels_train": len({q["label"] for r in by_split["train"] for q in r["queries"]}),
        "labels_test_unseen_in_train": len(
            {q["label"] for r in by_split["test"] for q in r["queries"]}
            - {q["label"] for r in by_split["train"] for q in r["queries"]}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
