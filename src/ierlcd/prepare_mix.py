"""Round-1 source mixture: FiNERWeb + Few-NERD + Pile-NER (type, definition) + MASSIVE.

Stage `inventory` writes the Few-NERD / MASSIVE label inventories for
label_bank.py. Stage `build` writes data/mix/<source>/<split>.jsonl in the
unified format (SPEC 2.2) plus manifest.json.

Query status per source:
  finerweb   positive only (schema lists only labels with mentions)
  fewnerd    all 8 coarse + 66 fine types; unmentioned -> scope_assumed_absent
             (dataset_closed_world: Few-NERD is annotated over its full ontology)
  massive    all train slot types; unmentioned -> scope_assumed_absent
  pile       type / definition questions; [] -> explicit_absent
             (upstream negative sampling); answers aligned to every
             word-boundary occurrence; any answer not found -> unknown;
             a label asked with both [] and mentions keeps only the positive
Train rows whose text overlaps any eval text (exact normalized match, or
word-5-gram containment >= 0.8 in either direction) are dropped. Eval files
carry a fixed query subset (positives + seeded absent sample, SPEC 2.5 cap).
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from .prepare_data import SPLITS, norm_text, sha, split_groups, word_ngrams

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
RAW = ROOT / "raw"
FEWNERD = RAW / "DFKI-SLT__few-nerd/main/supervised"
PILE_TYPE = RAW / "Universal-NER__Pile-NER-type/main/train.json"
PILE_DEF = RAW / "Universal-NER__Pile-NER-definition/main/train.json"
MASSIVE = RAW / "AmazonScience__massive/parquet/en-US"
EVAL_SPLITS = ("dev_select", "dev_calib", "test")
FINE_WORDS = {
    "broadcastprogram": "broadcast program", "writtenart": "written art",
    "sportsfacility": "sports facility", "militaryconflict": "military conflict",
    "sportsevent": "sports event", "GPE": "geopolitical entity", "bodiesofwater": "body of water",
    "governmentagency": "government agency", "politicalparty": "political party",
    "showorganization": "show organization", "sportsleague": "sports league",
    "sportsteam": "sports team", "astronomything": "astronomical object",
    "biologything": "biological entity", "chemicalthing": "chemical",
    "educationaldegree": "educational degree", "livingthing": "living thing",
}
SLOT_RE = re.compile(r"\[(\w+) : ([^\]]*)\]")
QUESTION_RE = re.compile(r"^What describes (.*) in the text\?$", re.S)


# ---------------------------------------------------------------- query sampling
def absent_cap(n_pos_labels):
    return min(8, max(2, n_pos_labels)) if n_pos_labels else 8


def sample_queries(queries, rng):
    """SPEC 2.5 training/eval query distribution: all positives, explicit absent
    before scoped absent up to the cap, unknown queries kept (masked). Shuffled."""
    by = collections.defaultdict(list)
    for q in queries:
        by[q["status"]].append(q)
    exp, sco = list(by["explicit_absent"]), list(by["scope_assumed_absent"])
    rng.shuffle(exp)
    rng.shuffle(sco)
    out = by["positive"] + (exp + sco)[:absent_cap(len(by["positive"]))] + by["unknown"]
    rng.shuffle(out)
    return out


def row_seed(row_id):
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------- Few-NERD
def feature_names(path, column):
    meta = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"])
    def find(node):
        if isinstance(node, dict):
            if "names" in node:
                return node["names"]
            for v in node.values():
                r = find(v)
                if r:
                    return r
        return None
    return find(meta["info"]["features"][column])


def fewnerd_name(raw):
    if "-" not in raw:
        return "other entity" if raw == "other" else raw
    parent, fine = raw.split("-", 1)
    if fine == "other":
        return f"other {parent}"
    return " / ".join(FINE_WORDS.get(p, p) for p in fine.split("/"))


def fewnerd_inventory():
    path = FEWNERD / "train-00000-of-00001.parquet"
    coarse = [n for n in feature_names(path, "ner_tags") if n != "O"]
    fine = [n for n in feature_names(path, "fine_ner_tags") if n != "O"]
    inv = [{"raw": c, "name": fewnerd_name(c), "parent": None} for c in coarse]
    inv += [{"raw": f, "name": fewnerd_name(f), "parent": fewnerd_name(f.split("-")[0])} for f in fine]
    names = [x["name"] for x in inv]
    assert len(names) == len(set(names)), collections.Counter(names).most_common(3)
    return inv


def tokens_to_text(tokens):
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1
    return " ".join(tokens), starts


def adapt_fewnerd(split, bank, stats):
    path = FEWNERD / f"{split}-00000-of-00001.parquet"
    coarse_names = feature_names(path, "ner_tags")
    fine_names = feature_names(path, "fine_ner_tags")
    labels = [(raw, bank[raw]) for raw in coarse_names[1:] + fine_names[1:]]
    rows = []
    for rec in pq.read_table(path).to_pylist():
        text, starts = tokens_to_text(rec["tokens"])
        ents, k, tags = [], 0, rec["fine_ner_tags"]
        while k < len(tags):                       # IO tagging: a run of one fine tag = one entity
            if tags[k] == 0:
                k += 1
                continue
            j = k
            while j + 1 < len(tags) and tags[j + 1] == tags[k]:
                j += 1
            s, e = starts[k], starts[j] + len(rec["tokens"][j])
            fine = fine_names[tags[k]]
            ents.append({"start": s, "end": e, "label": bank[fine]["name"]})
            ents.append({"start": s, "end": e, "label": bank[fine.split("-")[0]]["name"]})
            k = j + 1
        present = {e["label"] for e in ents}
        queries = [{"label": v["name"], "description": v["description"],
                    "status": "positive" if v["name"] in present else "scope_assumed_absent",
                    "description_origin": v["description_origin"],
                    "supervision_origin": "source" if v["name"] in present else "dataset_closed_world"}
                   for _, v in labels]
        ents = [dict(t) for t in {tuple(sorted(e.items())) for e in ents}]
        rows.append({"id": f"fewnerd:{split}:{rec['id']}", "source": "fewnerd", "text": text,
                     "queries": queries, "entities": sorted(ents, key=lambda e: (e["start"], e["label"]))})
    stats[f"fewnerd_{split}_rows"] = len(rows)
    return rows


# ---------------------------------------------------------------- MASSIVE
def parse_massive(annot, utt):
    text, ents, pos, last = [], [], 0, 0
    for m in SLOT_RE.finditer(annot):
        text.append(annot[last:m.start()])
        pos += m.start() - last
        value = m.group(2)
        ents.append({"start": pos, "end": pos + len(value), "slot": m.group(1)})
        text.append(value)
        pos += len(value)
        last = m.end()
    text.append(annot[last:])
    text = "".join(text)
    return (text, ents) if text == utt else (None, None)


def massive_inventory():
    slots = set()
    for rec in pq.read_table(MASSIVE / "train/0000.parquet").to_pylist():
        slots.update(m.group(1) for m in SLOT_RE.finditer(rec["annot_utt"]))
    return [{"raw": s, "name": s.replace("_", " "), "parent": None} for s in sorted(slots)]


def adapt_massive(split, bank, stats):
    rows = []
    for rec in pq.read_table(MASSIVE / f"{split}/0000.parquet").to_pylist():
        text, ents = parse_massive(rec["annot_utt"], rec["utt"])
        if text is None:
            stats["massive_parse_mismatch"] += 1
            continue
        kept = []
        for e in ents:
            if e["slot"] not in bank:
                stats["massive_unsupported_slot_mentions"] += 1
                continue
            kept.append({"start": e["start"], "end": e["end"], "label": bank[e["slot"]]["name"]})
        present = {e["label"] for e in kept}
        queries = [{"label": v["name"], "description": v["description"],
                    "status": "positive" if v["name"] in present else "scope_assumed_absent",
                    "description_origin": v["description_origin"],
                    "supervision_origin": "source" if v["name"] in present else "dataset_closed_world"}
                   for v in bank.values()]
        rows.append({"id": f"massive:{split}:{rec['id']}", "source": "massive", "text": text,
                     "queries": queries, "entities": kept})
    stats[f"massive_{split}_rows"] = len(rows)
    return rows


# ---------------------------------------------------------------- Pile-NER
def find_all(text, s):
    """Every occurrence of s whose alphanumeric ends are not inside a word."""
    out, k = [], text.find(s)
    while k >= 0:
        e = k + len(s)
        left_ok = not (s[0].isalnum() and k > 0 and text[k - 1].isalnum())
        right_ok = not (s[-1].isalnum() and e < len(text) and text[e].isalnum())
        if left_ok and right_ok:
            out.append((k, e))
        k = text.find(s, k + 1)
    return out


def adapt_pile(path, kind, stats):
    rows = []
    for rec in json.load(open(path)):
        conv = rec["conversations"]
        first = conv[0]["value"]
        if not first.startswith("Text: "):
            stats["pile_bad_text_turn"] += 1
            continue
        text = first[len("Text: "):]
        asked = collections.OrderedDict()
        for k in range(2, len(conv) - 1, 2):
            m = QUESTION_RE.match(conv[k]["value"].strip())
            if not m or conv[k]["from"] != "human":
                stats["pile_bad_question"] += 1
                continue
            try:
                answers = json.loads(conv[k + 1]["value"])
                assert isinstance(answers, list) and all(isinstance(a, str) for a in answers)
            except Exception:
                stats["pile_bad_answer"] += 1
                continue
            label = m.group(1).strip()
            prev = asked.get(label)
            answers = [a.strip() for a in answers if a.strip()]
            if prev is not None:
                stats["pile_repeated_label"] += 1
                if bool(prev) != bool(answers):
                    stats["pile_pos_neg_conflict"] += 1
                answers = sorted(set(prev) | set(answers))
            asked[label] = answers
        queries, ents = [], []
        for label, answers in asked.items():
            q = {"label": label, "description": "",
                 "description_origin": "definition_as_label" if kind == "definition" else "name_only"}
            if not answers:
                q.update(status="explicit_absent", supervision_origin="upstream_negative_sampling")
            else:
                spans = [find_all(text, a) for a in answers]
                if all(spans):
                    q.update(status="positive", supervision_origin="source_string_all_occurrences")
                    ents += [{"start": s, "end": e, "label": label} for sp in spans for s, e in sp]
                else:
                    q.update(status="unknown", supervision_origin="unaligned_answer")
                    stats["pile_unaligned_queries"] += 1
            queries.append(q)
        ents = [dict(t) for t in {tuple(sorted(e.items())) for e in ents}]
        rows.append({"id": f"pile_{kind}:train:{rec['id']}", "source": "pile", "text": text,
                     "queries": queries, "entities": sorted(ents, key=lambda e: (e["start"], e["label"]))})
    stats[f"pile_{kind}_rows"] = len(rows)
    return rows


# ---------------------------------------------------------------- dedup / eval subsets
def contamination_filter(rows, eval_texts, stats, key, thr=0.8, df_cap=2000):
    exact = {sha(norm_text(t)) for t in eval_texts}
    grams = [word_ngrams(t) for t in eval_texts]
    index = collections.defaultdict(list)
    for k, g in enumerate(grams):
        for x in g:
            index[x].append(k)
    capped = {x for x, v in index.items() if len(v) > df_cap}
    for x in capped:
        del index[x]
    stats[f"{key}_capped_grams"] = len(capped)
    kept = []
    for r in rows:
        if sha(norm_text(r["text"])) in exact:
            stats[f"{key}_drop_exact"] += 1
            continue
        g = word_ngrams(r["text"])
        hit = False
        if g:
            shared = collections.Counter(k for x in g for k in index.get(x, ()))
            hit = any(s / len(grams[k]) >= thr or s / len(g) >= thr for k, s in shared.items())
        if hit:
            stats[f"{key}_drop_near"] += 1
            continue
        kept.append(r)
    return kept


def eligible(row):
    return any(q["status"] != "unknown" for q in row["queries"])


def fixed_eval_queries(rows):
    out = []
    for r in rows:
        r = dict(r)
        r["queries"] = sample_queries(r["queries"], random.Random(row_seed(r["id"])))
        out.append(r)
    return out


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def status_counts(rows):
    return dict(collections.Counter(q["status"] for r in rows for q in r["queries"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inventory", "build"], required=True)
    ap.add_argument("--out", default=str(ROOT / "data/mix"))
    ap.add_argument("--bank", help="label_bank.py output (build stage)")
    ap.add_argument("--finerweb", default=str(ROOT / "data/finerweb"))
    ap.add_argument("--ood", default=str(ROOT / "data/ood"))
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=512, help="text window (tokens)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "inventory":
        inv = {"fewnerd": fewnerd_inventory(), "massive": massive_inventory()}
        (out / "inventory.json").write_text(json.dumps(inv, indent=2))
        print({k: len(v) for k, v in inv.items()})
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)
    bank = json.loads(Path(args.bank).read_text())["bank"]
    stats = collections.Counter()
    splits = {}

    load = lambda p: [json.loads(line) for line in open(p)]
    splits["finerweb"] = {s: load(Path(args.finerweb) / f"{s}.jsonl") for s, _ in SPLITS}
    fn = {s: adapt_fewnerd(s, bank["fewnerd"], stats) for s in ("train", "validation", "test")}
    ms = {s: adapt_massive(s, bank["massive"], stats) for s in ("train", "validation", "test")}
    for name, d in (("fewnerd", fn), ("massive", ms)):
        val = d["validation"]
        splits[name] = {"train": d["train"], "dev_select": val[0::2], "dev_calib": val[1::2],
                        "test": d["test"]}

    pile = adapt_pile(PILE_TYPE, "type", stats) + adapt_pile(PILE_DEF, "definition", stats)
    lengths = tok([r["text"] for r in pile], add_special_tokens=False)["input_ids"]
    pile = [r for r, ids in zip(pile, lengths) if len(ids) <= args.max_tokens]
    stats[f"pile_drop_over_{args.max_tokens}_tokens"] = len(lengths) - len(pile)
    by_hash = collections.defaultdict(list)
    for i, r in enumerate(pile):
        by_hash[sha(norm_text(r["text"]))].append(i)
    assign = split_groups(list(by_hash.values()), len(pile), args.seed)
    splits["pile"] = {s: [pile[i] for i in range(len(pile)) if assign[i] == s] for s, _ in SPLITS}

    for src in splits:
        for s in splits[src]:
            before = len(splits[src][s])
            splits[src][s] = [r for r in splits[src][s] if eligible(r)]
            stats[f"{src}_{s}_drop_no_known_query"] = before - len(splits[src][s])

    eval_texts = [r["text"] for src in splits for s in EVAL_SPLITS for r in splits[src][s]]
    ood_manifest = json.loads((Path(args.ood) / "manifest.json").read_text())
    for name in ood_manifest["sets"]:
        eval_texts += [r["text"] for r in load(Path(args.ood) / f"{name}.jsonl")]
    stats["eval_texts"] = len(eval_texts)
    for src in splits:
        splits[src]["train"] = contamination_filter(splits[src]["train"], eval_texts, stats, src)

    fetched = json.loads((RAW / "fetch_summary.json").read_text())
    manifest = {"sources": {}, "bank": args.bank, "seed": args.seed,
                "raw_revisions": {repo: {"sha": v["sha"], "parquet_sha": v.get("parquet_sha")}
                                  for repo, v in fetched.items()},
                "eval_query_policy": "positives + explicit-then-scoped absent, cap min(8,max(2,#pos)), "
                                     "seeded by sha256(row id); unknown kept (masked)",
                "language_id": "not applied", "pile_near_dup_within": "exact normalized hash only"}
    for src, d in splits.items():
        info = {}
        for s, rows in d.items():
            rows_out = rows if s == "train" else fixed_eval_queries(rows)
            write(out / src / f"{s}.jsonl", rows_out)
            info[s] = {"rows": len(rows_out), "entities": sum(len(r["entities"]) for r in rows_out),
                       "query_status": status_counts(rows_out)}
        info["description_origin"] = dict(collections.Counter(
            q["description_origin"] for r in d["train"] for q in r["queries"]))
        manifest["sources"][src] = info
    manifest["stats"] = dict(stats)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"stats": dict(stats), "sources": {k: {s: v[s]["rows"] for s in v if s != "description_origin"}
                                                        for k, v in manifest["sources"].items()}}, indent=1))


if __name__ == "__main__":
    main()
