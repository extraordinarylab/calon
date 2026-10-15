"""Seed pools for target-seeded synthesis (calon.synth.seed_synth), one per evaluation set.

Each pool holds what a generator may see about one evaluation set:
  labels       the set's own types with the descriptions its evaluation rows carry (NER, RE),
               relation types (RE), event types with their roles (EE)
  labeled      annotated examples from the benchmark's train / validation splits, converted by the
               set's own builder (same label names as the evaluation rows); never the test split
  texts        the evaluation texts themselves, as unlabeled style references (text only: their
               labels are never shown to a generator)
  genre        one line naming the text type, for the prompt
Sets without train / validation data (GENTLE, SPY, M2E2, SPEED++, LitBank, the *_new sets...)
get texts and labels only.

Output: $CALON_ROOT/seed_pools/<set>.json and summary.json (pool sizes per set).
"""
import argparse
import collections
import json
import random

from calon.data.roots import ROOT

OUT = ROOT / "seed_pools"
D = ROOT / "data"
MAX_LABELED = 3000                 # examples kept per pool (sampled with a fixed seed)
MAX_TEXTS = 3000

# (evaluation group, task, genre). Genres describe the benchmark's text type only.
SETS = {
    "crossner_ai": ("ood", "NER", "Wikipedia sentences about artificial intelligence"),
    "crossner_literature": ("ood", "NER", "Wikipedia sentences about literature and writers"),
    "crossner_music": ("ood", "NER", "Wikipedia sentences about music, musicians and albums"),
    "crossner_politics": ("ood", "NER", "Wikipedia sentences about politics, parties and elections"),
    "crossner_science": ("ood", "NER", "Wikipedia sentences about the natural sciences"),
    "mit_movie": ("ood", "NER", "spoken queries to a movie search assistant, lowercase without punctuation"),
    "mit_restaurant": ("ood", "NER", "spoken queries to a restaurant search assistant, lowercase without punctuation"),
    "crossner_ai_new": ("ood_new", "NER", "recent Wikipedia passages about artificial intelligence"),
    "crossner_literature_new": ("ood_new", "NER", "recent Wikipedia passages about literature and writers"),
    "crossner_music_new": ("ood_new", "NER", "recent Wikipedia passages about music, musicians and albums"),
    "crossner_politics_new": ("ood_new", "NER", "recent Wikipedia passages about politics, parties and elections"),
    "crossner_science_new": ("ood_new", "NER", "recent Wikipedia passages about the natural sciences"),
    "mit_movie_new": ("ood_new", "NER", "spoken queries to a movie search assistant, lowercase without punctuation"),
    "mit_restaurant_new": ("ood_new", "NER",
                           "spoken queries to a restaurant search assistant, lowercase without punctuation"),
    "pii_ai4privacy": ("ood_pii", "NER",
                       "short personal texts, forms and messages that contain personal information"),
    "spy_legal": ("ood_spy", "NER",
                  "legal documents (contracts, court filings, letters) that contain personal information"),
    "spy_medical": ("ood_spy", "NER",
                    "medical documents (clinical notes, letters, records) that contain personal information"),
    "buster": ("ood_x", "NER", "excerpts of SEC filings about mergers and acquisitions"),
    "dynamicner_en": ("ood_x", "NER", "encyclopedic sentences with fine-grained named entities"),
    "scier": ("ood_x", "NER", "sentences from machine learning and NLP research papers"),
    "scier_ood": ("ood_x", "NER", "sentences from research papers outside machine learning and NLP"),
    "gentle": ("ood_x2", "NER", "unusual English genres: dictionary entries, esports commentary, legal texts, "
                                "medical notes, poetry, mathematical proofs, course syllabi, forum threads"),
    "biored": ("ood_x2", "NER", "PubMed biomedical abstracts"),
    "measeval": ("ood_x2", "NER", "scientific article passages that report measurements"),
    "crossre_ai": ("ood_re", "RE", "Wikipedia sentences about artificial intelligence"),
    "crossre_literature": ("ood_re", "RE", "Wikipedia sentences about literature and writers"),
    "crossre_music": ("ood_re", "RE", "Wikipedia sentences about music, musicians and albums"),
    "crossre_news": ("ood_re", "RE", "news sentences"),
    "crossre_politics": ("ood_re", "RE", "Wikipedia sentences about politics, parties and elections"),
    "crossre_science": ("ood_re", "RE", "Wikipedia sentences about the natural sciences"),
    "biored_re": ("ood_re_x2", "RE", "PubMed biomedical abstracts"),
    "dwie_re": ("ood_re_x2", "RE", "Deutsche Welle English news articles"),
    "retacred": ("ood_re_x2", "RE", "news and web sentences about people and organisations"),
    "aste_14lap": ("ood_re_x2", "RE", "laptop reviews"),
    "aste_14res": ("ood_re_x2", "RE", "restaurant reviews"),
    "aste_15res": ("ood_re_x2", "RE", "restaurant reviews"),
    "aste_16res": ("ood_re_x2", "RE", "restaurant reviews"),
    "gap": ("ood_re_x2", "RE", "Wikipedia passages about people, with pronouns"),
    "litbank": ("ood_re_x2", "RE", "excerpts of English novels"),
    "maven_ere": ("ood_re_x2", "RE", "Wikipedia articles about historical events"),
    "measeval_re": ("ood_re_x2", "RE", "scientific article passages that report measurements"),
    "casie": ("ood_ee", "EE", "cybersecurity news"),
    "mlee": ("ood_ee", "EE", "biomedical research abstracts"),
    "m2e2": ("ood_ee", "EE", "Voice of America news"),
    "speed_mpox": ("ood_ee2", "EE", "social media posts about the mpox epidemic"),
    "speed_zika": ("ood_ee2", "EE", "social media posts about the Zika epidemic"),
    "sentivent": ("ood_ee2", "EE", "company-specific economic news"),
    "mailex": ("ood_ee2", "EE", "business email threads"),
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compact(row):
    """An evaluation-format row -> the fields a prompt needs."""
    out = {"text": row["text"], "entities": [[e["start"], e["end"], e["label"]] for e in row.get("entities", [])]}
    if "relations" in row:
        out["relations"] = [[*x["head"], *x["tail"], x["label"]] for x in row["relations"]]
    if "events" in row:
        out["events"] = row["events"]
    return out


def labels_of(rows):
    """Types asked by the evaluation rows, with their descriptions, in first-seen order."""
    ent, rel = {}, {}
    for r in rows:
        for q in r.get("queries", []):
            ent.setdefault(q["label"], q.get("description") or "")
        for q in r.get("rel_queries", []):
            rel.setdefault(q["label"], q.get("description") or "")
    return ent, rel


def mneb_rows(path, kind):
    """mneb / CrossRE style rows ({input, output{entities, relations}}) -> evaluation format."""
    out = []
    for rec in read_jsonl(path):
        text = rec["input"]
        ents = [{"start": m["start"], "end": m["end"], "label": lab}
                for lab, ms in (rec["output"].get("entities") or {}).items() for m in ms
                if text[m["start"]:m["end"]] == m["text"]]
        row = {"text": text, "entities": ents}
        if kind == "RE":
            row["relations"] = [{"head": [m["head"]["start"], m["head"]["end"]],
                                 "tail": [m["tail"]["start"], m["tail"]["end"]], "label": lab}
                                for lab, ms in (rec["output"].get("relations") or {}).items() for m in ms]
        out.append(row)
    return out


def bank_rows(raw, src, bank, kind):
    """Builder rows (codes) -> evaluation format with the label bank's names, as each builder's finish."""
    b, rb = bank[src], bank.get(f"{src}_rel", {})
    out = []
    for r in raw:
        if any(e["code"] not in b for e in r["ents"]):
            continue
        row = {"text": r["text"], "entities": [{"start": e["start"], "end": e["end"], "label": b[e["code"]]["name"]}
                                               for e in r["ents"]]}
        if kind == "RE":
            if any(x["label"] not in rb for x in r.get("rels", [])):
                continue
            row["relations"] = [dict(x, label=rb[x["label"]]["name"]) for x in r.get("rels", [])]
        out.append(row)
    return out


def bank_events(raw, src, bank):
    eb, rb = bank[src], bank[f"{src}_role"]
    return [{"text": r["text"], "events": [{"type": eb[e["type"]]["name"], "trigger": e["trigger"],
                                             "args": [{"role": rb[a["role"]]["name"], "span": a["span"]}
                                                      for a in e["args"] if a["role"] in rb]}
                                            for e in r["events"] if e["type"] in eb]} for r in raw]


def labeled_pools(stats):
    """{set: [labeled rows]} from train / validation splits that exist on disk."""
    from . import prepare_evalsets as ev1
    from . import prepare_evalsets2 as ev2
    pools = collections.defaultdict(list)

    def guard(name, fn):
        try:
            fn()
        except Exception as exc:                  # a missing or odd split costs that set its seeds only
            stats[f"{name}_missing"] = repr(exc)[:160]
            print(f"{name}: no labeled seeds ({repr(exc)[:160]})", flush=True)

    # CrossNER / MIT train + CrossNER validation (benchmarks/, from calon-data)
    def crossner_mit():
        for split_dir in ("crossner_mit_train", "crossner_validation"):
            for p in sorted((ROOT / "benchmarks" / split_dir).glob("*.jsonl")):
                name = p.stem                              # evaluation-format rows, one file per set
                rows = read_jsonl(p)
                pools[name] += rows
                stats[f"{name}_{split_dir}"] += len(rows)
    guard("crossner_mit", crossner_mit)

    # CrossRE train + validation
    def crossre():
        for d in ("ai", "literature", "music", "news", "politics", "science"):
            for sp in ("train", "validation"):
                pools[f"crossre_{d}"] += mneb_rows(D / "cross_re" / d / f"{sp}.jsonl", "RE")
    guard("crossre", crossre)

    b1 = json.loads((D / "ood_x/label_bank.json").read_text())["bank"]
    b2 = json.loads((D / "ood_x2/label_bank.json").read_text())["bank"]
    st = collections.Counter()

    def dynamicner():
        for sp in ("train", "dev"):
            rows, _ = ev1.dynamicner(st, sp)
            pools["dynamicner_en"] += bank_rows(rows, "dynamicner", b1, "NER")
    guard("dynamicner_en", dynamicner)
    guard("buster", lambda: pools["buster"].extend(bank_rows(ev1.buster(st, (1, 2, 3, 4))[0], "buster", b1, "NER")))

    def scier():
        rows, _, _ = ev1.scier(st, (("train", "scier"), ("dev", "scier")))
        pools["scier"] += bank_rows(rows, "scier", b1, "NER")
        pools["scier_ood"] += pools["scier"]          # SciER has no out-of-domain train split
    guard("scier", scier)

    def biored():
        for sp in ("Train", "Dev"):
            ner, rel, _, _ = ev2.biored(st, sp)
            pools["biored"] += bank_rows(ner, "biored", b2, "NER")
            pools["biored_re"] += bank_rows(rel, "biored", b2, "RE")
    guard("biored", biored)

    def measeval():
        for sp in ("train", "trial"):
            ner, rel, _, _ = ev2.measeval(st, sp)
            pools["measeval"] += bank_rows(ner, "measeval", b2, "NER")
            pools["measeval_re"] += bank_rows(rel, "measeval", b2, "RE")
    guard("measeval", measeval)
    guard("dwie_re", lambda: pools["dwie_re"].extend(bank_rows(ev2.dwie(st, "train")[0], "dwie", b2, "RE")))

    def retacred():
        for sp in ("train", "dev"):
            pools["retacred"] += bank_rows(ev2.retacred(st, sp)[0], "retacred", b2, "RE")
    guard("retacred", retacred)

    def aste():
        for sp in ("train", "dev"):
            for r in ev2.aste(st, sp)[0]:
                pools[r["set"]] += bank_rows([r], "aste", b2, "RE")
    guard("aste", aste)

    def gap():
        for sp in ("development", "validation"):
            pools["gap"] += bank_rows(ev2.gap(st, sp)[0], "gap", b2, "RE")
    guard("gap", gap)
    guard("maven_ere", lambda: pools["maven_ere"].extend(bank_rows(ev2.maven_ere(st, "train")[0], "maven", b2, "RE")))

    def sentivent():
        for sp in ("train", "validation"):
            pools["sentivent"] += bank_events(ev2.sentivent(st, sp), "sentivent", b2)
    guard("sentivent", sentivent)

    def mailex():
        for sp in ("train", "dev"):
            pools["mailex"] += bank_events(ev2.mailex(st, sp), "mailex", b2)
    guard("mailex", mailex)

    def casie_mlee():
        from .prepare_ee import casie_index, load_split
        index = casie_index()
        for name in ("casie", "mlee"):
            rows, _, _ = load_split(name, "train", index)
            pools[name] += rows
    guard("casie_mlee", casie_mlee)

    def pii():
        # English validation rows of ai4privacy that are not in the evaluation set
        from .prepare_pii import readable
        used = {r["id"].split(":", 2)[2] for r in read_jsonl(D / "ood_pii/pii_ai4privacy.jsonl")}
        path = next((ROOT / "raw/hf").glob("datasets--ai4privacy--pii-masking-openpii-1.5m/snapshots/*/"
                                            "data/validation.jsonl"))
        for line in open(path):
            rec = json.loads(line)
            if str(rec.get("language", "")).lower() not in ("en", "english") or str(rec.get("uid")) in used:
                continue
            mask = rec["privacy_mask"]
            mask = json.loads(mask) if isinstance(mask, str) else mask
            text = rec["source_text"]
            ents = [{"start": int(m["start"]), "end": int(m["end"]), "label": readable(m["label"])} for m in mask or []]
            if all(text[e["start"]:e["end"]] for e in ents):
                pools["pii_ai4privacy"].append({"text": text, "entities": ents})
            if len(pools["pii_ai4privacy"]) >= MAX_LABELED:
                break
    guard("pii_ai4privacy", pii)
    stats.update({f"builder_{k}": v for k, v in st.items() if isinstance(v, int)})
    return pools


def ee_schema(group, name):
    m = json.loads((D / group / "manifest.json").read_text())["sets"][name]
    roles = m.get("roles", {})
    return {t: [d, {r: roles.get(r, "") for r in m.get("event_roles", {}).get(t, [])}]
            for t, d in m.get("event_types", {}).items()}


def build(args):
    stats = collections.Counter()
    pools = labeled_pools(stats)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (group, task, genre) in SETS.items():
        path = D / group / f"{name}.jsonl"
        if not path.exists():
            summary[name] = {"missing_eval_file": str(path)}
            continue
        rows = read_jsonl(path)
        rng = random.Random(name)
        texts = sorted({r["text"] for r in rows})
        rng.shuffle(texts)
        # the *_new sets ask the same types as CrossNER / MIT: their train splits are the labeled seeds
        src = name[:-len("_new")] if name.endswith("_new") else name
        labeled = [compact(r) for r in pools.get(src, []) if r.get("text")]
        if task == "EE":
            # long documents (MLEE abstracts, emails) cut into sentence-length pieces, events kept whole
            from ..synth.ee_gen import pieces
            labeled = [p for x in labeled for p in pieces(x) if p["events"]]
            pool = {"schema": ee_schema(group, name)}
        else:
            ent, rel = labels_of(rows)
            pool = {"labels": sorted(ent.items()), "rel_labels": sorted(rel.items())}
            known = set(ent)
            # a labeled example keeps only the set's own types
            for x in labeled:
                x["entities"] = [e for e in x["entities"] if e[2] in known]
            labeled = [x for x in labeled if x["entities"]]
        rng.shuffle(labeled)
        pool.update(set=name, group=group, task=task, genre=genre, labeled=labeled[:MAX_LABELED],
                    texts=texts[:MAX_TEXTS])
        (OUT / f"{name}.json").write_text(json.dumps(pool, ensure_ascii=False))
        summary[name] = {"task": task, "labeled": len(labeled), "texts": len(texts),
                         "types": len(pool.get("labels") or pool.get("schema") or []),
                         "rel_types": len(pool.get("rel_labels") or []),
                         "median_chars": sorted(len(t) for t in texts)[len(texts) // 2] if texts else 0}
    (OUT / "summary.json").write_text(json.dumps({"sets": summary, "stats": dict(stats)}, indent=1))
    for k, v in summary.items():
        print(f"{k:26s} {json.dumps(v)}")
    print(json.dumps(dict(stats))[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build"])
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
