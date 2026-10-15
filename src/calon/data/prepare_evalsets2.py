"""Third batch of zero-shot evaluation sets (evaluation only, never trained on).

NER  -> data/ood_x2      gentle (GENTLE, all nominal mentions, 8 unusual genres),
                         biored (BioRED test, 6 biomedical types), measeval (MeasEval eval spans)
RE   -> data/ood_re_x2   biored_re and dwie_re (concept level: one representative mention per
                         concept, the first in the row; BioRED relations are undirected),
                         retacred (Re-TACRED test: the subject and object of each sentence),
                         measeval_re, aste_{14lap,14res,15res,16res} (aspect -> opinion with the
                         sentiment as the relation), gap (pronoun -> name), litbank (mention ->
                         nearest earlier mention of the same entity), maven_ere (MAVEN-ERE valid,
                         event and time mentions; temporal, causal, subevent relations between
                         the first mentions of two events, coreference from a later mention of an
                         event to its first mention)
EE   -> data/ood_ee2     speed_mpox and speed_zika (SPEED++ English), sentivent (SENTiVENT test
                         sentences), mailex (MAILEX test emails)

Long texts are cut at whitespace outside every annotated span into pieces of at most MAX_CHARS
characters and at most MAX_ENTS gold entities (the relation head pairs at most 24 spans); a
relation or argument whose two ends fall in different pieces is dropped from the gold for every
model alike (counted in build_meta.json). Sets with a "settings" list are scored only in those
settings (concept-level and pair-classification sets: gold entities given).
Absent types are scope_assumed_absent (benchmark_closed_world). Descriptions come from
label_bank.py (name, parent and a fixed dataset context only).
Stages: build -> label_bank (inventory in data/ood_x2) -> finish.
"""
import argparse
import ast
import collections
import csv
import glob
import json
import re
import subprocess
from pathlib import Path

from calon.data.roots import ROOT

from .prepare_spy import chunks

RAW, RECAST = ROOT / "raw/evalsets2", ROOT / "raw/recast"
OUT_NER, OUT_RE, OUT_EE = ROOT / "data/ood_x2", ROOT / "data/ood_re_x2", ROOT / "data/ood_ee2"
MAX_CHARS, MAX_ENTS = 2000, 24
PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


def words(code):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", code))).strip().lower()


def join_tokens(tokens):
    """Text of tokens joined by single spaces and the (start, end) of each token."""
    pos, offs = 0, []
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    return " ".join(tokens), offs


def pieces(text, marks, count, max_chars=MAX_CHARS, cap=MAX_ENTS, base=0):
    """chunks() at max_chars, then pieces with more than `cap` units (count(a, b)) cut again
    at half the size, down to 300 characters."""
    out = []
    local = [{"start": m["start"] - base, "end": m["end"] - base} for m in marks
             if base <= m["start"] and m["end"] <= base + len(text)]
    for a, b in chunks(text, local, max_chars):
        if count(base + a, base + b) > cap and b - a > 300:
            out += pieces(text[a:b], marks, count, max(300, (b - a) // 2), cap, base + a)
        else:
            out.append((base + a, base + b))
    return out


def inside(span, a, b):
    return a <= span[0] and span[1] <= b


def shift(span, a):
    return [span[0] - a, span[1] - a]


# ---------------------------------------------------------------- NER

ENTITY_RE = re.compile(r"\((\d+)-([^()\-]+)[^()]*(\))?|(\d+)\)")


def gentle(stats):
    rows = []
    for f in sorted((RAW / "gentle/dep").glob("*.conllu")):
        for k, block in enumerate(open(f, encoding="utf-8").read().strip().split("\n\n")):
            lines = block.split("\n")
            text = next((x[len("# text = "):] for x in lines if x.startswith("# text = ")), None)
            if text is None:
                stats["gentle_no_text"] += 1
                continue
            pos, open_, ents = 0, collections.defaultdict(list), []
            for line in lines:
                cols = line.split("\t")
                if line.startswith("#") or len(cols) < 10 or not cols[0].isdigit():
                    continue
                j = text.find(cols[1], pos)
                if j < 0:
                    stats["gentle_token_unaligned"] += 1
                    continue
                s, e = j, j + len(cols[1])
                pos = e
                m = re.search(r"Entity=([^|]+)", cols[9])
                for mm in ENTITY_RE.finditer(m.group(1) if m else ""):
                    if mm.group(1):
                        if mm.group(3):
                            ents.append({"start": s, "end": e, "code": mm.group(2)})
                        else:
                            open_[mm.group(1)].append((s, mm.group(2)))
                    elif open_.get(mm.group(4)):
                        a, t = open_[mm.group(4)].pop()
                        ents.append({"start": a, "end": e, "code": t})
            stats["gentle_unclosed"] += sum(len(v) for v in open_.values())
            ents = [dict(t) for t in {tuple(sorted(x.items())) for x in ents}]
            rows.append({"set": "gentle", "id": f"gentle:{f.stem}:{k}", "text": text,
                         "ents": sorted(ents, key=lambda x: (x["start"], x["end"]))})
    types = sorted({e["code"] for r in rows for e in r["ents"]})
    for r in rows:
        r["ask"] = types
    stats["gentle_rows"], stats["gentle_entities"] = len(rows), sum(len(r["ents"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in types]


def biored_docs(stats, split="Test"):
    docs = []
    for block in open(RAW / f"BioRED/{split}.PubTator", encoding="utf-8").read().strip().split("\n\n"):
        lines = block.strip().split("\n")
        pmid, title = lines[0].split("|t|", 1)
        text = title + " " + lines[1].split("|a|", 1)[1]
        ments, rels = [], []
        for line in lines[2:]:
            p = line.split("\t")
            if len(p) >= 6 and p[1].isdigit():
                s, e = int(p[1]), int(p[2])
                if text[s:e] != p[3]:
                    stats["biored_mention_mismatch"] += 1
                    continue
                ids = [x for x in re.split(r"[,;]", p[5]) if x and x != "-"]
                ments.append({"start": s, "end": e, "code": words(p[4]), "ids": ids})
            elif len(p) >= 4:
                rels.append((words(p[1]), p[2], p[3]))
        docs.append((pmid, text, ments, rels))
    return docs


def concept_rows(name, key, text, ments, rels, ent_label, stats, sym=False):
    """Rows of one document at concept level. ments: [{start, end, ids}]; rels: [(label,
    concept, concept)]. Each row: one representative mention per concept (the first in the row
    that names that concept alone, else the first naming it)."""
    by_concept = collections.defaultdict(list)
    for m in sorted(ments, key=lambda x: (x["start"], x["end"])):
        for c in m["ids"]:
            by_concept[c].append(m)

    def concepts(a, b):
        return len({c for m in ments if inside((m["start"], m["end"]), a, b) for c in m["ids"]})
    out = []
    for a, b in pieces(text, ments, concepts):
        rep, used = {}, set()
        for c, ms in by_concept.items():
            ms = [m for m in ms if inside((m["start"], m["end"]), a, b)]
            pick = next((m for m in ms if len(m["ids"]) == 1 and (m["start"], m["end"]) not in used),
                        next((m for m in ms if (m["start"], m["end"]) not in used), None))
            if ms and pick is None:
                stats[f"{name}_concept_without_free_mention"] += 1
            if pick is not None:
                rep[c] = pick
                used.add((pick["start"], pick["end"]))
        ents = [{"start": m["start"] - a, "end": m["end"] - a, "code": ent_label(c, m)} for c, m in rep.items()]
        rr = set()
        for lab, h, t in rels:
            if h in rep and t in rep and h != t:
                x = (tuple(shift((rep[h]["start"], rep[h]["end"]), a)),
                     tuple(shift((rep[t]["start"], rep[t]["end"]), a)))
                rr.add((min(x), max(x), lab) if sym else (x[0], x[1], lab))
        out.append({"set": name, "id": f"{name}:{key}:{a}", "text": text[a:b],
                    "ents": sorted(ents, key=lambda x: (x["start"], x["end"])),
                    "rels": [{"head": list(h), "tail": list(t), "label": lab} for h, t, lab in sorted(rr)]})
        stats[f"{name}_over_cap"] += len(ents) > MAX_ENTS
    gold = {(lab, h, t) for lab, h, t in rels if h != t}
    stats[f"{name}_gold_relations_doc"] += len(gold)
    return out


def biored(stats, split="Test"):
    docs = biored_docs(stats, split)
    ner, rel = [], []
    for pmid, text, ments, rels in docs:
        for a, b in pieces(text, ments, lambda a, b: 0):
            ents = [{"start": m["start"] - a, "end": m["end"] - a, "code": m["code"]} for m in ments
                    if inside((m["start"], m["end"]), a, b)]
            ner.append({"set": "biored", "id": f"biored:{pmid}:{a}", "text": text[a:b], "ents": ents})
        cm = {}
        for m in ments:
            for c in m["ids"]:
                cm.setdefault(c, m["code"])
        rel += concept_rows("biored_re", pmid, text, [m for m in ments if m["ids"]], rels,
                            lambda c, m: cm.get(c, m["code"]), stats, sym=True)
    types = sorted({m["code"] for _, _, ms, _ in docs for m in ms})
    rtypes = sorted({x[0] for _, _, _, rs in docs for x in rs})
    for r in ner + rel:
        r["ask"] = types
    for r in rel:
        r["rel_ask"] = rtypes
    stats["biored_rows"], stats["biored_re_rows"] = len(ner), len(rel)
    stats["biored_re_relations"] = sum(len(r["rels"]) for r in rel)
    return ner, rel, [{"raw": t, "name": t, "parent": None} for t in types], rtypes


# ---------------------------------------------------------------- RE

DWIE_TYPES = ("person", "organization", "gpe", "location", "event")


def dwie(stats, split="test"):
    rows = []
    for f in sorted(glob.glob(str(RAW / "DWIE/data/annos_with_content/*.json"))):
        d = json.load(open(f))
        if split not in d["tags"]:
            continue
        ctype = {}
        for c in d["concepts"]:
            tags = [t[6:] for t in c.get("tags") or [] if t.startswith("type::")]
            ctype[c["concept"]] = next((t for t in DWIE_TYPES if t in tags), "other")
        ments = [{"start": m["begin"], "end": m["end"], "ids": [m["concept"]]} for m in d["mentions"]]
        rels = [(words(r["p"].rstrip("0123456789")), r["s"], r["o"]) for r in d["relations"]
                if not r["p"].endswith("-x")]
        stats["dwie_x_relations_skipped"] += sum(r["p"].endswith("-x") for r in d["relations"])
        rows += concept_rows("dwie_re", d["id"], d["content"], ments, rels, lambda c, m: ctype[c], stats)
    rtypes = sorted({x["label"] for r in rows for x in r["rels"]})
    for r in rows:
        r["ask"], r["rel_ask"] = list(DWIE_TYPES) + ["other"], rtypes
    stats["dwie_re_rows"], stats["dwie_re_relations"] = len(rows), sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in list(DWIE_TYPES) + ["other"]], rtypes


def tacred_relation(code):
    head, _, rest = code.partition(":")
    return words({"per": "person", "org": "organization"}.get(head, head) + " " + rest)


def retacred(stats, split="test"):
    tdir = RAW / "tacred"
    if not tdir.exists():
        subprocess.run(["tar", "xzf", str(RAW / "tacred_LDC2018T24.tgz"), "-C", str(RAW)], check=True)
    test = next(Path(p) for p in glob.glob(str(RAW / f"**/data/json/{split}.json"), recursive=True))
    labels = json.load(open(RECAST / f"Re-TACRED/Re-TACRED/{split}_id2label.json"))
    rows = []
    for rec in json.load(open(test)):
        if rec["id"] not in labels:
            stats["retacred_removed_by_patch"] += 1
            continue
        toks = [PTB.get(t, t) for t in rec["token"]]
        text, offs = join_tokens(toks)
        h = (offs[rec["subj_start"]][0], offs[rec["subj_end"]][1])
        t = (offs[rec["obj_start"]][0], offs[rec["obj_end"]][1])
        if h == t:
            stats["retacred_same_span"] += 1
            continue
        lab = labels[rec["id"]]
        ents = [{"start": h[0], "end": h[1], "code": words(rec["subj_type"])},
                {"start": t[0], "end": t[1], "code": words(rec["obj_type"])}]
        rels = [] if lab == "no_relation" else [{"head": list(h), "tail": list(t), "label": tacred_relation(lab)}]
        rows.append({"set": "retacred", "id": f"retacred:{split}:{rec['id']}", "text": text, "ents": ents,
                     "rels": rels})
    etypes = sorted({e["code"] for r in rows for e in r["ents"]})
    rtypes = sorted({tacred_relation(x) for x in set(labels.values()) if x != "no_relation"})
    for r in rows:
        r["ask"], r["rel_ask"] = sorted({e["code"] for e in r["ents"]}), rtypes
    stats["retacred_rows"] = len(rows)
    stats["retacred_positive"] = sum(bool(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in etypes], rtypes


ASTE_SENT = {"POS": "positive sentiment", "NEG": "negative sentiment", "NEU": "neutral sentiment"}


def aste(stats, split="test"):
    rows = []
    for sub in ("14lap", "14res", "15res", "16res"):
        f = RECAST / f"SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/{sub}/{split}_triplets.txt"
        for k, line in enumerate(open(f, encoding="utf-8")):
            sent, trip = line.rstrip("\n").split("####")
            text, offs = join_tokens(sent.split(" "))
            ents, rels = set(), set()
            for asp, opi, pol in ast.literal_eval(trip):
                a = (offs[min(asp)][0], offs[max(asp)][1])
                o = (offs[min(opi)][0], offs[max(opi)][1])
                ents |= {(a, "aspect term"), (o, "opinion term")}
                rels.add((a, o, ASTE_SENT[pol]))
            rows.append({"set": f"aste_{sub}", "id": f"aste_{sub}:{split}:{k}", "text": text,
                         "ents": [{"start": s[0], "end": s[1], "code": c} for s, c in sorted(ents)],
                         "ask": ["aspect term", "opinion term"], "rel_ask": sorted(ASTE_SENT.values()),
                         "rels": [{"head": list(h), "tail": list(t), "label": lab} for h, t, lab in sorted(rels)]})
        stats[f"aste_{sub}_rows"] = sum(r["set"] == f"aste_{sub}" for r in rows)
    stats["aste_relations"] = sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in ("aspect term", "opinion term")], \
        sorted(ASTE_SENT.values())


def gap(stats, split="test"):
    rows = []
    for rec in csv.DictReader(open(RECAST / f"gap-coreference/gap-{split}.tsv", encoding="utf-8"), delimiter="\t"):
        text = rec["Text"]
        p = (int(rec["Pronoun-offset"]), int(rec["Pronoun-offset"]) + len(rec["Pronoun"]))
        ents, rels = [{"start": p[0], "end": p[1], "code": "pronoun"}], []
        for x in ("A", "B"):
            s = (int(rec[f"{x}-offset"]), int(rec[f"{x}-offset"]) + len(rec[x]))
            if text[s[0]:s[1]] != rec[x] or text[p[0]:p[1]] != rec["Pronoun"]:
                stats["gap_offset_mismatch"] += 1
            ents.append({"start": s[0], "end": s[1], "code": "person name"})
            if rec[f"{x}-coref"].upper() == "TRUE":
                rels.append({"head": list(p), "tail": list(s), "label": "refers to"})
        rows.append({"set": "gap", "id": f"gap:{split}:{rec['ID']}", "text": text, "ents": ents, "rels": rels,
                     "ask": ["pronoun", "person name"], "rel_ask": ["refers to"]})
    stats["gap_rows"], stats["gap_relations"] = len(rows), sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in ("pronoun", "person name")], ["refers to"]


def conll_docs(path):
    """LitBank coreference CoNLL: sentences of (word, coref column)."""
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                sents.append(cur)
            cur = []
            continue
        cols = [c for c in line.split("\t")]
        while cols and cols[-1] == "":
            cols.pop()
        cur.append((cols[3], cols[-1]))
    if cur:
        sents.append(cur)
    return sents


def window_rows(name, key, sents, ments_of, rels_of, stats, cap=MAX_ENTS):
    """Consecutive sentences, joined by spaces, while the row holds at most `cap` mentions.
    sents: token lists; ments_of(k): [(token start, token end exclusive, mention id, label)] of
    sentence k; rels_of(ids): [(head id, tail id, label)] among the mention ids of a row."""
    out, k = [], 0
    while k < len(sents):
        j, n = k, 0
        while j < len(sents) and (j == k or n + len(ments_of(j)) <= cap):
            n += len(ments_of(j))
            j += 1
        toks, spans, ents = [], {}, []
        for s in range(k, j):
            base = len(toks)
            toks += sents[s]
            for a, b, mid, lab in ments_of(s):
                spans[mid] = (base + a, base + b, lab)
        text, offs = join_tokens(toks)
        char = {m: (offs[a][0], offs[b - 1][1]) for m, (a, b, _) in spans.items()}
        seen = set()
        for m, (a, b, lab) in spans.items():
            if (char[m], lab) not in seen:
                seen.add((char[m], lab))
                ents.append({"start": char[m][0], "end": char[m][1], "code": lab})
        rels = {(char[h], char[t], lab) for h, t, lab in rels_of(set(spans)) if char[h] != char[t]}
        out.append({"set": name, "id": f"{name}:{key}:{k}", "text": text,
                    "ents": sorted(ents, key=lambda x: (x["start"], x["end"])),
                    "rels": [{"head": list(h), "tail": list(t), "label": lab} for h, t, lab in sorted(rels)]})
        stats[f"{name}_over_cap"] += n > cap
        k = j
    return out


def litbank(stats):
    rows = []
    for f in sorted((RECAST / "litbank/coref/conll").glob("*.conll")):
        sents = conll_docs(f)
        ments, stack, mid = collections.defaultdict(list), collections.defaultdict(list), 0
        cluster = {}
        for k, sent in enumerate(sents):
            for i, (_, col) in enumerate(sent):
                for part in col.split("|") if col not in ("_", "-") else []:
                    m = re.fullmatch(r"\(?(\d+)\)?", part)
                    if not m:
                        continue
                    c = m.group(1)
                    if part.startswith("(") and part.endswith(")"):
                        ments[k].append((i, i + 1, mid, "mention"))
                        cluster[mid], mid = c, mid + 1
                    elif part.startswith("("):
                        stack[c].append((k, i))
                    elif stack.get(c):
                        sk, si = stack[c].pop()
                        if sk == k:
                            ments[k].append((si, i + 1, mid, "mention"))
                            cluster[mid], mid = c, mid + 1
                        else:
                            stats["litbank_cross_sentence_mention"] += 1
        order = {m[2]: (k, m[0], m[1]) for k in ments for m in ments[k]}

        def rels_of(ids):
            # each mention -> the nearest earlier mention of its entity in the same row
            out, last = [], {}
            for m in sorted(ids, key=lambda x: order[x]):
                if cluster[m] in last:
                    out.append((m, last[cluster[m]], "coreference"))
                last[cluster[m]] = m
            return out
        rows += window_rows("litbank", f.stem.split("_")[0], [[w for w, _ in s] for s in sents],
                            lambda k: ments.get(k, []), rels_of, stats)
    for r in rows:
        r["ask"], r["rel_ask"] = ["mention"], ["coreference"]
    stats["litbank_rows"], stats["litbank_relations"] = len(rows), sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": "mention", "name": "mention", "parent": None}], ["coreference"]


def maven_ere(stats, split="valid"):
    rows = []
    for line in open(RECAST / f"maven_ere/MAVEN_ERE/{split}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        ments, event_of = collections.defaultdict(list), {}
        for ev in d["events"]:
            for m in ev["mention"]:
                if " ".join(d["tokens"][m["sent_id"]][m["offset"][0]:m["offset"][1]]) != m["trigger_word"]:
                    stats["maven_trigger_mismatch"] += 1
                ments[m["sent_id"]].append((m["offset"][0], m["offset"][1], m["id"], "event"))
                event_of[m["id"]] = ev["id"]
        for tx in d.get("TIMEX", []):
            ments[tx["sent_id"]].append((tx["offset"][0], tx["offset"][1], tx["id"], "time expression"))
            event_of[tx["id"]] = tx["id"]
        pos = {m[2]: (k, m[0]) for k in ments for m in ments[k]}
        links = []
        for kind in ("temporal_relations", "causal_relations"):
            for lab, pairs in (d.get(kind) or {}).items():
                links += [(a, b, words(lab)) for a, b in pairs]
        links += [(a, b, "subevent") for a, b in d.get("subevent_relations") or []]

        def rels_of(ids):
            first = {}
            for m in sorted(ids, key=lambda x: pos[x]):
                first.setdefault(event_of[m], m)
            out = [(m, first[event_of[m]], "coreference") for m in ids if first[event_of[m]] != m]
            out += [(first[a], first[b], lab) for a, b, lab in links if a in first and b in first and a != b]
            return out
        rows += window_rows("maven_ere", d["id"], d["tokens"], lambda k: ments.get(k, []), rels_of, stats)
        stats["maven_doc_links"] += len(links)
    rtypes = sorted({x["label"] for r in rows for x in r["rels"]})
    for r in rows:
        r["ask"], r["rel_ask"] = ["event", "time expression"], rtypes
    stats["maven_ere_rows"], stats["maven_ere_relations"] = len(rows), sum(len(r["rels"]) for r in rows)
    return rows, [{"raw": t, "name": t, "parent": None} for t in ("event", "time expression")], rtypes


MEAS = {"Quantity": "quantity", "MeasuredEntity": "measured entity", "MeasuredProperty": "measured property",
        "Qualifier": "qualifier"}
MEAS_REL = {"HasQuantity": "has quantity", "HasProperty": "has property", "Qualifies": "qualifies"}


def measeval(stats, split="eval"):
    ner, rel = [], []
    for f in sorted((RECAST / f"MeasEval/data/{split}/tsv").glob("*.tsv")):
        doc = f.stem
        text = open(RECAST / f"MeasEval/data/{split}/text/{doc}.txt", encoding="utf-8").read()
        recs = list(csv.DictReader(open(f, encoding="utf-8"), delimiter="\t"))
        span = {}
        for x in recs:
            s, e = int(x["startOffset"]), int(x["endOffset"])
            if text[s:e] != x["text"]:
                stats["measeval_offset_mismatch"] += 1
            span[(x["annotSet"], x["annotId"])] = ((s, e), MEAS[x["annotType"]])
        ents = {v for v in span.values()}
        rr = set()
        for x in recs:
            other = json.loads(x["other"]) if x.get("other", "").strip().startswith("{") else {}
            for k, lab in MEAS_REL.items():
                if k in other:
                    tgt = span.get((x["annotSet"], other[k]))
                    if tgt is None:
                        stats["measeval_dangling_link"] += 1
                        continue
                    rr.add((span[(x["annotSet"], x["annotId"])][0], tgt[0], lab))
        marks = [{"start": s[0], "end": s[1]} for s, _ in ents]
        for a, b in pieces(text, marks, lambda a, b: len({s for s, _ in ents if inside(s, a, b)})):
            ce = [{"start": s[0] - a, "end": s[1] - a, "code": c} for s, c in sorted(ents) if inside(s, a, b)]
            cr = [{"head": shift(h, a), "tail": shift(t, a), "label": lab} for h, t, lab in sorted(rr)
                  if inside(h, a, b) and inside(t, a, b)]
            stats["measeval_relations_cross_piece"] += sum(inside(h, a, b) != inside(t, a, b) for h, t, _ in rr)
            row = {"id": f"measeval:{doc}:{a}", "text": text[a:b], "ents": ce, "ask": sorted(MEAS.values())}
            ner.append(dict(row, set="measeval"))
            rel.append(dict(row, set="measeval_re", id=row["id"].replace("measeval:", "measeval_re:"), rels=cr,
                            rel_ask=sorted(MEAS_REL.values())))
    stats["measeval_rows"], stats["measeval_relations"] = len(ner), sum(len(r["rels"]) for r in rel)
    return ner, rel, [{"raw": t, "name": t, "parent": None} for t in sorted(MEAS.values())], \
        sorted(MEAS_REL.values())


# ---------------------------------------------------------------- EE

def ee_rows(name, key, text, evs, stats):
    """Pieces of one text with the events whose trigger and arguments lie inside."""
    marks = [{"start": e["trigger"][0], "end": e["trigger"][1]} for e in evs]
    marks += [{"start": a["span"][0], "end": a["span"][1]} for e in evs for a in e["args"]]
    out = []
    for a, b in (chunks(text, marks, MAX_CHARS) if len(text) > MAX_CHARS else [(0, len(text))]):
        cev = []
        for e in evs:
            if not inside(e["trigger"], a, b):
                continue
            keep = [x for x in e["args"] if inside(x["span"], a, b)]
            stats[f"{name}_args_cross_piece"] += len(e["args"]) - len(keep)
            cev.append({"type": e["type"], "trigger": shift(e["trigger"], a),
                        "args": [{"role": x["role"], "span": shift(x["span"], a)} for x in keep]})
        out.append({"set": name, "id": f"{name}:{key}:{a}", "text": text[a:b], "events": cev})
    return out


def load_json_any(path):
    raw = open(path, encoding="utf-8").read()
    try:
        x = json.loads(raw)
        return x if isinstance(x, list) else [x]
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def speed(stats):
    d = RAW / "data-speed++"
    if not d.exists():
        subprocess.run(["unzip", "-q", "-o", str(RAW / "data-speed++.zip"), "-x", "__MACOSX/*", "-d", str(RAW)],
                       check=True)
    rows = []
    for sub in ("mpox", "zika"):
        for k, rec in enumerate(load_json_any(d / f"en-{sub}.json")):
            toks = rec["tokens"]
            text, offs = join_tokens(toks)

            def span(x):
                s, e = int(x["start"]), int(x["end"])
                if (x.get("text") is not None and " ".join(toks[s:e]) != x["text"]
                        and " ".join(toks[s:e + 1]) == x["text"]):
                    e += 1
                if e <= s or e > len(toks):
                    return None
                return [offs[s][0], offs[e - 1][1]]
            ents = {m["id"]: span(m) for m in rec.get("entity_mentions") or []}
            evs = []
            for ev in rec.get("event_mentions") or []:
                tr = span(ev["trigger"])
                if tr is None:
                    stats["speed_bad_trigger"] += 1
                    continue
                args = [{"role": a["role"], "span": ents[a["entity_id"]]} for a in ev.get("arguments") or []
                        if ents.get(a.get("entity_id"))]
                stats["speed_args_unresolved"] += len(ev.get("arguments") or []) - len(args)
                evs.append({"type": ev["event_type"], "trigger": tr, "args": args})
            rows += ee_rows(f"speed_{sub}", rec.get("sent_id", rec.get("doc_id", k)), text, evs, stats)
    return rows


def sentivent(stats, split="test"):
    import pandas as pd
    f = glob.glob(str(ROOT / "raw/hf/datasets--GillesJacobs--sentivent/snapshots/*/data/"
                       f"ere_omnievent_sentence/{split}.parquet"))[0]
    rows = []
    for k, rec in enumerate(pd.read_parquet(f).to_dict("records")):
        text, evs = rec["text"], []
        for ev in rec["events"]:
            for tr in ev["triggers"]:
                s, e = (int(x) for x in tr["position"])
                if text[s:e] != tr["trigger_word"]:
                    # discontinuous triggers ("maintains ... profit margin"): kept as the
                    # position's full extent
                    stats["sentivent_trigger_discontinuous_extent"] += 1
                args = []
                for a in tr["arguments"]:
                    for m in a["mentions"]:
                        ms, me = (int(x) for x in m["position"])
                        if text[ms:me] != m["mention"]:
                            stats["sentivent_arg_discontinuous_extent"] += 1
                        args.append({"role": a["role"], "span": [ms, me]})
                evs.append({"type": ev["type"], "trigger": [s, e], "args": args})
        rows += ee_rows("sentivent", f"{split}:{k}", text, evs, stats)
    return rows


def mailex(stats, split="test"):
    rows = []
    for f in sorted((RAW / f"mailex/data/{split}").glob("*.json")):
        d = json.load(open(f))
        sents = d["sentences"]
        for turn, etypes in d["events"].items():
            toks = sents[turn] if isinstance(sents, dict) else sents[int(turn.split("_")[-1])]
            toks = toks.split(" ") if isinstance(toks, str) else list(toks)
            if toks and isinstance(toks[0], list):     # the turn's sentences: one token sequence
                stats["mailex_nested_turns"] += 1
                toks = [t for s in toks for t in s]
            elif any(isinstance(t, list) for t in toks):  # train split: some sentences nested
                stats["mailex_mixed_turns"] += 1
                toks = [t for s in toks for t in (s if isinstance(s, list) else [s])]
            text, offs = join_tokens(toks)
            evs = []
            for et, v in etypes.items():
                if et == "O":                            # no event: not an event type
                    stats["mailex_O_entries"] += len(v["triggers"])
                    continue
                for labels, trig in zip(v["labels"], v["triggers"]):
                    m = re.search(r"'indices':\s*'([\d ]*)'", trig) if isinstance(trig, str) else None
                    idx = [int(x) for x in (m.group(1) if m else str((trig or {}).get("indices", "")
                                                                       if isinstance(trig, dict) else "")).split()]
                    if not idx or max(idx) >= len(toks) or len(labels) != len(toks):
                        if stats["mailex_misaligned"] < 4:
                            print("mailex misaligned", et, repr(trig)[:120], len(labels), len(toks), flush=True)
                        stats["mailex_misaligned"] += 1
                        continue
                    args, cur = [], None
                    for i, lab in enumerate(labels + ["O"]):
                        tag = lab.split(":", 1)[1] if ":" in lab else lab
                        if cur and not (tag.startswith("I-") and tag[2:] == cur[0]):
                            args.append({"role": cur[0], "span": [offs[cur[1]][0], offs[cur[2]][1]]})
                            cur = None
                        if tag.startswith("B-") or (tag.startswith("I-") and cur is None):
                            cur = [tag[2:], i, i]
                        elif tag.startswith("I-") and cur:
                            cur[2] = i
                    evs.append({"type": et, "trigger": [offs[min(idx)][0], offs[max(idx)][1]], "args": args})
            rows += ee_rows("mailex", f"{f.stem}:{turn}", text, evs, stats)
    return rows


# ---------------------------------------------------------------- stages

def build(args):
    stats = collections.Counter()
    ner, rel, ee, inv = [], [], [], {}
    g, inv["gentle"] = gentle(stats)
    bn, br, inv["biored"], rt = biored(stats)
    inv["biored_rel"] = rt
    mn, mr, inv["measeval"], inv["measeval_rel"] = measeval(stats)
    ner += g + bn + mn
    rel += br + mr
    for fn, key in ((dwie, "dwie"), (retacred, "retacred"), (aste, "aste"), (gap, "gap"), (litbank, "litbank"),
                    (maven_ere, "maven")):
        rows, inv[key], inv[f"{key}_rel"] = fn(stats)
        rel += rows
    for fn, key in ((speed, "speed"), (sentivent, "sentivent"), (mailex, "mailex")):
        rows = fn(stats)
        ee += rows
        types = collections.defaultdict(set)
        for r in rows:
            for e in r["events"]:
                types[e["type"]] |= {a["role"] for a in e["args"]}
        inv[key] = [{"raw": c, "name": words(c), "parent": None, "kind": "event"} for c in sorted(types)]
        inv[f"{key}_role"] = [{"raw": x, "name": words(x), "parent": None, "kind": "role"}
                              for x in sorted({x for v in types.values() for x in v})]
        inv[f"{key}_schema"] = {c: sorted(v) for c, v in types.items()}
    for k in [k for k in inv if k.endswith("_rel")]:
        inv[k] = [{"raw": t, "name": t, "parent": None, "kind": "relation"} for t in inv[k]]
    for name in sorted({r["set"] for r in ee}):
        rs = [r for r in ee if r["set"] == name]
        stats[f"{name}_rows"] = len(rs)
        stats[f"{name}_events"] = sum(len(r["events"]) for r in rs)
        stats[f"{name}_args"] = sum(len(e["args"]) for r in rs for e in r["events"])
    stats["max_chars"] = max(len(r["text"]) for r in ner + rel + ee)
    OUT_NER.mkdir(parents=True, exist_ok=True)
    (OUT_NER / "raw_rows.json").write_text(json.dumps({"ner": ner, "re": rel, "ee": ee}))
    schemas = {k: inv.pop(k) for k in [k for k in inv if k.endswith("_schema")]}
    (OUT_NER / "inventory.json").write_text(json.dumps(inv, indent=1))
    (OUT_NER / "build_meta.json").write_text(json.dumps({"stats": dict(stats), "schemas": schemas}, indent=1))
    print(json.dumps(dict(sorted(stats.items()))))
    for k, v in inv.items():
        print(k, len(v), [x["name"] for x in v][:14])


SOURCE_OF = {"gentle": "gentle", "biored": "biored", "biored_re": "biored", "measeval": "measeval",
             "measeval_re": "measeval", "dwie_re": "dwie", "retacred": "retacred", "gap": "gap",
             "litbank": "litbank", "maven_ere": "maven", "speed_mpox": "speed", "speed_zika": "speed",
             "sentivent": "sentivent", "mailex": "mailex"}
GIVEN_ONLY = {"biored_re", "dwie_re", "retacred", "gap", "litbank", "maven_ere"}
SYMMETRIC = {"biored_re"}


def source_of(name):
    return "aste" if name.startswith("aste_") else SOURCE_OF[name]


def finish(args):
    bank = json.loads((OUT_NER / "label_bank.json").read_text())["bank"]
    raw = json.loads((OUT_NER / "raw_rows.json").read_text())
    meta = json.loads((OUT_NER / "build_meta.json").read_text())
    train_labels = set()
    for line in open(ROOT / "data/mix/finerweb/train.jsonl"):
        train_labels |= {q["label"].lower().replace("_", " ") for q in json.loads(line)["queries"]}

    def queries(b, ask, present):
        return [{"label": b[c]["name"], "description": b[c]["description"],
                 "status": "positive" if b[c]["name"] in present else "scope_assumed_absent",
                 "description_origin": b[c]["description_origin"],
                 "supervision_origin": "source" if b[c]["name"] in present else "benchmark_closed_world"}
                for c in ask]
    sets = {"ner": collections.defaultdict(list), "re": collections.defaultdict(list)}
    for kind in ("ner", "re"):
        for r in raw[kind]:
            src = source_of(r["set"])
            b = bank[src]
            ents = [{"start": e["start"], "end": e["end"], "label": b[e["code"]]["name"]} for e in r["ents"]]
            row = {"id": r["id"], "source": r["set"], "text": r["text"],
                   "queries": queries(b, r["ask"], {e["label"] for e in ents}), "entities": ents}
            if kind == "re":
                rb = bank[f"{src}_rel"]
                row["relations"] = [dict(x, label=rb[x["label"]]["name"]) for x in r["rels"]]
                used = {x["label"] for x in row["relations"]}
                row["rel_queries"] = [{"label": rb[c]["name"], "description": rb[c]["description"],
                                       "status": "positive" if rb[c]["name"] in used else "scope_assumed_absent"}
                                      for c in r["rel_ask"]]
                if r["set"] in SYMMETRIC:
                    row["symmetric_labels"] = [q["label"] for q in row["rel_queries"]]
            sets[kind][r["set"]].append(row)
    for kind, out_dir in (("ner", OUT_NER), ("re", OUT_RE)):
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"scope_policy": "benchmark_closed_world",
                    "threshold_policy": "zero-shot: training sources' dev_calib T and threshold",
                    "descriptions": "authored by label_bank.py from the label name, parent and dataset context",
                    "max_chars": MAX_CHARS, "max_entities": MAX_ENTS, "sets": {}}
        for name, rows in sorted(sets[kind].items()):
            with open(out_dir / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            labels = sorted({q["label"] for r in rows for q in r["queries"]})
            info = {"rows": len(rows), "entities": sum(len(r["entities"]) for r in rows), "labels": labels,
                    "seen_labels": [x for x in labels if x in train_labels],
                    "unseen_labels": [x for x in labels if x not in train_labels]}
            if kind == "re":
                info["relations"] = sum(len(r["relations"]) for r in rows)
                info["settings"] = ["given"] if name in GIVEN_ONLY else ["given", "end2end"]
                info["symmetric"] = name in SYMMETRIC
            manifest["sets"][name] = info
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(kind, json.dumps({k: {x: v[x] for x in v if "labels" not in x} for k, v in manifest["sets"].items()}))
    by_set = collections.defaultdict(list)
    for r in raw["ee"]:
        by_set[r["set"]].append(r)
    OUT_EE.mkdir(parents=True, exist_ok=True)
    manifest = {"splits_used": "test (SPEED++: the English mpox and zika sets)",
                "scope_policy": "benchmark_closed_world",
                "descriptions": "authored by label_bank.py from the label name, parent and dataset context",
                "chunk_chars": MAX_CHARS, "sets": {}}
    for name, rows in sorted(by_set.items()):
        src = source_of(name)
        eb, rb, schema = bank[src], bank[f"{src}_role"], meta["schemas"][f"{src}_schema"]
        out = [{"id": r["id"], "source": name, "text": r["text"],
                "events": [{"type": eb[e["type"]]["name"], "trigger": e["trigger"],
                            "args": [{"role": rb[a["role"]]["name"], "span": a["span"]} for a in e["args"]]}
                           for e in r["events"]]} for r in rows]
        with open(OUT_EE / f"{name}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        present = {e["type"] for r in out for e in r["events"]}
        manifest["sets"][name] = {
            "rows": len(out), "events": sum(len(r["events"]) for r in out),
            "args": sum(len(e["args"]) for r in out for e in r["events"]),
            "event_types": {v["name"]: v["description"] for v in eb.values() if v["name"] in present},
            "roles": {v["name"]: v["description"] for v in rb.values()},
            "event_roles": {eb[c]["name"]: [rb[x]["name"] for x in rs] for c, rs in schema.items()
                            if eb[c]["name"] in present}}
    (OUT_EE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("ee", json.dumps({k: {x: v[x] for x in ("rows", "events", "args")} for k, v in manifest["sets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
