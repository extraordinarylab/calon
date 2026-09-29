"""Round-2 Qwen3.8 data (GLiNER2-style diversity), sharded by data.

Products, all written under synth2/:
  g   synthetic texts over a genre x domain grid; each generation also writes
      its own schema (6-12 types with definitions, including confusable pairs),
      the mentions, and 3-5 in-domain types that do not occur (explicit absent).
  c2  more CrossNER/MIT-style sentences from their *train* splits (5 seeds, the
      target label set and descriptions); each request asks for a different
      sub-topic. Not zero-shot.
  r1  open-type annotation of real text (news, PubMed, arXiv, US case law;
      no Wikipedia), same prompt as round-1 web text.
  r2  absent-type questions for r1 passages with hard negatives: half the asked
      types share a content word with a listed type, half are frequent types.
  n   the same absent-type questions for FiNERWeb train rows and Pile train rows
      that have no explicit negative, so every row carries some absent labels.

Subcommands: fetch-real, prepare-g, prepare-c2, prepare-r2, prepare-n, run, assemble.
"""
import argparse
import collections
import json
import os
import random
import re
import time
from pathlib import Path

from .llm_annotate import (A1_PROMPT, A2_PROMPT, C_DOMAIN_NAMES, C_PROMPT, CROSSNER_DOMAINS,
                           align, chunk_passages, clean_type, format_example, parse_a1,
                           parse_json, read_jsonl, schema_items, seed_of, seed_pool, write_jsonl)

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
S2 = ROOT / "synth2"
MIX_DIR = Path(os.environ.get("IE_MIX", str(ROOT / "data/mix2")))   # target data dir for fill / g3
PROMPT_VERSION = "v2"
STOP = set("a an the of and or for to in on at by with from other type name names".split())

GENRES = ["email", "text message conversation", "internal business report", "meeting notes",
          "social media post", "product review", "short news brief", "customer support ticket",
          "invoice or order confirmation", "job posting", "contract clause", "clinical note",
          "research paper abstract", "recipe", "travel itinerary", "forum discussion post",
          "press release", "resume summary", "encyclopedia-style paragraph", "podcast transcript excerpt",
          "spoken request to a voice assistant", "search query log"]
DOMAINS = ["finance and banking", "healthcare", "pharmaceuticals", "law and courts", "politics and elections",
           "government and public policy", "military and defense", "sports", "music", "film and television",
           "literature and publishing", "visual arts", "video games", "fashion", "food and restaurants",
           "travel and tourism", "hotels and hospitality", "real estate", "construction and architecture",
           "automotive", "aviation", "shipping and logistics", "retail and e-commerce", "consumer electronics",
           "telecommunications", "software engineering", "cybersecurity", "artificial intelligence research",
           "data science", "physics", "chemistry", "biology and genetics", "astronomy and space",
           "earth science and climate", "agriculture", "energy and utilities", "oil and gas", "mining",
           "manufacturing", "human resources", "education", "academia and universities", "religion",
           "history", "archaeology", "philosophy", "linguistics", "journalism and media", "advertising and marketing",
           "insurance", "cryptocurrency and blockchain", "nonprofit and charity", "environmental conservation",
           "veterinary and animals", "sports betting and gaming", "social work", "public transport",
           "weddings and events", "fitness and wellness", "parenting and family"]
LENGTHS = [30, 60, 120, 200, 300]

G_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English. Make it specific and natural, with concrete names, places, numbers and dates where the genre calls for them.

Then design an entity-extraction schema for this text: 6 to 12 entity types a user working in {domain} would want to extract from texts like it. Include at least two pairs of closely related types that must be told apart (for example a general type and a more specific one, or two similar roles). Give every type a one-sentence definition that would apply to other texts too.
Annotate every mention of every schema type exactly as it appears in the text; schema types that do not occur get no entries.
Finally list 3 to 5 further entity types that are relevant to {domain} but do not occur anywhere in this text, each with a one-sentence definition.

Return JSON only: {{"text": "...", "types": {{"type name": "definition"}}, "entities": [{{"text": "...", "type": "..."}}], "absent_types": {{"type name": "definition"}}}}"""

G2_GROUPS = {   # round-3 targeted synthetic data: no target-dataset examples or labels
    "music": (["pop music", "rock and metal", "hip hop and rap", "jazz and blues", "classical music and opera",
               "electronic and dance music", "country and folk music", "K-pop and J-pop", "music festivals and tours",
               "record labels and the music industry", "musical instruments and gear", "film and game soundtracks",
               "music awards and charts", "music history", "songwriting and production"],
              ["encyclopedia-style paragraph", "news article", "album review", "artist biography",
               "concert review", "interview excerpt", "forum discussion post", "liner notes"]),
    "literature": (["classic novels", "contemporary fiction", "poetry", "science fiction and fantasy",
                    "crime and mystery fiction", "children's and young adult books", "literary awards and prizes",
                    "publishing houses and magazines", "drama and playwrights", "comics and graphic novels",
                    "literary movements and history", "non-fiction and biography", "translation and world literature"],
                   ["encyclopedia-style paragraph", "book review", "author biography", "news article",
                    "literary criticism excerpt", "reading-group discussion post", "publisher catalogue entry"]),
    "spoken": (["finding a movie to watch", "movie trivia", "booking a restaurant", "finding a place to eat",
                "hotel booking", "flights and trains", "playing music", "podcasts and audiobooks", "shopping",
                "local events and tickets", "weather and time", "navigation and traffic", "sports scores",
                "recipes and cooking", "smart home"],
               ["spoken request to a voice assistant", "search query typed into a phone",
                "spoken question to a voice assistant", "chat message to a booking assistant"]),
}
G2_EXTRA = """
Include at least one general type together with more specific subtypes of it (for example a general person type and a more specific role); annotate each mention only with the most specific type that applies."""
SPOKEN_LENGTHS = [6, 10, 15, 25]

GENRES3 = GENRES + ["legal contract summary", "government notice", "patent abstract", "wiki talk page comment",
                    "product specification sheet", "hospital discharge summary", "sports match report", "obituary",
                    "museum exhibit label", "travel guide paragraph", "restaurant menu description", "movie synopsis",
                    "tweet thread", "answer on a question-and-answer site", "tutorial step", "software changelog entry",
                    "science news article", "court judgment excerpt"]
DOMAINS3 = DOMAINS + ["ophthalmology", "dentistry", "mental health", "nutrition", "tennis", "association football",
                      "basketball", "baseball", "cricket", "motorsport", "cycling", "the Olympic Games", "chess",
                      "board games", "anime and manga", "theatre", "dance", "photography", "sculpture", "museums",
                      "classical antiquity", "medieval history", "military history", "genealogy", "urban planning",
                      "maritime industry", "railways", "space industry", "robotics", "semiconductors",
                      "cloud computing", "open-source software", "mobile apps", "social networks", "e-sports",
                      "tax law", "intellectual property", "immigration", "criminal justice", "human rights",
                      "international relations", "diplomacy", "local government", "labor unions", "supply chains",
                      "wine and beverages", "coffee and tea", "gardening", "pets", "zoology", "botany",
                      "marine biology", "ecology", "geology", "meteorology", "oceanography", "epidemiology",
                      "neuroscience", "materials science", "mathematics"]
FILL_PROMPT = """Text:
{text}

Entity types:
{types}

Current annotations:
{current}

Some mentions of these types may be missing from the current annotations. List only the missing mentions, exactly as they appear in the text, per type; use [] when nothing is missing for a type.
Return JSON only: {{"type name": ["span", ...]}}"""
FILL_SOURCES = ["finerweb", "pile", "synth_a", "synth_r", "synth_g", "synth_g2"]

C2_EXTRA = """
For diversity: give each new example a different, specific sub-topic that the examples above do not cover, and vary sentence structure and length."""

REAL_SOURCES = {   # repo, file filter, text field candidates, target passages
    "news": ("vblagoje/cc_news", "plain_text/train-00000", ["text"], 74000),
    "legal": ("scottyjmp5/courtlistener-legal-corpus", "", ["text", "opinion", "plain_text", "content"], 20000),
    "pubmed": ("omnibioai/pubmed-abstracts-36M", "", ["abstract", "text", "AbstractText"], 16000),
    "arxiv": ("gfissore/arxiv-abstracts-2021", "", ["abstract", "text"], 7000),
}


# ---------------------------------------------------------------- real text
def iter_records(path, max_records):
    """Records from a parquet / jsonl / json(.gz) file, at most max_records."""
    import gzip
    p = str(path)
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        n = 0
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                yield r
                n += 1
                if n >= max_records:
                    return
        return
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt") as f:
        first = f.read(1)
    if first == "[":                            # one JSON array
        with opener(p, "rt") as f:
            yield from json.load(f)[:max_records]
        return
    with opener(p, "rt") as f:                  # JSON lines
        for k, line in enumerate(f):
            if k >= max_records:
                return
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def fetch_real(args):
    from huggingface_hub import HfApi, hf_hub_download
    from .prepare_mix import contamination_filter
    api = HfApi()
    stats, meta, passages = collections.Counter(), {}, []
    rng = random.Random(args.seed)
    for name, (repo, filt, fields, target) in REAL_SOURCES.items():
        info = api.dataset_info(repo)
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=info.sha)
                       if filt in f and re.search(r"\.(parquet|jsonl|json|jsonl\.gz|json\.gz)$", f))
        got, used = [], []
        for fname in files:
            path = hf_hub_download(repo, fname, repo_type="dataset", revision=info.sha,
                                   cache_dir=str(ROOT / "raw/hf"))
            used.append(fname)
            for rec in iter_records(path, target * 4):
                text = next((rec[k] for k in fields if isinstance(rec.get(k), str) and rec[k].strip()), None)
                if not text:
                    continue
                for k, p in enumerate(chunk_passages(text, max_words=300, min_words=30)):
                    got.append({"id": f"{name}:{len(got)}", "text": p, "genre": name})
                    if k >= 2:          # at most 3 passages per document
                        break
            if len(got) >= target * 1.3:
                break
        rng.shuffle(got)
        passages += got[:int(target * 1.2)]
        meta[name] = {"repo": repo, "revision": info.sha, "files": used, "passages": len(got)}
        stats[f"{name}_passages"] = len(got)
    eval_texts = []
    for split in ("dev_select", "dev_calib", "test"):
        for src in ("finerweb", "fewnerd", "pile", "massive"):
            eval_texts += [r["text"] for r in read_jsonl(ROOT / f"data/mix2/{src}/{split}.jsonl")]
    for f in (ROOT / "data/ood").glob("*.jsonl"):
        eval_texts += [r["text"] for r in read_jsonl(f)]
    passages = contamination_filter(passages, eval_texts, stats, "real")
    by = collections.defaultdict(list)
    for p in passages:
        by[p["genre"]].append(p)
    passages = [p for name, (_, _, _, target) in REAL_SOURCES.items() for p in by[name][:target]]
    write_jsonl(S2 / "r1_requests.jsonl", passages)
    (S2 / "r1_requests.meta.json").write_text(json.dumps({"sources": meta, "stats": dict(stats),
                                                         "n": len(passages)}, indent=2))
    print(json.dumps({"stats": dict(stats), "n": len(passages)}))


# ---------------------------------------------------------------- request builders
def prepare_g(args):
    reqs, k = [], 0
    rng = random.Random(args.seed)
    cells = [(g, d) for g in GENRES for d in DOMAINS]
    per = -(-args.n // len(cells))
    for g, d in cells:
        for j in range(per):
            reqs.append({"id": f"g:{k}", "genre": g, "domain": d, "length": rng.choice(LENGTHS)})
            k += 1
    rng.shuffle(reqs)
    write_jsonl(S2 / "g_requests.jsonl", reqs[:args.n])
    print(f"g requests {min(args.n, len(reqs))} over {len(cells)} genre x domain cells")


def prepare_g2(args):
    rng = random.Random(args.seed + 2)
    reqs, k = [], 0
    per_group = args.n // len(G2_GROUPS)
    for group, (domains, genres) in G2_GROUPS.items():
        cells = [(g, d) for g in genres for d in domains]
        for j in range(per_group):
            g, d = cells[j % len(cells)]
            length = rng.choice(SPOKEN_LENGTHS if group == "spoken" else LENGTHS)
            reqs.append({"id": f"g2:{k}", "group": group, "genre": g, "domain": d, "length": length})
            k += 1
    rng.shuffle(reqs)
    write_jsonl(S2 / "g2_requests.jsonl", reqs)
    print(f"g2 requests {len(reqs)}: " + ", ".join(f"{g} {per_group}" for g in G2_GROUPS))


def prepare_g3(args):
    """Round-4 diverse synthetic texts: 40 genres x 120 domains, general + specific types."""
    rng = random.Random(args.seed + 3)
    cells = [(g, d) for g in GENRES3 for d in DOMAINS3]
    reqs = [{"id": f"g3:{k}", "genre": cells[k % len(cells)][0], "domain": cells[k % len(cells)][1],
             "length": rng.choice(LENGTHS)} for k in range(args.n)]
    rng.shuffle(reqs)
    write_jsonl(S2 / "g3_requests.jsonl", reqs)
    print(f"g3 requests {len(reqs)} over {len(cells)} genre x domain cells")


def prepare_fill(args):
    """Missing-mention check for LLM-annotated and FiNERWeb/Pile train rows (data/mix2)."""
    reqs = []
    for src in FILL_SOURCES:
        for r in read_jsonl(MIX_DIR / src / "train.jsonl"):
            qs = [q for q in r["queries"] if q["status"] in ("positive", "explicit_absent")]
            if not qs:
                continue
            reqs.append({"id": r["id"], "source": src, "text": r["text"],
                         "types": [(q["label"], q["description"]) for q in qs],
                         "current": {q["label"]: [r["text"][e["start"]:e["end"]] for e in r["entities"]
                                                  if e["label"] == q["label"]] for q in qs}})
    write_jsonl(S2 / "fill_requests.jsonl", reqs)
    print("fill requests", len(reqs), dict(collections.Counter(r["source"] for r in reqs)))


def apply_fill(args):
    """Add Qwen-found missing mentions to the train rows (backup *.pre_fill)."""
    from .prepare_mix import find_all
    reqs = {r["id"]: r for r in read_jsonl(S2 / "fill_requests.jsonl")}
    outs = load_outputs("fill")
    stats = collections.Counter()
    for src in FILL_SOURCES:
        path = MIX_DIR / src / "train.jsonl"
        rows = read_jsonl(path)
        for r in rows:
            o = outs.get(r["id"])
            if o is None or r["id"] not in reqs:
                continue
            js = parse_json(o["output"])
            if not isinstance(js, dict):
                stats[f"{src}_parse_fail"] += 1
                continue
            ans = {clean_type(k): v for k, v in js.items() if isinstance(v, list)}
            have = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
            for q in r["queries"]:
                v = [str(x).strip() for x in ans.get(clean_type(q["label"]), []) if str(x).strip()]
                if not v or q["status"] not in ("positive", "explicit_absent"):
                    continue
                occ = align(r["text"], v, find_all)
                if occ is None:
                    stats[f"{src}_unaligned"] += 1
                    continue
                new = [(s, e, q["label"]) for s, e in occ if (s, e, q["label"]) not in have]
                if not new:
                    continue
                if q["status"] == "explicit_absent":
                    stats[f"{src}_absent_to_positive"] += 1
                    q["status"], q["supervision_origin"] = "positive", "llm_fill_found"
                have |= set(new)
                stats[f"{src}_added_mentions"] += len(new)
            r["entities"] = sorted(({"start": s, "end": e, "label": l} for s, e, l in have),
                                   key=lambda x: (x["start"], x["label"]))
        path.rename(path.with_suffix(".jsonl.pre_fill"))
        write_jsonl(path, rows)
    (S2 / "fill_summary.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def prepare_c2(args):
    """Seeded requests: 5 CrossNER / MIT train-split examples each (tag c2, or c3 for
    round 5 with fresh example draws)."""
    tag = getattr(args, "tag", "c2")
    pools = seed_pool()
    reqs = []
    for dom, pool in pools.items():
        labels = schema_items(pool[0]["schema"])
        for k in range(args.per_domain):
            rng = random.Random(seed_of(f"{tag}:{dom}:{k}"))
            ex = rng.sample(pool, min(5, len(pool)))
            reqs.append({"id": f"{tag}:{dom}:{k}", "domain": dom, "labels": labels,
                         "examples": [format_example(e) for e in ex]})
    write_jsonl(S2 / f"{tag}_requests.jsonl", reqs)
    print(f"{tag} requests", len(reqs))


def type_pool(entries):
    """entries: iterable of (row_id, set(types), {type: definition}) -> pool helpers."""
    count, desc, co = collections.Counter(), {}, collections.defaultdict(collections.Counter)
    for _, types, d in entries:
        for t in types:
            count[t] += 1
            if t not in desc and d.get(t):
                desc[t] = d[t]
    by_word = collections.defaultdict(set)
    for t in count:
        for w in re.findall(r"[a-z]+", t.lower()):
            if w not in STOP and len(w) > 2:
                by_word[w].add(t)
    return count, desc, by_word


def pick_absent(present, count, desc, by_word, frequent, rng, n=8, need_desc=True):
    """Half confusable (shares a content word with a present type), half frequent."""
    near = set()
    for p in present:
        for w in re.findall(r"[a-z]+", p.lower()):
            near |= by_word.get(w, set())
    ok = lambda t: t not in present and (desc.get(t) or not need_desc) and count[t] >= 3
    near = sorted(t for t in near if ok(t))
    rng.shuffle(near)
    picked = near[:n // 2]
    rest = [t for t in frequent if ok(t) and t not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return picked, len(near[:n // 2])


def prepare_r2(args):
    reqs = {r["id"]: r for r in read_jsonl(S2 / "r1_requests.jsonl")}
    outs = load_outputs("r1")
    parsed = {pid: p for pid, o in outs.items() if (p := parse_a1(o["output"]))}
    count, desc, by_word = type_pool((pid, {t for _, t in e}, d) for pid, (e, d) in parsed.items())
    frequent = [t for t, _ in count.most_common(3000)]
    out, n_near = [], 0
    for pid, (ents, _) in parsed.items():
        rng = random.Random(seed_of(f"r2:{pid}"))
        picked, k = pick_absent({t for _, t in ents}, count, desc, by_word, frequent, rng)
        n_near += k
        out.append({"id": pid, "text": reqs[pid]["text"], "query_types": [(t, desc[t]) for t in picked]})
    write_jsonl(S2 / "r2_requests.jsonl", out)
    print(f"r2 requests {len(out)}, confusable picks {n_near}, pool {len(count)}")


def prepare_n(args):
    """FiNERWeb rows and Pile rows without explicit negatives, from data/mix2."""
    mix = ROOT / "data/mix2"
    fw = read_jsonl(mix / "finerweb/train.jsonl")
    pile = read_jsonl(mix / "pile/train.jsonl")
    out = []
    for name, rows, need_desc in (("finerweb", fw, True), ("pile", pile, False)):
        entries = [(r["id"], {q["label"] for q in r["queries"]},
                    {q["label"]: q["description"] for q in r["queries"]}) for r in rows]
        count, desc, by_word = type_pool(entries)
        frequent = [t for t, _ in count.most_common(3000)]
        n_rows = n_near = 0
        for r in rows:
            if name == "pile" and any(q["status"] == "explicit_absent" for q in r["queries"]):
                continue
            rng = random.Random(seed_of(f"n:{r['id']}"))
            present = {q["label"] for q in r["queries"]}
            picked, k = pick_absent(present, count, desc, by_word, frequent, rng, need_desc=need_desc)
            if not picked:
                continue
            n_rows += 1
            n_near += k
            out.append({"id": r["id"], "text": r["text"], "source": name,
                        "query_types": [(t, desc.get(t, "")) for t in picked]})
        print(f"n/{name}: {n_rows} rows, confusable picks {n_near}")
    write_jsonl(S2 / "n_requests.jsonl", out)


# ---------------------------------------------------------------- prompts / run
def build_prompt(task, r):
    if task == "g":
        return G_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"])
    if task in ("g2", "g3"):
        return G_PROMPT.format(genre=r["genre"], domain=r["domain"], length=r["length"]) + G2_EXTRA
    if task == "fill":
        types = "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in r["types"])
        current = "\n".join(f'- "{t}": {json.dumps(v, ensure_ascii=False)}' for t, v in r["current"].items())
        return FILL_PROMPT.format(text=r["text"], types=types, current=current)
    if task == "r1":
        return A1_PROMPT.format(text=r["text"])
    if task in ("r2", "n"):
        types = "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in r["query_types"])
        return A2_PROMPT.format(text=r["text"], types=types)
    types = "\n".join(f'- "{t}": {d}' if d else f'- "{t}"' for t, d in r["labels"])
    examples = "\n\n".join(f"Example {i + 1}:\n{e}" for i, e in enumerate(r["examples"]))
    return C_PROMPT.format(domain=C_DOMAIN_NAMES[r["domain"]], types=types, examples=examples,
                           n=5) + C2_EXTRA


def run(args):
    from vllm import LLM, SamplingParams
    reqs = read_jsonl(S2 / f"{args.task}_requests.jsonl")
    mine = [r for i, r in enumerate(reqs) if i % args.num_shards == args.shard][:args.limit]
    out_path = S2 / args.task / f"shard-{args.shard:05d}-of-{args.num_shards:05d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in mine if r["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} items, {len(done)} done, {len(todo)} to go",
          flush=True)
    if not todo:
        return
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, max_num_seqs=256, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sampled = args.task in ("g", "g2", "g3", "c2", "c3")
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        t0 = time.time()
        msgs = [[{"role": "user", "content": build_prompt(args.task, r)}] for r in chunk]
        params = ([SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens,
                                  seed=seed_of(r["id"])) for r in chunk] if sampled else
                  SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False})
        with open(out_path, "a") as f:
            for r, o in zip(chunk, outs):
                g = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "output": g.text, "finish_reason": g.finish_reason},
                                   ensure_ascii=False) + "\n")
        dt = time.time() - t0
        print(f"chunk {c // args.chunk}: {len(chunk)} items {dt:.0f}s {len(chunk) / dt:.2f} items/s", flush=True)


def load_outputs(task):
    out = {}
    for p in sorted((S2 / task).glob("shard-*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------- assemble
def mk_query(label, desc, status, origin, desc_origin="llm_authored"):
    return {"label": label, "description": desc or "", "status": status, "supervision_origin": origin,
            "description_origin": desc_origin if desc else "name_only"}


def dedup_entities(ents):
    return sorted([dict(x) for x in {tuple(sorted(e.items())) for e in ents}],
                  key=lambda e: (e["start"], e["label"]))


def absent_answers(output, asked):
    """Parse an r2 / n answer: {type: [spans]} over the asked (type, desc) list."""
    js = parse_json(output)
    if not isinstance(js, dict):
        return None
    ans = {clean_type(k): v for k, v in js.items()}
    res = []
    for t, d in asked:
        v = ans.get(clean_type(t))
        if isinstance(v, list):
            res.append((t, d, [str(x).strip() for x in v if str(x).strip()]))
    return res


def assemble(args):
    """Round 2: synth_g, synth_r, synth_c into data/mix2, then absent labels (n)
    added to FiNERWeb and Pile train rows."""
    from transformers import AutoTokenizer
    from .prepare_data import word_ngrams
    from .prepare_mix import find_all
    stats = collections.Counter()
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    mix = ROOT / "data/mix2"
    rows_by = {"synth_g": g_rows("g", "synth_g", stats, find_all),
               "synth_r": r_rows(stats, find_all),
               "synth_c": c_rows(stats, find_all, word_ngrams)}
    summary = finalize(args, rows_by, stats, tok, mix)
    apply_n(mix, stats, summary, find_all)
    write_summary(summary, stats, "assemble_summary.json")


def write_summary(summary, stats, name):
    summary["stats"] = dict(stats)
    (S2 / name).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:5000])


def g_rows(task, source, stats, find_all):
    """Parse synthetic genre x domain generations (their own schema) into rows."""
    rows = []
    for rid, o in load_outputs(task).items():
        js = parse_json(o["output"])
        if not isinstance(js, dict) or not isinstance(js.get("text"), str):
            stats[f"{task}_parse_fail"] += 1
            continue
        text = js["text"].strip()
        types = {clean_type(k): str(v) for k, v in (js.get("types") or {}).items()}
        absent = {clean_type(k): str(v) for k, v in (js.get("absent_types") or {}).items()}
        by = collections.defaultdict(list)
        for e in js.get("entities") or []:
            if isinstance(e, dict) and e.get("text") and clean_type(e.get("type", "")) in types:
                by[clean_type(e["type"])].append(str(e["text"]).strip())
        queries, ents = [], []
        for t, d in types.items():
            if t in by:
                occ = align(text, by[t], find_all)
                if occ is None:
                    queries.append(mk_query(t, d, "unknown", "llm_unaligned"))
                    stats[f"{task}_unaligned_types"] += 1
                else:
                    queries.append(mk_query(t, d, "positive", "llm_generated"))
                    ents += [{"start": s, "end": e, "label": t} for s, e in occ]
            else:
                queries.append(mk_query(t, d, "explicit_absent", "llm_schema_unused"))
        for t, d in absent.items():
            if t not in types:
                queries.append(mk_query(t, d, "explicit_absent", "llm_listed_absent"))
        if any(q["status"] == "positive" for q in queries):
            rows.append({"id": f"{source}:{rid}", "source": source, "text": text, "queries": queries,
                         "entities": dedup_entities(ents)})
    return rows


def assemble_c(args):
    """Rebuild MIX_DIR/synth_c from all seeded generations (c2 + c3 + round 1)."""
    from transformers import AutoTokenizer
    from .prepare_data import word_ngrams
    from .prepare_mix import find_all
    stats = collections.Counter()
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    summary = finalize(args, {"synth_c": c_rows(stats, find_all, word_ngrams)}, stats, tok, MIX_DIR)
    write_summary(summary, stats, "assemble_c_summary.json")


def assemble_g2(args):
    """Build data/mix2/synth_g2 (round-3 music / literature / spoken synthetic text) only."""
    from transformers import AutoTokenizer
    from .prepare_mix import find_all
    stats = collections.Counter()
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    task = getattr(args, "task", "g2")
    rows_by = {f"synth_{task}": g_rows(task, f"synth_{task}", stats, find_all)}
    summary = finalize(args, rows_by, stats, tok, MIX_DIR)
    write_summary(summary, stats, f"assemble_{task}_summary.json")


def r_rows(stats, find_all):
    """Real text: r1 open types + r2 absent answers (hard negatives)."""
    reqs = {r["id"]: r for r in read_jsonl(S2 / "r1_requests.jsonl")}
    r2reqs = {r["id"]: r for r in read_jsonl(S2 / "r2_requests.jsonl")} if (S2 / "r2_requests.jsonl").exists() else {}
    r2 = load_outputs("r2")
    rows = []
    for pid, o in load_outputs("r1").items():
        p = parse_a1(o["output"])
        if p is None:
            stats["r1_parse_fail"] += 1
            continue
        text, (ents_raw, types) = reqs[pid]["text"], p
        by = collections.defaultdict(list)
        for s, t in ents_raw:
            by[t].append(s)
        queries, ents = [], []
        for t, spans in by.items():
            occ = align(text, spans, find_all)
            if occ is None:
                queries.append(mk_query(t, types.get(t, ""), "unknown", "llm_unaligned"))
            else:
                queries.append(mk_query(t, types.get(t, ""), "positive", "llm_annotated"))
                ents += [{"start": s, "end": e, "label": t} for s, e in occ]
        if pid in r2 and pid in r2reqs:
            for t, d, v in absent_answers(r2[pid]["output"], r2reqs[pid]["query_types"]) or []:
                if not v:
                    queries.append(mk_query(t, d, "explicit_absent", "llm_absent_hard"))
                    stats["r2_absent"] += 1
                elif (occ := align(text, v, find_all)) is not None:
                    queries.append(mk_query(t, d, "positive", "llm_r2_found"))
                    ents += [{"start": s, "end": e, "label": t} for s, e in occ]
                    stats["r2_found"] += 1
        if any(q["status"] != "unknown" for q in queries):
            rows.append({"id": f"synth_r:{pid}", "source": "synth_r", "genre": reqs[pid]["genre"],
                         "text": text, "queries": queries, "entities": dedup_entities(ents)})
    return rows


def c_rows(stats, find_all, word_ngrams):
    """Target-style rows seeded with CrossNER / MIT train examples: round-1 seeded
    train rows plus c2 and (if generated) c3."""
    tags = [t for t in ("c2", "c3") if (S2 / f"{t}_requests.jsonl").exists()]
    reqs = {r["id"]: r for t in tags for r in read_jsonl(S2 / f"{t}_requests.jsonl")}
    pools = seed_pool()
    seed_grams = {d: [word_ngrams(r["input"]) for r in pool] for d, pool in pools.items()}
    rows = []
    outputs = {rid: o for t in tags for rid, o in load_outputs(t).items()}
    for rid, o in outputs.items():
        req, js = reqs[rid], parse_json(o["output"])
        if not isinstance(js, dict) or not isinstance(js.get("examples"), list):
            stats["c2_parse_fail"] += 1
            continue
        labels = dict(req["labels"])
        for k, ex in enumerate(js["examples"]):
            if not isinstance(ex, dict) or not isinstance(ex.get("text"), str):
                continue
            text = ex["text"].strip()
            g = word_ngrams(text)
            if g and any(len(g & sg) / len(g) >= 0.8 for sg in seed_grams[req["domain"]] if sg):
                stats["c2_copied_seed"] += 1
                continue
            by, bad = collections.defaultdict(list), False
            for e in ex.get("entities") or []:
                if not isinstance(e, dict) or e.get("type") not in labels or not e.get("text"):
                    bad = True
                    continue
                by[e["type"]].append(str(e["text"]).strip())
            queries, ents = [], []
            for lab, desc in labels.items():
                if lab in by:
                    occ = align(text, by[lab], find_all)
                    if occ is None:
                        queries.append(mk_query(lab, desc, "unknown", "llm_unaligned", "source"))
                    else:
                        queries.append(mk_query(lab, desc, "positive", "llm_generated", "source"))
                        ents += [{"start": s, "end": e, "label": lab} for s, e in occ]
                else:
                    queries.append(mk_query(lab, desc, "unknown" if bad else "scope_assumed_absent",
                                            "llm_generated_closed_world", "source"))
            rows.append({"id": f"synth_c:{rid}:{k}", "source": "synth_c", "domain": req["domain"],
                         "text": text, "queries": queries, "entities": dedup_entities(ents)})
    old = ROOT / "data/mix/synth_c/train.jsonl"
    return rows + (read_jsonl(old) if old.exists() else [])


def finalize(args, rows_by, stats, tok, mix):
    """Schema / length filters, OOD-test dedup, 80/10/5/5 split, write per source."""
    from .prepare_data import SPLITS, norm_text, sha, split_groups
    from .prepare_mix import contamination_filter, fixed_eval_queries
    ood_test = [r["text"] for f in (ROOT / "data/ood").glob("*.jsonl") for r in read_jsonl(f)]
    summary = {}
    for src, rows in rows_by.items():
        before = len(rows)
        rows = [r for r in rows if sum(len(q["label"]) + len(q["description"]) + 3 for q in r["queries"]) <= 4000]
        stats[f"{src}_drop_long_schema"] = before - len(rows)
        lengths = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
        before = len(rows)
        rows = [r for r, ids in zip(rows, lengths) if len(ids) <= args.max_tokens]
        stats[f"{src}_drop_over_{args.max_tokens}"] = before - len(rows)
        rows = contamination_filter(rows, ood_test, stats, src)
        by_hash = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_hash[sha(norm_text(r["text"]))].append(i)
        assign = split_groups(list(by_hash.values()), len(rows), args.seed)
        info = {}
        for s, _ in SPLITS:
            part = [rows[i] for i in range(len(rows)) if assign[i] == s]
            part = part if s == "train" else fixed_eval_queries(part)
            write_jsonl(mix / src / f"{s}.jsonl", part)
            info[s] = {"rows": len(part), "status": dict(collections.Counter(
                q["status"] for r in part for q in r["queries"]))}
        summary[src] = info
    return summary


def apply_n(mix, stats, summary, find_all):
    """Add verified absent labels (task n) to FiNERWeb and Pile train rows."""
    nreqs = {r["id"]: r for r in read_jsonl(S2 / "n_requests.jsonl")} if (S2 / "n_requests.jsonl").exists() else {}
    nout = load_outputs("n")
    for src in ("finerweb", "pile"):
        path = mix / src / "train.jsonl"
        rows = read_jsonl(path)
        for r in rows:
            if r["id"] not in nout or r["id"] not in nreqs:
                continue
            present = {q["label"] for q in r["queries"]}
            for t, d, v in absent_answers(nout[r["id"]]["output"], nreqs[r["id"]]["query_types"]) or []:
                if t in present:
                    continue
                if not v:
                    r["queries"].append(mk_query(t, d, "explicit_absent", "llm_absent_hard",
                                                 "source" if d else "name_only"))
                    stats[f"n_{src}_absent"] += 1
                elif (occ := align(r["text"], v, find_all)) is not None:
                    r["queries"].append(mk_query(t, d, "positive", "llm_n_found", "source" if d else "name_only"))
                    r["entities"] = dedup_entities(r["entities"] + [{"start": s, "end": e, "label": t}
                                                                    for s, e in occ])
                    stats[f"n_{src}_found"] += 1
        write_jsonl(path, rows)
        summary[f"{src}_train_after_n"] = {"rows": len(rows), "status": dict(collections.Counter(
            q["status"] for r in rows for q in r["queries"]))}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-real"); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-g"); p.add_argument("--n", type=int, default=120000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-c2"); p.add_argument("--per-domain", type=int, default=2060)
    p.add_argument("--tag", default="c2", choices=["c2", "c3"])
    p = sub.add_parser("assemble-c"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024)
    sub.add_parser("prepare-r2")
    sub.add_parser("prepare-n")
    p = sub.add_parser("assemble"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024)
    p = sub.add_parser("prepare-g2"); p.add_argument("--n", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("assemble-g2"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--task", default="g2", choices=["g2", "g3"])
    p = sub.add_parser("prepare-g3"); p.add_argument("--n", type=int, default=370000)
    p.add_argument("--seed", type=int, default=42)
    sub.add_parser("prepare-fill")
    sub.add_parser("apply-fill")
    p = sub.add_parser("run")
    p.add_argument("--task", choices=["g", "g2", "g3", "c2", "c3", "r1", "r2", "n", "fill"], required=True)
    p.add_argument("--model", default="/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-model-len", type=int, default=6144)
    args = ap.parse_args()
    S2.mkdir(parents=True, exist_ok=True)
    {"fetch-real": fetch_real, "prepare-g": prepare_g, "prepare-c2": prepare_c2, "prepare-r2": prepare_r2,
     "prepare-n": prepare_n, "assemble": assemble, "prepare-g2": prepare_g2, "assemble-g2": assemble_g2,
     "prepare-g3": prepare_g3, "prepare-fill": prepare_fill, "apply-fill": apply_fill, "assemble-c": assemble_c,
     "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
