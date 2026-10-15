"""Verified absent event types (round 8, from the case study against gpt-oss): in training most asked
event types occur in the text (MAVEN-Arg 90%, CC-News 57%, multi-domain 58%), in the evaluation sets
few do (M2E2 2%, SENTiVENT 6%), and the model over-predicts triggers. Each trigger document gets
further absent event types until about TARGET of its asked types are positive (at most MAX_ADD more):
candidates from the same source's type pool (three quarters the types closest in meaning to a present
type, by sentence embeddings of the names; the rest random),
kept only when gpt-oss answers that the text has no event of that type.

  prepare  requests evneg (one per document, 2 x the needed candidates)
  apply    docs with the confirmed absent types added to "absent" (explicit_absent in the trigger
           row), written to <out>/ee/<source>/docs_*.jsonl (dev / test copied unchanged), then rows
"""
import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from .llm_annotate import read_jsonl, write_jsonl
from .paths import S2

TARGET, MAX_ADD = 0.2, 8
EVNEG_PROMPT = """Text:
{text}

Event types:
{types}

For each event type above, answer "yes" if the text mentions at least one event of that type (an occurrence, action or change of state that fits the definition), otherwise "no".
Return JSON only: {{"T1": "yes" or "no", ...}}"""


def prompt_for(task, r):
    return EVNEG_PROMPT.format(text=r["text"], types="\n".join(f'T{k + 1}: "{t}"' + (f": {d}" if d else "")
                                                               for k, (t, d) in enumerate(r["cands"])))


def src_dir(mix, src):
    return Path(mix).joinpath(src).resolve()


def neighbours(names, k=30):
    """{type: the k type names closest in meaning} (sentence embeddings of the names): hard negatives."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    e = torch.tensor(np.asarray(model.encode(names, batch_size=512, normalize_embeddings=True)), device="cuda").half()
    out = {}
    for i in range(0, len(names), 4096):
        s = e[i:i + 4096] @ e.T
        top = s.topk(k + 1, dim=1).indices.cpu().numpy()
        for j, row in enumerate(top):
            out[names[i + j]] = [names[x] for x in row if x != i + j][:k]
    return out


def prepare(args):
    reqs, stats = [], collections.Counter()
    for src in args.sources.split(","):
        docs = read_jsonl(src_dir(args.mix, src) / "docs_train.jsonl")
        pool = {}
        for d in docs:
            for t, v in d["schema"].items():
                pool.setdefault(t, v[0])
            for t, v in d["absent"].items():
                pool.setdefault(t, v)
        names = sorted(pool)
        nn = neighbours(names)
        for k, d in enumerate(docs):
            pos = {t for _, _, t in d["triggers"]}
            asked = set(d["schema"]) | set(d["absent"])
            need = min(MAX_ADD, max(0, int(len(pos) / TARGET) - len(asked)))
            stats[f"{src}_docs"] += 1
            if not need:
                continue
            rng = random.Random(f"evneg:{src}:{k}")
            near = list(dict.fromkeys(x for t in sorted(pos) for x in nn.get(t, [])[:12] if x not in asked))
            rng.shuffle(near)
            cands = near[:need + need // 2]          # mostly near types, some random ones
            while len(cands) < 2 * need:
                x = names[rng.randrange(len(names))]
                if x not in asked and x not in cands:
                    cands.append(x)
            reqs.append({"id": f"{src}#{k}", "text": d["text"], "need": need,
                         "cands": [[t, pool[t]] for t in cands]})
            stats[f"{src}_requests"] += 1
            stats[f"{src}_candidates"] += len(cands)
    write_jsonl(S2 / "evneg_requests.jsonl", reqs)
    print(json.dumps(dict(stats)))


def apply(args):
    from .ee_gen import load_outputs, parse
    reqs = {r["id"]: r for r in read_jsonl(S2 / "evneg_requests.jsonl")}
    ans = load_outputs("evneg")
    stats = collections.Counter()
    for src in args.sources.split(","):
        d0 = src_dir(args.mix, src)
        o = Path(args.out) / "ee" / src
        o.mkdir(parents=True, exist_ok=True)
        for sp in ("dev_select", "dev_calib", "test"):
            shutil.copy(d0 / f"docs_{sp}.jsonl", o / f"docs_{sp}.jsonl")
        docs = read_jsonl(d0 / "docs_train.jsonl")
        pos_share = [0, 0]
        for k, d in enumerate(docs):
            q = reqs.get(f"{src}#{k}")
            o_ = ans.get(f"{src}#{k}")
            if q and o_:
                js = parse(o_["output"])
                added = 0
                for i, (t, desc) in enumerate(q["cands"]):
                    v = str(js.get(f"T{i + 1}", "")).strip().lower() if isinstance(js, dict) else ""
                    if v.startswith("n") and added < q["need"]:
                        d["absent"][t] = desc
                        added += 1
                    elif v.startswith("y"):
                        stats[f"{src}_candidate_present"] += 1
                stats[f"{src}_absent_added"] += added
            npos = len({t for _, _, t in d["triggers"]})
            pos_share[0] += npos
            pos_share[1] += len(set(d["schema"]) | set(d["absent"]))
        write_jsonl(o / "docs_train.jsonl", docs)
        stats[f"{src}_positive_share_after"] = round(pos_share[0] / max(1, pos_share[1]), 3)
    print(json.dumps(dict(stats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "apply"])
    ap.add_argument("--mix", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"prepare": prepare, "apply": apply}[args.stage](args)


if __name__ == "__main__":
    main()
