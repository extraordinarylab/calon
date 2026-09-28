"""Leakage audit: training texts vs OOD test texts, insensitive to punctuation spacing.

Round-1/2 dedup normalized only case and whitespace, so "classifier s, Gaussian"
and "classifier s , Gaussian" were different strings. Here texts are reduced to
lowercase alphanumeric words: exact = same word sequence; near = word-5-gram
containment >= 0.8 in either direction. Reports counts per training source and
OOD set, plus a few examples.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def key(t):
    return hashlib.sha1(" ".join(words(t)).encode()).hexdigest()


def grams(t, n=5):
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", nargs="+", default=[str(ROOT / "data/mix"), str(ROOT / "data/mix2")])
    ap.add_argument("--out", default=str(ROOT / "runs2/leak_audit.json"))
    args = ap.parse_args()
    ood = {}
    for d in (ROOT / "data/ood", ROOT / "data/ood_pii"):
        for f in sorted(d.glob("*.jsonl")):
            ood[f.stem] = [json.loads(l)["text"] for l in open(f)]
    exact, idx, og = {}, collections.defaultdict(list), []
    for name, texts in ood.items():
        for t in texts:
            exact.setdefault(key(t), name)
            g = grams(t)
            og.append((name, g, t))
            for x in g:
                idx[x].append(len(og) - 1)
    report = {}
    for mix in args.mix:
        for src_dir in sorted(Path(mix).iterdir()):
            f = src_dir / "train.jsonl"
            if not f.exists():
                continue
            c, ex = collections.Counter(), []
            for line in open(f):
                t = json.loads(line)["text"]
                k = key(t)
                if k in exact:
                    c[f"exact/{exact[k]}"] += 1
                    if len(ex) < 3:
                        ex.append(("exact", exact[k], t[:160]))
                    continue
                g = grams(t)
                if not g:
                    continue
                shared = collections.Counter(j for x in g for j in idx.get(x, ()))
                for j, s in shared.items():
                    name, gj, tj = og[j]
                    if gj and (s / len(gj) >= 0.8 or s / len(g) >= 0.8):
                        c[f"near/{name}"] += 1
                        if len(ex) < 3:
                            ex.append(("near", name, t[:120], tj[:120]))
                        break
            report[f"{Path(mix).name}/{src_dir.name}"] = {"counts": dict(c), "examples": ex}
            print(f"{Path(mix).name}/{src_dir.name}: {dict(c)}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
