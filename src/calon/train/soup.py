"""Model soup (Wortsman et al. 2022): the element-wise mean of several checkpoints' weights.
Our seeds share the pretrained encoder but initialise the heads differently, so a uniform soup
may fall outside every seed's basin; the greedy soup (seeds added in order of in-domain dev F1,
kept only if the dev F1 does not drop) is the safer variant and is chosen by the caller."""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acc = None
    for p in args.ckpts:
        sd = torch.load(p, map_location="cpu")
        if acc is None:
            acc = {k: v.double().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
        del sd
    n = len(args.ckpts)
    ref = torch.load(args.ckpts[0], map_location="cpu")
    out = {k: (v / n).to(ref[k].dtype) if v.is_floating_point() else v for k, v in acc.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("soup of", n, "->", args.out)


if __name__ == "__main__":
    main()
