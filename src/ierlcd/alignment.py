"""Token-span candidates mapped back to character offsets (SPEC 3)."""
import numpy as np

MAX_TEXT_TOKENS = 1024        # one window; texts are pre-filtered to fit (was 512 in round 1)


def build_candidates_np(text, offsets, W):
    """Vectorized build_candidates with identical output (see tests)."""
    n = len(offsets)
    if n == 0:
        return []
    L = len(text)
    space = np.array([c.isspace() for c in text], dtype=bool)
    # next_ns[c]: first non-space index >= c ; prev_ns[c]: (last non-space index < c) + 1
    idx = np.arange(L + 1)
    ns = np.where(~space[:L], idx[:L], L)
    next_ns = np.minimum.accumulate(np.append(ns, L)[::-1])[::-1]
    last = np.where(~space[:L], idx[:L] + 1, 0)
    prev_ns = np.concatenate([[0], np.maximum.accumulate(last)])
    off = np.asarray(offsets, dtype=np.int64).reshape(n, 2)
    i = np.repeat(np.arange(n), W)
    j = i + np.tile(np.arange(W), n)
    keep = j < n
    i, j = i[keep], j[keep]
    s = next_ns[off[i, 0]]
    e = prev_ns[off[j, 1]]
    ok = s < e
    s, e, i, j = s[ok], e[ok], i[ok], j[ok]
    # per (s, e) keep min (j - i, i, j)
    order = np.lexsort((j, i, j - i, e, s))
    s, e, i, j = s[order], e[order], i[order], j[order]
    first = np.ones(len(s), dtype=bool)
    first[1:] = (s[1:] != s[:-1]) | (e[1:] != e[:-1])
    s, e, i, j = s[first], e[first], i[first], j[first]
    order = np.lexsort((j, i))
    return list(zip(s[order].tolist(), e[order].tolist(), i[order].tolist(), j[order].tolist()))


def build_candidates(text, offsets, W):
    """All token spans [i, j] with j - i < W, as right-open char spans.

    Outer whitespace is stripped deterministically. When several token spans
    map to the same char span, keep the one with fewest tokens, then smallest
    (i, j). Returns a list of (start, end, i, j) sorted by (i, j).
    """
    best = {}
    n = len(offsets)
    for i in range(n):
        for j in range(i, min(n, i + W)):
            s, e = offsets[i][0], offsets[j][1]
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s >= e:
                continue
            key = (j - i, i, j)
            if (s, e) not in best or key < best[(s, e)]:
                best[(s, e)] = key
    return sorted(((s, e, i, j) for (s, e), (_, i, j) in best.items()),
                  key=lambda c: (c[2], c[3]))


def tokenize_text(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]


def coverage(text, offsets, entities, W):
    """Boundary alignment, width coverage and total candidate recall for gold spans.

    boundary: some token starts at gold start and some token ends at gold end
    (after whitespace stripping); width: additionally within W tokens.
    """
    starts, ends = {}, {}
    for k, (s, e) in enumerate(offsets):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            starts.setdefault(s, k)
            ends[e] = k
    out = {"gold": 0, "boundary": 0, "width": 0}
    for ent in entities:
        out["gold"] += 1
        i, j = starts.get(ent["start"]), ends.get(ent["end"])
        if i is not None and j is not None and j >= i:
            out["boundary"] += 1
            if j - i < W:
                out["width"] += 1
    return out
