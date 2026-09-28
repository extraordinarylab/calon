"""Token-span candidates mapped back to character offsets (SPEC 3)."""

MAX_TEXT_TOKENS = 512


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
