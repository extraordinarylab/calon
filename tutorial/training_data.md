# Training data format

One JSON object per line. One directory per source, four splits:

```
data/<mix>/<source>/train.jsonl
data/<mix>/<source>/dev_select.jsonl   # checkpoint selection
data/<mix>/<source>/dev_calib.jsonl    # temperature and threshold
data/<mix>/<source>/test.jsonl
```

Offsets are character offsets into `text`, `end` exclusive (`text[start:end]` is the span).

## NER row

```json
{
  "id": "mysource:0",
  "source": "mysource",
  "text": "Maya Lin designed the Vietnam Veterans Memorial.",
  "queries": [
    {"label": "person",   "description": "Names of real or fictional people.", "status": "positive"},
    {"label": "artwork",  "description": "Titles of works of art.",            "status": "positive"},
    {"label": "location", "description": "Names of places.",                   "status": "explicit_absent"},
    {"label": "award",    "description": "",                                   "status": "unknown"}
  ],
  "entities": [
    {"start": 0,  "end": 8,  "label": "person"},
    {"start": 22, "end": 47, "label": "artwork"}
  ]
}
```

`queries` are the types asked in this row. `description` may be `""` (name only).

| `status` | meaning | trained |
|---|---|---|
| `positive` | the type occurs; all its mentions are in `entities` | yes |
| `explicit_absent` | the type was checked and does not occur | yes |
| `scope_assumed_absent` | not checked, but the source annotates every type it lists (closed world) | yes |
| `unknown` | may occur unannotated (partial annotation) | no |

Every entity `label` must be one of the row's `queries`.

## RE row

A NER row plus `rel_queries` and `relations`:

```json
{
  "id": "mysource:1",
  "source": "mysource",
  "text": "Maya Lin designed the Vietnam Veterans Memorial.",
  "queries":  [{"label": "person", "description": "", "status": "positive"},
               {"label": "artwork", "description": "", "status": "positive"}],
  "entities": [{"start": 0, "end": 8, "label": "person"},
               {"start": 22, "end": 47, "label": "artwork"}],
  "rel_queries": [
    {"label": "creator of", "description": "The head made the tail.", "status": "positive"},
    {"label": "owner of",   "description": "The head owns the tail.", "status": "explicit_absent"}
  ],
  "relations": [
    {"head": [0, 8], "tail": [22, 47], "label": "creator of"}
  ]
}
```

- Relations are directed (head to tail) and hold between spans listed in `entities`.
- `rel_queries` use the same `status` values as `queries`.
- Ordered pairs are formed over at most 24 entity spans per row.

## EE rows

Each text becomes two rows.

Trigger row: event types are span labels, triggers are their spans.

```json
{
  "id": "mysource:2:trig",
  "source": "mysource",
  "text": "Hackers demanded $2 million in Bitcoin from the hospital.",
  "queries": [
    {"label": "ransom",      "description": "", "status": "positive"},
    {"label": "data breach", "description": "", "status": "explicit_absent"}
  ],
  "entities": [{"start": 8, "end": 16, "label": "ransom"}]
}
```

Argument row: event types and roles are span labels; roles are also relation types from
the trigger to each argument.

```json
{
  "id": "mysource:2:args",
  "source": "mysource",
  "text": "Hackers demanded $2 million in Bitcoin from the hospital.",
  "queries": [
    {"label": "ransom",         "description": "", "status": "positive"},
    {"label": "attacker",       "description": "", "status": "positive"},
    {"label": "price",          "description": "", "status": "positive"},
    {"label": "payment method", "description": "", "status": "positive"},
    {"label": "victim",         "description": "", "status": "positive"}
  ],
  "entities": [
    {"start": 8,  "end": 16, "label": "ransom"},
    {"start": 0,  "end": 7,  "label": "attacker"},
    {"start": 17, "end": 27, "label": "price"},
    {"start": 31, "end": 38, "label": "payment method"},
    {"start": 44, "end": 56, "label": "victim"}
  ],
  "rel_queries": [
    {"label": "attacker",       "description": "", "status": "positive"},
    {"label": "price",          "description": "", "status": "positive"},
    {"label": "payment method", "description": "", "status": "positive"},
    {"label": "victim",         "description": "", "status": "positive"}
  ],
  "relations": [
    {"head": [8, 16], "tail": [0, 7],   "label": "attacker"},
    {"head": [8, 16], "tail": [17, 27], "label": "price"},
    {"head": [8, 16], "tail": [31, 38], "label": "payment method"},
    {"head": [8, 16], "tail": [44, 56], "label": "victim"}
  ]
}
```

An argument may be another event's trigger: give it the role label as well and add the
relation from the outer trigger to it.

## Limits

```python
MAX_TEXT_TOKENS = 1024   # longer rows are dropped (filter_long)
W = 32                   # a span is at most 32 tokens wide
MAX_PAIR_ENTITIES = 24   # entity spans used for relation pairs per row
```

A span whose boundaries do not fall on token boundaries is counted as unrepresented and
not trained.

## Using a source

Add the directory under the mix and give it a share of each step:

```bash
# --mix: shares are normalised; the --relations flags only when rows have relations
torchrun --nproc_per_node=4 -m calon.train.train --data data/mix --out runs/calon \
    --mix fewnerd:0.3,mysource:0.2,rebel:0.5 --encoder Qwen/Qwen3-0.6B --head rlcd --mode bce \
    --none_weight 0.1 --desc_dropout 0.3 --batch 128 --micro 8 --updates 6000 --enc_lr 4e-5 --head_lr 2e-4 \
    --relations --re_weight 1.0 --re_none_weight 0.1 --seed 42
```
