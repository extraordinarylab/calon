# 訓練資料格式

每行一個 JSON 物件。每個資料來源一個資料夾，分四個切分：

```
data/<mix>/<source>/train.jsonl
data/<mix>/<source>/dev_select.jsonl   # 挑選存檔點
data/<mix>/<source>/dev_calib.jsonl    # 挑溫度與門檻
data/<mix>/<source>/test.jsonl
```

位置一律是 `text` 裡的字元位置，`end` 不含（`text[start:end]` 就是片段）。

## NER 一列

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

`queries` 是這一列要問的類型；`description` 可以是 `""`（只給名稱）。

| `status` | 意思 | 會不會拿來訓練 |
|---|---|---|
| `positive` | 這個類型有出現，所有提及都列在 `entities` | 會 |
| `explicit_absent` | 確認過，這個類型沒有出現 | 會 |
| `scope_assumed_absent` | 沒逐一確認，但這個來源會把列出的類型標全（封閉世界） | 會 |
| `unknown` | 可能有出現但沒標（部分標注） | 不會 |

每個實體的 `label` 都必須在這一列的 `queries` 裡。

## RE 一列

NER 一列再加 `rel_queries` 和 `relations`：

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

- 關係有方向（頭 → 尾），頭和尾都必須是 `entities` 裡的片段。
- `rel_queries` 的 `status` 跟 `queries` 一樣。
- 每列最多取 24 個實體片段來組成有序配對。

## EE 兩列

每段文字拆成兩列。

觸發詞列：事件類型當片段類型，觸發詞是它的片段。

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

論元列：事件類型和角色都當片段類型；角色同時也是「觸發詞 → 論元」的關係類型。

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

論元也可以是另一個事件的觸發詞：那個片段同時加上角色類型，再加一條從外層觸發詞指向它的關係。

## 限制

```python
MAX_TEXT_TOKENS = 1024   # 超過就整列丟掉（filter_long）
W = 32                   # 片段最長 32 個 token
MAX_PAIR_ENTITIES = 24   # 每列用來組關係配對的實體片段上限
```

片段邊界沒有落在 token 邊界上時，會被記為「無法表示」，不會拿來訓練。

## 加入訓練

把資料夾放進混合資料目錄，並在每一步裡給它一個比例：

```bash
# --mix：比例會自動正規化；--relations 這幾個參數只在資料有關係時加
torchrun --nproc_per_node=4 -m calon.train.train --data data/mix --out runs/calon \
    --mix fewnerd:0.3,mysource:0.2,rebel:0.5 --encoder Qwen/Qwen3-0.6B --head rlcd --mode bce \
    --none_weight 0.1 --desc_dropout 0.3 --batch 128 --micro 8 --updates 6000 --enc_lr 4e-5 --head_lr 2e-4 \
    --relations --re_weight 1.0 --re_none_weight 0.1 --seed 42
```
