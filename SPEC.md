# CalOn：英文 schema-conditioned 資訊抽取 — Spec v0.3

更新：2026-09-28。範圍：英文 NER、character offsets、label descriptions；RE／EE 等 RLCD 贏過基準模型後再加入。

**命名（2026-09-29）**：方法改名為 CalOn（讀作 KAL-on）。它是「校準後的型別或 null 決策」：每個候選片段（或有方向的實體對）在 null 和各型別之間選一個，只有 1 − p(null) 超過門檻才輸出。現在的訓練目標沒有強化學習的步驟，Brier 權重是 0，所以不再稱為 RLCD。下文與舊報告中的「RLCD」都是指同一個方法；程式碼裡的 `--head rlcd` 會自動當成 `--head calon`。

**第 0 節是目前實際在跑的做法（第 4 輪實驗）。第 1–6 節和附錄是 v0.2 的原始規格，保留作紀錄；兩者不一致時以第 0 節為準。**

## 0. 目前實際做法（v0.3，第 4 輪）

### 0.1 模型

| 項目 | 目前做法 | 和 v0.2 的差別 |
|---|---|---|
| 編碼器（把文字變成向量的主體） | **Qwen3-0.6B，改成雙向**：每個字都能看到前後文（關掉只能看左邊的遮罩，只遮 padding） | v0.2 用 ModernBERT-base；ModernBERT 和 Qwen2.5-0.5B 在前幾輪比過，Qwen3-0.6B 最好，也能處理 1,024 個 token 的長文 |
| 權重精度 | 訓練時主權重用 fp32 | 下載的權重是 bf16，直接訓練不穩 |
| 輸入格式、候選片段、打分 | 同 v0.2：`<LABEL> 名稱 : 說明 ... <TEXT> 原文`；候選片段最長 32 個 token；分數 = 片段向量 · 類別向量 / √256 + 偏差 | 文字上限 1,024 token，schema＋文字總長 3,072 |
| 訓練規模 | 4 台機器 × 4 張卡一起訓練；一次 128 筆，6,000 步；學習率 encoder 4e-5、head 2e-4 | v0.2 是單卡、一次 32 筆 |
| 類別說明隨機拿掉 | 訓練時 30% 的類別只給名稱、不給說明（desc dropout） | 讓模型在只有類別名稱時也能用 |

### 0.2 RLCD 現在怎麼做（依 blog 版本改寫）

參考：<https://di-zhang-llm.github.io/blog/what-is-rlcd-the-secret-behind-jev/>

v0.2 的做法是「每個片段對每個類別各自判斷是或不是，加上隨機抖動後用 policy gradient 更新」。這個做法在前幾輪沒有比單純的 BCE（每個片段對每個類別各自判斷是不是）好，所以已經不用了。

**現在的 RLCD：每個候選片段在「不是實體」和這次詢問的所有類別之間只選一個。** 例：片段「Taylor Swift」，這次問 [person, musical artist, city]，模型給出 [不是實體 3%、person 40%、musical artist 55%、city 2%]。

1. **分數**：每個類別一個分數，另外加一個「不是實體」的分數。「不是實體」用一個學出來的向量表示。所有分數經過 softmax 變成機率，加總為 1。
2. **訓練目標**：
   - 正確類別可能不只一個，例如同一片段同時是 person 和 musical artist。所以把正確類別的機率加總，要求這個總和越大越好（log loss）。
   - 可以另外加上 Brier 分數（機率和正確答案差距的平方）。目前設定的權重是 0，因為第 3 輪只看訓練資料的驗證集時，0 最好。
3. **哪些片段參與訓練**：
   - 類別狀態是 unknown（沒標不代表不存在）的那一筆，只有標了正確答案的片段參與訓練。
   - 其他片段若都不是實體，正確答案就是「不是實體」。
4. **判定為實體的規則（entity 解碼）**：
   - 1 − P(不是實體) 超過門檻，就判定這是實體，類別取機率最高的那一個。
   - 第 3 輪的錯誤分析發現舊的規則有問題。舊規則是每個類別各自比門檻，但同一個人的機率會被 person 和 musical artist 分掉，兩個都不到門檻，結果整個漏掉。
5. **溫度和門檻**：只在訓練來源的校準集（dev_calib）上決定。溫度用 Brier 最小來選，再選 F1 最高的門檻。測試資料完全不參與。

**第 4 輪改了什麼（之前的實作錯誤和新加的設定）：**

| 改動 | 原因 |
|---|---|
| **修正挑選模型的錯誤** | 訓練中每 500 步評估一次，要留下最好的一版。之前 RLCD 在評估時用錯了規則：用「每個類別各自比 0.5」，而不是實際使用的 entity 解碼。現在評估改用 entity 解碼，並且在 0.05–0.95 之間取最好的門檻再算 F1，BCE 和 RLCD 都一樣，才公平 |
| **「不是實體」片段的權重（none weight）** | 一段文字裡約 99% 的候選片段都不是實體，它們可能壓過真正的實體，讓模型太保守；模型漏掉的實體（約 51%）比基準模型多（約 38%）。現在沒有任何正確類別的片段在訓練目標裡乘上一個權重，試 1、0.3、0.1 三種 |
| **Few-NERD 的一般類別和特定類別（precedence 模式，選用）** | 同一片段只保留最特定的類別（例如 politician 就不再同時標 person）。「other person」這種類別改成一般類別 person，並在 person 的說明後面加一句「若有更特定的類別則選特定類別」。第 4 輪第 1 步還是用 fine 模式（只用細類別），第 2 步再比較 |

超參數只在訓練來源的驗證集（dev_select）上選，不看要測的 7 個新領域。

### 0.3 資料

**訓練資料（第 4 輪第 1 步，data/mix2）**，每一步依下表機率抽來源：

| 來源 | 內容 | 訓練筆數 | 抽樣機率 |
|---|---|---:|---:|
| finerweb | FiNERWeb 細粒度 NER，有原始類別說明 | 16,968 | 8% |
| fewnerd | Few-NERD，類別說明由 Qwen 撰寫；沒出現的類別當作不存在 | 125,842 | 12% |
| pile | Pile-NER，恢復上游的空答案（明確不存在） | 73,870 | 12% |
| massive | MASSIVE 口語指令 | 11,064 | 4% |
| synth_a | Qwen3.8-27B 生成的文字與標註（第一批） | 79,476 | 8% |
| synth_g | Qwen 仿照 GLiNER2 生成：多種文體 × 多個領域 | 95,692 | 20% |
| synth_r | 真實網頁文字，由 Qwen 標註 | 90,124 | 16% |
| synth_g2 | Qwen 生成：音樂、文學、口語等之前較弱的領域 | 31,868 | 20% |

- 每一筆資料都放入所有正確類別，再加最多 min(8, max(2, 正確類別數)) 個不存在的類別。
- 用目標領域訓練資料當種子生成的 synth_c **不算零樣本（zero-shot，訓練時完全沒看過目標領域資料），不放進上表**；它的結果另外報告。
- 已用 5-gram 重疊（≥0.8）檢查並刪除和測試集重複的文字。

**第 4 輪第 2 步的新資料（data/mix3，正在生成）**：

| 新資料 | 做法 | 規模 |
|---|---|---|
| synth_g3 | Qwen3.8-27B 在 120 種文體 × 40 個領域的組合下寫文字；**同一段文字同時問一般類別和特定類別**（例如 person 和 musical artist） | 37 萬段 |
| 補標漏掉的實體（fill） | 把現有訓練資料（finerweb、pile、synth_a/r/g/g2，約 39 萬筆）交給 Qwen 重新看一次，補上漏標的實體；原本標為「不存在」的類別若找到實體，改成有實體 | 38.8 萬筆 |

生成用 vLLM，按資料切成很多份，每台機器 4 張卡同時跑。

**評估資料（都不參與訓練和選模型）：**

| 評估 | 資料 |
|---|---|
| 零樣本 NER（主要指標） | CrossNER 5 個領域（AI、文學、音樂、政治、科學）＋ MIT Restaurant ＋ MIT Movie，共 7 個領域 |
| 個人資料（PII） | ai4privacy，只當測試 |
| 兩種詢問方式 | 給類別說明、只給類別名稱，兩種都報 |

報告的指標：
- F1：邊界與類別都要完全正確。
- 誤報率：不存在的類別卻輸出實體的比例。
- 漏報率：該抓的實體沒抓到的比例。
- ECE：機率準不準。

**比較的基準模型**：GLiNER2-large（有說明時 7 領域平均 0.597）、gliformer-large（只給名稱 0.653）。兩者都由我們自己跑推論，沒有照抄論文上的數字。

**之後才做的資料**：
- REBEL：全部 345 萬句都已經由 Qwen 標好實體和關係，還沒拿來訓練。
- CrossRE：已轉成 mneb 格式。它的測試句和 CrossNER 測試集相同，加入訓練時必須排除這些句子。

---

以下是 v0.2 原始規格（歷史紀錄）。

本文件中的 RLCD 是自訂的 calibrated-decision 研究方案，非 TypeSafe 官方演算法復現。

## 1. v0.2 改了什麼

1. 納入 FiNERWeb、Pile-NER、Open-NER-English、NuNER、Nemotron-PII，為每個來源分配明確用途。
2. 新增 `positive / explicit_absent / scope_assumed_absent / unknown`，避免把「沒標」誤當「不存在」。
3. 恢復 Pile-NER／Open-NER 上游的空答案；Nemotron 在其自身 PII ontology 內建立 scoped negatives。
4. 缺失 description 先用 name-only；不捏造原始資料含有 descriptions。
5. RLCD 主流程只使用 log score。Spherical score 移至可選消融；取消不必要的 RE／EE、F1-RL、teacher 分支。
6. 保留兩個 estimator，是為公平比較同一目標，不是要求每步同時訓練兩套 loss。

## 2. 資料來源：加入什麼、解決什麼

| 來源 | 用途 | description 策略 | absent-label 策略 |
|---|---|---|---|
| `mneb/finerweb-english` | 細粒度、description-conditioned NER | 保留逐筆原文 description | 不由 label 缺失推定不存在 |
| `mneb/pile-ner` | 通用類型與領域擴充 | name-only 起步 | 從 Universal-NER 原始 QA 恢復有名字的空答案 |
| `mneb/open-ner-english` | 通用 NER 的另一份轉換資料 | name-only 起步 | 從候選上游恢復空 entity_mentions；須先確認逐筆來源對應 |
| `mneb/nuner` | 增加語境與 label 多樣性 | name-only 起步 | 預設只用已提供 label；其他 label 為 unknown |
| `mneb/nemotron-pii` | PII 領域與 absent-query 訓練 | 建立版本化 PII description bank | 在固定 PII ontology、明確封閉世界假設下建立負查詢 |

**這些資料的加入有價值，但新增資料量本身不會自動解決 absent-label 問題。關鍵是恢復／建立「問了哪個 label、答案為空」的監督。**

### 2.1 本次確認的事實與抽查範圍

FiNERWeb 沿用 v0.1 全檔稽核：21,224 rows、58,231 annotations，全部符合 `text[start:end] == entity.text`；4,351 個 label 字串；所有 schema labels 都有正例。

其餘來源本次逐檔讀取 **前 2,000 筆**。這是格式抽查，不是隨機抽樣，不代表全檔錯誤率或類別分布。

| 資料檔 | 抽查 rows | schema 為空 | output 無實體 | 有名稱但 value=[] 的 output labels | offset 正確／annotations |
|---|---:|---:|---:|---:|---:|
| Pile-NER train | 2,000 | 2,000 | 20 | 0 | 38,902 / 38,902 |
| Open-NER train | 2,000 | 2,000 | 25 | 0 | 44,193 / 44,193 |
| Open-NER validation | 2,000 | 2,000 | 14 | 0 | 43,661 / 43,661 |
| Nemotron train | 2,000 | 2,000 | 0 | 0 | 16,414 / 16,414 |
| Nemotron test | 2,000 | 2,000 | 0 | 0 | 16,887 / 16,887 |
| NuNER train | 2,000 | 2,000 | 10 | 0 | 8,779 / 8,779 |

所以 `output.entities={}` 且 `schema=[]` 不能單獨解讀成「所有可能 entity types 都不存在」。這種 row 在恢復查詢範圍之前不產生監督。

上游核實：Universal-NER/Pile-NER-type 的說明包含 negative sampling；本次讀取前三筆 QA，第一筆就有 `database`、`Date` 的空答案，而對應 mneb 第一筆僅保留三個正類別。`yongsun-yoon/open-ner-english` 的公開資料也保留帶 entity_type 的空 entity_mentions，但它與 mneb 的完整逐筆關係尚未驗證。來源負標註可能是合成或抽樣產生，不等同人工核實。

Nemotron 官方將資料描述為合成 PII/PHI span annotations。這支持把它當作**有明確任務範圍**的來源，但不證明每篇的每一種 PII 都完整標註。封閉世界負例必須標記為 assumption，不偽裝成 explicit 或 human-verified negatives。

### 2.2 統一資料格式

```json
{
  "id": "source:split:row_id",
  "source": "pile-ner",
  "text": "Acme hired Maya.",
  "queries": [
    {"label": "person", "description": "", "status": "positive",
     "description_origin": "name_only", "supervision_origin": "source"},
    {"label": "database", "description": "", "status": "explicit_absent",
     "description_origin": "name_only", "supervision_origin": "upstream_negative_sampling"}
  ],
  "entities": [{"start": 11, "end": 15, "label": "person"}]
}
```

這是自造格式示例。原始資料的 schema 可為 `{"entities":[...]}` 或 list；adapter 必須兼容。offset 統一為原始 Python Unicode 字元右開區間 `[start,end)`，不能改成 byte offsets 或用搜尋第一個同名實體代替位置。

| query status | target y | loss mask | 條件 |
|---|---|---|---|
| positive | gold 邊界為 1，其餘候選為 0 | 有效候選為 1 | 假定該來源對此 label 的已提供提及近似完整 |
| explicit_absent | 此 label 所有候選為 0 | 有效候選為 1 | 來源明確提供該 label 的空答案；保留合成來源資訊 |
| scope_assumed_absent | 此 label 所有候選為 0 | 有效候選為 1 | 限定 ontology 的完整標註假設；單獨可關閉 |
| unknown | 任意 placeholder | 全部為 0 | 不知道，不等於負例 |

部分完整標註：若 label 只有確知的部分 positive spans、其餘未檢查，只對確知 positives 設 mask=1，其他 candidates 為 unknown。這种資料不能用來估計完整部署分布的校準。

### 2.3 具體 adapter 規則

**FiNERWeb**：保留每筆 name+description。schema 中已有 labels 的非 gold spans 作候選負例；不能添加其他資料的類別後一律標 0。

**Pile-NER／Open-NER**：保留 mneb 已對齊的正 span，另外建立 `negative_query_sidecar`，儲存上游 record ID、text hash、label、空答案來源。只回填已確認是同一段 text 的 negative queries，不改正 span offsets。

- 比對優先使用穩定 ID；mneb 無 ID 時，先精確 text hash，再用版本化空白／tokenization 正規化比對副本。
- 非精確比對須唯一對應，並以正 label／mention inventory 作第二重驗證；有碰撞或資訊不足就不回填。不得以目前看到前三筆相似推定全檔 row-order 對齊。
- 上游 name 的大小寫／空格／底線用明確 alias map 對應；不得自動把 organization、company、institution 合成同義類。
- label 同時有正例與空答案時，記錄衝突並不使用該負查詢。
- 无法回填的 row 仍可做 positive extraction；但須報告復原率，不能說它已提供 absent supervision。

**NuNER**：由 positive output keys 建立 query，description 空值。先不從其他例子隨機抽 label 當真負例。跨句隨機 label negatives 可另列 weak-negative 消融，預設關閉。

**Nemotron**：從固定來源 ontology 建立 source-scoped label bank。若上游沒有完整 enum，就從 **train** label inventory 建立並與來源說明核對，凍結後套用到 dev/test；不能用 test labels 補 bank。超出 bank 的 test labels仍列為 unsupported 並報告，不刪掉後假裝完整覆蓋。

- `scope_policy=dataset_closed_world` 是預設研究假設；缺失的 in-scope label 設 `scope_assumed_absent`。
- 先按 label 抽樣檢查未標註但語義存在的例子，報告錯誤率。若某 label 的假負例率超過預先設定的 5% gate（至少審核 50 個假設負查詢），該 label 關閉 assumed-negative，改 unknown，並重新比較。此 gate 是品質控制設定，不是資料已通過的結果。
- 把 broad organization、person 等跨來源 label 加到 Nemotron，仍是 unknown；PII ontology 不等於所有通用 NER 類別。
- description bank 優先使用官方定義；沒有定義時先 name-only。自行撰寫的定義要標 `authored`、版本化，且不引用當筆答案或 test examples。

### 2.4 Description 不完整怎麼訓練

同一套模型兼容 name-only 與 name+description；空 description 就不加該段，不能填入「本句沒有此 entity」等答案洩漏文字。

FiNERWeb 保留現有描述；其他來源不用先付出大量生成描述成本。後續需要補描述時，以 source ontology、train labels 和固定文件建立 bank。未確認語義一致前，不跨來源自動共享同名字串的描述。

記錄 description_origin，但不把 positive／absent 狀態或 supervision_origin 當模型輸入。新 label 推論可提供 description，不需要全域固定 label ID。

### 2.5 混合、去重與 split

先做跨來源去重，再建立 split 和 source sampler。Pile/Open 可能共享語料；目前未量測完整重疊率，不能把它們的 row 數直接視為獨立新增樣本。

- 精確 text hash + normalized copy hash；近重複用 word-5-gram Jaccard >= 0.8 的連通群組。原文 offset 不因去重副本而改動。
- 同文但不同 queries 可保留，不跨 split；互相衝突的 ontology 標註不得盲目 union。
- 保留 mneb Nemotron test 作最終 test；其重複群組從所有其他 train/dev 排除。Open validation 作 dev-select，亦從其他 train 移除重複群。
- FiNERWeb／NuNER／Pile 等 train-only 來源按群組切 80/10/5/5：train/dev-select/dev-calibration/test。已有來源 test 不回灌，優先級 test > dev-calibration > dev-select > train。跨角色碰撞只從較低優先級用途移除，保留報告。
- train-only 的切分候選須先排除已鎖定的來源 eval 群組；test labels/描述不可用來寫 train label bank。
- mneb/Pile 抽查可見非英文文字。採用固定 language-ID 模型與閾值（起始 en probability >= 0.8），對短句／code 判不明者另列，不默認全是英文。保存偵測模型 revision 與排除數。

建議的初始 **source sampling probability**（不是把全部資料複製這個比例）：

| Source family | 每步抽樣機率 |
|---|---:|
| FiNERWeb | 30% |
| 去重後的 Pile-NER + Open-NER family | 25% |
| NuNER | 25% |
| Nemotron-PII | 20% |

先抽 family，再均勻抽 eligible row。Pile/Open family 內按 unique-text group 抽取，避免同文多份轉換加倍加權。這些比例是起始超參數。

對每個 row，保留可用 positives；有來源負查詢時，最多加入 `min(8,max(2,正類別數))` 個 absent labels；explicit 優先，其次 scoped，組內均勻抽樣。對只有已知 absent 的 row，最多抽 8 個。這是刻意定義的訓練 query 分布，不宣稱仍代表原始部署 prevalence；校準必須使用固定的 eval query 分布。

空答案但沒有任何 known query 的 row 跳過。不要為湊 absent 比例把 unknown 填成 0。日誌必須列 explicit/scoped/unknown 數量與每來源比例；不能只記錄 span negatives 總數。

## 3. 模型與邊界：沿用主線，保持簡單

**ModernBERT-base → text/label representations → span MLP + label MLP → 每個 span–label 一個 logit z。**

- 輸入：`<LABEL> name : description ... <TEXT> original_text`；兩種控制 token 加入 tokenizer 並 resize embeddings。
- 原文獨立 tokenization 取得 offsets，再拼接 schema token IDs，避免 prefix offset 混淆。
- label 向量：name+description token mean pooling → MLP。
- span 向量：`concat(start_hidden,end_hidden,mean_hidden,width_embedding)` → MLP。
- 兩個 MLP：hidden=512、output=256、GELU、dropout=0.1；width embedding=32。
- score：`z = dot(span,label)/sqrt(256) + shared_bias`，機率 `sigmoid(z)`。
- 允許 overlapping spans 與多 label；不做跨 label softmax，不強迫每個 query 至少一個輸出。
- 暫無 BiLSTM／額外 Transformer head；BiLSTM 是後續消融。GLiNER 的 token BiLSTM 和 GLiNER2 的 count 模組不是同一功能。

**候選與長文本**：W=32 subwords 起步；枚舉全部 token spans，再用 offsets 回到原文。僅 deterministic 地去除候選外側空白。同 char offsets 的候選，保留 token 數最少者，再按 i,j 決定。依 train 統計調 W，固定後使用在 eval。

新增資料可能長於原先 FiNERWeb。先把 text 分成最多 512 subwords 的 windows，重疊 64；若 W>65，overlap 至少 W-1。每個 schema block 的 name/description tokens 預算 1,500，總長不超過 2,048（實際另扣特殊 tokens）。超過時按完整 label 分塊，不能截斷某個 description；单个過長 description 報錯。

每個 char-span candidate 只在一個完整包含它的 window 計算：選離 window 中心最近者，平手選最早 window；其他 copies mask 掉。每個 label 只在一個 schema block，decoder 合併結果。candidate ownership 規則與 gold 無關。跨邊界但不能完整表示的 gold 仍保留於全文件 F1 分母。

先報 boundary alignment recall、width coverage、總 candidate recall。train 目標 >=99.5%；不符就修候選表示，不 silent drop。此輪未執行 tokenizer coverage 測試。

跨 window/block 的 loss 按原始 row 的全部有效 decisions 聚合，不能讓切出較多 windows 的文件自動增重。若實作先用 window minibatch，需以預計算 N_row 和該 window 的有效數量給權重。

Qwen2.5-0.5B 雙向版仍作比較：修改實際 causal mask 和 backend flag，不只是 `is_decoder=False`；關閉 KV cache。測試右側有效 token 會影響左側 hidden states、masked padding 不會影響。先用 eager/已驗證 SDPA，再換優化 backend。不能在可看見 target 的雙向原文上直接用普通 next-token loss。

## 4. RLCD：先用一句話理解

**同一個 span–label，模型嘗試 K 個略有不同的機率；比較哪個機率更符合標註，再把模型往高分的方向更新。**

例：候選「Maya」對 label「person」，y=1。嘗試 0.65、0.75、0.85 時，log score 會偏好 0.85；若 query 是已知不存在的 database，y=0，則偏好較小的機率。這個小例子只說明單筆更新方向；校準是整個資料分布上的性質。

### 4.1 只記四個變數

| 變數 | 定義 |
|---|---|
| z | 模型輸出的 scalar logit |
| u | 抽樣 action：帶 Gaussian noise 的 logit |
| p | `sigmoid(u)`：這次嘗試提交的機率 |
| y | 此 span–label 的標註，0 或 1 |

**Action 是 u；不要再從 p 抽一個 entity label。** 我們獎勵的是機率預測，所以要對產生這個機率的 Gaussian policy 計算 log-probability。

### 4.2 六個步驟

1. **一次 forward** 得到所有 `z[B,M,C]`。
2. **嘗試 K 次**：`u_k ~ Normal(z,sigma)`，`p_k=sigmoid(u_k)`。
3. **評分**：`R_k = y log(p_k) + (1-y) log(1-p_k)`。越大越好。
4. **同 decision 比較**：`A_k = R_k - mean(R_j for j != k)`。
5. **更新 policy**：`L_pg = -mean(detach(A_k) * log Normal(detach(u_k); z,sigma))`。
6. **保留監督錨點**：`L_total = BCEWithLogits(z,y) + lambda * L_pg`。

Baseline 僅沿 K 軸，不能跨不同 spans、labels 或 sentences。padding/unknown 不參與 loss。每一筆 row 先對所有 valid pairs 平均，再對 rows 平均，K 也取平均。

起始設定：K=8、sigma=0.2、lambda=0.1，sigma 固定不學習。PG 要求 K>=2 且 sigma>0。不加 reward clipping、advantage/std normalization 或 PPO replay；每組 samples 只更新一次。

### 4.3 detach 到底放哪裡

| 位置 | 是否保留梯度 |
|---|---|
| Gaussian mean z | 是；回到 span head 和 encoder |
| 已抽出的 action u | 否；當作固定觀測值 |
| reward、baseline、advantage、y | 否 |
| `log Normal(u;z,sigma)` | 是；只經 mean z 回傳 |

最常見錯誤：`u=z+noise` 不 detach，直接代回同一 Gaussian.log_prob。此時 `(u-z)` 的 z 互相抵消，policy mean 的學習訊號消失。

原偽代碼的另一個錯誤是抽 `a~Bernoulli(p)`，reward 卻不依 a，然後乘 `log P(a)`：其條件期望梯度為零。此規格以 Gaussian action u 修正兩者的對應。

### 4.4 為什麼還要有「直接反向傳播」對照

我們要最大化的目標只有一個：

\[
J(\theta)=\mathbb E_{\epsilon\sim N(0,1)}[R(z_\theta+\sigma\epsilon,y)].
\]

這個 reward 可微分，所以有兩種方法得到梯度：

| 訓練模式 | 做法 | 性質 |
|---|---|---|
| `pg` | detach action，使用 advantage × Gaussian log-probability | Policy gradient，即本 spec 的 RL 版本 |
| `direct` | 不 detach action，直接最小化 `-mean(R)` | 可微分的 noisy supervised objective，不稱為 RL |

固定相同 K/sigma、mask、抽样分布與 loss reduction，兩者估計同一個 J 的梯度。PG 未必更好，通常只是增加抽樣方差；如果 direct 更好，就應採用 direct，而不是把它重新命名成 RL。

兩者都加相同 clean BCE，且從同一 warm-up checkpoint 出發。這樣能回答「PG 本身有沒有價值」，不把 backbone／資料／reward 變更混在一起。

## 5. 最小 PyTorch loss

`z,y,valid` 的 shape 均為 `[B,M,C]`。下面程式假設同一 batch row 已聚合完整原始 row 的 candidates；長文分塊需按第 3 節保留同一 reduction。mask=0 的 y 用 0 placeholder，不可放 NaN。

```python
import math
import torch
import torch.nn.functional as F


def loss_fn(z, y, valid, mode="pg", K=8, sigma=0.2, lam=0.1):
    z, y = z.float(), y.detach().float()
    mask = valid.float()
    counts = mask.sum((-2, -1))                 # [B]
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("Skip all-invalid batch before optimizer.step")

    def mean_valid(v):                         # v: [K,B,M,C]
        per_row = (v * mask[None]).sum((-2, -1)) / counts.clamp_min(1)[None]
        return per_row[:, active].mean()

    clean = mean_valid(F.binary_cross_entropy_with_logits(
        z, y, reduction="none")[None])
    if mode == "bce":
        return clean
    if mode not in {"pg", "direct"}:
        raise ValueError(mode)
    if K < 1 or sigma < 0 or (mode == "pg" and (K < 2 or sigma <= 0)):
        raise ValueError("PG needs K>=2 and sigma>0")

    noise = torch.randn((K,) + tuple(z.shape), device=z.device, dtype=z.dtype)
    u = z[None] + sigma * noise
    if mode == "pg":
        u = u.detach()                         # fixed sampled action

    # log score, computed stably; expand y to the K-rollout shape
    reward = -F.binary_cross_entropy_with_logits(
        u, y[None].expand_as(u), reduction="none")

    if mode == "direct":
        auxiliary = -mean_valid(reward)         # reward -> z -> model
    else:
        reward = reward.detach()
        baseline = (reward.sum(0, keepdim=True) - reward) / (K - 1)
        advantage = reward - baseline
        log_policy = (-0.5 * ((u - z[None]) / sigma).square()
                      - math.log(sigma) - 0.5 * math.log(2 * math.pi))
        auxiliary = -mean_valid(advantage * log_policy)

    return clean + lam * auxiliary
```

每個 update：`zero_grad → model(batch) → loss_fn → backward → clip_grad_norm_(1.0) → optimizer.step`。所有 K samples 共用同次 forward 的 z，包括同一 dropout realization。Loss/reward/density 使用 float32，backbone 可用 bf16。

`L_pg.item()` 是梯度 surrogate，不是模型品質分數；另記錄 clean BCE、mean reward、gradient norm、F1、NLL。大量抽樣時 PG 與 direct 的平均梯度應一致，不要求每一個 minibatch 相同。

## 6. 訓練、推論與最低驗收

### 訓練

- AdamW，encoder lr=2e-5，head lr=1e-4，weight_decay=0.01（bias/norm 除外）。effective batch=32，microbatch 從 4 起；warmup=10%，gradient clip=1。
- 以混合 source sampler 做 10,000 updates clean BCE warm-up，保存 dev-select macro-source F1 最佳 checkpoint。這是固定預算起點，不是已驗證收斂步數。
- 從相同 checkpoint 分叉：`bce`、`direct`、`pg`，各 2,000 updates，重設同規則 optimizer/scheduler；後兩者 lambda 在前 200 updates 從 0 升至 0.1。
- 先用 seed 42，選定設定後再跑 43、44。三組保持資料/query sampler、模型、K、sigma、訓練預算相同。
- v0.2 不用 focal loss、positive class weighting 或 label smoothing，以免先改變 probability target。若抽樣 candidate negatives，需用 inverse-inclusion weighting 和原全集分母；query/source 混合本身是明確定義的新訓練分布。
- 檢查資料抽樣：每個來源、name-only/name+description、explicit/scoped absent 都能進入訓練，unknown 的 gradient contribution 必須為 0。

### 推論

不加 noise、不抽 action，使用 `p=sigmoid(z)`。若需校準，在 dev-calibration 擬合 scalar temperature T，以 `sigmoid(z/T)` 推論，再選 global F1 threshold；test 不參與。保留達門檻的 spans，允許空輸出、重疊和多 label。

機率校準不是單靠 log score 即保證。訓練 source/query 配比、標註噪聲與 clean/noisy predictor 差異都會影響結果。使用與實際 query 場景一致的 held-out 分布，報告 raw/calibrated 兩版。

### 三份分開的評估

| 評估面板 | Query 集合 | 報告 |
|---|---|---|
| Extraction | 每來源固定的已提供／已恢復 schema，eval 時不動態抽樣 | strict typed character-span F1、candidate recall |
| Absent rejection | explicit 與 scoped 分開，另留少量人工覆核集 | absent-query false-positive rate：無實體 query 中，至少輸出一個 span 的比例 |
| Calibration | 已知監督、固定候選/query 分布，按來源分開 | NLL、Brier、reliability/ECE、prediction precision–coverage |

Nemotron eval 固定全 train-frozen PII bank（必要時分 schema blocks）；Pile/Open eval 固定所有成功恢復的 queries；FiNERWeb/NuNER 原始查詢面板不能被稱為通用 absent-label benchmark。

全負 candidate 很多，需另報 threshold 後 predictions 的 reliability；不能只看全部 decisions 的 ECE。也不要把 `1-max(span_probability)` 宣稱為已校準的「該 label 完全不存在」機率；v0.2 沒有建模該 joint event。

模型選擇：先對各 source family 的 dev F1 取宏平均，再以 NLL 打破平手；另保留逐來源結果，不能用總 micro-F1 隱藏 PII 或稀有類別退步。固定 query 集和 label bank 在比較各模型前生成。

### 必要 correctness tests

1. Unicode／重複名稱／空白的 offset round-trip；完整資料 adapter audit。
2. 增加 padding 不改 loss；unknown labels 的梯度為 0；absent query 全 y=0。
3. train/dev/test 跨來源 text-group 不洩漏；sidecar 回填唯一且無正負衝突。
4. K 個 samples 的 baseline 沿 K 軸；`u.detach` 不可移除；PG sigma=0 報錯。
5. scalar Monte Carlo PG 梯度與 direct/有限差分一致；原錯誤 Bernoulli 寫法作零梯度負控制。
6. Qwen 雙向性、padding isolation、attention-backend parity；只測 shape 不足。

工程檔案：`prepare_data.py / recover_queries.py / alignment.py / collator.py / model.py / losses.py / train.py / evaluate.py / calibrate.py`。Forward 只讀 text/schema/candidates，不讀 gold 數量或 query status。

## 附錄 A：何時才加 spherical score

先跑純 log score，確認梯度與資料有效後，再對 pg/direct **同時**改 reward：

\[
R= y\log p+(1-y)\log(1-p)
+\alpha\left(\frac{yp+(1-y)(1-p)}{\sqrt{p^2+(1-p)^2}}-1\right),
\quad \alpha=0.1.
\]

alpha=0 是主規格；alpha=0.1 是獨立消融。此為二元分布 `(p,1-p)` 的 spherical score，不是 `yp/|p|`。RPS 不用，因 entity types 沒有 ordinal 順序。Teacher probability 也不在此版本範圍。

## 附錄 B：版本與證據

本次 mneb revisions（訓練前應固定 model/tokenizer/library versions）：

| Dataset | Revision |
|---|---|
| finerweb-english | `46a29532929a2cd439bf5c3df2281d7760d28731` |
| pile-ner | `05c389c709b6844e52d82b9f429a56c3f055eb21` |
| open-ner-english | `cc90f02eecbaa0d5d7aa25f9a57e6ccb9bafe3fc` |
| nemotron-pii | `a4bdef0402f75f8196578f7d6f556e28ad7f49a4` |
| nuner | `5f85697724cc2bd7306d4d6faa5729f3f9faa01c` |

mneb 的四個新增 README 只列 schema/features/splits，沒有足夠說明轉換、標註完整性或每份資料 provenance。恢復上游時必須建立實際對照與來源記錄；未驗證的候選上游不當成既成對應。上游 Pile-NER-type 標示 CC BY-NC 4.0；mneb 轉換不應被視為移除上游限制，研究資料 manifest 要保留來源及授權資訊。

主要連結：

- <https://huggingface.co/datasets/mneb/finerweb-english>
- <https://huggingface.co/datasets/mneb/pile-ner>
- <https://huggingface.co/datasets/mneb/open-ner-english>
- <https://huggingface.co/datasets/mneb/nemotron-pii>
- <https://huggingface.co/datasets/mneb/nuner>
- Pile 原始 QA：<https://huggingface.co/datasets/Universal-NER/Pile-NER-type>
- Open-NER 候選上游：<https://huggingface.co/datasets/yongsun-yoon/open-ner-english>
- Nemotron 原始來源：<https://huggingface.co/datasets/nvidia/Nemotron-PII>
- NuNER 原始來源參考：<https://huggingface.co/datasets/numind/NuNER>
- <https://huggingface.co/answerdotai/ModernBERT-base>
- <https://huggingface.co/Qwen/Qwen2.5-0.5B>
- <https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/layers.py>
- <https://github.com/McGill-NLP/llm2vec>

驗證狀態：新增六個檔案各前 2,000 rows 的格式與 offsets 已檢查；沒有完成全檔語言篩選、跨來源去重、negative sidecar recovery 或 tokenizer coverage。文件 loss code 通過語法檢查與 scalar 数學驗證不等同 PyTorch 整合測試；沒有模型品質或訓練速度結果。

本版純 log-score scalar 驗證：seed=42、K=8、sigma=0.2、20,000 組抽樣；表內是最大化 reward 的梯度，loss 梯度符號相反。

| z | y | direct 平均梯度 | PG 平均梯度 | 差值標準誤 |
|---:|---:|---:|---:|---:|
| -1.000000 | 1.000000 | 0.729191 | 0.730274 | 0.002754 |
| 0.700000 | 0.000000 | -0.666611 | -0.667729 | 0.002544 |

兩組差值皆在 5 個 Monte Carlo 標準誤內。驗證使用 Python 標準函式庫，尚未執行 PyTorch loss 的整合測試。
