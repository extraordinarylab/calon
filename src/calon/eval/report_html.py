"""Version comparison page (HTML) from round_summary JSON: what each CalOn version changed, then
one table per benchmark group with every set, our versions (descriptions on, 5-seed mean) and the
baselines side by side, the best value per row in bold, and an average row per metric."""
import argparse
import html
import json
from pathlib import Path

VERSIONS = [
    ("ee1", "第一個同時做 NER、RE、EE 的版本。NER / RE 用前幾輪的訓練來源，另加事件資料：多領域生成事件"
            "（synth_gee）和仿 CASIE / MLEE 的 seeded 事件（synth_cee）。所有 NER 與 RE 來源都經過第二個模型複標和 judge 過篩。",
     "18,000"),
    ("ee2", "ee1 ＋ 錯誤分析後的針對性資料：細粒度實體階層（synth_fine，針對 DynamicNER）、用 SciER 關係名稱的"
            "科學關係（synth_sci）、巢狀事件（synth_nest，針對 MLEE）。", "20,000"),
    ("ee3", "ee2 ＋ 模型從沒學過的五種關係（第五輪）：代名詞與 coreference（synth_coref）、aspect–opinion 情感"
            "（synth_opin）、事件先後與因果（synth_temp）、人物與機構屬性（synth_slot）、生醫關聯（synth_bio）。", "24,000"),
    ("ee4", "ee3 的資料變得更多樣（第六輪）：第五輪、科學、seeded 事件這幾種來源加上隨機地區、年代、敘述者並禁止常見人名後"
            "重新生成；所有生成來源移除 near-duplicate；新增日常短句（synth_daily）、真實 CC-News 事件（ccnews_ee）、"
            "人工標註事件資料（MAVEN-Arg train、GENEVA、RAMS、PHEE）。", "24,000"),
    ("ee5", "ee4 的資料修正與補強（第七、八輪，mix14），2 個 seed（GPU 額度不足，第三個取消）：同一事件的重複 trigger 合併；RAMS、GENEVA 沒標的類型改成 unknown；"
            "新增所有 mention（synth_ment）、上下位類型與 miscellaneous（synth_misc）、person identity（synth_slot2）、片語 trigger 與子句論元"
            "（synth_phr）、評論慣用語與反諷（synth_opin2）、BioRED 方向與跨句（synth_bio2）、MeasEval 數量綁定（synth_meas）；關係來源補標"
            "漏掉的關係（Nemotron 補、gpt-oss 審）；事件資料加入經 gpt-oss 確認不存在的相近事件類型；事件比例 23% → 20%，synth_c 恢復 ee3 比例。"
            "（先前試過的 blind label descriptions 讓使用定義時 recall 減半，已放棄，不在此版本。）", "24,000"),
]
# annotation problems: left out of the comparison (still evaluated, not reported)
EXCLUDE = {"dynamicner_en", "buster", "dwie_re"}
INTRO_NOTE = """<section><h2>先看名詞</h2>
<p><b>新版 Wikipedia 文章（汙染檢查）</b>：CrossNER 和 MIT 是公開多年的資料集，模型（包括 baselines）可能在預訓練時看過這些句子，分數會虛高。為了檢查這件事，我們另外做了一份同樣類型清單的測試集：CrossNER 五個領域的文字取自 gliformer 發布日（2026-09-11）之後才新建的英文 Wikipedia 條目，所以沒有任何受測模型可能看過；MIT 兩個領域找不到新的真實來源，改由非 Qwen 模型模仿 MIT train 的口語查詢寫新句子。每段文字由三個非 Qwen 模型用原資料集的類型清單和定義各標一次，至少兩個模型同意的 span 才保留。如果某個模型在原版很高、在新版明顯下降，就代表它可能記住了原版測試集。</p>
<div class='tbl'><table><thead><tr><th>來源</th><th>文字從哪來</th><th>標註</th><th>內容</th></tr></thead><tbody>
<tr><td>synth_a、synth_r</td><td>真實文字（synth_a：FineWeb 網頁；synth_r：新聞、PubMed、arXiv、美國判決）</td><td>Qwen 開放類型標註</td><td>實體</td></tr>
<tr><td>synth_g、synth_g2、synth_g3</td><td>生成，genre × domain 網格（g：22 × 60；g2：按場景分組；g3：40 × 120）</td><td>生成模型自己設計類型並標註；g2、g3 要求同時有上位與下位類型</td><td>實體</td></tr>
<tr><td>synth_fine</td><td>生成，40 × 120</td><td>兩層細粒度類型階層，含 other X</td><td>實體</td></tr>
<tr><td>synth_c</td><td>生成，仿 CrossNER / MIT 風格</td><td>以它們 train split 的 5 個例子和類型清單當種子</td><td>實體</td></tr>
<tr><td>synth_daily</td><td>生成，日常短句（查詢、訊息、短評）</td><td>生成模型定類型，gpt-oss 檢查</td><td>實體</td></tr>
<tr><td>synth_ment、synth_misc</td><td>生成（第七輪）</td><td>ment：所有專名、名詞片語、代名詞；misc：上位＋下位類型＋miscellaneous</td><td>實體</td></tr>
<tr><td>synth_gre</td><td>生成，40 × 120</td><td>生成模型自己設計關係類型</td><td>關係</td></tr>
<tr><td>synth_cre</td><td>生成，仿 CrossRE 風格</td><td>以 CrossRE train split 當種子</td><td>關係</td></tr>
<tr><td>synth_sci、synth_bio、synth_slot、synth_slot2</td><td>生成</td><td>分別用 SciER、BioRED、Re-TACRED 的關係名稱；slot2 必含 person identity 與 alternate names</td><td>關係</td></tr>
<tr><td>synth_coref、synth_opin、synth_temp</td><td>生成（第五輪）</td><td>代名詞指代與 coreference、aspect–opinion 情感、事件先後與因果</td><td>關係</td></tr>
<tr><td>synth_gee、synth_nest</td><td>生成，40 × 120</td><td>生成模型自己設計事件類型與論元；nest 至少一半事件的論元是另一個事件</td><td>事件</td></tr>
<tr><td>synth_cee</td><td>生成，仿 CASIE / MLEE 風格</td><td>以它們 train split 當種子</td><td>事件</td></tr>
<tr><td>synth_phr</td><td>生成（第七輪）</td><td>trigger 用完整片語、論元可為完整子句</td><td>事件</td></tr>
<tr><td>ccnews_ee</td><td>真實新聞（CC-News）</td><td>Qwen 標註，Nemotron 複標，gpt-oss 裁決</td><td>事件</td></tr>
</tbody></table></div>
<p class='note'>BUSTER 與 DynamicNER 的標註有問題、DWIE 的關係名稱與文件層級設定很特殊，三者已從比較表移除。</p></section>"""
GAP_NOTE = """<section><h2>為什麼輸給 gpt-oss-120b：錯誤分析與案例</h2>
<p>每個測試集抽 200 筆，ee4（seed 42）和 gpt-oss 在同一批句子上預測，再逐筆比對。另外把我們模型的「回答門檻」（1 − p(無關係) 要超過 0.85 才輸出，這個值由 REBEL dev 決定）換成事後最佳值看差距有多少是門檻造成的，這只是診斷，不能拿來選模型。</p>
<div class='tbl'><table><thead><tr><th>測試集</th><th class='n'>我們 F1</th><th class='n'>門檻換成最佳值</th><th class='n'>gpt-oss F1</th><th>主要原因</th></tr></thead><tbody>
<tr><td>SciER</td><td class='n'>0.49</td><td class='n'>0.55</td><td class='n'>0.54</td><td>差距全部來自門檻：我們 precision 0.65、recall 0.41，gpt-oss 0.50 / 0.61。例如 region proposal network part-of Faster R-CNN，我們標籤對但分數 0.76，低於 0.85 就不輸出。</td></tr>
<tr><td>GAP</td><td class='n'>0.68</td><td class='n'>0.75</td><td class='n'>0.85</td><td>錯誤全是「分數不夠」：例如 "her" refers to "Amanda" 分數 0.76。另有一部分是真的分不清兩個同性別的人。</td></tr>
<tr><td>LitBank</td><td class='n'>0.54</td><td class='n'>0.57</td><td class='n'>0.61</td><td>同上，代名詞指向前文 mention 的分數普遍在 0.7–0.85。</td></tr>
<tr><td>Re-TACRED</td><td class='n'>0.39</td><td class='n'>0.45</td><td class='n'>0.57</td><td>門檻佔一部分；其餘是 person identity 幾乎完全找不到（2,036 個，召回 0），以及縮寫對全名（PSIA 與 Poverty and Social Impact Analysis，分數 0.00）。</td></tr>
<tr><td>MeasEval</td><td class='n'>0.43</td><td class='n'>0.40</td><td class='n'>0.61</td><td>不是門檻問題。一段文字有很多數量時，我們把數量接錯實體（例如把 3.4% 接到 MEX orbits），誤報 450 個。</td></tr>
<tr><td>BioRED</td><td class='n'>0.21</td><td class='n'>0.15</td><td class='n'>0.36</td><td>不是門檻問題。positive / negative correlation 常被判成 association（154 次）；跨句的變異與疾病關聯幾乎抓不到（例如 C88Y 與 anemia 分數 0.02）。</td></tr>
<tr><td>ASTE（給定實體）</td><td class='n'>0.69–0.83</td><td class='n'>0.73–0.85</td><td class='n'>0.49–0.62</td><td>給定正確實體時我們比 gpt-oss 好；輸在 end-to-end，因為我們找 aspect 與 opinion 詞的能力弱。剩下的錯誤多是慣用語的情感判反：pasta is out of this world、soup rocks my world、to die for 都被判成負面，信心 0.98 以上。</td></tr>
</tbody></table></div>
<p><b>事件</b>：我們找 trigger 的 recall 其實比 gpt-oss 高（M2E2 0.75 對 0.59，CASIE 0.23 對 0.17，MLEE 0.41 對 0.13），輸在 precision：M2E2 我們誤報 6,806 個、gpt-oss 1,476 個，例如把 crisis、conflict 當成 attack 事件。原因是訓練資料裡被問到的事件類型大多真的出現（MAVEN-Arg 90%、CC-News 57%、多領域合成 58%），但 M2E2 只有 2%、SENTiVENT 6%，模型學到「被問就有」。論元方面 gpt-oss 明顯較好：MAILEX action members 0.54 對 0.02、SENTiVENT time 0.50 對 0.09、SPEED++ infected 0.55 對 0.04；MAILEX 和 SENTiVENT 的 trigger 多是片語（"Attached are the talking points"、"will send Al's phone number"），我們只標到單字 "Attached"、"send"。</p></section>"""
EE34_NOTE = """<section><h2>為什麼有些地方 ee3 比 ee4 好</h2>
<p>ee3 與 ee4 用同一套訓練方式和步數，差別只在資料（mix10 → mix11）。和下降有關的資料變化：</p>
<div class='tbl'><table><thead><tr><th>資料變化</th><th>影響</th></tr></thead><tbody>
<tr><td>新增 CC-News 事件（5%）與 MAVEN-Arg（1.5%），事件總比例 18% → 23%</td><td>這兩個來源被問到的事件類型分別有 57% 與 90% 真的出現，且 MAVEN 每 100 字有 8.5 個 trigger（連 crisis 這類抽象名詞也算），使模型更容易誤報：M2E2 trigger precision 0.17 → 0.11，SENTiVENT 誤報 385 → 1,030。</td></tr>
<tr><td>CC-News 與多領域事件精修時，同一事件被標成單字與片語兩個 trigger（CC-News 98,217 個）</td><td>同一事件有兩個正確答案，鼓勵模型多輸出 span，也讓論元分散到兩個 trigger 上。</td></tr>
<tr><td>RAMS、GENEVA 的部分標註被當成完整標註</td><td>沒標的事件和角色被當成負例，論元召回下降：SENTiVENT time 0.25 → 0.09，SPEED++ infected 0.30 → 0.04。</td></tr>
<tr><td>synth_c（仿 CrossNER / MIT）去重刪掉 35%，比例 5.1% → 4.0%</td><td>MIT Restaurant 0.660 → 0.626。</td></tr>
<tr><td>synth_opin 重新生成（更多樣，但句子變短）</td><td>ASTE end-to-end 下降 3–5 點（找 aspect / opinion 詞變弱），給定實體則上升 3–7 點。</td></tr>
</tbody></table></div>
<p class='note'>前三項已在 mix13 修正或可在比例上調整；這是根據資料差異與錯誤型態的推論，要確認需要做 ablation（例如只拿掉 MAVEN-Arg 重訓）。</p></section>"""
R7_NOTE = """<section><h2>ee4 錯誤分析與第七輪對策</h2>
<p>ee4 相對 ee3：關係進步（ASTE given +3–7、MAVEN-ERE +5、Re-TACRED +2），NER 大多小幅進步，但事件變差：M2E2 trigger −10
（precision 0.17 → 0.11，例如 movement:transport 誤報 911 → 1,582），給定 trigger 的論元在 MAILEX −6、SENTiVENT −4、SPEED −2–3、
CASIE −2；DynamicNER −3.6、MIT Restaurant −3.4。</p>
<div class='tbl'><table><thead><tr><th>問題</th><th>證據</th><th>對策（已做進 mix13）</th></tr></thead><tbody>
<tr><td>同一事件被標成兩個 trigger</td><td>精修時兩個標註模型一個標詞、一個標片語（"sent" 與 "sent the invoice"），judge 兩個都留：ccnews_ee 98,217 個、synth_gee 55,744 個、synth_phr 67,826 個。同一事件兩個正確答案，模型學到多輸出 span，是 ee4 trigger 誤報變多的可能原因之一</td><td>同類型、互相包含的 trigger 只留外層片語，內層的論元移過去（ccnews_ee、synth_gee、synth_cee、synth_nest、synth_phr）</td></tr>
<tr><td>部分標註的資料被當成完整標註</td><td>RAMS 每段只標一個事件、GENEVA 只標部分 FrameNet frame，但沒標的事件類型和角色被當成負例，教模型「別的事件不是事件」</td><td>RAMS、GENEVA 沒標的類型與角色改成 unknown（不計 loss）</td></tr>
<tr><td>trigger 是片語、論元是子句</td><td>CASIE、SENTiVENT、MAILEX 的 gold trigger 多數是 2–4 字以上的片語，我們幾乎只預測單字；MAILEX「action description」、SPEED「means」「information source」召回 0–4%</td><td>synth_phr：42,579 段文字，trigger 用完整片語（75% 為多字），論元可為完整子句</td></tr>
<tr><td>GENTLE 把所有名詞片語都算 mention</td><td>recall 0.03（common noun 0.02、pronoun 0.00），abstract 2,842 個只找到 1%</td><td>synth_ment：31,594 段，標出所有專名、普通名詞片語、代名詞</td></tr>
<tr><td>CrossNER miscellaneous 與上下位類型</td><td>miscellaneous F1 0.20–0.33；writer → person 75 次、person → scientist 51 次</td><td>synth_misc：21,376 段，每段有上位類型＋下位類型＋miscellaneous，標最具體的類型</td></tr>
<tr><td>Re-TACRED person identity</td><td>2,036 個 gold，召回 0.00</td><td>synth_slot2：22,215 段，每段必含 person identity 與 organization alternate names，附定義</td></tr>
<tr><td>blind label descriptions</td><td>ee5 seed 42 使用定義時 recall 減半</td><td>不採用（mix13 以 mix11 為底，不含 mix12 的 blind 定義）</td></tr>
</tbody></table></div>
<p class='note'>尚未處理、需在訓練比例上調整：ee4 的事件比例 23%（ee3 18%）可能讓 trigger 偏多，MIT Restaurant 下降可能與 synth_c 被去重 35% 有關。
建議 ee6 事件降回約 20%（maven_arg、ccnews_ee 減量），synth_c 比例恢復到 ee3 的水準。</p></section>"""
SOUP_NOTE = ("<section><h2>Model soup（ee2 試驗）</h2><p>把同一設定的 5 個 seed 權重平均成一個模型。結果無效：每個 seed 單獨的 "
             "in-domain dev F1 約 0.70，5 個直接平均（uniform soup）掉到 0.058；greedy soup 從最好的 seed 46（0.709）開始，"
             "加入任何第二個 seed 都掉到 0.47–0.52，所以只剩單一 seed，等於沒有 soup。原因：5 個 seed 共用預訓練 encoder，但上面的 "
             "extraction head 是各自隨機初始化，權重不在同一個 basin，平均後失效。若要用 soup，需要讓所有 seed 的 head 從同一組初始權重出發"
             "（只換資料順序），這需要重新訓練。</p></section>")
GROUPS = [
    ("ner", "NER：CrossNER 與 MIT", "F1，span 與 type 都要完全正確。"),
    ("ner_new", "NER：新版 Wikipedia 文章（汙染檢查）", "encoder 訓練截止後才寫的文章，類型清單相同。"),
    ("ner_x", "NER：SciER", ""),
    ("ner_x2", "NER：GENTLE、BioRED、MeasEval", ""),
    ("re", "Relation：CrossRE", "given：給定正確實體，只判斷關係。end2end：先自己找實體再判斷關係。"),
    ("re_x", "Relation：SciER", ""),
    ("re_x2", "Relation：第三批測試集", "BioRED、DWIE 以 concept 為單位計分；GAP、LitBank、MAVEN-ERE 是改寫成關係的任務"
                                     "（代名詞或 mention 指向、事件先後）。GLiNER2 / GLiFormer 沒有「給定實體」模式，"
                                     "given 欄是它們 end-to-end 的答案只保留落在正確實體上的部分。"),
    ("ee", "Event：CASIE、MLEE、M2E2", "given AC：給定正確 trigger，判斷論元與角色。e2e TC：自己找 trigger 並分類。"
                                       "e2e AC：自己找 trigger 再找論元。"),
    ("ee_x2", "Event：第三批測試集", ""),
]
SET_NAMES = {"crossner_ai": "CrossNER AI", "crossner_literature": "CrossNER Literature", "crossner_music": "CrossNER Music",
             "crossner_politics": "CrossNER Politics", "crossner_science": "CrossNER Science", "mit_movie": "MIT Movie",
             "mit_restaurant": "MIT Restaurant", "dynamicner_en": "DynamicNER", "buster": "BUSTER", "scier": "SciER",
             "scier_ood": "SciER (out of domain)", "gentle": "GENTLE", "biored": "BioRED", "measeval": "MeasEval",
             "crossre_ai": "CrossRE AI", "crossre_literature": "CrossRE Literature", "crossre_music": "CrossRE Music",
             "crossre_news": "CrossRE News", "crossre_politics": "CrossRE Politics", "crossre_science": "CrossRE Science",
             "aste_14lap": "ASTE 14lap", "aste_14res": "ASTE 14res", "aste_15res": "ASTE 15res", "aste_16res": "ASTE 16res",
             "biored_re": "BioRED", "dwie_re": "DWIE", "gap": "GAP", "litbank": "LitBank", "maven_ere": "MAVEN-ERE",
             "measeval_re": "MeasEval", "retacred": "Re-TACRED", "casie": "CASIE", "mlee": "MLEE", "m2e2": "M2E2",
             "crossner_ai_new": "CrossNER AI（新文章）", "crossner_literature_new": "CrossNER Literature（新文章）",
             "crossner_music_new": "CrossNER Music（新文章）", "crossner_politics_new": "CrossNER Politics（新文章）",
             "crossner_science_new": "CrossNER Science（新文章）", "mit_movie_new": "MIT Movie（新文章）",
             "mit_restaurant_new": "MIT Restaurant（新文章）", "mailex": "MAILEX", "sentivent": "SENTiVENT", "speed_mpox": "SPEED++ mpox", "speed_zika": "SPEED++ zika"}


def model_name(m):
    if m.startswith("CalOn"):
        return m.split(" (")[0]
    s = m.replace("ner-", "").replace("re-", "").replace("ee-", "")
    for a, b in (("gliner2-large-v1-gf-desc", "GLiNER2-large"), ("gliformer-large-v1-gf-name_only", "GLiFormer-large"),
                 ("gliner2-large-v1-desc", "GLiNER2-large"), ("gliformer-large-v1-name_only", "GLiFormer-large"),
                 ("gpt-oss-120b-given-desc", "gpt-oss-120b"), ("gpt-oss-120b-end2end-desc", "gpt-oss-120b"),
                 ("gpt-oss-120b-desc", "gpt-oss-120b"), ("GNER-LLaMA-7B (name_only)", "GNER-7B"),
                 ("gollie-7b-desc", "GoLLIE-7B"), ("gollie-7b-name_only", "GoLLIE-7B")):
        if s == a:
            return b
    return s


def table(group, raw):
    # one column per model: a baseline's given and end-to-end files merge into one column
    data = {}
    for m, c in raw.items():
        if m.startswith("CalOn") and "(desc" not in m:
            continue
        data.setdefault(model_name(m), {}).update(c)
    cols = [m for m in data if m.startswith("CalOn")] + [m for m in data if not m.startswith("CalOn")]
    names = cols
    ours_keys = {k for m in cols if m.startswith("CalOn") for k in data[m] if k.split(" | ")[0] not in EXCLUDE}
    keys = sorted({k for m in cols for k in data[m] if k in ours_keys}, key=lambda k: (k.split(" | ")[1], k))
    metrics = sorted({k.split(" | ")[1] for k in keys})
    head = "".join(f"<th class='n'>{html.escape(n)}</th>" for n in names)
    rows = []
    for metric in metrics:
        ks = [k for k in keys if k.endswith(" | " + metric)]
        rows.append(f"<tr class='group'><td colspan='{len(cols) + 1}'>{html.escape(metric)}</td></tr>")
        for k in ks + ["__avg__"]:
            if k == "__avg__":
                vals = []
                for m in cols:
                    v = [data[m][x][0] for x in ks if x in data[m]]
                    vals.append(sum(v) / len(v) if len(v) == len(ks) else None)
                label = "Average（平均）"
            else:
                vals = [data[m][k][0] if k in data[m] else None for m in cols]
                s = k.split(" | ")[0]
                label = SET_NAMES.get(s, s)
            best = max((v for v in vals if v is not None), default=None)
            cells = []
            for m, v in zip(cols, vals):
                if v is None:
                    cells.append("<td class='n dim'>—</td>")
                    continue
                sd = data[m][k][1] if k != "__avg__" and k in data[m] else None
                txt = f"{v:.3f}" + (f"<span class='sd'> ±{sd:.3f}</span>" if sd else "")
                cells.append(f"<td class='n{' best' if v == best else ''}'>{txt}</td>")
            rows.append(f"<tr class='{'avg' if k == '__avg__' else ''}'><td>{html.escape(label)}</td>{''.join(cells)}</tr>")
    return f"<div class='tbl'><table class='res'><thead><tr><th>Dataset</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = json.loads(Path(args.summary).read_text())
    present = {m.split(" (")[0].split()[1] for g in res.values() for m in g if m.startswith("CalOn")}
    vers = "".join(f"<tr><td class='key'><b>{v}</b></td><td>{html.escape(d)}</td><td class='n'>{u}</td>"
                   f"<td class='key'>{'已有結果' if v in present else '進行中'}</td></tr>" for v, d, u in VERSIONS)
    body = [f"<section><h2>各版本差在哪裡</h2><div class='tbl'><table><thead><tr><th>版本</th><th>改了什麼</th>"
            f"<th class='n'>訓練步數</th><th>狀態</th></tr></thead><tbody>{vers}</tbody></table></div>"
            "<p class='note'>所有版本都用 Qwen3-0.6B encoder 與 RLCD head；ee1–ee4 各 5 個 seed、ee5 2 個 seed，表中是 seed 的平均（使用類型定義），"
            "± 是 seed 之間的標準差。Baselines：GLiNER2-large、GLiFormer-large（span 模型）、GNER-7B（生成式 NER）、"
            "gpt-oss-120b（prompt 大型語言模型）。粗體是該列最好的分數；「—」代表該模型不支援這個設定或尚未跑。</p></section>"]
    for g, title, note in GROUPS:
        if g in res and res[g]:
            body.append(f"<section><h2>{html.escape(title)}</h2>"
                        + (f"<p class='note'>{html.escape(note)}</p>" if note else "") + table(g, res[g]) + "</section>")
    body.insert(1, INTRO_NOTE.replace("\n", ""))
    body += [GAP_NOTE.replace("\n", ""), EE34_NOTE.replace("\n", ""), R7_NOTE.replace("\n", "")]
    css =Path(__file__).with_name("report_html.css").read_text()
    fonts = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@600;700&"
             "family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap'>")
    page = (f"<title>CalOn 版本比較</title>{fonts}<style>{css}</style><div class='wrap'><header><div class='eyebrow'>"
            "CalOn · zero-shot IE</div><h1>CalOn 版本比較</h1><p class='lede'>所有測試集、我們每一版模型、所有 baselines "
            "放在同一處。每一節的 Average 列只在該模型每個子集都有分數時才計算。</p></header>" + "".join(body) + "</div>")
    Path(args.out).write_text(page)
    print("wrote", args.out, len(page))


if __name__ == "__main__":
    main()
