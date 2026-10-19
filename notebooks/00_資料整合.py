# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 資料整合：從 25 個原始檔到兩個分析檔
#
# **這本 notebook 做什麼**：把 PSFD 主樣本 RR2007–RR2024（12 波）的原始 CSV 整理成兩個分析用檔案。
#
# | 輸出檔 | 內容 | 用在哪裡 |
# |---|---|---|
# | `data/long_2007_2024.csv` | 每人每波一列：快樂、生活評價、工作/家庭/經濟/婚姻滿意度、健康、人口變項、婚姻、子女、收入 | 18 年權重趨勢、追蹤資料個人斜率、固定效果、交叉延宕 |
# | `data/main_2020_2024.csv` | RR2020/2022/2024 三波，加上多題幸福指數、態度與重要性題、寂寞感、職家衝突、工作特性、個性 | 主模型、個人權重、問卷與計算機 |
# | `data/attitudes_long.csv` | 2011/2018/2020/2022/2024 的婚姻態度與性別角色題 | 家庭導向分數的跨年穩定度、預測個人斜率 |
#
# **三個重要的清理決策**
# 1. **遺漏值**：每個變數只保留問卷定義的有效範圍（例如滿意度 1–4），其餘（跳答 0、不知道 6/96、拒答 8/98、遺漏 9/99、2022 年後的負值代碼）一律設為缺失。依據是各波編碼簿，下方第 1 節會把每個用到的變數的數值標籤印出來逐一核對。
# 2. **方向統一**：滿意度（1=很滿意…4=很不滿意）、自評健康（1=很好…5=很不好）全部反轉成「數字越大越好」。
# 3. **跨波編號**：PSFD 自 RR2024 起樣本編號改為 8 碼（舊 7 碼 ×10，官方〈資料合併原則〉），因此 RR2024 的編號先 ÷10 才能和前面波次對上。我們先前（v3）合併檔沒有做這一步，導致 RR2022 與 RR2024 只對上 5 人，這裡修正。

# %%
import re, json
import numpy as np, pandas as pd
from psfd_utils import *
pd.set_option("display.width", 200, "display.max_columns", 50)

# %% [markdown]
# ## 1. 變數對照表（每一波用哪個原始變數）
#
# 題目來源：各波編碼簿（`data/cleaned_data/RR_YYYY/RR_YYYY_編碼簿.md`）與 PSFD 官方「資料檢索及合併系統」的跨波對照表（快樂 = s34、生活好不好 = s35a、家庭生活滿意 = s17a、自評健康 = s36）。

# %%
COMMON_2008 = dict(happy="a03a", good="a03b", health="a04a", job="c07a", fam="c07b", sex="a01", byr="a02a")
MAP = {
    2007: dict(happy="a03a", good="a03b", health="a04a", job="d07d", fam="d07e", sex="a01", byr="a02a"),
    2008: COMMON_2008, 2009: COMMON_2008, 2010: COMMON_2008, 2011: COMMON_2008,
    2012: dict(happy="a03a", good="a03b", health="a04a", job="e01a", fam="e01b", sex="a01", byr="a02a01"),
    2014: COMMON_2008,
    2016: dict(happy="a03a", good="a03b", health="a04a", job="a06b", fam="b12", econ="c12", sex="a01", byr="a02a",
               emp="a06a01", mar="a17a"),
    2018: dict(happy="a03c", good="a03d", health="a04a", job="a06b", fam="b08", econ="c19", marsat="b24a",
               sex="a01", byr="a02a", emp="a06a", mar="a17a", mode="x01f"),
    2020: dict(happy="w01a", good="w01c", health="w01e", job="w04", fam="b08", econ="c19a", marsat="w54",
               sex="a01", byr="a02a", emp="w03", mar="w20b", mode="x01f", nkids="b11", inc="c04a02", hours="w06g",
               edu="a03a"),
    2022: dict(happy="a06", good="a09", health="a11", job="c03", fam="f17", econ="k32", marsat="d53",
               sex="a01", byr="a02z01", emp="c02r1", mar="d02", mode="x01f", nkids="g02z01", inc="k04z02r1",
               incb="k05r1", hours="c19r1", edu="b01"),
    2024: dict(happy="a06", good="a09", health="a11", job="c03", fam="f17", econ="k32", marsat="d53",
               sex="a01", byr="a02z01", emp="c02r1", mar="d02b", mode="x01f", nkids="g02z01", inc="k04z02r1",
               incb="k05r1", hours="c19br1", edu="b01"),
}
WAVES = list(MAP)


def labels(year, var):
    """從編碼簿抓某變數的題目與數值標籤（核對用）"""
    p = RAW / f"RR_{year}" / f"RR_{year}_編碼簿.md"
    for line in open(p, encoding="utf-8"):
        if f"| `{var}` |" in line:
            parts = [x.strip() for x in line.split("|")]
            return parts[3][:40], parts[5][:160]
    return None, None


rows = []
for y, m in MAP.items():
    for k, v in m.items():
        q, lab = labels(y, v)
        rows.append(dict(year=y, key=k, var=v, question=q, labels=lab))
check = pd.DataFrame(rows)
# 只印出核心題，確認方向與有效範圍
check[check.key.isin(["happy", "good", "health", "job", "fam"])].pivot(index="year", columns="key", values="labels").map(lambda s: (s or "")[:70])

# %% [markdown]
# 上表確認：快樂與生活評價都是 1=很不快樂/很不好 … 7=很快樂/很好；滿意度都是 1=很滿意 … 4=很不滿意（0 = 目前沒有工作/跳答）；健康都是 1=很好 … 5=很不好。所以快樂、生活評價直接用，滿意度與健康反轉。

# %% [markdown]
# ## 2. 長資料：RR2007–RR2024（12 波）

# %%
# 子女名冊出生年變數的字首（後面接 1..6 代表第幾位子女）
ROSTER = {2007: "b13c01c", 2008: "b13c01c", 2009: "b13c01c", 2010: "b13c01c", 2011: "b13c01c", 2012: "b13c01c",
          2014: "b14c01c", 2016: "b16c01c", 2018: "b12c01z01c", 2020: "b12c01z01c", 2022: "g08z01c", 2024: "g08z01c"}
MAR_MAP_2022 = {1: "married", 2: "married", 3: "married", 8: "married", 4: "separated", 10: "separated",
                5: "divorced", 9: "divorced", 6: "widowed", 11: "widowed", 7: "never"}
MAR_MAP_STD = {1: "never", 2: "married", 3: "separated", 4: "divorced", 5: "widowed"}


def income_bracket_midpoints(year, var):
    """把所得級距（如『20—30萬元（不含）』）轉成中點（元）"""
    _, lab = labels(year, var)
    p = RAW / f"RR_{year}" / f"RR_{year}_編碼簿.md"
    full = [l for l in open(p, encoding="utf-8") if f"| `{var}` |" in l][0].split("|")[5]
    mids = {}
    for code, text in re.findall(r"`(\d+)`=\d+ ([^;`]+)", full):
        code = int(code); t = text.replace("—", "-").replace("–", "-")
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", t.split("萬")[0] if "萬" in t else t)]
        if "無收入" in t: mids[code] = 0.0
        elif "以下" in t and nums: mids[code] = nums[0] / 2 * 1e4
        elif "以上" in t and nums: mids[code] = nums[0] * 1.25 * 1e4
        elif len(nums) >= 2: mids[code] = (nums[0] + nums[1]) / 2 * 1e4
    return mids


frames = []
for y, m in MAP.items():
    d = load_wave("RR", y)
    pid = pd.to_numeric(d["x01"], errors="coerce")
    o = pd.DataFrame({"pid": (pid // 10 if y == 2024 else pid).astype("Int64"), "year": y})
    o["pid_raw"] = pid
    o["happy"] = V(d[m["happy"]], 1, 7)
    o["good"] = V(d[m["good"]], 1, 7)
    o["health"] = rev(V(d[m["health"]], 1, 5), 1, 5)
    o["jobsat"] = rev(V(d[m["job"]], 1, 4), 1, 4)
    o["famsat"] = rev(V(d[m["fam"]], 1, 4), 1, 4)
    o["econsat"] = rev(V(d[m["econ"]], 1, 4), 1, 4) if "econ" in m else np.nan
    o["marsat"] = rev(V(d[m["marsat"]], 1, 4), 1, 4) if "marsat" in m else np.nan
    o["sex"] = V(d[m["sex"]], 1, 2)
    byr = V(d[m["byr"]], 1, 100)
    o["birth_year"] = byr + 1911
    o["web"] = (pd.to_numeric(d[m["mode"]], errors="coerce") == 4).astype(float) if "mode" in m else 0.0
    # 是否有工作：有工作問題時直接用；早期波次用「工作滿意度 0 = 目前沒有工作」判斷
    if "emp" in m:
        # 「目前有沒有工作」：『沒有』的代碼 2016–2020 是 3、2022 是 4、2024 是 5；比它小的代碼都是某種有工作
        no_code = {2016: 3, 2018: 3, 2020: 3, 2022: 4, 2024: 5}[y]
        e = pd.to_numeric(d[m["emp"]], errors="coerce")
        o["employed"] = np.where(e.between(1, no_code - 1), 1.0, np.where(e == no_code, 0.0, np.nan))
        o.loc[o.jobsat.notna(), "employed"] = 1.0
    else:
        jraw = pd.to_numeric(d[m["job"]], errors="coerce")
        o["employed"] = np.where(jraw.between(1, 4), 1.0, np.where(jraw == 0, 0.0, np.nan))
    if "mar" in m:
        mr = pd.to_numeric(d[m["mar"]], errors="coerce")
        o["marital"] = mr.map(MAR_MAP_2022 if y == 2022 else MAR_MAP_STD)
    else:
        o["marital"] = np.nan
    if "nkids" in m:
        o["nkids_direct"] = V(d[m["nkids"]], 0, 20)
    # 子女名冊：每位子女的出生年（民國），用來算子女數與最小子女年齡
    pfx = ROSTER[y]
    kcols = [c for c in d.columns if re.fullmatch(pfx + r"\d+", c)]
    B = d[kcols].apply(pd.to_numeric, errors="coerce")
    B = B.where((B >= 1) & (B <= y - 1911))
    o["nkids_roster"] = B.notna().sum(axis=1)
    o["youngest_child_age"] = (y - 1911) - B.max(axis=1)
    if "inc" in m:
        inc = pd.to_numeric(d[m["inc"]], errors="coerce")
        inc = inc.where((inc > 0) & (inc < 9999990))          # 0 = 跳答／沒有工作，不當作收入 0
        if "incb" in m:
            mids = income_bracket_midpoints(y, m["incb"])
            inc = inc.fillna(pd.to_numeric(d[m["incb"]], errors="coerce").map(mids).replace(0, np.nan))
        o["income_raw"] = inc
    if "hours" in m:
        o["hours_raw"] = V(d[m["hours"]], 1, 120)            # 0 = 跳答（沒有工作）
    frames.append(o)
    print(y, len(o), "pid NA:", o.pid.isna().sum())

long = pd.concat(frames, ignore_index=True)
long.shape

# %% [markdown]
# ### 2.1 補上舊合併檔（v3）的變數：教育年數、2007–2018 的收入與工時
#
# 先前整理的 `psfd_full_panel_clean.csv`（v3）已換算好教育年數與早期波次的收入、工時，這裡沿用；但它的 RR2024 編號沒有 ÷10，所以先修正再合併。
# ⚠️ v3 的「最小子女年齡」在 2018 年後明顯錯誤（最小值逐波 +2 歲，代表新生兒沒有被記錄），所以子女數與最小子女年齡改由上面直接從各波子女名冊重新計算。若找不到 v3 檔，這一步會自動跳過。

# %%
V3 = RAW.parent / "psfd_full_panel_clean.csv"   # 先前階段的合併檔（不公開）；沒有時見下方說明
if V3.exists():
    v3 = pd.read_csv(V3, low_memory=False)
    v3 = v3[v3.wave.str.startswith("RR")].copy()
    v3["year"] = v3.wave.str[2:].astype(int)
    v3["pid"] = np.where(v3.year == 2024, v3.pid // 10, v3.pid)
    keep = ["pid", "year", "education_years",
            "annual_work_income", "work_hours_week", "gender_filled", "birth_year_ad_filled"]
    v3 = v3[keep].drop_duplicates(["pid", "year"])
    long = long.merge(v3, on=["pid", "year"], how="left")
    print("v3 合併成功；與 RR2024 對上的比例：",
          long.loc[long.year == 2024, "education_years"].notna().mean().round(3))
else:
    # 沒有 v3 檔：教育年數與 2007–2018 的收入、工時會是缺失值，其餘分析不受影響
    for c in ["education_years", "annual_work_income", "work_hours_week"]:
        long[c] = np.nan
    print("找不到 v3 檔，略過（教育年數與 2020 年前的收入、工時將為缺失值）")

# %% [markdown]
# ### 2.2 時間不變的特質：性別、出生年用同一人各波的眾數補齊；年齡 = 調查年 − 出生年

# %%
def mode_or_nan(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan

sex_p = long.groupby("pid")["sex"].agg(mode_or_nan)
byr_p = long.groupby("pid")["birth_year"].agg(mode_or_nan)
long["sex"] = long["sex"].fillna(long.pid.map(sex_p))
long["birth_year"] = long["birth_year"].fillna(long.pid.map(byr_p))
if "gender_filled" in long:
    long["sex"] = long["sex"].fillna(long["gender_filled"])
    long["birth_year"] = long["birth_year"].fillna(long["birth_year_ad_filled"])
long["female"] = (long.sex == 2).astype(float).where(long.sex.notna())
long["age"] = long.year - long.birth_year
# 收入：2020 起用原始題（含 2022/2024 級距中點），之前用 v3
long["income"] = long.get("income_raw").combine_first(long.get("annual_work_income"))
long["log_income"] = np.log1p(long.income / 10000)  # log(1 + 萬元)
long["hours"] = long.get("hours_raw").combine_first(long.get("work_hours_week"))
# 子女：2020 起有直接題，優先使用；其餘用名冊換算（名冊與直接題在 2020–2024 一致率 96–99%）
long["nkids"] = long.get("nkids_direct").combine_first(long["nkids_roster"])
long["has_kids"] = (long.nkids > 0).astype(float).where(long.nkids.notna())
# 有 6 歲以下子女：沒有小孩 = 0；有小孩但不知道最小子女年齡 = 缺失
long["kid_under6"] = np.where(long.nkids == 0, 0.0,
                              np.where(long.youngest_child_age.notna(), (long.youngest_child_age < 6).astype(float), np.nan))
long["married"] = (long.marital == "married").astype(float).where(long.marital.notna())
# 教育年數：成年後幾乎不變，取同一人最大值補齊
edu_p = long.groupby("pid")["education_years"].max()
long["edu_years"] = long.pid.map(edu_p)

# %%
# 檢查：子女名冊算出的子女數 vs「您有幾個小孩」直接題（兩者都有的波次）
_k = long.dropna(subset=["nkids_direct", "nkids_roster"])
kid_check = _k.groupby("year").apply(lambda g: pd.Series({
    "N": len(g),
    "完全一致%": (g.nkids_direct == g.nkids_roster).mean() * 100,
    "有無子女一致%": ((g.nkids_direct > 0) == (g.nkids_roster > 0)).mean() * 100})).round(1)
kid_check

# %% [markdown]
# ### 2.3 幸福分數（長資料版）
#
# 2007–2024 每一波都有的幸福題只有兩題（快樂、生活過得好不好），所以長資料用兩題版：把 1–7 分轉成 0–100 後平均。
# 主分析（2020–2024）再加上情緒 6 題成為三面向版（見第 3 節）。

# %%
to100 = lambda s, lo, hi: (s - lo) / (hi - lo) * 100
long["swb2"] = pd.concat([to100(long.happy, 1, 7), to100(long.good, 1, 7)], axis=1).mean(axis=1, skipna=False)
long.groupby("year")[["happy", "good", "swb2", "jobsat", "famsat", "econsat", "health", "employed", "married", "nkids", "age", "female"]].agg(lambda s: s.notna().mean()).round(2)

# %%
long.groupby("year")[["swb2", "jobsat", "famsat", "health", "age"]].mean().round(2)

# %% [markdown]
# ## 3. 主分析檔：RR2020 / RR2022 / RR2024
#
# **情緒 6 題**：三波共同題（2020 的 15 題情緒量表與 2022/2024 的 10 題量表重疊的部分），正向 3 題（安心、穩重沈著冷靜、知足）＋負向 3 題反轉（緊張不安、困難一直增加、太擔心無關緊要的事），1–4 分。
#
# **家庭導向（傳統家庭觀）分數 FO**：先讀入婚姻態度 6 題（2024 沒有第 5 題，只用四波共同的 6 題）與性別角色 8 題（2018–2024 共同的 a–h），每題轉成「越高 = 越重視婚姻與傳統家庭」。探索性因素分析（見 notebook 01）顯示性別角色中「平權」那幾題（a、e、f、g、h）和其他題目不在同一個因素上，所以 FO 只用 9 題：婚姻態度 6 題＋性別角色 b、c、d 三題（母親工作對幼兒不好、妻子全職對家庭不好、女人真正想要的是家和小孩）。
#
# ⚠️ **重要發現**：婚姻態度、性別角色、個性題在 RR2022 只問了約 270 人、RR2024 只問了約 135 人（其餘為跳答 −10，應是只給部分樣本的題組）；RR2018 也只有約 1,780 人。真正全樣本都有的是 **RR2020**（約 5,700 人）與 **RR2011 的婚姻態度**（約 4,880 人）。因此 FO 的主要分析用 RR2020。唯一例外是「丈夫的責任是賺錢養家，太太是照顧家庭」這一題，在 2020/2022/2024 都是全樣本，另外存成 `breadwinner`。
#
# **家庭價值重要性（RR2022 獨有）**：「您認為下列觀念對您自己的重要性」8 題（結婚成家、盡量維持婚姻、教導子女、家人感情好、為了子女不離婚、家庭幫助成長、賺錢養家、有困難家人幫忙），1=不重要…5=絕對重要。這是唯一直接問「重要性」的題組，最接近「權重」的概念。

# %%
AFF = {
    2020: dict(pos=["h03c", "h03d", "h03l"], neg=["h03a", "h03e", "h03f"],
               full_pos=["h03b", "h03c", "h03d", "h03g", "h03j", "h03l", "h03n"],
               full_neg=["h03a", "h03e", "h03f", "h03h", "h03i", "h03k", "h03m", "h03o"]),
    2022: dict(pos=["m02z04", "m02z05", "m02z06"], neg=["m02z01", "m02z02", "m02z03"]),
    2024: dict(pos=["m02z04", "m02z05", "m02z06"], neg=["m02z01", "m02z02", "m02z03"]),
}
LONE = {2020: dict(pos=["h13a", "h13d", "h13f"], neg=["h13b", "h13c", "h13e"]),
        2022: dict(pos=["m01z01", "m01z04", "m01z06"], neg=["m01z02", "m01z03", "m01z05"]),
        2024: dict(pos=["m01z01", "m01z04", "m01z06"], neg=["m01z02", "m01z03", "m01z05"])}
# 婚姻態度：+1 = 同意代表重視婚姻/家庭；-1 = 同意代表不重視
MAR_ITEMS = {"a": -1, "b": -1, "c": +1, "d": -1, "f": +1, "g": +1}
# 性別角色：+1 = 同意代表傳統家庭分工
GEN_ITEMS = {"a": -1, "b": +1, "c": +1, "d": +1, "e": +1, "f": -1, "g": -1, "h": -1}
LET2NUM = {c: f"{i+1:02d}" for i, c in enumerate("abcdefghij")}


def att_vars(year):
    if year == 2011:
        mar = {k: f"d02z{LET2NUM[k]}" for k in MAR_ITEMS}
        gen = {}
    elif year == 2018:
        mar = {k: f"h02{k}" for k in MAR_ITEMS}; gen = {k: f"h03{k}" for k in GEN_ITEMS}
    elif year == 2020:
        mar = {k: f"h07{k}" for k in MAR_ITEMS}; gen = {k: f"h06{k}" for k in GEN_ITEMS}
    else:
        mar = {k: f"e04z{LET2NUM[k]}" for k in MAR_ITEMS}; gen = {k: f"e03z{LET2NUM[k]}" for k in GEN_ITEMS}
    return mar, gen


def oriented(d, varmap, signs):
    cols = {}
    for k, v in varmap.items():
        x = V(d[v], 1, 5)
        cols[f"{k}"] = x if signs[k] > 0 else 6 - x
    return pd.DataFrame(cols)


att_frames, main_frames = [], []
for y in [2011, 2018, 2020, 2022, 2024]:
    d = load_wave("RR", y)
    pid = pd.to_numeric(d["x01"], errors="coerce")
    pid = pid // 10 if y == 2024 else pid
    mar, gen = att_vars(y)
    M = oriented(d, mar, MAR_ITEMS).add_prefix("mar_")
    G = oriented(d, gen, GEN_ITEMS).add_prefix("gen_") if gen else pd.DataFrame(index=d.index)
    a = pd.concat([pd.DataFrame({"pid": pid, "year": y}), M, G], axis=1)
    a["mar_att"] = mean_if(M); a["gen_trad"] = mean_if(G) if gen else np.nan
    att_frames.append(a)
    if y < 2020:
        continue
    o = pd.DataFrame({"pid": pid, "year": y})
    A = pd.concat([V(d[c], 1, 4) for c in AFF[y]["pos"]] + [5 - V(d[c], 1, 4) for c in AFF[y]["neg"]], axis=1)
    A.columns = [f"aff_{i}" for i in range(6)]
    o = pd.concat([o, A], axis=1)
    o["affect6"] = mean_if(A, 0.67)
    if "full_pos" in AFF[y]:
        Af = pd.concat([V(d[c], 1, 4) for c in AFF[y]["full_pos"]] + [5 - V(d[c], 1, 4) for c in AFF[y]["full_neg"]], axis=1)
        o["affect15"] = mean_if(Af, 0.67)
    L = pd.concat([V(d[c], 1, 5) for c in LONE[y]["pos"]] + [6 - V(d[c], 1, 5) for c in LONE[y]["neg"]], axis=1)
    o["social6"] = mean_if(L, 0.67)          # 高 = 社會連結好、不寂寞
    o["mar_att"] = a["mar_att"].values; o["gen_trad"] = a["gen_trad"].values
    for c in M.columns: o[c] = M[c].values
    for c in G.columns: o[c] = G[c].values
    if y == 2022:
        I = pd.concat([V(d[f"e02z0{i}"], 1, 5) for i in range(1, 9)], axis=1)
        I.columns = [f"imp_{i}" for i in range(1, 9)]
        o = pd.concat([o, I], axis=1); o["fam_importance"] = mean_if(I)
        F = pd.concat([V(d[f"e01z0{i}"], 1, 5) for i in range(1, 9)], axis=1)
        o["filial"] = mean_if(F)
    if y in (2022, 2024):     # 個性（簡版五大）1=很符合 → 反轉成 5=很符合
        P = pd.concat([6 - V(d[f"e05z{i:02d}"], 1, 5) for i in range(1, 11)], axis=1)
        P.columns = [f"pers_{i}" for i in range(1, 11)]
        o = pd.concat([o, P], axis=1)
    if y == 2020:
        W = pd.concat([V(d[f"w07z{i:02d}"], 1, 5) for i in range(0, 13)], axis=1)
        W.columns = [f"jobchar_{i:02d}" for i in range(0, 13)]
        o = pd.concat([o, W], axis=1)
        wfc = pd.concat([V(d[f"w08z0{i}"], 1, 5) for i in range(1, 5)], axis=1)
        o["wfc"] = mean_if(wfc); o["wfc_w2f"] = mean_if(wfc.iloc[:, :2]); o["wfc_f2w"] = mean_if(wfc.iloc[:, 2:])
        o["subj_class"] = V(d["h10a"], 1, 10)
    if y == 2024:
        wfc = pd.concat([V(d[f"c19i0{i}"], 1, 5) for i in range(1, 5)], axis=1)
        o["wfc"] = mean_if(wfc); o["wfc_w2f"] = mean_if(wfc.iloc[:, :2]); o["wfc_f2w"] = mean_if(wfc.iloc[:, 2:])
    bw = {2020: "h06j", 2022: "e03z10", 2024: "e03z10"}[y]
    o["breadwinner"] = V(d[bw], 1, 5)        # 同意「丈夫賺錢養家、太太照顧家庭」= 高分
    o["respondent_type"] = pd.to_numeric(d["x01a"], errors="coerce")   # 1 主樣本、3 子女轉主樣本
    main_frames.append(o)

att = pd.concat(att_frames, ignore_index=True)
main = pd.concat(main_frames, ignore_index=True)
main = main.merge(long, on=["pid", "year"], how="left", validate="1:1")
main.shape

# %% [markdown]
# ### 3.1 三面向幸福指數
#
# 依 OECD《主觀福祉測量指引》，主觀幸福至少包含「生活評價」與「情緒」兩個面向；我們用三個面向等權平均：
# 快樂（1–7）、生活過得好不好（1–7）、情緒 6 題平均（1–4），各自線性轉為 0–100 後平均 → `swb`（0–100 分）。
# 三個面向都有值才計分（避免只答一部分的人分數被拉偏）。

# %%
main["swb"] = pd.concat([to100(main.happy, 1, 7), to100(main.good, 1, 7), to100(main.affect6, 1, 4)], axis=1).mean(axis=1, skipna=False)
# 也保留同一套的兩題版，方便與長資料比較
main["swb2"] = pd.concat([to100(main.happy, 1, 7), to100(main.good, 1, 7)], axis=1).mean(axis=1, skipna=False)
for y in [2020, 2022, 2024]:
    s = main[main.year == y]
    items = pd.concat([z(s.happy), z(s.good), z(s.affect6)], axis=1)
    print(y, "三面向 alpha =", round(cronbach_alpha(items), 3),
          "| 8 題 alpha =", round(cronbach_alpha(pd.concat([s.happy, s.good, s[[f'aff_{i}' for i in range(6)]]], axis=1)), 3),
          "| N(有幸福分數) =", s.swb.notna().sum())

# %% [markdown]
# ### 3.2 家庭導向分數（FO）與家庭價值重要性
#
# FO = 婚姻態度 6 題與性別角色 8 題兩個分量表的平均（1–5 分，高 = 越重視婚姻、越傳統的家庭分工）。

# %%
FO_ITEMS = ["mar_a", "mar_b", "mar_c", "mar_d", "mar_f", "mar_g", "gen_b", "gen_c", "gen_d"]
att["fo"] = mean_if(att.reindex(columns=FO_ITEMS), 0.67)
att.loc[att.year == 2011, "fo"] = np.nan          # 2011 只有婚姻態度，FO 不完整
att["mar6"] = mean_if(att[[c for c in FO_ITEMS if c.startswith("mar_")]], 0.67)
main["fo"] = mean_if(main.reindex(columns=FO_ITEMS), 0.67)
print(att.groupby("year")[["mar6", "fo"]].count())
for y in [2011, 2018, 2020, 2022, 2024]:
    s = att[att.year == y]
    print(y, "N =", s.mar6.notna().sum(), "| 婚姻態度 6 題 alpha", round(cronbach_alpha(s[FO_ITEMS[:6]]), 3),
          "| FO 9 題 alpha", round(cronbach_alpha(s[FO_ITEMS]), 3) if y != 2011 else "—")
s = main[main.year == 2022]
print("家庭價值重要性 8 題 alpha", round(cronbach_alpha(s.filter(like="imp_")), 3))

# %% [markdown]
# ## 4. 存檔與總覽

# %%
keep_long = ["pid", "year", "happy", "good", "swb2", "health", "jobsat", "famsat", "econsat", "marsat",
             "female", "age", "birth_year", "web", "employed", "marital", "married", "nkids", "has_kids",
             "kid_under6", "youngest_child_age", "edu_years", "income", "log_income", "hours"]
long[keep_long].to_csv(DERIVED / "long_2007_2024.csv", index=False, encoding="utf-8-sig")
main.drop(columns=[c for c in ["pid_raw", "gender_filled", "birth_year_ad_filled", "income_raw", "hours_raw",
                               "nkids_direct", "annual_work_income", "work_hours_week",
                               "education_years"] if c in main]).to_csv(
    DERIVED / "main_2020_2024.csv", index=False, encoding="utf-8-sig")
att.to_csv(DERIVED / "attitudes_long.csv", index=False, encoding="utf-8-sig")

summary = (main.groupby("year").agg(N=("pid", "size"), 有幸福分數=("swb", lambda s: s.notna().sum()),
                                     有工作=("employed", "sum"), 平均年齡=("age", "mean"),
                                     女性比例=("female", "mean"), 網路自填=("web", "mean"))
           .round(2))
summary

# %% [markdown]
# ### 4.1 追蹤資料的樣本結構：同一個人被訪問幾次？

# %%
ids = {y: set(main.loc[main.year == y, "pid"]) for y in [2020, 2022, 2024]}
print("2020∩2022:", len(ids[2020] & ids[2022]), " 2022∩2024:", len(ids[2022] & ids[2024]),
      " 三波都有:", len(ids[2020] & ids[2022] & ids[2024]))
ok = long.dropna(subset=["swb2", "jobsat", "famsat"])
cnt = ok.groupby("pid").size()
print("2007–2024 有工作且幸福/工作/家庭滿意度都有效的人：", cnt.size, "人；觀測", len(ok), "筆")
print(cnt.value_counts().sort_index().to_string())
