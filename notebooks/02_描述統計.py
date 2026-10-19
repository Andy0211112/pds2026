# %% [markdown]
# # 02 描述統計：台灣人過得怎麼樣？
#
# 先看幾個基本事實：幸福分數的分布、隨年齡與人生狀態的差異、工作/家庭/經濟三個領域的滿意度，以及 18 年來的變化。
# 這些事實是故事的「場景」：讀者要先知道大家現在大概在哪裡，後面才看得懂「什麼讓人更幸福」。

# %%
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from psfd_utils import *
plt = setup_plot()
main = pd.read_csv(DERIVED / "main_2020_2024.csv", low_memory=False)
long = pd.read_csv(DERIVED / "long_2007_2024.csv", low_memory=False)
main["age_g"] = pd.cut(main.age, [0, 34, 44, 54, 64, 120], labels=["≤34", "35–44", "45–54", "55–64", "65+"])
main["inc_q"] = main.groupby("year").income.transform(lambda s: pd.qcut(s.where(s > 0), 5, labels=False, duplicates="drop") + 1)

# %% [markdown]
# ## 1. 幸福分數（0–100）的整體分布與各群體平均（2020–2024 合併）

# %%
def grp(col, d=main):
    t = d.groupby(col, observed=True).swb.agg(["mean", "std", "count"])
    t["se"] = t["std"] / np.sqrt(t["count"])
    return t.round(2)

print("全體：平均", round(main.swb.mean(), 1), "標準差", round(main.swb.std(), 1))
for col in ["year", "age_g", "female", "marital", "employed", "has_kids", "kid_under6", "inc_q"]:
    print(f"\n== {col}"); print(grp(col).to_string())

# %% [markdown]
# ## 2. 三個領域的滿意度分布

# %%
lab = {1: "很不滿意", 2: "不太滿意", 3: "還算滿意", 4: "很滿意"}
dist = pd.DataFrame({k: main[v].map(lab).value_counts(normalize=True) for k, v in
                     [("工作（有工作者）", "jobsat"), ("家庭生活", "famsat"), ("個人經濟", "econsat"), ("婚姻（已婚）", "marsat")]}).reindex(list(lab.values()))
(dist * 100).round(1)

# %%
fig, ax = plt.subplots(figsize=(8, 2.8))
cols = ["#d03b3b", "#ec835a", "#86b6ef", "#2a78d6"]
left = np.zeros(4)
for i, k in enumerate(lab.values()):
    v = dist.loc[k].values * 100
    ax.barh(dist.columns, v, left=left, color=cols[i], label=k, edgecolor="white", linewidth=1.5)
    for j, (l, w) in enumerate(zip(left, v)):
        if w > 6: ax.text(l + w / 2, j, f"{w:.0f}%", ha="center", va="center", fontsize=9, color="white" if i in (0, 3) else "#0b0b0b")
    left += v
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("%"); ax.grid(False)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_title("經濟是最多人不滿意的領域；家庭與婚姻最多人「很滿意」")
savefig(fig, "fig02_domain_satisfaction")

# %% [markdown]
# ## 3. 年齡與幸福：U 型嗎？

# %%
d = main.dropna(subset=["swb", "age"])
d["age5"] = (d.age // 5 * 5).clip(25, 80)
t = d.groupby("age5").swb.agg(["mean", "count", "std"])
t["se"] = t["std"] / np.sqrt(t["count"])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.errorbar(t.index + 2, t["mean"], yerr=1.96 * t.se, color=PALETTE["blue"], marker="o", capsize=3)
ax.set_xlabel("年齡"); ax.set_ylabel("幸福分數（0–100）")
ax.set_title("幸福分數隨年齡變化（2020–2024 合併，95% 信賴區間）")
savefig(fig, "fig02b_swb_by_age")
t.round(2)

# %%
m = smf.ols("swb ~ age + I(age**2) + female + web + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": d.pid})
print(m.summary().tables[1])
if m.params["I(age ** 2)"] > 0:
    print("U 型最低點年齡 ≈", round(-m.params["age"] / (2 * m.params["I(age ** 2)"]), 1))

# %% [markdown]
# ## 4. 18 年趨勢：兩題版幸福分數與三個領域滿意度（2007–2024）

# %%
tr = long.groupby("year")[["swb2", "jobsat", "famsat", "health"]].mean()
tr["N"] = long.groupby("year").size()
tr.round(2)

# %%
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(tr.index, tr.swb2, marker="o", color=PALETTE["blue"])
ax.set_ylim(50, 75); ax.set_ylabel("兩題版幸福分數（0–100）")
ax.set_xticks([2007, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
ax.set_title("2007–2024 平均幸福分數：小幅上升，2020 疫情年略降")
for yy in [2009, 2020]:
    ax.annotate(f"{tr.loc[yy, 'swb2']:.1f}", (yy, tr.loc[yy, "swb2"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9)
savefig(fig, "fig02c_swb_trend")

# %% [markdown]
# ## 5. 「家庭比工作滿意」的人有多少？

# %%
e = main[main.employed == 1].dropna(subset=["jobsat", "famsat"])
print("家庭滿意度 > 工作滿意度：", round((e.famsat > e.jobsat).mean() * 100, 1), "%")
print("家庭滿意度 = 工作滿意度：", round((e.famsat == e.jobsat).mean() * 100, 1), "%")
print("家庭滿意度 < 工作滿意度：", round((e.famsat < e.jobsat).mean() * 100, 1), "%")
print("\n工作 × 家庭滿意度交叉表（%）")
(pd.crosstab(e.jobsat.map(lab), e.famsat.map(lab), normalize=True) * 100).round(1).reindex(index=list(lab.values()), columns=list(lab.values()))

# %%
print("工作滿意度與家庭滿意度的相關 r =", round(e.jobsat.corr(e.famsat), 3))
print("工作與經濟滿意度 r =", round(e.jobsat.corr(e.econsat), 3), "；家庭與經濟 r =", round(e.famsat.corr(e.econsat), 3))
