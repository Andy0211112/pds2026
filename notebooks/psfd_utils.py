"""
psfd_utils.py — 本專案所有 notebook 共用的小工具
================================================
放在 notebooks/ 資料夾裡，各 notebook 用 `from psfd_utils import *` 載入。

路徑約定（相對於 repo 根目錄）：
    data/cleaned_data/RR_2020/RR_2020.csv   ← 逐年原始資料（已轉 CSV，不隨 repo 公開，見 data/README.md）
    data/raw_sav/<原始資料夾>/*.sav          ← 若 CSV 不在，退而讀 SPSS 檔
    data/                                   ← 本專案產生的衍生資料與結果表
    figures/                                ← 圖檔
也可用環境變數 PSFD_DATA 指定 cleaned_data 的位置。
"""
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # repo 根目錄
_RAW_CANDIDATES = [ROOT / "data" / "cleaned_data", ROOT.parent / "data" / "cleaned_data"]
RAW = Path(os.environ["PSFD_DATA"]) if "PSFD_DATA" in os.environ else \
    next((p for p in _RAW_CANDIDATES if p.exists()), _RAW_CANDIDATES[0])
RAW_SAV = ROOT / "data" / "raw_sav"
DERIVED = ROOT / "data"
FIG = ROOT / "figures"
DERIVED.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 讀檔
_SAV_ENCODING = {"RII_2000": "big5hkscs", "RII_2001": "big5hkscs",
                 "RIII_2001": "big5hkscs", "RR_2005": "big5hkscs"}


def load_wave(code: str, year: int, usecols=None) -> pd.DataFrame:
    """讀取某一波資料，例如 load_wave('RR', 2020)。
    優先讀 cleaned_data 的 CSV；若不存在（例如 RR_2018/RR_2024 的 CSV 太大未同步），
    改讀 data/data 底下的原始 .sav。"""
    name = f"{code}_{year}"
    csv = RAW / name / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv, usecols=usecols, low_memory=False)
    import pyreadstat, glob
    pat = f"{code}{year}"
    cands = [p for p in RAW_SAV.glob("*") if pat in p.name.replace("（", "(").replace("）", ")")]
    for folder in cands:
        savs = list(folder.glob("*.sav"))
        if savs:
            kw = {"encoding": _SAV_ENCODING[name]} if name in _SAV_ENCODING else {}
            df, _ = pyreadstat.read_sav(str(savs[0]), apply_value_formats=False, usecols=usecols, **kw)
            return df
    raise FileNotFoundError(f"找不到 {name} 的 CSV 或 SAV")


# ---------------------------------------------------------------- 清理
def V(s, lo, hi):
    """只保留介於 [lo, hi] 的有效值，其餘（跳答、不知道、拒答、遺漏值代碼）→ NaN"""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(lo, hi))


def rev(s, lo, hi):
    """反向計分：例如 1=很滿意…4=很不滿意 → 4=很滿意…1=很不滿意"""
    return (lo + hi) - s


def z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / s.std()


def cronbach_alpha(df: pd.DataFrame) -> float:
    d = df.dropna()
    k = d.shape[1]
    if k < 2 or len(d) < 3:
        return np.nan
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))


def mean_if(df: pd.DataFrame, min_valid: float = 0.5):
    """題組平均：至少答了 min_valid 比例的題目才計分"""
    ok = df.notna().mean(axis=1) >= min_valid
    return df.mean(axis=1).where(ok)


# ---------------------------------------------------------------- 繪圖風格
PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "grey": "#9a9993", "grey_light": "#d6d5d0", "ink": "#0b0b0b", "ink2": "#52514e",
}
C_FAMILY, C_WORK, C_ECON, C_HEALTH = PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["grey"]


def setup_plot():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    wanted = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC",
              "Noto Sans TC", "Source Han Sans TC", "Arial Unicode MS"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in wanted if f in have] or ["sans-serif"]
    mpl.rcParams.update({
        "font.family": fam, "axes.unicode_minus": False,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": PALETTE["ink2"],
        "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def savefig(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p)
    return p


# ---------------------------------------------------------------- 模型小工具
def cv_r2(X, y, k=5, seed=1, model=None):
    """k 折交叉驗證 R²（預設線性迴歸）"""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.base import clone
    X = np.asarray(X, float); y = np.asarray(y, float)
    m0 = model if model is not None else LinearRegression()
    out = []
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m = clone(m0).fit(X[tr], y[tr])
        p = m.predict(X[te])
        out.append(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(out)), float(np.std(out))


def coef_table(res, names=None):
    """statsmodels 結果 → 係數表（係數、標準誤、95% CI、p）"""
    t = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    ci = res.conf_int(); t["ci_low"], t["ci_high"] = ci[0], ci[1]
    if names is not None:
        t = t.loc[[n for n in names if n in t.index]]
    return t.round(4)
