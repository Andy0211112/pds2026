"""把 SRDA 下載的 PSFD SPSS 檔（.sav）轉成本專案使用的格式：一份 CSV＋一份 Markdown 編碼簿。

用法（在 repo 根目錄執行）：
    python tools/sav_to_csv.py <.sav 檔路徑> RR_2020
    python tools/sav_to_csv.py <.sav 檔路徑> RR_2005 --encoding big5hkscs

輸出：data/cleaned_data/RR_2020/RR_2020.csv 與 RR_2020_編碼簿.md
- CSV 保留原始數值代碼（不套用標籤）。
- 編碼簿每列：| 序號 | `變數名稱` | 變數說明 | 型態 | `代碼`=標籤; ... |，notebook 00 會用它查數值標籤與所得級距。
- 少數舊檔（RII_2000、RII_2001、RIII_2001、RR_2005）中文會亂碼，需加 --encoding big5hkscs。
"""
import argparse
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]


def fmt_code(k):
    try:
        f = float(k)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(k)


def clean(x):
    return str(x if x is not None else "").replace("|", "/").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sav", help="SRDA 下載的 .sav 檔")
    ap.add_argument("name", help="輸出資料夾名稱，格式為 {問卷類型}_{年份}，例如 RR_2020")
    ap.add_argument("--out", default=str(ROOT / "data" / "cleaned_data"), help="輸出根目錄（預設 data/cleaned_data）")
    ap.add_argument("--encoding", default=None, help="強制指定編碼，例如 big5hkscs")
    a = ap.parse_args()

    kw = {"encoding": a.encoding} if a.encoding else {}
    df, meta = pyreadstat.read_sav(a.sav, apply_value_formats=False, **kw)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.name}.csv", index=False, encoding="utf-8")

    lines = [f"# {a.name} 編碼簿 (Codebook)", "",
             f"- 來源檔案：{Path(a.sav).name}",
             f"- 對應資料檔：{a.name}.csv",
             f"- 受訪者筆數（列數）：{len(df)}",
             f"- 變數個數（欄位數）：{df.shape[1]}", "",
             "> 說明：本編碼簿由原始 SPSS 檔（.sav）之變數屬性自動整理而成；CSV 欄位皆為原始數值代碼。", "",
             "| # | 變數名稱 | 變數說明 | 型態 | 數值標籤（代碼=意義） |",
             "|---|---|---|---|---|"]
    labels = dict(zip(meta.column_names, meta.column_labels))
    for i, c in enumerate(df.columns, 1):
        vl = meta.variable_value_labels.get(c, {})
        vals = "; ".join(f"`{fmt_code(k)}`={clean(v)}" for k, v in vl.items())
        typ = "數值 (numeric)" if meta.readstat_variable_types.get(c) == "double" else "文字 (string)"
        lines.append(f"| {i} | `{c}` | {clean(labels.get(c))} | {typ} | {vals} |")
    (out / f"{a.name}_編碼簿.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"完成：{out}（{len(df)} 列 × {df.shape[1]} 欄）")


if __name__ == "__main__":
    main()
