# data/

這個資料夾放兩種東西：

1. **彙總結果表（隨 repo 公開）**：notebooks 產生的迴歸係數、各組平均與模型比較，不含任何受訪者的個別紀錄。括號內是產生它的 notebook 編號。
2. **原始資料與個人層級資料（不公開）**：依 SRDA 使用規範不可散布，已列在 `.gitignore`。重現分析時請自行申請資料，並用 `tools/sav_to_csv.py` 轉檔到 `data/cleaned_data/`。

## 執行 notebook 時需要的輸入（不公開）

| 路徑 | 說明 |
|---|---|
| `data/cleaned_data/RR_2007/RR_2007.csv`、`RR_2007_編碼簿.md` …… `RR_2024` | 12 波原始資料（RR2007–2012、2014、2016、2018、2020、2022、2024），由 `tools/sav_to_csv.py` 產生 |
| `data/psfd_full_panel_clean.csv` | （選用）前一階段的合併檔，提供教育年數與 2007–2018 年的收入、工時 |

## notebook 00 產生、但不公開的個人層級檔

`long_2007_2024.csv`、`main_2020_2024.csv`、`attitudes_long.csv`、`importance_subscales.csv`、`personal_slopes_blup.csv`

## 公開的結果表

| 檔案 | 內容 |
|---|---|
| `children_effects.csv` | 父母依子女階段的幸福差異（08） |
| `clusters_2020.csv` | 家庭導向 K-means 兩群的權重（05） |
| `clusters_2022.csv` | 重要性 K-means 兩群的權重（05） |
| `cross_lagged.csv` | 交叉延宕係數（一般版與個人中心化版）（07） |
| `cv_models_2022.csv` | 六種權重設定的 2022 年 5 折交叉驗證 R²（04） |
| `event_studies.csv` | 第一胎、失業事件研究的係數與 95% 區間（07） |
| `fe_vs_pooled.csv` | 一般迴歸 vs 個人固定效果的係數（07） |
| `importance_quintile_slopes.csv` | 家庭價值重要性五等分的家庭／工作斜率（04） |
| `job_characteristics_2020.csv` | 工作特性對工作滿意度的係數（08） |
| `latent_class_results.csv` | 潛在類別迴歸 K = 2–4 各類的比例、斜率、權重與殘差（05） |
| `life_stage_weights.csv` | 五個人生階段的家庭／工作斜率與家庭權重 w（05） |
| `ml_additive_vs_full.csv` | 線性 vs 梯度提升樹（加法／完整）的交叉驗證 R²（09） |
| `moderation_results.csv` | 重要性、家庭導向、傳統分工 × 滿意度的交互作用（04） |
| `objective_total_effects.csv` | 婚姻、子女、就業對幸福的總效果與中介比例（08） |
| `oot_2024_models.csv` | 六種權重設定以 2022 估計、預測 2024 的 R²（04） |
| `personal_weight_prediction.csv` | 用態度與人生階段預測個人權重的 R²（06） |
| `random_slope_summary.csv` | 隨機斜率模型的平均斜率、個人標準差、相關與 LR（06） |
| `robustness.csv` | 16 種替代設定下的家庭與工作係數（09） |
| `shapley_recipe.csv` | 各領域占 R² 的 Shapley 分解（03） |
| `stage1_家庭滿意.csv` | 客觀條件 → 家庭滿意度的係數（08） |
| `stage1_工作滿意.csv` | 客觀條件 → 工作滿意度的係數（08） |
| `stage1_經濟滿意.csv` | 客觀條件 → 經濟滿意度的係數（08） |
| `weights_subgroups.csv` | 各群體（性別、年齡、婚姻、子女、教育、收入）的係數（03） |
| `weights_trend_2007_2024.csv` | 2007–2024 各年工作、家庭、健康的標準化係數與 95% 區間（03） |
