"""
check_dataset_health.py
======================
對 reid_features_merged.csv 做自動化「資料健檢」，確認資料集沒有明顯問題。
檢查項目：
  1. 缺漏 / 無限值
  2. 數值合理範圍 (距離不可為負、方向一致性須在 -1~1、LCSS 在 0~1、n_points>=5 等)
  3. 特徵方向與區別力 (正樣本是否如預期比負樣本小/大，並計算 Cohen's d)
  4. 正負樣本比例是否合理
  5. 每一輪(來源)是否正常貢獻樣本
  6. 重複列檢查
  7. f3 與 ED 的相關性 (解釋為何三類方法分數相近)

輸出：主控台報告 + dataset_health_report.txt

用法：
    python check_dataset_health.py  D:\CARLA_Experiments\reid_features_merged.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# 特徵方向：正樣本(同一人)預期比負樣本『小』或『大』
DIRECTION = {
    "f1_len_diff": "smaller", "f2_speed_diff": "smaller", "f3_mean_dist": "smaller",
    "f4_dir_consistency": "larger", "f5_acc_consistency": "smaller",
    "ed_dist": "smaller", "dtw_dist": "smaller", "lcss_sim": "larger",
}
# 合理值範圍
NONNEG = ["f1_len_diff", "f2_speed_diff", "f3_mean_dist", "f5_acc_consistency",
          "ed_dist", "dtw_dist", "n_points"]
RANGE_11 = ["f4_dir_consistency"]      # [-1, 1]
RANGE_01 = ["lcss_sim"]                # [0, 1]
MIN_LEN = 5


def cohens_d(pos, neg):
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = pos.std(ddof=1), neg.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (pos.mean() - neg.mean()) / pooled


def main(csv_path):
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("資料集健檢報告  (reid_features_merged.csv)")
    out(f"檔案: {csv_path}")
    out("=" * 70)

    df = pd.read_csv(csv_path)
    warnings = 0

    # ---- 0. 基本資訊 ----
    out(f"總列數: {len(df)}   欄位數: {df.shape[1]}")
    out(f"欄位: {', '.join(df.columns)}\n")

    # ---- 1. 缺漏 / 無限值 ----
    out("[1] 缺漏值 / 無限值檢查")
    n_nan = df.isna().sum().sum()
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if n_nan == 0 and n_inf == 0:
        out("   ✓ 沒有缺漏值、沒有無限值")
    else:
        out(f"   ⚠ 缺漏值 {n_nan} 個、無限值 {n_inf} 個")
        warnings += 1
    out()

    # ---- 2. 數值範圍 ----
    out("[2] 數值合理範圍檢查")
    for c in NONNEG:
        if c in df.columns:
            bad = (df[c] < 0).sum()
            if bad == 0:
                out(f"   ✓ {c}: 皆 >= 0")
            else:
                out(f"   ⚠ {c}: 有 {bad} 筆為負值 (不合理)")
                warnings += 1
    for c in RANGE_11:
        if c in df.columns:
            bad = ((df[c] < -1.0001) | (df[c] > 1.0001)).sum()
            out(f"   {'✓' if bad == 0 else '⚠'} {c}: 落在 [-1,1] "
                f"{'正常' if bad == 0 else f'有 {bad} 筆超出'}")
            warnings += (bad != 0)
    for c in RANGE_01:
        if c in df.columns:
            bad = ((df[c] < -0.0001) | (df[c] > 1.0001)).sum()
            out(f"   {'✓' if bad == 0 else '⚠'} {c}: 落在 [0,1] "
                f"{'正常' if bad == 0 else f'有 {bad} 筆超出'}")
            warnings += (bad != 0)
    if "n_points" in df.columns:
        bad = (df["n_points"] < MIN_LEN).sum()
        out(f"   {'✓' if bad == 0 else '⚠'} n_points: 皆 >= {MIN_LEN} "
            f"{'正常' if bad == 0 else f'有 {bad} 筆過短'}")
        warnings += (bad != 0)
    if "label" in df.columns:
        bad = (~df["label"].isin([0, 1])).sum()
        out(f"   {'✓' if bad == 0 else '⚠'} label: 只有 0/1 "
            f"{'正常' if bad == 0 else f'有 {bad} 筆異常'}")
        warnings += (bad != 0)
    out()

    # ---- 3. 特徵方向與區別力 ----
    out("[3] 特徵方向與區別力 (Cohen's d，正樣本應如預期比負樣本小/大)")
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    out(f"   {'特徵':<20}{'正均值':>10}{'負均值':>10}{'|d|':>7}  判讀")
    for c, want in DIRECTION.items():
        if c not in df.columns:
            continue
        mp, mn = pos[c].mean(), neg[c].mean()
        d = cohens_d(pos[c], neg[c])
        pos_smaller = mp < mn
        ok = (pos_smaller == (want == "smaller"))
        tag = "✓方向符合" if ok else "⚠方向相反!"
        if not ok:
            warnings += 1
        dv = f"{abs(d):.2f}" if not np.isnan(d) else " NA"
        out(f"   {c:<20}{mp:>10.4f}{mn:>10.4f}{dv:>7}  {tag}")
    out()

    # ---- 4. 正負樣本比例 ----
    out("[4] 正負樣本比例")
    n_pos, n_neg = len(pos), len(neg)
    out(f"   正樣本 {n_pos} / 負樣本 {n_neg}  (比例 1 : {n_neg / max(n_pos,1):.2f})")
    if n_pos == 0 or n_neg == 0:
        out("   ⚠ 只有單一類別，無法訓練")
        warnings += 1
    elif n_pos > n_neg:
        out("   ⚠ 正樣本多於負樣本，與『每輪3人→3正6負』的預期不符，建議檢查配對邏輯")
        warnings += 1
    else:
        out("   ✓ 比例合理 (負樣本較多，符合多對配對的預期)")
    out()

    # ---- 5. 每一輪(來源)貢獻 ----
    if "source" in df.columns:
        out("[5] 每一輪來源檢查")
        g = df.groupby("source")["label"].agg(["count", "sum"])
        g.columns = ["對數", "正樣本"]
        g["負樣本"] = g["對數"] - g["正樣本"]
        n_rounds = len(g)
        no_pos = (g["正樣本"] == 0).sum()
        out(f"   共 {n_rounds} 輪；平均每輪 {g['對數'].mean():.1f} 對")
        out(f"   完全沒有正樣本的輪次: {no_pos} 輪 "
            f"{'(正常，該輪行人沒被兩車同時看到)' if no_pos>0 else ''}")
        out("   ✓ 來源分佈已統計 (詳見下方每輪明細若需要)")
        out()

    # ---- 6. 重複列 ----
    out("[6] 重複列檢查")
    key = [c for c in ["pair_id", "source"] if c in df.columns]
    if key:
        dup = df.duplicated(subset=key).sum()
        out(f"   {'✓' if dup == 0 else '⚠'} 以 {key} 為鍵，重複列: {dup} 筆")
        warnings += (dup != 0)
    else:
        out("   (無 pair_id/source 欄位，略過)")
    out()

    # ---- 7. f3 與 ED 的相關性 ----
    if "f3_mean_dist" in df.columns and "ed_dist" in df.columns:
        out("[7] f3_mean_dist 與 ed_dist 相關性")
        corr = df["f3_mean_dist"].corr(df["ed_dist"])
        out(f"   相關係數 = {corr:.4f}")
        out("   (接近 1.0 屬正常：f3 與 ED 本質是同一個量，")
        out("    這解釋了為何『傳統距離法』與『用 f3 的 ML』分數會很接近。)")
        out()

    # ---- 總結 ----
    out("=" * 70)
    if warnings == 0:
        out("健檢結論：✅ 未發現異常，資料集健康，可放心用於實驗與論文。")
    else:
        out(f"健檢結論：⚠ 發現 {warnings} 項需注意的地方，請對照上方標記 ⚠ 的項目。")
    out("=" * 70)

    # 存報告
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    report_path = os.path.join(out_dir, "dataset_health_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 報告已存: {report_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
