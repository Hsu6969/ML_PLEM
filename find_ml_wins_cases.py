"""
find_ml_wins_cases.py
=====================
找出「傳統法(ED)判錯、但機器學習法(RandomForest)判對」的配對，用來佐證
RQ1 的論點：ML 靠融合多個特徵，補足了單一距離法(ED)的盲點。

作法：用與 evaluate 相同的 5 折交叉驗證，取得每一筆的『跨折(out-of-fold)預測』，
再比對 ED 與 ML 的預測誰對誰錯。特別標出最有說服力的一類：
   不同人(label=0) + ED 誤判成同一人(距離太近) + ML 正確判為不同人(靠方向 f4)。
並畫出 ED 距離 vs 方向一致性 的散布圖，把這些案例圈出來，可直接放進論文。

用法：
    python find_ml_wins_cases.py  D:\CARLA_Experiments\reid_features_merged.csv
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ML_FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
               'f4_dir_consistency', 'f5_acc_consistency']
RANDOM_STATE = 42


def best_threshold(scores_oriented, y):
    thrs = np.unique(scores_oriented)
    best_thr, best_f1 = thrs[0], -1.0
    for t in thrs:
        pred = (scores_oriented >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return best_thr


def oof_threshold_pred(oriented, y, splits):
    """傳統分數的跨折預測：每折在訓練部分找門檻、套到驗證部分。"""
    pred = np.zeros(len(y), dtype=int)
    for tr, va in splits:
        thr = best_threshold(oriented[tr], y[tr])
        pred[va] = (oriented[va] >= thr).astype(int)
    return pred


def main(csv_path):
    print("=" * 72)
    print("找出『傳統ED判錯、ML判對』的佐證案例")
    print(f"資料表: {csv_path}")
    print("=" * 72)

    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ML_FEATURES + ["ed_dist", "label"]).reset_index(drop=True)
    y = df["label"].astype(int).values
    X_ml = df[ML_FEATURES].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_ml, y))

    # ML(RandomForest) 的跨折預測
    rf = Pipeline([("s", StandardScaler()),
                   ("c", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])
    ml_pred = cross_val_predict(rf, X_ml, y, cv=splits, method="predict")

    # ED 的跨折預測 (距離 -> 越大越像用 -ed_dist)
    ed_oriented = -df["ed_dist"].values.astype(float)
    ed_pred = oof_threshold_pred(ed_oriented, y, splits)

    # 也把 DTW / LCSS 算出來，方便標記「三個傳統法全錯」
    dtw_pred = oof_threshold_pred(-df["dtw_dist"].values.astype(float), y, splits)
    lcss_pred = oof_threshold_pred(df["lcss_sim"].values.astype(float), y, splits)

    ml_ok = (ml_pred == y)
    ed_wrong = (ed_pred != y)
    all_trad_wrong = (ed_pred != y) & (dtw_pred != y) & (lcss_pred != y)

    # ML 對、ED 錯
    ml_wins = ml_ok & ed_wrong
    # 最有說服力的一類：不同人 + ED 誤判成同一人 + ML 判對
    star = (y == 0) & (ed_pred == 1) & (ml_pred == 0)

    print(f"總樣本: {len(y)} 對")
    print(f"ML 對、ED 錯 的案例: {ml_wins.sum()} 筆")
    print(f"  其中『不同人 + ED誤判成同人(距離近) + ML靠方向判對』: {star.sum()} 筆  ← 最佳佐證")
    print(f"  ML 對、且三個傳統法(ED/DTW/LCSS)全錯: {(ml_ok & all_trad_wrong).sum()} 筆\n")

    cols = ["pair_id", "source", "label", "n_points",
            "ed_dist", "dtw_dist", "lcss_sim",
            "f3_mean_dist", "f4_dir_consistency"]
    cols = [c for c in cols if c in df.columns]

    show = df[star].copy()
    show["ed_pred"] = 1
    show["ml_pred"] = 0
    # 按「距離最近(最容易騙過ED) + 方向最相反(f4最負)」排序，越前面越戲劇性
    show = show.sort_values(["ed_dist", "f4_dir_consistency"]).reset_index(drop=True)

    if len(show) > 0:
        print("最佳佐證案例 (依 ED距離小、方向相反 排序，取前 10 筆)：")
        print(show[cols + ["ed_pred", "ml_pred"]].head(10).round(4).to_string(index=False))
    else:
        print("這批資料裡沒有『不同人被ED誤判、ML救回』的案例 (可等資料量更多再找)。")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    # 存所有 ML-win 案例
    win_df = df[ml_wins].copy()
    win_df["ed_pred"] = ed_pred[ml_wins]
    win_df["ml_pred"] = ml_pred[ml_wins]
    win_df.to_csv(os.path.join(out_dir, "ml_wins_cases.csv"), index=False, encoding="utf-8-sig")

    # ---- 散布圖：ED距離 vs 方向一致性，圈出佐證案例 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        m_same = (y == 1)
        m_diff = (y == 0)
        ax.scatter(df.loc[m_diff, "ed_dist"], df.loc[m_diff, "f4_dir_consistency"],
                   c="#E45756", label="different person (0)", alpha=0.6, s=30)
        ax.scatter(df.loc[m_same, "ed_dist"], df.loc[m_same, "f4_dir_consistency"],
                   c="#4C78A8", label="same person (1)", alpha=0.6, s=30)
        # 圈出 star 案例
        if star.sum() > 0:
            ax.scatter(df.loc[star, "ed_dist"], df.loc[star, "f4_dir_consistency"],
                       facecolors="none", edgecolors="black", s=140, linewidths=1.6,
                       label="ED wrong, ML correct")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("ED distance (ed_dist, meters)  — smaller = closer")
        ax.set_ylabel("Direction consistency (f4)  — higher = same direction")
        ax.set_title("Why ML beats ED: close-distance different-person pairs\nare separated by direction (f4)")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(out_dir, "ml_vs_ed_cases.png")
        fig.savefig(out_png, dpi=120)
        print(f"\n✅ 佐證案例清單已存: {os.path.join(out_dir, 'ml_wins_cases.csv')}")
        print(f"✅ 散布圖已存: {out_png}")
    except ImportError:
        print(f"\n✅ 佐證案例清單已存: {os.path.join(out_dir, 'ml_wins_cases.csv')}")
        print("⚠️ 未安裝 matplotlib，略過散布圖。")
    print("=" * 72)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
