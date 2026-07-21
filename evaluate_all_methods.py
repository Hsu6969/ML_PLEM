"""
evaluate_all_methods.py
=======================
在「同一份資料、同一組交叉驗證切分」下，公平比較 RQ1 的三類方法：
  - 傳統軌跡相似度 (門檻式)：ED、DTW、LCSS
  - 機器學習法 (5 維特徵)：SVM、RandomForest、XGBoost、MLP

作法：用 StratifiedKFold(5) 產生固定的 5 組切分，所有方法都用「完全相同的訓練/驗證折」。
  - 傳統方法：在每折的訓練部分找『最佳門檻』(最大化 F1)，套用到驗證部分。
  - ML 方法：在每折訓練部分 fit，於驗證部分預測。
最後輸出一張「傳統 vs ML」比較表 (計畫書 RQ1 核心結果)。

用法：
    python evaluate_all_methods.py  D:\CARLA_Experiments\reid_features_merged.csv

需要套件：pandas numpy scikit-learn xgboost (可選) matplotlib (可選)
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ML_FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
               'f4_dir_consistency', 'f5_acc_consistency']

# 傳統方法：欄位 + 方向 (distance=越小越像 / similarity=越大越像)
TRADITIONAL = {
    "ED (ed_dist)":   ("ed_dist", "distance"),
    "DTW (dtw_dist)": ("dtw_dist", "distance"),
    "LCSS (lcss_sim)": ("lcss_sim", "similarity"),
}
RANDOM_STATE = 42


def build_ml_models():
    models = {
        "SVM": Pipeline([("s", StandardScaler()),
                         ("c", SVC(kernel="rbf", C=1.0, gamma="scale",
                                   probability=True, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("s", StandardScaler()),
                                  ("c", RandomForestClassifier(n_estimators=300,
                                                               random_state=RANDOM_STATE))]),
        "MLP": Pipeline([("s", StandardScaler()),
                         ("c", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                             random_state=RANDOM_STATE))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("s", StandardScaler()),
                                      ("c", XGBClassifier(n_estimators=300, max_depth=3,
                                                          learning_rate=0.1, subsample=0.9,
                                                          eval_metric="logloss",
                                                          random_state=RANDOM_STATE))])
    return models


def best_threshold(scores_oriented, y):
    """在『越大越像』的分數上，找最大化 F1 的門檻 (預測 1 if score >= thr)。"""
    thrs = np.unique(scores_oriented)
    best_thr, best_f1 = thrs[0], -1.0
    for t in thrs:
        pred = (scores_oriented >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return best_thr


def metrics(y_true, pred, score_oriented):
    auc = np.nan
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, score_oriented)
    return (accuracy_score(y_true, pred),
            precision_score(y_true, pred, zero_division=0),
            recall_score(y_true, pred, zero_division=0),
            f1_score(y_true, pred, zero_division=0),
            auc)


def agg(rows):
    arr = np.array(rows, dtype=float)
    return arr.mean(axis=0), np.nanmean(arr, axis=0), arr.std(axis=0)


def main(csv_path):
    print("=" * 74)
    print("RQ1 三類方法公平比較 (同一份資料、同一組交叉驗證切分)")
    print(f"資料表: {csv_path}")
    print("=" * 74)

    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    need = ML_FEATURES + [c for c, _ in TRADITIONAL.values()] + ["label"]
    df = df.dropna(subset=need)
    y = df["label"].astype(int).values
    X_ml = df[ML_FEATURES].values

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"樣本: {len(y)} 對 (正 {n_pos} / 負 {n_neg})")
    if not HAS_XGB:
        print("⚠️ 未安裝 xgboost，略過 XGBoost。")

    # 固定的 5 折切分：所有方法共用，確保完全公平
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_ml, y))

    results = []

    # ---- 傳統方法 ----
    for disp, (col, kind) in TRADITIONAL.items():
        s_all = df[col].values.astype(float)
        oriented = s_all if kind == "similarity" else -s_all   # 統一成「越大越像」
        fold_rows = []
        for tr, va in splits:
            thr = best_threshold(oriented[tr], y[tr])
            pred = (oriented[va] >= thr).astype(int)
            fold_rows.append(metrics(y[va], pred, oriented[va]))
        mean, mean_nan, std = agg(fold_rows)
        results.append({"method": disp, "type": "傳統",
                        "acc": mean[0], "prec": mean[1], "recall": mean[2],
                        "f1": mean[3], "f1_std": std[3], "auc": mean_nan[4]})

    # ---- 機器學習法 ----
    for name, pipe in build_ml_models().items():
        fold_rows = []
        for tr, va in splits:
            pipe.fit(X_ml[tr], y[tr])
            pred = pipe.predict(X_ml[va])
            proba = pipe.predict_proba(X_ml[va])[:, 1]
            fold_rows.append(metrics(y[va], pred, proba))
        mean, mean_nan, std = agg(fold_rows)
        results.append({"method": name, "type": "ML",
                        "acc": mean[0], "prec": mean[1], "recall": mean[2],
                        "f1": mean[3], "f1_std": std[3], "auc": mean_nan[4]})

    res_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    # ---- 印比較表 ----
    print("\n比較表 (5-fold 交叉驗證平均，依 F1 由高到低)：")
    print("-" * 74)
    print(f"{'方法':<18}{'類別':<6}{'Acc':>7}{'Prec':>8}{'Recall':>8}{'F1(±std)':>16}{'AUC':>8}")
    for _, r in res_df.iterrows():
        print(f"{r['method']:<18}{r['type']:<6}{r['acc']:>7.3f}{r['prec']:>8.3f}"
              f"{r['recall']:>8.3f}   {r['f1']:.3f}±{r['f1_std']:.3f}{r['auc']:>8.3f}")
    print("-" * 74)

    best_ml = res_df[res_df["type"] == "ML"].iloc[0]
    best_trad = res_df[res_df["type"] == "傳統"].iloc[0]
    gap = best_ml["f1"] - best_trad["f1"]
    print(f"🏆 最佳 ML: {best_ml['method']} (F1={best_ml['f1']:.3f})")
    print(f"   最佳傳統: {best_trad['method']} (F1={best_trad['f1']:.3f})")
    print(f"   ML 比傳統法 F1 高出約 {gap:.3f} —— 這就是 RQ1 要證明的貢獻。")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    out_csv = os.path.join(out_dir, "method_comparison_all.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---- 長條圖 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#4C78A8" if t == "ML" else "#E45756" for t in res_df["type"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(res_df["method"], res_df["f1"], yerr=res_df["f1_std"],
               color=colors, capsize=4)
        ax.set_ylabel("F1 (5-fold CV mean)")
        ax.set_title("RQ1: Traditional (red) vs Machine-Learning (blue) methods")
        ax.set_ylim(0, 1)
        for i, v in enumerate(res_df["f1"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        plt.xticks(rotation=20)
        fig.tight_layout()
        out_png = os.path.join(out_dir, "method_f1_comparison.png")
        fig.savefig(out_png, dpi=120)
        print(f"\n✅ 比較表已存: {out_csv}")
        print(f"✅ 長條圖已存: {out_png}")
    except ImportError:
        print(f"\n✅ 比較表已存: {out_csv}")
        print("⚠️ 未安裝 matplotlib，略過長條圖。")
    print("=" * 74)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
