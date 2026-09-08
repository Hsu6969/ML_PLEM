"""
evaluate_all_methods.py  (三類方法同框版)
========================================
依計畫書 RQ1，在「同一份資料、同一組交叉驗證切分」下公平比較三類方法：

  第(1)類 門檻值法 (Threshold-based)：DistOnly1、DistOnly2、DistAngle
  第(2)類 傳統軌跡相似度：ED、DTW、LCSS
  第(3)類 機器學習法：SVM、RandomForest、XGBoost、MLP

門檻/上下界一律「從每折的訓練部分自動學出」，再套到驗證部分，確保公平。

用法：
    python evaluate_all_methods_3groups.py  D:\CARLA_Experiments\reid_features_merged.csv
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
RANDOM_STATE = 42


# ============================================================
# 機器學習模型 (第 3 類)
# ============================================================
def build_ml_models():
    """
    ★ 參數已依 tune_ml_hyperparams.py 的 Grid Search 結果更新為各模型的最佳參數：
        SVM:          C=10, gamma=0.1        (F1: 0.886 -> 0.905，+0.018)
        RandomForest: max_depth=5, min_samples_leaf=4, n_estimators=500  (+0.003)
        MLP:          hidden_layer_sizes=(32,), alpha=0.0001            (+0.009)
        XGBoost:      max_depth=6, learning_rate=0.05, n_estimators=300,
                      subsample=0.8                                     (+0.013)
    """
    models = {
        "SVM": Pipeline([("s", StandardScaler()),
                         ("c", SVC(kernel="rbf", C=10, gamma=0.1,
                                   probability=True, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("s", StandardScaler()),
                                  ("c", RandomForestClassifier(n_estimators=500,
                                                               max_depth=5,
                                                               min_samples_leaf=4,
                                                               random_state=RANDOM_STATE))]),
        "MLP": Pipeline([("s", StandardScaler()),
                         ("c", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.0001,
                                             max_iter=2000, random_state=RANDOM_STATE))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("s", StandardScaler()),
                                      ("c", XGBClassifier(n_estimators=300, max_depth=6,
                                                          learning_rate=0.05, subsample=0.8,
                                                          eval_metric="logloss",
                                                          random_state=RANDOM_STATE))])
    return models


def metrics(y_true, pred, score_oriented):
    auc = np.nan
    if len(np.unique(y_true)) > 1 and score_oriented is not None:
        auc = roc_auc_score(y_true, score_oriented)
    return (accuracy_score(y_true, pred),
            precision_score(y_true, pred, zero_division=0),
            recall_score(y_true, pred, zero_division=0),
            f1_score(y_true, pred, zero_division=0),
            auc)


def agg(rows):
    arr = np.array(rows, dtype=float)
    return arr.mean(axis=0), np.nanmean(arr, axis=0), arr.std(axis=0)


# ============================================================
# 門檻值法 (第 1 類) — 逐點距離序列的相似度機率 P_v
#   輸入：兩軌跡對齊後、每個時間點的「逐點距離」序列 (公尺)
#   計畫書用 dist(l_v, l~u)；本資料集裡「平均逐點距離」= f3_mean_dist，
#   而每一對只有一個 f3 值(已對整條軌跡平均)，因此在此以 f3 當該對的代表距離，
#   方向部分用 f4_dir_consistency。這與計畫書的機率化定義一致(對整條軌跡平均後的分數)。
# ============================================================
def best_threshold_generic(score, y, predict_fn):
    """在候選門檻中，挑使訓練集 F1 最高者。predict_fn(score, thr)->0/1 陣列。"""
    best_thr, best_f1 = None, -1.0
    for t in np.unique(score):
        pred = predict_fn(score, t)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return best_thr


def dist_only1_fit_predict(dist_tr, y_tr, dist_va):
    """DistOnly1：dist < d_th 判為同一人(1)。門檻由訓練集學。"""
    thr = best_threshold_generic(dist_tr, y_tr, lambda s, t: (s < t).astype(int))
    return (dist_va < thr).astype(int), -dist_va      # score: 越大越像 -> -dist


def dist_only2_fit_predict(dist_tr, y_tr, dist_va):
    """DistOnly2：以 d_low、d_high 做線性遞減的連續相似度 P，再取 0.5 為判定。
       d_low、d_high 由訓練集正負樣本距離的百分位自動決定。"""
    d_low = np.percentile(dist_tr[y_tr == 1], 75)      # 同一人距離多在此以下
    d_high = np.percentile(dist_tr[y_tr == 0], 25)     # 不同人距離多在此以上
    if d_high <= d_low:
        d_low, d_high = np.median(dist_tr[y_tr == 1]), np.median(dist_tr[y_tr == 0])
    if d_high <= d_low:
        d_high = d_low + 1e-6

    def prob(d):
        p = np.clip((d_high - d) / (d_high - d_low), 0.0, 1.0)  # d<d_low->1, d>d_high->0
        return p

    p_va = prob(dist_va)
    return (p_va >= 0.5).astype(int), p_va             # score: 機率本身(越大越像)


def dist_angle_fit_predict(dist_tr, y_tr, dir_tr, dist_va, dir_va):
    """DistAngle：距離判定 × 方向分數。方向分數 D=(1+cossim)/2 ∈[0,1]。
       綜合分數 = P_dist(線性遞減) × D，再由訓練集學一個門檻。"""
    d_low = np.percentile(dist_tr[y_tr == 1], 75)
    d_high = np.percentile(dist_tr[y_tr == 0], 25)
    if d_high <= d_low:
        d_low, d_high = np.median(dist_tr[y_tr == 1]), np.median(dist_tr[y_tr == 0])
    if d_high <= d_low:
        d_high = d_low + 1e-6

    def combined(dist, direc):
        p_dist = np.clip((d_high - dist) / (d_high - d_low), 0.0, 1.0)
        d_score = (1.0 + direc) / 2.0                  # cossim(-1~1) -> 0~1
        return p_dist * d_score

    s_tr = combined(dist_tr, dir_tr)
    s_va = combined(dist_va, dir_va)
    thr = best_threshold_generic(s_tr, y_tr, lambda s, t: (s >= t).astype(int))
    return (s_va >= thr).astype(int), s_va


def main(csv_path):
    print("=" * 80)
    print("RQ1 三類方法同框比較 (門檻值法 / 傳統相似度 / 機器學習)")
    print(f"資料表: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    need = ML_FEATURES + ["ed_dist", "dtw_dist", "lcss_sim",
                          "f3_mean_dist", "f4_dir_consistency", "label"]
    df = df.dropna(subset=need).reset_index(drop=True)
    y = df["label"].astype(int).values
    X_ml = df[ML_FEATURES].values
    dist = df["f3_mean_dist"].values.astype(float)     # 代表整條軌跡的平均逐點距離
    direc = df["f4_dir_consistency"].values.astype(float)

    print(f"樣本: {len(y)} 對 (正 {int((y==1).sum())} / 負 {int((y==0).sum())})")
    if not HAS_XGB:
        print("⚠️ 未安裝 xgboost，略過 XGBoost。")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_ml, y))
    results = []

    # ---------- 第 1 類：門檻值法 ----------
    for name in ["DistOnly1", "DistOnly2", "DistAngle"]:
        fold_rows = []
        for tr, va in splits:
            if name == "DistOnly1":
                pred, score = dist_only1_fit_predict(dist[tr], y[tr], dist[va])
            elif name == "DistOnly2":
                pred, score = dist_only2_fit_predict(dist[tr], y[tr], dist[va])
            else:
                pred, score = dist_angle_fit_predict(dist[tr], y[tr], direc[tr],
                                                     dist[va], direc[va])
            fold_rows.append(metrics(y[va], pred, score))
        mean, mean_nan, std = agg(fold_rows)
        results.append({"method": name, "type": "門檻值",
                        "acc": mean[0], "prec": mean[1], "recall": mean[2],
                        "f1": mean[3], "f1_std": std[3], "auc": mean_nan[4]})

    # ---------- 第 2 類：傳統相似度 (門檻自動學) ----------
    TRAD = {"ED": ("ed_dist", "distance"),
            "DTW": ("dtw_dist", "distance"),
            "LCSS": ("lcss_sim", "similarity")}
    for disp, (col, kind) in TRAD.items():
        s_all = df[col].values.astype(float)
        oriented = s_all if kind == "similarity" else -s_all
        fold_rows = []
        for tr, va in splits:
            thr = best_threshold_generic(oriented[tr], y[tr],
                                         lambda s, t: (s >= t).astype(int))
            pred = (oriented[va] >= thr).astype(int)
            fold_rows.append(metrics(y[va], pred, oriented[va]))
        mean, mean_nan, std = agg(fold_rows)
        results.append({"method": disp, "type": "傳統",
                        "acc": mean[0], "prec": mean[1], "recall": mean[2],
                        "f1": mean[3], "f1_std": std[3], "auc": mean_nan[4]})

    # ---------- 第 3 類：機器學習 ----------
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

    # ---------- 印表 ----------
    print("\n比較表 (5-fold 交叉驗證平均，依 F1 由高到低)：")
    print("-" * 80)
    print(f"{'方法':<14}{'類別':<8}{'Acc':>7}{'Prec':>8}{'Recall':>8}{'F1(±std)':>16}{'AUC':>8}")
    for _, r in res_df.iterrows():
        print(f"{r['method']:<14}{r['type']:<8}{r['acc']:>7.3f}{r['prec']:>8.3f}"
              f"{r['recall']:>8.3f}   {r['f1']:.3f}±{r['f1_std']:.3f}{r['auc']:>8.3f}")
    print("-" * 80)

    # 各類最佳
    for t in ["門檻值", "傳統", "ML"]:
        sub = res_df[res_df["type"] == t]
        if len(sub) > 0:
            b = sub.iloc[0]
            print(f"  {t:<4} 最佳: {b['method']} (F1={b['f1']:.3f})")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    res_df.to_csv(os.path.join(out_dir, "method_comparison_3groups.csv"),
                  index=False, encoding="utf-8-sig")

    # ---------- 長條圖 (依類別上色) ----------
        # ---------- 長條圖 (依類別上色，標數值，縱軸讓最優方法頂到滿) ----------
        # ---------- 長條圖 (依類別上色，標 F1 與準確率，縱軸讓最優方法頂到滿) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = {"門檻值": "#F58518", "傳統": "#E45756", "ML": "#4C78A8"}
        colors = [cmap[t] for t in res_df["type"]]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(res_df["method"], res_df["f1"], yerr=res_df["f1_std"],
                      color=colors, capsize=4)

        # ★ 縱軸區間：讓最優方法幾乎頂到圖表頂端
        f1_max = res_df["f1"].max()
        f1_min = res_df["f1"].min()
        y_top = f1_max + res_df["f1_std"].max() + 0.02   # 留空間放誤差棒與兩行文字
        y_bottom = max(0, f1_min - 0.15)
        ax.set_ylim(y_bottom, y_top)

        ax.set_ylabel("F1 (5-fold CV mean)")
        ax.set_title("RQ1: Threshold (orange) vs Traditional (red) vs ML (blue)")

        # ★ 每根長條旁標上 F1 (主要指標) 與 Accuracy (準確率)
        for bar, f1v, stdv, accv in zip(bars, res_df["f1"], res_df["f1_std"], res_df["acc"]):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, f1v + stdv + 0.003, f"F1={f1v:.3f}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(x, f1v + stdv + 0.003 + (y_top - y_bottom) * 0.035,
                    f"Acc={accv:.3f}", ha="center", va="bottom", fontsize=8, color="dimgray")

        plt.xticks(rotation=20)
        fig.tight_layout()
        out_png = os.path.join(out_dir, "method_comparison_3groups.png")
        fig.savefig(out_png, dpi=120)
        print(f"\n✅ 比較表已存: {os.path.join(out_dir, 'method_comparison_3groups.csv')}")
        print(f"✅ 長條圖已存: {out_png}")
    except ImportError:
        print(f"\n✅ 比較表已存: {os.path.join(out_dir, 'method_comparison_3groups.csv')}")
        print("⚠️ 未安裝 matplotlib，略過長條圖。")
    print("=" * 80)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
