"""
evaluate_hard_subset.py
=======================
承接 evaluate_all_methods.py 的發現：當資料量夠大時，純距離法(ED)在「全體平均」上
已逼近機器學習法。本程式進一步檢驗一個假設：
    機器學習法的優勢，主要展現在『距離特徵失效的困難區域』，而非全域平均。

作法：
  - 所有方法照常用『全部資料』做 5-fold 交叉驗證，取得每一筆的跨折(OOF)預測。
  - 但『評估』時分成兩組看：全體 vs 困難子集。
  - 困難子集 = ED 距離落在「正負樣本重疊的模糊地帶」的配對
    (此區域光靠距離分不開，最需要方向等其他特徵救援)。

用法：
    python evaluate_hard_subset.py  D:\CARLA_Experiments\reid_features_merged.csv
    # 或手動指定模糊區間 (公尺)：
    python evaluate_hard_subset.py  ...\reid_features_merged.csv  2.5  5.0
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
    thrs = np.unique(scores_oriented)
    best_thr, best_f1 = thrs[0], -1.0
    for t in thrs:
        pred = (scores_oriented >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return best_thr


def oof_threshold(oriented, y, splits):
    pred = np.zeros(len(y), dtype=int)
    for tr, va in splits:
        thr = best_threshold(oriented[tr], y[tr])
        pred[va] = (oriented[va] >= thr).astype(int)
    return pred


def metrics_on(mask, y, pred, score):
    yy, pp, ss = y[mask], pred[mask], score[mask]
    if len(yy) == 0:
        return (np.nan,) * 5
    auc = roc_auc_score(yy, ss) if len(np.unique(yy)) > 1 else np.nan
    return (accuracy_score(yy, pp),
            precision_score(yy, pp, zero_division=0),
            recall_score(yy, pp, zero_division=0),
            f1_score(yy, pp, zero_division=0),
            auc)


def main(csv_path, band=None):
    print("=" * 78)
    print("困難子集分析：ML 在『距離分不開的區域』是否更強")
    print(f"資料表: {csv_path}")
    print("=" * 78)

    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    need = ML_FEATURES + [c for c, _ in TRADITIONAL.values()] + ["label"]
    df = df.dropna(subset=need).reset_index(drop=True)
    y = df["label"].astype(int).values
    X_ml = df[ML_FEATURES].values
    ed = df["ed_dist"].values.astype(float)

    # ---- 決定「模糊地帶」的 ED 距離區間 ----
    if band is not None:
        LOW, HIGH = band
        how = "手動指定"
    else:
        pos_ed, neg_ed = ed[y == 1], ed[y == 0]
        LOW = float(np.percentile(pos_ed, 75))   # 多數同一人的距離在此以下
        HIGH = float(np.percentile(neg_ed, 25))  # 多數不同人的距離在此以上
        how = "自動(正樣本75百分位 ~ 負樣本25百分位)"
        if LOW >= HIGH:  # 幾乎不重疊時退回用中位數
            LOW, HIGH = float(np.median(pos_ed)), float(np.median(neg_ed))
            how = "自動(改用中位數，因兩類重疊很小)"

    hard = (ed >= LOW) & (ed <= HIGH)
    print(f"模糊地帶 (ED 距離): {LOW:.2f} ~ {HIGH:.2f} 公尺  [{how}]")
    print(f"全體樣本: {len(y)} 對 (正 {int((y==1).sum())} / 負 {int((y==0).sum())})")
    print(f"困難子集: {int(hard.sum())} 對 "
          f"(正 {int((y[hard]==1).sum())} / 負 {int((y[hard]==0).sum())})\n")

    if hard.sum() < 10 or len(np.unique(y[hard])) < 2:
        print("⚠️ 困難子集太小或只有單一類別，無法可靠比較。可手動指定較寬的區間再試。")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_ml, y))

    rows = []
    # 傳統方法
    for disp, (col, kind) in TRADITIONAL.items():
        s = df[col].values.astype(float)
        oriented = s if kind == "similarity" else -s
        pred = oof_threshold(oriented, y, splits)
        f1_all = metrics_on(np.ones(len(y), bool), y, pred, oriented)[3]
        acc_h, prec_h, rec_h, f1_h, auc_h = metrics_on(hard, y, pred, oriented)
        rows.append({"method": disp, "type": "傳統", "f1_all": f1_all,
                     "f1_hard": f1_h, "acc_hard": acc_h, "auc_hard": auc_h})

    # 機器學習法
    for name, pipe in build_ml_models().items():
        pred = cross_val_predict(pipe, X_ml, y, cv=splits, method="predict")
        proba = cross_val_predict(pipe, X_ml, y, cv=splits, method="predict_proba")[:, 1]
        f1_all = metrics_on(np.ones(len(y), bool), y, pred, proba)[3]
        acc_h, prec_h, rec_h, f1_h, auc_h = metrics_on(hard, y, pred, proba)
        rows.append({"method": name, "type": "ML", "f1_all": f1_all,
                     "f1_hard": f1_h, "acc_hard": acc_h, "auc_hard": auc_h})

    res = pd.DataFrame(rows).sort_values("f1_hard", ascending=False).reset_index(drop=True)

    print("比較表 (依『困難子集 F1』排序)：")
    print("-" * 78)
    print(f"{'方法':<18}{'類別':<6}{'全體 F1':>10}{'困難 F1':>10}{'困難 Acc':>10}{'困難 AUC':>10}")
    for _, r in res.iterrows():
        print(f"{r['method']:<18}{r['type']:<6}{r['f1_all']:>10.3f}"
              f"{r['f1_hard']:>10.3f}{r['acc_hard']:>10.3f}{r['auc_hard']:>10.3f}")
    print("-" * 78)

    best_ml = res[res["type"] == "ML"].iloc[0]
    best_trad = res[res["type"] == "傳統"].iloc[0]
    gap_all = best_ml["f1_all"] - best_trad["f1_all"]
    gap_hard = best_ml["f1_hard"] - best_trad["f1_hard"]
    print(f"最佳 ML: {best_ml['method']}   最佳傳統: {best_trad['method']}")
    print(f"  全體平均 F1 差距: {gap_all:+.3f}")
    print(f"  困難子集 F1 差距: {gap_hard:+.3f}   ← 若明顯為正，代表 ML 在困難區域勝出")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    res.to_csv(os.path.join(out_dir, "hard_subset_comparison.csv"),
               index=False, encoding="utf-8-sig")

    # ---- 長條圖：全體 vs 困難子集 (每個方法兩根) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(res))
        w = 0.38
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w/2, res["f1_all"], w, label="All samples", color="#B0B0B0")
        colors = ["#4C78A8" if t == "ML" else "#E45756" for t in res["type"]]
        ax.bar(x + w/2, res["f1_hard"], w, label="Hard subset", color=colors)
        ax.set_xticks(x); ax.set_xticklabels(res["method"], rotation=20)
        ax.set_ylabel("F1"); ax.set_ylim(0, 1)
        ax.set_title("F1 on all samples (gray) vs hard subset "
                     "(blue=ML, red=Traditional)")
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(out_dir, "hard_subset_f1.png")
        fig.savefig(out_png, dpi=120)
        print(f"\n✅ 比較表已存: {os.path.join(out_dir, 'hard_subset_comparison.csv')}")
        print(f"✅ 長條圖已存: {out_png}")
    except ImportError:
        print(f"\n✅ 比較表已存: {os.path.join(out_dir, 'hard_subset_comparison.csv')}")
        print("⚠️ 未安裝 matplotlib，略過長條圖。")
    print("=" * 78)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    band = None
    if len(sys.argv) >= 4:
        band = (float(sys.argv[2]), float(sys.argv[3]))
    main(csv_path, band)
