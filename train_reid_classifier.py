"""
train_reid_classifier.py
========================
Step3：把 5 維軌跡特徵餵給二元分類模型，學習判斷「兩條軌跡是否為同一行人」。
對應計畫書 p_v = Sim_ML(T_v^overlap, T_u^overlap)。

會做的事：
  1. 讀 reid_features_merged.csv (merge_inspect 產生的合併表)。
  2. 用 f1~f5 當輸入、label 當答案。
  3. 訓練 SVM / Random Forest / XGBoost / MLP 四個模型。
  4. 用 5-fold 交叉驗證 + 獨立測試集雙重評估 (accuracy / precision / recall / F1 / AUC)。
  5. 算特徵重要度 (permutation importance)。
  6. 把每個訓練好的模型 (含標準化) 存成 .joblib，供之後推論。

用法：
    python train_reid_classifier.py  D:\CARLA_Experiments\reid_features_merged.csv

需要套件：pandas numpy scikit-learn joblib xgboost
    pip install pandas numpy scikit-learn joblib xgboost
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
            'f4_dir_consistency', 'f5_acc_consistency']
RANDOM_STATE = 42


def build_models():
    """每個模型都包成 Pipeline(標準化 + 分類器)，SVM/MLP 需要標準化，樹模型加了也無害；
       包成 pipeline 可避免交叉驗證時的資料洩漏，也方便整包存檔直接推論。"""
    models = {
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                                     probability=True, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("scaler", StandardScaler()),
                                  ("clf", RandomForestClassifier(n_estimators=300,
                                                                 random_state=RANDOM_STATE))]),
        "MLP": Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                               random_state=RANDOM_STATE))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("scaler", StandardScaler()),
                                      ("clf", XGBClassifier(n_estimators=300, max_depth=3,
                                                            learning_rate=0.1, subsample=0.9,
                                                            eval_metric="logloss",
                                                            random_state=RANDOM_STATE))])
    return models


def main(csv_path):
    print("=" * 70)
    print("Step3：訓練跨車行人重識別分類器")
    print(f"資料表: {csv_path}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["label"])

    X = df[FEATURES].values
    y = df["label"].astype(int).values

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"樣本: {len(y)} 對 (正 {n_pos} / 負 {n_neg})")
    if not HAS_XGB:
        print("⚠️ 未安裝 xgboost，將略過 XGBoost (pip install xgboost 後可加入)。")
    print()

    # 切一個獨立測試集 (stratify 保持正負比例)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    results = []
    fitted = {}
    for name, pipe in models.items():
        # 交叉驗證 (在全部資料上，看 F1 的平均與波動)
        cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
        # 用訓練集 fit、測試集評估
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        results.append({
            "model": name,
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "test_acc": accuracy_score(y_te, pred),
            "test_prec": precision_score(y_te, pred, zero_division=0),
            "test_recall": recall_score(y_te, pred, zero_division=0),
            "test_f1": f1_score(y_te, pred, zero_division=0),
            "test_auc": roc_auc_score(y_te, proba),
        })

    res_df = pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)

    # ---- 印比較表 ----
    print("模型比較 (依交叉驗證 F1 由高到低)：")
    print("-" * 70)
    print(f"{'模型':<14}{'CV_F1(平均±標準差)':<22}{'測試Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}")
    for _, r in res_df.iterrows():
        print(f"{r['model']:<14}"
              f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}      "
              f"{r['test_acc']:>7.3f}{r['test_prec']:>8.3f}{r['test_recall']:>8.3f}"
              f"{r['test_f1']:>8.3f}{r['test_auc']:>8.3f}")
    print("-" * 70)

    best_name = res_df.iloc[0]["model"]
    print(f"🏆 交叉驗證表現最佳: {best_name}")

    # ---- 最佳模型的混淆矩陣 ----
    best_pipe = fitted[best_name]
    cm = confusion_matrix(y_te, best_pipe.predict(X_te))
    print(f"\n{best_name} 在測試集的混淆矩陣：")
    print(f"                預測:不同人  預測:同一人")
    print(f"  實際:不同人      {cm[0,0]:>6}      {cm[0,1]:>6}")
    print(f"  實際:同一人      {cm[1,0]:>6}      {cm[1,1]:>6}")

    # ---- 特徵重要度 (permutation importance，對任何模型都適用) ----
    print(f"\n{best_name} 的特徵重要度 (permutation importance，越大越關鍵)：")
    imp = permutation_importance(best_pipe, X_te, y_te, scoring="f1",
                                 n_repeats=30, random_state=RANDOM_STATE)
    order = np.argsort(imp.importances_mean)[::-1]
    for idx in order:
        print(f"  {FEATURES[idx]:<22}{imp.importances_mean[idx]:.4f} ± {imp.importances_std[idx]:.4f}")

    # ---- 存模型 ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "reid_models")
    os.makedirs(out_dir, exist_ok=True)
    for name, pipe in fitted.items():
        joblib.dump({"pipeline": pipe, "features": FEATURES,
                     "label_map": {1: "same_person", 0: "different_person"}},
                    os.path.join(out_dir, f"reid_model_{name}.joblib"))
    res_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "-" * 70)
    print(f"✅ 四個模型已存於: {out_dir}  (reid_model_*.joblib)")
    print(f"✅ 比較表已存: {os.path.join(out_dir, 'model_comparison.csv')}")
    print("推論時載入方式：")
    print("    d = joblib.load('reid_model_XGBoost.joblib')")
    print("    prob = d['pipeline'].predict_proba([[f1,f2,f3,f4,f5]])[0,1]  # = p_v，同一人的機率")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
