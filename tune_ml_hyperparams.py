"""
tune_ml_hyperparams.py
=====================
對四個機器學習模型 (SVM / RandomForest / XGBoost / MLP) 做網格搜尋 (Grid Search)，
在 5-fold 交叉驗證下自動試多組超參數，找出各模型的最佳設定。

輸出：
  - 每個模型的「預設參數 F1」vs「調參後最佳 F1」對照，以及最佳參數。
  - 存下調參後的最佳模型 (reid_models_tuned/) 供之後使用。

用途說明：此步驟的主要目的是「證明已系統性尋找最佳參數」(論文嚴謹度)，
         提升幅度通常有限，因為瓶頸多在特徵而非模型能力。

用法：
    python tune_ml_hyperparams.py  D:\CARLA_Experiments\reid_features_merged.csv

需要套件：pandas numpy scikit-learn joblib xgboost(可選)
注意：Grid Search 會試很多組合，可能需要數分鐘。
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ML_FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
               'f4_dir_consistency', 'f5_acc_consistency']
RANDOM_STATE = 42


def make_configs():
    """回傳 {模型名稱: (預設pipeline, 調參pipeline, 參數網格)}。"""
    configs = {}

    # --- SVM (Grid Search 時 probability=False 以加速；只需 predict 算 F1) ---
    svm_default = Pipeline([("s", StandardScaler()),
                            ("c", SVC(kernel="rbf", C=1.0, gamma="scale",
                                      random_state=RANDOM_STATE))])
    svm_grid_pipe = Pipeline([("s", StandardScaler()),
                              ("c", SVC(kernel="rbf", random_state=RANDOM_STATE))])
    svm_grid = {"c__C": [0.1, 1, 10, 100],
                "c__gamma": ["scale", 0.1, 1]}
    configs["SVM"] = (svm_default, svm_grid_pipe, svm_grid)

    # --- RandomForest ---
    rf_default = Pipeline([("s", StandardScaler()),
                           ("c", RandomForestClassifier(n_estimators=300,
                                                        random_state=RANDOM_STATE))])
    rf_grid_pipe = Pipeline([("s", StandardScaler()),
                             ("c", RandomForestClassifier(random_state=RANDOM_STATE))])
    rf_grid = {"c__n_estimators": [200, 300, 500],
               "c__max_depth": [None, 5, 10],
               "c__min_samples_leaf": [1, 2, 4]}
    configs["RandomForest"] = (rf_default, rf_grid_pipe, rf_grid)

    # --- MLP ---
    mlp_default = Pipeline([("s", StandardScaler()),
                            ("c", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                                random_state=RANDOM_STATE))])
    mlp_grid_pipe = Pipeline([("s", StandardScaler()),
                              ("c", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE))])
    mlp_grid = {"c__hidden_layer_sizes": [(16, 8), (32, 16), (32,), (64, 32)],
                "c__alpha": [0.0001, 0.001, 0.01]}
    configs["MLP"] = (mlp_default, mlp_grid_pipe, mlp_grid)

    # --- XGBoost ---
    if HAS_XGB:
        xgb_default = Pipeline([("s", StandardScaler()),
                                ("c", XGBClassifier(n_estimators=300, max_depth=3,
                                                    learning_rate=0.1, subsample=0.9,
                                                    eval_metric="logloss",
                                                    random_state=RANDOM_STATE))])
        xgb_grid_pipe = Pipeline([("s", StandardScaler()),
                                  ("c", XGBClassifier(eval_metric="logloss",
                                                      random_state=RANDOM_STATE))])
        xgb_grid = {"c__n_estimators": [200, 300],
                    "c__max_depth": [3, 4, 6],
                    "c__learning_rate": [0.05, 0.1],
                    "c__subsample": [0.8, 0.9, 1.0]}
        configs["XGBoost"] = (xgb_default, xgb_grid_pipe, xgb_grid)

    return configs


def main(csv_path):
    print("=" * 74)
    print("機器學習模型調參 (Grid Search)")
    print(f"資料表: {csv_path}")
    print("=" * 74)

    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna(subset=ML_FEATURES + ["label"])
    X = df[ML_FEATURES].values
    y = df["label"].astype(int).values
    print(f"樣本: {len(y)} 對 (正 {int((y==1).sum())} / 負 {int((y==0).sum())})")
    if not HAS_XGB:
        print("⚠️ 未安裝 xgboost，略過 XGBoost。")
    print("（Grid Search 進行中，可能需要數分鐘...）\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    configs = make_configs()

    rows = []
    tuned_models = {}
    for name, (default_pipe, grid_pipe, grid) in configs.items():
        # 預設參數 F1
        f1_default = cross_val_score(default_pipe, X, y, cv=cv, scoring="f1").mean()
        # 網格搜尋
        gs = GridSearchCV(grid_pipe, grid, cv=cv, scoring="f1", n_jobs=1)
        gs.fit(X, y)
        f1_tuned = gs.best_score_
        best_params = {k.replace("c__", ""): v for k, v in gs.best_params_.items()}
        tuned_models[name] = gs.best_estimator_

        rows.append({"model": name, "f1_default": f1_default,
                     "f1_tuned": f1_tuned, "improvement": f1_tuned - f1_default,
                     "best_params": str(best_params)})
        print(f"  {name:<14} 預設 F1={f1_default:.4f}  ->  調參後 F1={f1_tuned:.4f}  "
              f"(提升 {f1_tuned - f1_default:+.4f})")
        print(f"      最佳參數: {best_params}")

    res = pd.DataFrame(rows).sort_values("f1_tuned", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 74)
    print("調參結果總表 (依調參後 F1 排序)：")
    print(f"{'模型':<14}{'預設F1':>10}{'調參後F1':>12}{'提升':>10}")
    for _, r in res.iterrows():
        print(f"{r['model']:<14}{r['f1_default']:>10.4f}{r['f1_tuned']:>12.4f}"
              f"{r['improvement']:>+10.4f}")
    print("-" * 74)

    total_gain = res["improvement"].max()
    print(f"最大提升幅度: {total_gain:+.4f}")
    if total_gain < 0.01:
        print("→ 提升有限 (<0.01)，印證瓶頸在特徵而非模型；此表主要作為『已系統性調參』的佐證。")
    else:
        print("→ 有一定提升，可採用調參後的最佳參數。")

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    res.to_csv(os.path.join(out_dir, "ml_tuning_results.csv"), index=False, encoding="utf-8-sig")

    # 存調參後最佳模型
    tuned_dir = os.path.join(out_dir, "reid_models_tuned")
    os.makedirs(tuned_dir, exist_ok=True)
    for name, est in tuned_models.items():
        joblib.dump({"pipeline": est, "features": ML_FEATURES}, os.path.join(tuned_dir, f"reid_model_{name}_tuned.joblib"))

    print(f"\n✅ 調參結果已存: {os.path.join(out_dir, 'ml_tuning_results.csv')}")
    print(f"✅ 調參後最佳模型已存: {tuned_dir}")
    print("=" * 74)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\reid_features_merged.csv"
    main(csv_path)
