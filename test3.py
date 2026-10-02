import pandas as pd, numpy as np
from sklearn.metrics import f1_score
d = pd.read_csv(r"D:\CARLA_Experiments\reid_features_merged.csv")
ths = np.unique(d["f3_mean_dist"])
best = max(ths, key=lambda t: f1_score(d["label"], (d["f3_mean_dist"] < t).astype(int)))
d["pred"] = (d["f3_mean_dist"] < best).astype(int)
err = d[d["pred"] != d["label"]]
print("最佳距離門檻 %.2f m，判錯 %d / %d 對" % (best, len(err), len(d)))
print("label=0 → 不同人被判成同一人 (FP)；label=1 → 同一人被判成不同人 (FN)")
print(err.groupby(["scenario", "label"] if "scenario" in d else ["label"]).size())
print(err[["source", "pair_id", "label", "f3_mean_dist", "f4_dir_consistency",
           "f2_speed_diff", "n_points"]].round(2).to_string(index=False))